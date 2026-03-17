import importlib
import importlib.metadata as importlib_metadata
import logging
import os
import re
import statistics
from typing import Literal, Optional

import nltk
import requests
import torch
from nltk import word_tokenize
from nltk.translate import meteor_score
from packaging import version
from sacrebleu import sentence_bleu, sentence_chrf, sentence_ter
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lighteval.metrics.imports.bert_scorer import BERTScorer
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import BertScore, JudgeLLM, SampleLevelComputation
from lighteval.metrics.normalizations import remove_braces, remove_braces_and_strip
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SampleLevelMetricGrouping, SamplingMethod
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.multilingual.tasks.swiss_legal.prompts import (
    SLDS_JUDGE_ONE_SHOT_EXAMPLE_DE,
    SLDS_JUDGE_ONE_SHOT_EXAMPLE_FR,
    SLDS_JUDGE_ONE_SHOT_EXAMPLE_IT,
    SLDS_JUDGE_SYSTEM_PROMPT,
    SLDS_JUDGE_USER_PROMPT,
    SWISS_LEGAL_TRANSLATION_JUDGE_FEW_SHOT_EXAMPLES,
    SWISS_LEGAL_TRANSLATION_JUDGE_INSTRUCTION,
    SWISS_LEGAL_TRANSLATION_JUDGE_SYSTEM_PROMPT,
    SWISS_LEGAL_TRANSLATION_JUDGE_USER_PROMPT,
)
from lighteval.tasks.requests import Doc


logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# COMET (unbabel-comet) is temporarily unavailable in this task:
# lighteval requires numpy>=2 while stable unbabel-comet releases require numpy<2.
_DISABLED_COMET_METRICS = {"wmt22-comet-da", "xcomet_xl", "xcomet_xxl"}

if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if torch.cuda.get_device_capability()[0] >= 7:
        torch.set_float32_matmul_precision("medium")


def _load_comet():
    try:
        module = importlib.import_module("comet")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "COMET metric requires optional dependency `unbabel-comet`. "
            "Install lighteval with multilingual extras or add `unbabel-comet`."
        ) from exc
    return module.download_model, module.load_from_checkpoint


def _load_gemba():
    try:
        module = importlib.import_module("gemba")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "GEMBA metric requires optional dependency `gemba`. "
            "Install lighteval with multilingual extras or add `gemba`."
        ) from exc
    return module.get_gemba_scores


def process_judge_response_freeform_gpt(response: str) -> float:
    try:
        search = re.search(r"\[\[(\d.\d)\]\]", response)
        return float(search.group(1)) if search else 0.0
    except Exception as err:
        logger.warning("Error parsing judge response: %s", err)
        return 0.0


class BertScoreMultilingual(BertScore):
    def __init__(
        self, normalize_gold=None, normalize_pred=None, language=str, model_type=str, num_layers=int, device=str
    ):
        super().__init__(normalize_gold, normalize_pred)
        self.language = language
        self.model_type = model_type
        self.num_layers = num_layers
        self.device = device

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> dict[str, float]:
        # Make sure we load the correct bert_scorer before the parent class does
        if self.bert_scorer is None:
            self._init_bert_scorer()

        result = super().compute(model_response=model_response, doc=doc, **kwargs)

        # Multiply output by 100 for consistency
        return {k: v * 100 for k, v in result.items()}

    def _init_bert_scorer(self):
        language = self.language
        if language == "rm":
            language = "it"
            logger.warning("There is no BERTScore baseline file for Rumantsch, using Italian instead.")

        if self.device == "mps":
            raise ValueError("MPS is not supported for BERTScore")
        logger.info(
            f"Loading BERTScore with lang={language}, num_layers={self.num_layers}, model_type={self.model_type}, and device={device}..."
        )

        self.bert_scorer = BERTScorer(
            model_type=self.model_type,
            lang=language,  # Needs to be set if rescale_with_baseline is True
            num_layers=self.num_layers,  # Needs to be set if rescale_with_baseline is True
            rescale_with_baseline=True,
            baseline_path=None,
            device=self.device,
        )

        # Create directory structure if it doesn't exist
        os.makedirs(os.path.dirname(self.bert_scorer.baseline_path), exist_ok=True)

        # Download the baseline file if it doesn't exist
        if not os.path.exists(self.bert_scorer.baseline_path):
            raw_url = f"https://raw.githubusercontent.com/Tiiiger/bert_score/master/bert_score/rescale_baseline/{language}/{self.model_type}.tsv"
            logger.info(f"Downloading BERTScore baseline file from {raw_url}")
            response = requests.get(raw_url)
            if response.status_code == 200:
                with open(self.bert_scorer.baseline_path, "wb") as f:
                    f.write(response.content)
            else:
                raise RuntimeError(f"Failed to download baseline file from {raw_url}")


class GEMBA(SampleLevelComputation):
    def __init__(self, method: str = "GEMBA-MQM_norm", model: str = "gpt-4o"):
        self.method = method
        self.model = model
        self.name = f"{method.split('_')[0]}_{model}"

    def compute(
        self,
        responses: list[ModelResponse],
        docs: list[Doc],
        **kwargs,
    ) -> dict[str, float]:
        logger.info(f"Judging {len(docs)} samples with {self.name}...")
        source_langs = [doc.specific["source_lang"] for doc in docs]
        target_langs = [doc.specific["target_lang"] for doc in docs]

        # There should be only one language each in the batch
        assert len(set(source_langs)) == len(set(target_langs)) == 1
        sources = [doc.specific["source"] for doc in docs]
        predictions = [response.final_text for response in responses]

        get_gemba_scores = _load_gemba()
        answers, errors = get_gemba_scores(
            sources, predictions, source_langs[0], target_langs[0], method=self.method, model=self.model
        )

        # Handle cases where errors might be nan
        formatted_errors = []
        for error in errors:
            if isinstance(error, dict):
                # Convert defaultdict to dic
                formatted_errors.append([{key: value} for key, value in error.items()])
            else:
                formatted_errors.append([{"error": ["No error details available"]}])

        return [{self.name: answer, f"{self.name}_errors": error} for answer, error in zip(answers, formatted_errors)]


class BLEURT(SampleLevelComputation):
    def __init__(
        self,
        model_size: str = "tiny",
        seq_len: int = 512,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """Creates a BLEURT scorer based on the model size (tiny, base, large) and sequence length (128, 512)."""
        assert model_size in [
            "tiny",
            "base",
            "large",
        ], "Model size must be either tiny, base, or large"
        assert seq_len in [128, 512], "Sequence length must be either 128 or 512"
        if device == "mps":
            raise ValueError("MPS is not supported for BLEURT")

        self.metric_name = f"bleurt_{model_size}"
        self.model_size = model_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device

        # Lazy loading
        self.tokenizer = None
        self.model = None

    def _ensure_initialized(self):
        """Lazy initialization of model and tokenizer"""
        if self.tokenizer is None:
            logger.info(f"Loading BLEURT tokenizer {self.metric_name} lazily...")
            self.tokenizer = AutoTokenizer.from_pretrained(f"Elron/bleurt-{self.model_size}-{self.seq_len}")

        if self.model is None:
            logger.info(f"Loading BLEURT model {self.metric_name} lazily...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                f"Elron/bleurt-{self.model_size}-{self.seq_len}"
            )
            self.model = self.model.to(self.device)
            self.model.eval()

    def _process_batch(self, references: list[str], candidates: list[str]) -> list[float]:
        """Process a batch of references and candidates"""
        # Clean and prepare inputs
        references = [str(ref).strip() for ref in references]
        candidates = [str(cand).strip() for cand in candidates]

        # Tokenize
        inputs = self.tokenizer(
            references,
            candidates,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.seq_len,
        )

        # Log warning if any sequences were truncated
        if any(len(encoding) == self.seq_len for encoding in inputs["input_ids"]):
            logger.warning(f"Some inputs were truncated to max_length={self.seq_len} in BLEURT scoring")

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)[0]

        return outputs.squeeze().cpu().tolist()

    def compute(
        self,
        responses: list[ModelResponse],
        docs: list[Doc],
        **kwargs,
    ) -> dict[str, float]:
        """Compute BLEURT scores for a batch of translations"""
        self._ensure_initialized()

        logger.info(f"Scoring {len(docs)} samples with {self.metric_name}...")

        # Get references and predictions
        golds = [doc.get_golds()[0] for doc in docs]
        predictions = [response.final_text for response in responses]

        # Process in batches
        all_scores = []
        for i in tqdm(
            range(0, len(golds), self.batch_size),
            desc=f"Processing batches of size {self.batch_size} with {self.metric_name}",
        ):
            batch_refs = golds[i : i + self.batch_size]
            batch_preds = predictions[i : i + self.batch_size]
            try:
                scores = self._process_batch(batch_refs, batch_preds)
                all_scores.extend(scores if isinstance(scores, list) else [scores])
            except Exception as e:
                logger.error(f"Error processing batch {i}: {str(e)}")
                # Use minimum score for failed batches
                all_scores.extend([-1.0] * len(batch_refs))

        return [{self.metric_name: score * 100} for score in all_scores]


class COMET(SampleLevelComputation):
    def __init__(
        self,
        model_name: str = "Unbabel/wmt22-comet-da",
        batch_size: int = 8,
        gpus: int = 1,
        accelerator: str = "cpu",
    ):
        if accelerator == "mps":
            raise ValueError("MPS is not supported for COMET")

        self.metric_name = model_name.split("/")[-1]
        self.model = None  # Lazy loading of the model
        self.model_name = model_name
        self.batch_size = batch_size
        self.gpus = gpus
        self.accelerator = accelerator

    def compute(
        self,
        responses: list[ModelResponse],
        docs: list[Doc],
        **kwargs,
    ) -> dict[str, float]:
        # Only load the model here to save memory and time
        if self.model is None:
            logger.info(f"Loading COMET model {self.model_name} lazily...")
            download_model, load_from_checkpoint = _load_comet()
            self.model = load_from_checkpoint(download_model(self.model_name))

        logger.info(f"Scoring {len(docs)} samples with {self.metric_name}...")
        golds = [doc.get_golds()[0] for doc in docs]
        predictions = [response.final_text[0] for response in responses]
        sources = [doc.specific["source"] for doc in docs]

        data = [
            {"src": src, "mt": pred if isinstance(pred, str) else pred[0], "ref": gold}
            for src, pred, gold in zip(sources, predictions, golds)
        ]
        model_output = self.model.predict(
            data,
            batch_size=self.batch_size,
            gpus=self.gpus,
            accelerator=self.accelerator,
        )

        return [{self.metric_name: score * 100} for score in model_output["scores"]]


class METEOR(SampleLevelComputation):
    def __init__(self, alpha=0.9, beta=3, gamma=0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        NLTK_VERSION = version.parse(importlib_metadata.version("nltk"))
        assert NLTK_VERSION >= version.Version("3.9.0"), "NLTK version must be >= 3.9.0"

        nltk.download("punkt_tab", quiet=True)
        nltk.download("wordnet", quiet=True)

    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> float:
        """
        Compute METEOR score for a single prediction against its reference(s).
        """
        golds = doc.get_golds()
        prediction = model_response.final_text[0]

        if len(golds) > 1:  # multiple references
            score = meteor_score.meteor_score(
                [word_tokenize(gold) for gold in golds],
                word_tokenize(prediction),
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )
        else:
            score = meteor_score.single_meteor_score(
                word_tokenize(golds[0]),
                word_tokenize(prediction),
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )

        return score * 100


class BLEU(SampleLevelComputation):
    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> float:
        """
        Compute BLEU score for a single prediction against its reference.
        """
        # Get the first (and typically only) gold and prediction
        gold = doc.get_golds()[0]
        prediction = model_response.final_text[0]  # Get first prediction

        score = sentence_bleu(prediction, [gold]).score
        return score * 100


class CHRF(SampleLevelComputation):
    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> float:
        """
        Compute chrF score for a single prediction against its reference.
        """
        # Get the first (and typically only) gold and prediction
        gold = doc.get_golds()[0]
        prediction = model_response.final_text[0]  # Get first prediction

        score = sentence_chrf(prediction, [gold]).score
        return score * 100


class TER(SampleLevelComputation):
    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> float:
        """
        Compute TER score for a single prediction against its reference.
        """
        # Get the first (and typically only) gold and prediction
        gold = doc.get_golds()[0]
        prediction = model_response.final_text[0]  # Get first prediction

        score = sentence_ter(prediction, [gold]).score
        return score * 100


class JudgeSwissLegalTranslation(JudgeLLM):
    def compute(
        self,
        responses: list[ModelResponse],
        docs: list[Doc],
        **kwargs,
    ) -> dict[str, float]:
        logger.info(f"Judging {len(docs)} samples with {self.short_judge_name}...")
        questions = [doc.specific["source"] for doc in docs]
        options = [doc.choices for doc in docs]
        golds = [doc.get_golds()[0] for doc in docs]
        predictions = [response.text[0] for response in responses]

        scores, _, judgements = self.judge.evaluate_answer_batch(questions, predictions, options, golds)
        # Exclude the messages (user prompt) because they are too long
        return [
            {
                self.short_judge_name: score * 100,
            }
            for score, judgment in zip(scores, judgements)
        ]


class JudgeSwissLandmarkDecisionSummarization(JudgeLLM):
    SCORE_EXTRACTION_PATTERN = r"^\s*([A-Z_]+_SCORE):\s*(\d+)\s*$"
    RUBRIC_NAMES = (
        "ACCURACY_FAITHFULNESS_SCORE",
        "COMPLETENESS_RELEVANCE_SCORE",
        "CLARITY_COHERENCE_SCORE",
        "ARTICLES_SCORE",
        "CONSIDERATIONS_SCORE",
    )

    def __init__(
        self,
        language: Literal["de", "fr", "it"],
        **kwargs,
    ):
        self.language = language

        super().__init__(template=self._template, process_judge_response=self._process_judge_response, **kwargs)

    def _template(
        self,
        question: str,
        answer: str,
        options: Optional[list[str]] = None,
        gold: Optional[list[str]] = None,
    ) -> list[dict[str, str]]:
        """Template for evaluating the Swiss Landmark Decision Summarization task based only on the original and the generated headnotes."""

        # Remove landmark and trailing whitespaces
        system_prompt = SLDS_JUDGE_SYSTEM_PROMPT.strip()
        user_prompt = SLDS_JUDGE_USER_PROMPT.strip()

        if self.language == "de":
            one_shot_example = SLDS_JUDGE_ONE_SHOT_EXAMPLE_DE.strip()
        elif self.language == "fr":
            one_shot_example = SLDS_JUDGE_ONE_SHOT_EXAMPLE_FR.strip()
        elif self.language == "it":
            one_shot_example = SLDS_JUDGE_ONE_SHOT_EXAMPLE_IT.strip()

        # Fill template with original and generated headnote
        user_prompt = user_prompt.format(
            original_headnote=gold,
            generated_headnote=answer,
            one_shot_example=one_shot_example,
        )

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def _process_judge_response(self, response: str) -> float:
        """Process the judge responses and extract the scores for each category."""
        sample_scores = re.findall(pattern=self.SCORE_EXTRACTION_PATTERN, string=response, flags=re.MULTILINE)

        if len(sample_scores) != 5:
            logger.warning("Could only extract %d out of 5 scores from the response: %s", len(sample_scores), response)

        aggregated_score = 0
        for metric_name, score in sample_scores:
            if metric_name not in self.RUBRIC_NAMES:
                logger.warning("Invalid metric name: %s", metric_name)
                continue

            # Transform scale from 1-3 to 0-2
            aggregated_score += int(score) - 1

        # Divide be the maximum possible score
        aggregated_score /= len(self.RUBRIC_NAMES) * 2

        return aggregated_score

    def compute(
        self,
        responses: list[ModelResponse],
        docs: list[Doc],
        **kwargs,
    ) -> list[dict]:
        logger.info(f"Judging {len(docs)} samples with {self.short_judge_name}...")

        not_considered = [None for _ in docs]
        original_headnotes = [doc.get_golds()[0] for doc in docs]
        generated_headnotes = [response.text[0] for response in responses]

        # Exclude the messages (user prompt) because they are too long
        scores, _, judgements = self.judge.evaluate_answer_batch(
            questions=not_considered, answers=generated_headnotes, options=not_considered, golds=original_headnotes
        )
        return [
            {
                self.short_judge_name: score * 100,
            }
            for score, judgment in zip(scores, judgements)
        ]


def get_bert_score(
    language: str,
    num_layers: int = 24,
    model_type: str = "xlm-roberta-large",
    device: str = "cpu",
    metric_category: SamplingMethod = SamplingMethod.GENERATIVE,
):
    return SampleLevelMetricGrouping(
        metric_name=["BERTScore-P", "BERTScore-R", "BERTScore-F"],
        higher_is_better={
            "BERTScore-P": True,
            "BERTScore-R": True,
            "BERTScore-F": True,
        },
        category=metric_category,
        sample_level_fn=BertScoreMultilingual(
            normalize_gold=remove_braces,
            normalize_pred=remove_braces_and_strip,
            language=language,
            model_type=model_type,
            num_layers=num_layers,
            device=device,
        ),
        corpus_level_fn={
            "BERTScore-P": statistics.mean,
            "BERTScore-R": statistics.mean,
            "BERTScore-F": statistics.mean,
        },
        batched_compute=False,
    )


def get_swiss_legal_translation_judge(
    judge_model_name: str = "openai/gpt-4o-2024-11-20",
    short_judge_name: str = "slt_judge_gpt-4o",
    backend: Literal["litellm", "openai", "transformers", "vllm", "tgi", "inference-providers"] = "litellm",
    system_style: Literal["basic", "detailed", "codebook"] = "basic",
    few_shot_style: Literal["diverse", "single"] = "diverse",
    judgment_style: Literal["absolute", "deduction"] = "absolute",
):
    if system_style == "codebook" and judgment_style == "absolute":
        raise ValueError("The codebook system style can only be used with the deduction judgment style.")
    if system_style in ("basic", "detailed") and judgment_style == "deduction":
        raise ValueError(f"The {system_style} can only be used with the absolute judgment style.")

    def swiss_legal_translation_judge(question, options, answer, gold):
        system_prompt = SWISS_LEGAL_TRANSLATION_JUDGE_SYSTEM_PROMPT[system_style]
        user = SWISS_LEGAL_TRANSLATION_JUDGE_USER_PROMPT[system_style]
        few_shot_examples = SWISS_LEGAL_TRANSLATION_JUDGE_FEW_SHOT_EXAMPLES[f"{few_shot_style}_{judgment_style}"]
        instruction = SWISS_LEGAL_TRANSLATION_JUDGE_INSTRUCTION.format(question=question, gold=gold, answer=answer)
        user_prompt = user + few_shot_examples + instruction

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    return SampleLevelMetricGrouping(
        metric_name=[short_judge_name],
        higher_is_better={short_judge_name: True},
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=JudgeSwissLegalTranslation(
            judge_model_name=judge_model_name,
            template=swiss_legal_translation_judge,
            process_judge_response=process_judge_response_freeform_gpt,
            judge_backend=backend,
            short_judge_name=short_judge_name,
        ),
        corpus_level_fn={short_judge_name: statistics.mean},
        batched_compute=True,
    )


def get_swiss_landmark_decision_summarization_judge(
    language: Literal["de", "fr", "it"],
    model_name: str = "openrouter/deepseek/deepseek-chat",
    short_judge_name: str = "slds_judge_deepseek_v3",
    backend: str = "litellm",
):
    judge = JudgeSwissLandmarkDecisionSummarization(
        judge_model_name=model_name,
        judge_backend=backend,
        short_judge_name=short_judge_name,
        language=language,
    )

    judge.judge.API_MAX_RETRY = 60

    return SampleLevelMetricGrouping(
        metric_name=[short_judge_name],
        higher_is_better={short_judge_name: True},
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=judge,
        corpus_level_fn={short_judge_name: statistics.mean},
        batched_compute=True,
    )


def get_gemba_judge(method: str = "GEMBA-MQM_norm", model: str = "gpt-4o"):
    name = f"{method.split('_')[0]}_{model}"
    return SampleLevelMetricGrouping(
        metric_name=[name],
        higher_is_better={name: True},
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=GEMBA(method=method, model=model),
        corpus_level_fn={name: statistics.mean},
        batched_compute=True,
    )


def get_bleurt(
    model_size: str = "tiny",
    seq_len: int = 512,
    batch_size: int = 32,
    device: str = "cpu",
):
    logger.info(
        f"Loading BLEURT with model_size={model_size}, seq_len={seq_len}, batch_size={batch_size}, and device={device}..."
    )
    name = f"bleurt_{model_size}"
    return SampleLevelMetricGrouping(
        metric_name=[name],
        higher_is_better={name: True},
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=BLEURT(model_size=model_size, seq_len=seq_len, batch_size=batch_size, device=device),
        corpus_level_fn={name: statistics.mean},
        batched_compute=True,
    )


def get_comet(
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
    gpus: int = 1,
    device: str = "cpu",
):
    logger.info(
        f"Loading COMET with model_name={model_name}, batch_size={batch_size}, gpus={gpus}, and device={device}..."
    )
    name = model_name.split("/")[-1]
    return SampleLevelMetricGrouping(
        metric_name=[name],
        higher_is_better={name: True},
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=COMET(
            model_name=model_name,
            batch_size=batch_size,
            gpus=gpus,
            accelerator=device,
        ),
        corpus_level_fn={name: statistics.mean},
        batched_compute=True,
    )


def get_meteor(
    metric_category: SamplingMethod = SamplingMethod.GENERATIVE,
):
    return SampleLevelMetric(
        metric_name="meteor",
        higher_is_better=True,
        category=metric_category,
        sample_level_fn=METEOR(),
        corpus_level_fn=statistics.mean,
    )


def get_bleu_sentence():
    return SampleLevelMetric(
        metric_name="bleu_sentence",
        higher_is_better=True,
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=BLEU(),
        corpus_level_fn=statistics.mean,
    )


def get_chrf_sentence():
    return SampleLevelMetric(
        metric_name="chrf_sentence",
        higher_is_better=True,
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=CHRF(),
        corpus_level_fn=statistics.mean,
    )


def get_ter_sentence():
    return SampleLevelMetric(
        metric_name="ter_sentence",
        higher_is_better=False,
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=TER(),
        corpus_level_fn=statistics.mean,
    )


def get_extractiveness(language: Literal["de", "fr", "it"]) -> SampleLevelMetricGrouping:
    if language == "de":
        return Metrics.extractiveness_de
    if language == "fr":
        return Metrics.extractiveness_fr
    if language == "it":
        return Metrics.extractiveness_it

    raise ValueError(f"Unsupported language for extractiveness metric: {language}")


JUDGE_MODELS = {
    "o1": "openai/o1-2024-12-17",
    "o1-mini": "openai/o1-mini-2024-09-12",
    "gpt-4o-mini": "openai/gpt-4o-mini-2024-07-18",
    "gpt-4o": "openai/gpt-4o-2024-11-20",
    # The Gemini models are not very good judges.
    "gemini-1-5-flash": "gemini/gemini-1.5-flash-002",
    "gemini-1-5-pro": "gemini/gemini-1.5-pro-002",
    # The Claude models do not follow the required output format.
    # "claude-3-5-haiku": "anthropic/claude-3-5-haiku-20241022",
    # "claude-3-5-sonnet": "anthropic/claude-3-5-sonnet-20241022",
    "llama-3-3-70b": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "llama-3-1-405b": "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
}

LEXICAL_METRICS = [
    "bleu",
    "rouge1",
    "rouge2",
    "rougeL",
    "chrf",
    "bleu_sentence",
    "chrf_sentence",
    "ter_sentence",
    "meteor",
]

GPU_METRICS = [
    "bert_score",
    "bleurt_large",
    # "xcomet_xxl",  # Disabled: lighteval (numpy>=2) conflicts with stable unbabel-comet (numpy<2).
]

API_METRICS = [
    "gemba_mqm_gpt_4o",
    "slt_judge_gemini_1_5_flash_codebook_diverse_deduction",
]

JUDGE_METRICS = [
    f"slt_judge_{judge_model}-{system_style}-{few_shot_style}-{judgment_style}".replace("-", "_")
    for judge_model in JUDGE_MODELS
    for few_shot_style in ["diverse", "single"]
    for system_style, judgment_style in [
        ("basic", "absolute"),
        ("detailed", "absolute"),
        ("codebook", "deduction"),
        # Make sure that the codebook system style is used with the deduction judgment style
        # and the basic and detailed system styles are used with the absolute judgment style
    ]
]

metrics_to_evaluate = ["judge"]

METRICS_TO_USE = []
if metrics_to_evaluate == ["debug"]:
    METRICS_TO_USE = ["bleu"]
elif "lexical" in metrics_to_evaluate:
    METRICS_TO_USE += LEXICAL_METRICS
elif "gpu" in metrics_to_evaluate:
    METRICS_TO_USE += GPU_METRICS
elif "api" in metrics_to_evaluate:
    METRICS_TO_USE += API_METRICS
elif "judge" in metrics_to_evaluate:
    METRICS_TO_USE += JUDGE_METRICS
else:
    METRICS_TO_USE = LEXICAL_METRICS + GPU_METRICS + API_METRICS


logger.info(f"Available metrics: {METRICS_TO_USE}")

METRICS = {}


def init_lexical_metric(metric_name: str):  # noqa: C901
    # Corpus level metrics
    if metric_name == "bleu":
        METRICS["bleu"] = Metrics.bleu
    if metric_name == "rouge1":
        METRICS["rouge1"] = Metrics.rouge1
    if metric_name == "rouge2":
        METRICS["rouge2"] = Metrics.rouge2
    if metric_name == "rougeL":
        METRICS["rougeL"] = Metrics.rougeL
    if metric_name == "chrf":
        METRICS["chrf"] = Metrics.chrf
    if metric_name == "ter":
        # TER often hangs for a while and takes more than 10 minutes to compute
        METRICS["ter"] = Metrics.ter
    # Sample level metrics
    if metric_name == "bleu_sentence":
        METRICS["bleu_sentence"] = get_bleu_sentence()
    if metric_name == "chrf_sentence":
        METRICS["chrf_sentence"] = get_chrf_sentence()
    if metric_name == "ter_sentence":
        METRICS["ter_sentence"] = get_ter_sentence()
    if metric_name == "meteor":
        METRICS["meteor"] = get_meteor()


def init_model_based_metric(metric_name: str):
    if metric_name == "bert_score":
        METRICS["bert_score"] = {  # Create BERTScore metrics for each language
            lang: get_bert_score(language=lang, model_type="xlm-roberta-large", device=device)
            for lang in ["de", "fr", "it", "rm", "en"]
        }
    if metric_name == "bleurt_tiny":
        METRICS["bleurt_tiny"] = get_bleurt(model_size="tiny", seq_len=512, batch_size=256, device=device)
    if metric_name == "bleurt_base":
        METRICS["bleurt_base"] = get_bleurt(model_size="base", seq_len=512, batch_size=256, device=device)
    if metric_name == "bleurt_large":
        METRICS["bleurt_large"] = get_bleurt(model_size="large", seq_len=512, batch_size=256, device=device)
    # COMET metrics are intentionally disabled for now because lighteval depends on
    # numpy>=2, while stable unbabel-comet releases currently require numpy<2.
    if metric_name in _DISABLED_COMET_METRICS:
        logger.warning(
            "Skipping metric '%s': currently unavailable due to dependency conflicts between "
            "lighteval and unbabel-comet (numpy version incompatibility).",
            metric_name,
        )


def init_llm_judge_metric(metric_name: str):
    if metric_name == "gemba_mqm_gpt_4o":
        METRICS["gemba_mqm_gpt_4o"] = get_gemba_judge(method="GEMBA-MQM_norm", model="gpt-4o")

    if metric_name == "slt_judge_gpt_4o":
        METRICS["slt_judge_gpt_4o"] = get_swiss_legal_translation_judge(
            judge_model_name="openai/gpt-4o-2024-11-20",
            short_judge_name="slt_judge_gpt-4o",
        )

    # Check all the judge metric combinations
    for judge_model in JUDGE_MODELS:
        for few_shot_style in ("diverse", "single"):
            for system_style, judgment_style in (
                ("basic", "absolute"),
                ("detailed", "absolute"),
                ("codebook", "deduction"),
            ):
                short_judge_name = f"slt_judge_{judge_model}-{system_style}-{few_shot_style}-{judgment_style}"
                judge_metric_name = short_judge_name.replace("-", "_")
                if metric_name == judge_metric_name:
                    METRICS[metric_name] = get_swiss_legal_translation_judge(
                        judge_model_name=JUDGE_MODELS[judge_model],
                        short_judge_name=short_judge_name,
                        system_style=system_style,
                        few_shot_style=few_shot_style,
                        judgment_style=judgment_style,
                    )
                    break


def init_metric(metric_name: str):
    # Only load the metric once
    if metric_name in METRICS:
        logger.debug(f"Metric {metric_name} already initialized")
        return

    # ===== Lexical metrics =====
    init_lexical_metric(metric_name)
    # ===== Model-based metrics =====
    init_model_based_metric(metric_name)
    # ===== LLM Judge metrics =====
    init_llm_judge_metric(metric_name)


def get_metrics(METRICS_TO_USE, target_lang: str, generation_size: int):
    metrics = []
    for metric in METRICS_TO_USE:
        if metric in _DISABLED_COMET_METRICS:
            logger.warning(
                "Skipping metric '%s': currently unavailable due to dependency conflicts between "
                "lighteval and unbabel-comet (numpy version incompatibility).",
                metric,
            )
            continue

        # These metrics are sentence level metrics and we only want to use them for generation sizes up to 512.
        short_metrics = [
            "bleu_sentence",
            "chrf_sentence",
            "ter_sentence",
            "bert_score",
            "bleurt_tiny",
            "bleurt_base",
            "bleurt_large",
            # "wmt22-comet-da",
            # "xcomet_xl",
            # "xcomet_xxl",
        ]
        if generation_size > 512 and metric in short_metrics:
            logger.debug(
                f"Skipping {metric} for generation size {generation_size} because the maximum supported sequence length is 512."
            )
            continue

        init_metric(metric)
        if metric == "bert_score":
            # Add only the BERTScore for the target language
            metrics.append(METRICS["bert_score"][target_lang])
        else:
            metrics.append(METRICS[metric])
    return metrics
