# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
This module contains task configurations and prompt functions for evaluating
LLM models on Swiss legal datasets. Each task is defined using the
`LightevalTaskConfig` class with its respective prompt function. The tasks
cover a variety of benchmarks, including: translation of laws, court decisions
and press releases.

Author: Joel Niklaus
"""

import importlib.metadata as importlib_metadata
import statistics
from dataclasses import dataclass

import nltk
from comet import download_model, load_from_checkpoint
from nltk import word_tokenize
from nltk.translate import meteor_score
from packaging import version
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lighteval.logging.hierarchical_logger import hlog_warn
from lighteval.metrics.imports.bert_scorer import BERTScorer
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import BertScore, JudgeLLMMixEval
from lighteval.metrics.normalizations import remove_braces, remove_braces_and_strip
from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric,
    SampleLevelMetricGrouping,
)
from lighteval.tasks.extended.mix_eval.judge_prompts import (
    gpt_judge_for_closeended_freeform,
)
from lighteval.tasks.extended.mix_eval.main import process_judge_response_freeform_gpt
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# CUSTOM METRICS
# TODO: adjust prompt


def freeform_gpt_judge(judge_model_name: str = "gpt-4o"):
    return SampleLevelMetricGrouping(
        metric_name=[f"llm_judge_{judge_model_name}"],
        higher_is_better={"judge_score_{judge_model_name}": True},
        category=MetricCategory.LLM_AS_JUDGE,
        use_case=MetricUseCase.SUMMARIZATION,
        sample_level_fn=JudgeLLMMixEval(
            judge_model_name=judge_model_name,
            template=gpt_judge_for_closeended_freeform,
            process_judge_response=process_judge_response_freeform_gpt,
            judge_backend="openai",
            short_judge_name=judge_model_name,
        ).compute,
        corpus_level_fn={
            f"judge_score_{judge_model_name}": statistics.mean,
        },
    )


def get_bert_score(model_type: str = "xlm-roberta-large"):
    score = BertScore(normalize_gold=remove_braces, normalize_pred=remove_braces_and_strip)
    score.bert_scorer = BERTScorer(
        # We could download the files from here and set the baseline_path ourselves:
        # https://github.com/Tiiiger/bert_score/tree/master/bert_score/rescale_baseline
        model_type=model_type,
        lang=None,  # Needs to be set if rescale_with_baseline is True
        rescale_with_baseline=False,
        baseline_path=None,
    )
    return SampleLevelMetricGrouping(
        metric_name=["BERTScore-P", "BERTScore-R", "BERTScore-F"],
        sample_level_fn=score.compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.SUMMARIZATION,
        corpus_level_fn={
            "BERTScore-P": statistics.mean,
            "BERTScore-R": statistics.mean,
            "BERTScore-F": statistics.mean,
        },
        higher_is_better={
            "BERTScore-P": True,
            "BERTScore-R": True,
            "BERTScore-F": True,
        },
    )


bert_score = get_bert_score(model_type="xlm-roberta-large")


class BLEURT:
    def __init__(self, model_size: str = "tiny", seq_len: int = 512):
        """Creates a BLEURT scorer based on the model size (tiny, base, large) and sequence length (128, 512)."""
        assert model_size in [
            "tiny",
            "base",
            "large",
        ], "Model size must be either tiny, base, or large"
        assert seq_len in [128, 512], "Sequence length must be either 128 or 512"

        self.tokenizer = AutoTokenizer.from_pretrained(f"Elron/bleurt-{model_size}-{seq_len}")
        self.model = AutoModelForSequenceClassification.from_pretrained(f"Elron/bleurt-{model_size}-{seq_len}")
        self.max_length = seq_len
        self.model.eval()

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        """Uses the stored BLEURT scorer to compute the score on the current sample.

        Args:
            golds (list[str]): Reference targets
            predictions (list[str]): Predicted strings

        Returns:
            float: Score over the current sample's items.
        """
        if len(predictions) == 1:
            predictions = predictions * len(golds)
        inputs = self.tokenizer(
            golds,
            predictions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        if any(len(encoding) == self.max_length for encoding in inputs["input_ids"]):
            hlog_warn(f"Some inputs were truncated to max_length={self.max_length} in BLEURT scoring")
        scores = self.model(**inputs)[0].squeeze()
        return scores.item()


bleurt = SampleLevelMetric(
    metric_name="bleurt",
    sample_level_fn=BLEURT(model_size="tiny", seq_len=512).compute,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=statistics.mean,
    higher_is_better=True,
)


class COMET:
    def __init__(
        self,
        model_name: str = "Unbabel/wmt22-comet-da",
        batch_size: int = 1,
        gpus: int = 1,
        accelerator: str = "cpu",  # "mps" leads to errors
    ):
        model_path = download_model(model_name)
        self.model = load_from_checkpoint(model_path)
        self.batch_size = batch_size
        self.gpus = gpus
        self.accelerator = accelerator

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        data = [
            {"src": src, "mt": pred, "ref": gold}
            for src, pred, gold in zip(
                [kwargs["formatted_doc"].specific["source"]] * len(predictions),
                predictions,
                golds,
            )
        ]
        model_output = self.model.predict(
            data,
            batch_size=self.batch_size,
            gpus=self.gpus,
            accelerator=self.accelerator,
        )
        # model_output["scores"] contains the sentence level scores
        return model_output["system_score"]


comet = SampleLevelMetric(
    metric_name="comet",
    sample_level_fn=COMET(accelerator="cpu").compute,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=statistics.mean,
    higher_is_better=True,
)


class METEOR:
    def __init__(self, alpha=0.9, beta=3, gamma=0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        NLTK_VERSION = version.parse(importlib_metadata.version("nltk"))
        assert NLTK_VERSION >= version.Version("3.9.0"), "NLTK version must be >= 3.9.0"
        nltk.download("punkt_tab")
        nltk.download("wordnet")

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        if isinstance(golds[0], list):  # multiple references
            scores = [
                meteor_score.meteor_score(
                    [word_tokenize(ref) for ref in refs],
                    word_tokenize(pred),
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                )
                for refs, pred in zip(golds, predictions)
            ]
        else:
            scores = [
                meteor_score.single_meteor_score(
                    word_tokenize(ref),
                    word_tokenize(pred),
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                )
                for ref, pred in zip(golds, predictions)
            ]

        return statistics.mean(scores)


meteor = SampleLevelMetric(
    metric_name="meteor",
    sample_level_fn=METEOR().compute,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=statistics.mean,
    higher_is_better=True,
)

# EVALS WITH SUBSET
# This is how you create a subset task (like MMLU), which has several subset
# each being its own evaluation task.


def create_translation_pairs(langs_list: list) -> list[tuple]:
    """
    Create all possible translation pairs from a given list of languages.

    Args:
    langs_list (list): A list of languages.

    Returns:
    lang_pairs_list (list): A list of tuples representing a translation pair.
    """
    lang_pairs_list = []
    for i, lang1 in enumerate(langs_list):
        for lang2 in langs_list[i + 1 :]:
            lang_pairs_list.append((lang1, lang2))
            lang_pairs_list.append((lang2, lang1))
    return lang_pairs_list


@dataclass
class LevelConfig:
    name: str
    text_col_name: str
    metadata_cols: list[str]


@dataclass
class DatasetConfig:
    name: str
    hf_repo: str
    languages: list[str]
    subsets: dict[str, LevelConfig]

    def __post_init__(self):
        self.translation_pairs = create_translation_pairs(self.languages)


# Translation of Swiss Federal Supreme Court Decision Summaries on three levels: the entire decision, the regeste level and the text level.
SwissDecisionSummaryTranslations = DatasetConfig(
    name="sdst",
    hf_repo="joelniklaus/SwissDecisionSummaryTranslations",
    languages=["de", "fr", "it"],
    subsets={
        "bge_level": LevelConfig(
            name="bge_level",
            text_col_name="bgeText",
            metadata_cols=["bge"],
        ),
        "regeste_level": LevelConfig(
            name="regeste_level",
            text_col_name="regesteText",
            metadata_cols=["bge"],
        ),
        "text_level": LevelConfig(
            name="text_level",
            text_col_name="text",
            metadata_cols=["bge"],
        ),
    },
)

# Translation of Swiss Federal Laws on three levels: the entire law, the article level and the paragraph level.
SwissLawTranslations = DatasetConfig(
    name="slt",
    hf_repo="joelniklaus/SwissLawTranslations",
    languages=["de", "fr", "it", "rm", "en"],
    subsets={
        "law_level": LevelConfig(
            name="law_level",
            text_col_name="lawText",
            metadata_cols=["abbreviation", "url", "dateApplicability", "rsNr"],
        ),
        "article_level": LevelConfig(
            name="article_level",
            text_col_name="articleText",
            metadata_cols=["abbreviation", "url", "dateApplicability", "rsNr"],
        ),
        "paragraph_level": LevelConfig(
            name="paragraph_level",
            text_col_name="paragraphText",
            metadata_cols=["abbreviation", "url", "dateApplicability", "rsNr"],
        ),
    },
)

# Translation of Swiss Federal Supreme Court Press Releases on one level: the entire press release.
SwissSupremeCourtPressReleaseTranslations = DatasetConfig(
    name="sscprt",
    hf_repo="joelniklaus/SwissSupremeCourtPressReleaseTranslations",
    languages=["de", "fr", "it"],
    subsets={
        "press_release": LevelConfig(
            name="press_release",
            text_col_name="text",
            metadata_cols=["filename"],
        )
    },
)


def create_prompt_fn(level_config: LevelConfig, src_lang: str, target_lang: str):
    """
    Create a prompt function for a given level configuration.
    """
    text_col = level_config.text_col_name
    src_text_col = f"{src_lang}_{text_col}"
    target_text_col = f"{target_lang}_{text_col}"

    def prompt_fn(line: dict, task_name: str = None):
        # Following Template A from https://github.com/huggingface/lighteval/pull/389#issuecomment-2471580177
        custom_query = f"{src_lang.upper()}: {line[src_text_col]}\n{target_lang.upper()}:"

        return Doc(
            task_name=task_name,
            query=custom_query,
            choices=[str(line[target_text_col])],
            gold_index=0,
            specific={
                **{col: line[col] for col in level_config.metadata_cols},
                "question": custom_query,
                "source": line[src_text_col],
            },
        )

    return prompt_fn


class TranslationTask(LightevalTaskConfig):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        level_name: str,
        src_lang: str,
        target_lang: str,
    ):
        super().__init__(
            name=f"{dataset_config.name}-{level_name}:{src_lang}-{target_lang}",
            suite=["community"],
            prompt_function=create_prompt_fn(dataset_config.subsets[level_name], src_lang, target_lang),
            hf_repo=dataset_config.hf_repo,
            hf_subset=level_name,
            hf_filter=None,
            hf_avail_splits=["train", "validation", "test"],
            evaluation_splits=["test"],  # ["validation", "test"],
            few_shots_split="validation",
            few_shots_select=None,
            generation_size=10,
            metric=[
                Metrics.bleu,
                Metrics.bleu_1,
                Metrics.bleu_4,
                Metrics.chrf,
                Metrics.ter,
                bert_score,
                bleurt,
                comet,
                meteor,
                freeform_gpt_judge(judge_model_name="gpt-4o"),
                # Additionally we could consider adding the following open source judge models:
                # flowaicom/Flow-Judge-v0.1, prometheus-eval/prometheus-7b-v2.0
                # However, these are only fine-tuned on English data and we need multilingual support.
            ],
            stop_sequence=["\n"],
            trust_dataset=True,
        )


# STORE YOUR EVALS

# list of all the subsets to use for this eval
DATASETS = [
    SwissDecisionSummaryTranslations,
    SwissLawTranslations,
    SwissSupremeCourtPressReleaseTranslations,
]

TASKS_TABLE = [
    TranslationTask(
        dataset_config=dataset,
        level_name=subset,
        src_lang=src_lang,
        target_lang=target_lang,
    )
    for dataset in DATASETS
    for subset in dataset.subsets
    for src_lang, target_lang in dataset.translation_pairs
]


# MODULE LOGIC
# You should not need to touch this
# Convert to dict for lighteval
if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
