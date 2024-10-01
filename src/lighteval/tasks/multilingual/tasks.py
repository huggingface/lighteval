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

from functools import partial

from langcodes import Language as LangCodeLanguage
from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    loglikelihood_acc_metric,
    multilingual_quasi_exact_match_metric,
    multilingual_quasi_f1_score_metric,
)
from lighteval.metrics.normalizations import LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import (
    agieval_prompt,
    alghafa_adapter,
    ceval_adapter,
    get_m3exam_adapter,
    sciqa_adapter,
    thai_exams_adapter,
)
from lighteval.tasks.multilingual.utils.task_utils import normalize_subset
from lighteval.tasks.templates.copa import get_copa_prompt_function
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.nli import get_nli_prompt_function
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language, iso_639_3_ind_to_iso_639_3_macro


TASKS_TABLE = []
# ------------------------------- NLI Tasks ------------------------------- #
# NLI (Natural Language Inference) tasks involve determining the logical relationship
# between two given sentences: a premise and a hypothesis. The goal is to classify
# whether the hypothesis is entailed by, contradicts, or is neutral with respect to
# the premise. After our inspection we found the neutral label to be quite ambiguous
# and decided to exclude it. But you can easily add it by modifying the adapters


# The XNLI dataset is a multilingual variant of MultiNLI
# https://aclanthology.org/D18-1269/
xnli_tasks = [
    LightevalTaskConfig(
        name=f"xnli_{language.value}_{formulation.name.lower()}",
        suite=["lighteval"],
        metric=[loglikelihood_acc_metric(normalization=LogProbTokenNorm())],
        prompt_function=get_nli_prompt_function(
            language=language,
            adapter=lambda line: {
                "premise": line["premise"],
                "hypothesis": line["hypothesis"],
                # Since we ignore the neutral label
                "gold_idx": {0: 0, 2: 1}[line["label"]],
            },
            relations=["entailment", "contradiction"],
            formulation=formulation,
        ),
        hf_filter=lambda line: line["label"] in [0, 2],
        hf_repo="facebook/xnli",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=["validation"],
        few_shots_split="train",
    )
    for language in [
        Language.ARABIC,
        Language.ENGLISH,
        Language.FRENCH,
        Language.SPANISH,
        Language.BULGARIAN,
        Language.GERMAN,
        Language.GREEK,
        Language.ENGLISH,
        Language.FRENCH,
        Language.HINDI,
        Language.RUSSIAN,
        Language.SWAHILI,
        Language.THAI,
        Language.TURKISH,
        Language.URDU,
        Language.VIETNAMESE,
        Language.CHINESE,
    ]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

# Improvement on XNLI with better translation, from our experience models tend to
# perform better on XNLI2.0 than XNLI
# https://arxiv.org/abs/2301.06527
xnli2_tasks = [
    LightevalTaskConfig(
        name=f"xnli2.0_{language.value}_{formulation.name.lower()}",
        suite=["lighteval"],
        metric=[loglikelihood_acc_metric(normalization=LogProbTokenNorm())],
        prompt_function=get_nli_prompt_function(
            language=language,
            adapter=lambda line: {
                "premise": line["premise"],
                "hypothesis": line["hypothesis"],
                # Since we ignore the neutral label
                "gold_idx": {0: 0, 2: 1}[line["label"]],
            },
            relations=["entailment", "contradiction"],
            formulation=formulation,
        ),
        hf_filter=lambda line: line["label"] in [0, 2],
        hf_repo=f"Harsit/xnli2.0_train_{LangCodeLanguage(standardize_tag(language.value)).language_name().lower()}",
        hf_subset="default",
        evaluation_splits=["train"],
    )
    for language in [
        Language.ENGLISH,
        Language.FRENCH,
        Language.PUNJABI,
        Language.GUJARATI,
        Language.KANNADA,
        Language.ASSAMESE,
        Language.BENGALI,
        Language.MARATHI,
        Language.SANSKRIT,
        Language.TAMIL,
        Language.GERMAN,
        Language.ENGLISH,
        Language.URDU,
        Language.VIETNAMESE,
        Language.TURKISH,
        Language.THAI,
        Language.SWAHILI,
        Language.SPANISH,
        Language.RUSSIAN,
        Language.HINDI,
        Language.GREEK,
        Language.CHINESE,
        Language.BULGARIAN,
        Language.ARABIC,
        # Theoretically also: Bhojpuri, Gujarati, Odiya
    ]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

# Another variant of XNLI, with emphasis on Indic languages
# https://arxiv.org/abs/2204.08776
xnli_indic_tasks = [
    LightevalTaskConfig(
        name=f"indicnxnli_{language.value}_{formulation.name.lower()}",
        suite=["lighteval"],
        prompt_function=get_nli_prompt_function(
            language=language,
            adapter=lambda line: {
                "premise": line["premise"],
                "hypothesis": line["hypothesis"],
                # Since we ignore the neutral label
                "gold_idx": {0: 0, 2: 1}[line["label"]],
            },
            relations=["entailment", "contradiction"],
            formulation=formulation,
        ),
        hf_repo="Divyanshu/indicxnli",
        hf_subset=standardize_tag(language.value),
        # Ignore neutral
        hf_filter=lambda x: int(x["label"]) in [0, 2],
        evaluation_splits=["validation"],
        few_shots_split="train",
        few_shots_select=None,
        generation_size=-1,
        metric=[
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
        ],
    )
    for language in [
        Language.ASSAMESE,
        Language.BENGALI,
        Language.GUJARATI,
        Language.HINDI,
        Language.KANNADA,
        Language.MALAYALAM,
        Language.MARATHI,
        Language.ORIYA,
        Language.PUNJABI,
        Language.TAMIL,
        Language.TELUGU,
    ]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

# PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification
# This dataset contains paraphrase identification pairs in multiple languages.
# It's derived from PAWS (Paraphrase Adversaries from Word Scrambling) and
# We treat paraphrase as entailment and non-paraphrase as contradiction
# https://arxiv.org/abs/1908.11828

paws_x_tasks = [
    LightevalTaskConfig(
        name=f"pawsx_{language.value}_{formulation.name.lower()}",
        suite=("lighteval",),
        prompt_function=get_nli_prompt_function(
            language=language,
            adapter=lambda line: {
                "premise": line["sentence1"],
                "hypothesis": line["sentence2"],
                # Since we ignore the neutral label
                "gold_idx": int(line["label"]),
            },
            relations=["entailment", "contradiction"],
            formulation=formulation,
        ),
        hf_repo="google-research-datasets/paws-x",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="train",
        metric=[
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
        ],
    )
    for language in [
        Language.GERMAN,
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.JAPANESE,
        Language.KOREAN,
        Language.CHINESE,
    ]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

# Russian Commitment Bank (RCB) is a large-scale NLI dataset with Russian sentences,
# collected from the web and crowdsourcing.
# https://arxiv.org/abs/2401.04531
rcb_tasks = [
    LightevalTaskConfig(
        name=f"rcb_{Language.RUSSIAN.value}_{formulation.name.lower()}",
        prompt_function=get_nli_prompt_function(
            language=Language.RUSSIAN,
            adapter=lambda line: {
                "premise": line["inputs"]["premise"],
                "hypothesis": line["inputs"]["hypothesis"],
                # Since we ignore the neutral label
                "gold_idx": int(line["outputs"]) - 1,
            },
            relations=["entailment", "contradiction"],
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="ai-forever/MERA",
        hf_subset="rcb",
        # Ignore neutral label
        hf_filter=lambda x: int(x["outputs"] or "0") in [1, 2],
        evaluation_splits=("train", "validation"),
        metric=[
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
        ],
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

# Native Chinese NLI dataset based.
# https://arxiv.org/pdf/2010.05444
# We find this benchmark to have really good signal compared to other Chinese NLI
ocnli_tasks = [
    LightevalTaskConfig(
        name=f"ocnli_{Language.CHINESE.value}_{formulation.name.lower()}",
        prompt_function=get_nli_prompt_function(
            language=Language.CHINESE,
            adapter=lambda line: {
                "premise": line["sentence1"],
                "hypothesis": line["sentence2"],
                # Since we ignore the neutral label
                "gold_idx": {1: 0, 2: 1}[line["label"]],
            },
            relations=["entailment", "contradiction"],
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="clue/clue",
        hf_subset="ocnli",
        # Only keep the positive and negative examples
        hf_filter=lambda x: int(x["label"]) in [1, 2],
        evaluation_splits=("validation",),
        few_shots_split="train",
        metric=[
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
        ],
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

# https://arxiv.org/abs/2004.05986
# Native Chinese NLI dataset based on MNLI approach (Machine Translated)
cmnli_tasks = [
    LightevalTaskConfig(
        name=f"cmnli_{Language.CHINESE.value}_{formulation.name.lower()}",
        prompt_function=get_nli_prompt_function(
            language=Language.CHINESE,
            adapter=lambda line: {
                "premise": line["sentence1"],
                "hypothesis": line["sentence2"],
                # Since we ignore the neutral label
                "gold_idx": {"entailment": 0, "contradiction": 1}[line["label"]],
            },
            relations=["entailment", "contradiction"],
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="fenffef/cmnli",
        hf_subset="default",
        hf_filter=lambda x: x["label"] in ["entailment", "contradiction"],
        # Only keep the positive and negative examples
        evaluation_splits=("validation",),
        few_shots_split="train",
        metric=[
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
        ],
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

TASKS_TABLE.extend(
    [*xnli_tasks, *xnli2_tasks, *xnli_indic_tasks, *paws_x_tasks, *rcb_tasks, *ocnli_tasks, *cmnli_tasks]
)
# ------------------------------- Copa Tasks ------------------------------- #
# COPA (Choice of Plausible Alternatives) tasks involve determining the most plausible cause or effect
# for a given premise. These tasks test common sense reasoning and causal inference abilities.

# XCOPA: Cross-lingual Choice of Plausible Alternatives
# Paper: https://aclanthology.org/2020.emnlp-main.185/
# XCOPA extends the original English COPA task to 11 typologically diverse languages.
xcopa_tasks = [
    LightevalTaskConfig(
        name=f"xcopa_{language.value}_{formulation.name.lower()}",
        suite=["lighteval"],
        prompt_function=get_copa_prompt_function(
            language,
            adapter=lambda line: {
                "context": line["premise"],
                "cause_effect": line["question"],
                "continuations": [line["choice1"], line["choice2"]],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo=("OALL/AlGhafa-Arabic-LLM-Benchmark-Translated" if language == Language.ARABIC else "xcopa"),
        hf_subset=("copa_ext_ar" if language == Language.ARABIC else standardize_tag(language.value)),
        hf_revision="08663706ee7cab30c4b7dc1bb00042a3227ce1ff" if language == Language.ARABIC else None,
        evaluation_splits=["test"],
        few_shots_split="validation",
        generation_size=-1,
        metric=[
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
        ],
    )
    for language in [
        Language.ESTONIAN,
        Language.INDONESIAN,
        Language.ITALIAN,
        Language.SWAHILI,
        Language.TAMIL,
        Language.THAI,
        Language.TURKISH,
        Language.VIETNAMESE,
        Language.CHINESE,
        # Optionally: Haitian, Quechu
    ]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

# IndicCOPA: COPA for Indic Languages
# Paper: https://arxiv.org/pdf/2212.05409
# IndicCOPA extends COPA to 15 Indic languages, providing a valuable resource for
# evaluating common sense reasoning in these languages.
copa_indic_tasks = [
    LightevalTaskConfig(
        name=f"indicxcopa_{language.value}_{formulation.name.lower()}",
        suite=["lighteval"],
        prompt_function=get_copa_prompt_function(
            language,
            adapter=lambda line: {
                "context": line["premise"],
                "cause_effect": line["question"],
                "continuations": [line["choice1"], line["choice2"]],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo="ai4bharat/IndicCOPA",
        hf_subset=f"translation-{standardize_tag(language.value)}",
        evaluation_splits=["test"],
        metric=[
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
        ],
        trust_dataset=True,
    )
    for language in [
        Language.ASSAMESE,
        Language.BENGALI,
        Language.GUJARATI,
        Language.HINDI,
        Language.KANNADA,
        Language.MALAYALAM,
        Language.MARATHI,
        Language.NEPALI,
        Language.ORIYA,
        Language.PUNJABI,
        Language.SANSKRIT,
        Language.SINDHI,
        Language.TAMIL,
        Language.TELUGU,
        Language.URDU,
        # Optionally: Maithili, Santali, Sindhi, Konkani
    ]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

# PARus: Plausible Alternatives for Russian
# Paper: https://russiansuperglue.com/tasks/task_info/PARus
# PARus is the Russian adaptation of the COPA task, part of the Russian SuperGLUE benchmark.
# It evaluates common sense reasoning and causal inference abilities in Russian language models.
parus_tasks = [
    LightevalTaskConfig(
        name=f"parus_{Language.RUSSIAN.value}_{formulation.name.lower()}",
        suite=["lighteval"],
        prompt_function=get_copa_prompt_function(
            language=Language.RUSSIAN,
            adapter=lambda line: {
                "context": line["inputs"]["premise"],
                "cause_effect": line["meta"]["task"],
                "continuations": [line["inputs"]["choice1"], line["inputs"]["choice2"]],
                "gold_idx": int(line["outputs"]) - 1,
            },
            formulation=formulation,
        ),
        hf_repo="ai-forever/MERA",
        hf_subset="parus",
        evaluation_splits=["train"],
        few_shots_split="validation",
        metric=[
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
        ],
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]


TASKS_TABLE.extend([*xcopa_tasks, *copa_indic_tasks, *parus_tasks])
# ------------------------------- Hellaswag Tasks ------------------------------- #
# Hellaswag is a commonsense reasoning task that requires models to complete a given scenario
# with the most plausible ending. It tests the model's ability to understand and reason about
# everyday situations and human behavior.

# MLMM-Hellaswag: Multilingual adaptation of Hellaswag
# Paper: https://arxiv.org/abs/2306.07610
# This is a multilingual version of Hellaswag, part of the MLMM (Massive Language Model Meta-Evaluation) benchmark.
# It evaluates commonsense reasoning abilities across multiple languages.
mlmm_hellaswag_tasks = [
    LightevalTaskConfig(
        name=f"hellaswag_{lang.value}_{formulation.name.lower()}",
        suite=["lighteval"],
        prompt_function=get_hellaswag_prompt_function(
            language=lang,
            adapter=lambda line: {
                # We don't use activity_label as they are not available
                "ctx_a": line["ctx_a"],
                "ctx_b": line["ctx_b"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo="jon-tow/okapi_hellaswag",
        hf_subset=standardize_tag(lang.value),
        hf_revision="96ed8e0dfc6172dad1d3df338d7b8ba6c1ff9d83",
        trust_dataset=True,
        evaluation_splits=["validation"],
        metric=[
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
        ],
    )
    for lang in [
        Language.ARABIC,
        Language.BENGALI,
        Language.CATALAN,
        Language.DANISH,
        Language.GERMAN,
        Language.SPANISH,
        Language.BASQUE,
        Language.FRENCH,
        Language.GUJARATI,
        Language.HINDI,
        Language.CROATIAN,
        Language.HUNGARIAN,
        Language.ARMENIAN,
        Language.INDONESIAN,
        Language.ICELANDIC,
        Language.ITALIAN,
        Language.KANNADA,
        Language.MALAYALAM,
        Language.MARATHI,
        Language.NORWEGIAN,
        Language.NEPALI,
        Language.DUTCH,
        Language.PORTUGUESE,
        Language.ROMANIAN,
        Language.RUSSIAN,
        Language.SLOVAK,
        Language.SERBIAN,
        Language.SWEDISH,
        Language.TAMIL,
        Language.TELUGU,
        Language.UKRAINIAN,
        Language.VIETNAMESE,
        Language.CHINESE,
    ]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

# Hellaswag Turkish
# This is a Turkish adaptation of the Hellaswag task.
# While there's no specific paper for this version, it has been found to work well for evaluating
# Turkish language models on commonsense reasoning tasks.

# We don't handle them in single task as there is quite a lot of differences (dataset/subset, dot replacement, etc.)
# which would make it hard to read
hellaswag_tur_tasks = [
    LightevalTaskConfig(
        name=f"hellaswag_{Language.TURKISH.value}_{formulation.name.lower()}",
        suite=["lighteval"],
        prompt_function=get_hellaswag_prompt_function(
            language=Language.TURKISH,
            adapter=lambda line: {
                "ctx_a": line["ctx_a"],
                "ctx_b": line["ctx_b"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
            # https://github.com/malhajar17/lm-evaluation-harness_turkish/blob/main/lm_eval/tasks/hellaswag_tr-v0.2/utils.py
            dot_replacement=[" [title]", " [başlık]", " [adım]", " [header]"],
        ),
        hf_repo="malhajar/hellaswag_tr-v0.2",
        hf_subset="default",
        evaluation_splits=["validation"],
        metric=[
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
        ],
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

# Hellaswag Thai
# This is a Thai adaptation of the Hellaswag task.
# Similar to the Turkish version, there's no specific paper, but it has been found to be effective
# for evaluating Thai language models on commonsense reasoning tasks.
hellaswag_tha_tasks = [
    LightevalTaskConfig(
        name=f"hellaswag_{Language.THAI.value}_{formulation.name.lower()}",
        suite=["lighteval"],
        prompt_function=get_hellaswag_prompt_function(
            language=Language.THAI,
            adapter=lambda line: {
                "ctx_a": line["ctx_a"],
                "ctx_b": line["ctx_b"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo="HuggingFaceFW-Dev/hellaswag_thai",
        hf_subset="default",
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=[
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
        ],
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

TASKS_TABLE.extend(
    [
        *mlmm_hellaswag_tasks,
        *hellaswag_tur_tasks,
        *hellaswag_tha_tasks,
    ]
)
# ------------------------------- RC Tasks ------------------------------- #
# Reading Comprehension (RC) tasks evaluate a model's ability to understand and extract information from text passages.
# These tasks typically involve answering questions based on given contexts, spanning multiple languages and formats.
# Add RC tasks supporting about 130 unique languages/scripts.

# SQuAD - like

# XQuAD: Cross-lingual Question Answering Dataset, extending SQuAD to 11 languages.
# https://arxiv.org/abs/1910.11856
xquad_tasks = [
    LightevalTaskConfig(
        name=f"xquad_{language.value}",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="google/xquad",
        hf_subset=f"xquad.{standardize_tag(language.value)}",
        evaluation_splits=("validation",),
        few_shots_split="validation",
        generation_size=400,
        stop_sequence=("\n",),
        metric=(
            multilingual_quasi_exact_match_metric(language, "prefix"),
            multilingual_quasi_f1_score_metric(language),
        ),
    )
    for language in [
        Language.ARABIC,
        Language.GERMAN,
        Language.GREEK,
        Language.ENGLISH,
        Language.SPANISH,
        Language.HINDI,
        Language.ROMANIAN,
        Language.RUSSIAN,
        Language.THAI,
        Language.TURKISH,
        Language.VIETNAMESE,
        Language.CHINESE,
    ]
]

# ThaiQA: A question answering dataset for the Thai language.
thaiqa_tasks = [
    LightevalTaskConfig(
        name=f"thaiqa_{Language.THAI.value}",
        prompt_function=get_qa_prompt_function(
            Language.THAI,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["answer"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="HuggingFaceFW-Dev/thaiqa_squad_fixed",
        hf_subset="default",
        evaluation_splits=("train",),
        few_shots_split="validation",
        generation_size=400,
        stop_sequence=("\n",),
        metric=(
            multilingual_quasi_exact_match_metric(Language.THAI, "prefix"),
            multilingual_quasi_f1_score_metric(Language.THAI),
        ),
    )
]

# SberQuAD: A large-scale Russian reading comprehension dataset.
# https://arxiv.org/abs/1912.09723
sber_squad_tasks = [
    LightevalTaskConfig(
        name=f"sber_squad_{Language.RUSSIAN.value}",
        prompt_function=get_qa_prompt_function(
            Language.RUSSIAN,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="kuznetsoffandrey/sberquad",
        hf_subset="sberquad",
        evaluation_splits=("validation",),
        few_shots_split="train",
        metric=(
            multilingual_quasi_exact_match_metric(Language.RUSSIAN, "prefix"),
            multilingual_quasi_f1_score_metric(Language.RUSSIAN),
        ),
        generation_size=400,
        stop_sequence=("\n",),
    )
]

# ARCD: Arabic Reading Comprehension Dataset.
# https://arxiv.org/pdf/1906.05394
arcd_tasks = [
    LightevalTaskConfig(
        name=f"arcd_{Language.ARABIC.value}",
        prompt_function=get_qa_prompt_function(
            Language.ARABIC,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="hsseinmz/arcd",
        hf_subset="plain_text",
        evaluation_splits=("train", "validation"),
        trust_dataset=True,
        metric=(
            multilingual_quasi_exact_match_metric(Language.ARABIC, "prefix"),
            multilingual_quasi_f1_score_metric(Language.ARABIC),
        ),
        generation_size=400,
        stop_sequence=("\n",),
    )
]

# KenSwQuAD: A question answering dataset for Kenyan Swahili.
# https://arxiv.org/abs/2205.02364
kenswquad_tasks = [
    LightevalTaskConfig(
        name=f"kenswquad_{Language.SWAHILI.value}",
        prompt_function=get_qa_prompt_function(
            Language.SWAHILI,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [line["answer"]],
            },
        ),
        suite=("lighteval",),
        hf_repo="HuggingFaceFW-Dev/KenSwQuAD",
        hf_subset="default",
        evaluation_splits=("test",),
        few_shots_split="validation",
        metric=(
            multilingual_quasi_exact_match_metric(Language.SWAHILI, "prefix"),
            multilingual_quasi_f1_score_metric(Language.SWAHILI),
        ),
        generation_size=400,
        stop_sequence=("\n",),
    )
]

# ChineseSquad: A reading comprehension dataset for Chinese.
# https://github.com/pluto-junzeng/ChineseSquad
chinese_squad_tasks = [
    LightevalTaskConfig(
        name=f"chinese_squad_{Language.CHINESE.value}",
        prompt_function=get_qa_prompt_function(
            Language.CHINESE,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="HuggingFaceFW-Dev/ChineseSquad",
        hf_subset="default",
        evaluation_splits=("validation",),
        few_shots_split="train",
        metric=(
            multilingual_quasi_exact_match_metric(Language.CHINESE, "prefix"),
            multilingual_quasi_f1_score_metric(Language.CHINESE),
        ),
        generation_size=400,
        stop_sequence=("\n",),
    )
]

# CMRC 2018: A span-extraction machine reading comprehension dataset for Chinese.
# https://arxiv.org/abs/1810.07366
cmrc2018_tasks = [
    LightevalTaskConfig(
        name=f"cmrc2018_{Language.CHINESE.value}",
        prompt_function=get_qa_prompt_function(
            Language.CHINESE,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="clue/clue",
        hf_subset="cmrc2018",
        evaluation_splits=("trial",),
        few_shots_split="train",
        generation_size=400,
        metric=(
            multilingual_quasi_exact_match_metric(Language.CHINESE, "prefix"),
            multilingual_quasi_f1_score_metric(Language.CHINESE),
        ),
        stop_sequence=("\n",),
    )
]

# IndicQA: A reading comprehension dataset for 11 Indian languages.
# https://arxiv.org/abs/2407.13522
indicqa_tasks = [
    LightevalTaskConfig(
        name=f"indicqa_{language.value}",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="ai4bharat/IndicQA",
        hf_subset=f"indicqa.{LangCodeLanguage.get(language.value).language}",
        hf_revision="92d96092ae229950973dac3b9998f8b3a8949b0a",
        hf_filter=lambda line: any(len(ans) > 0 for ans in line["answers"]["text"]),
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="test",
        generation_size=400,
        metric=(
            multilingual_quasi_exact_match_metric(language, "prefix"),
            multilingual_quasi_f1_score_metric(language),
        ),
        stop_sequence=("\n",),
    )
    for language in [
        Language.ASSAMESE,
        Language.BENGALI,
        Language.GUJARATI,
        Language.HINDI,
        Language.KANNADA,
        Language.MALAYALAM,
        Language.MARATHI,
        Language.ORIYA,
        Language.PUNJABI,
        Language.TAMIL,
        Language.TELUGU,
    ]
]

# FQuAD v2: French Question Answering Dataset version 2.
# https://arxiv.org/abs/2002.06071
fquad_v2_tasks = [
    LightevalTaskConfig(
        name=f"fquadv2_{Language.FRENCH.value}",
        prompt_function=get_qa_prompt_function(
            Language.FRENCH,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="manu/fquad2_test",
        hf_subset="default",
        evaluation_splits=("test_hasAns",),
        few_shots_split="valid_hasAns",
        generation_size=400,
        stop_sequence=("\n",),
        metric=(
            multilingual_quasi_exact_match_metric(Language.FRENCH, "prefix"),
            multilingual_quasi_f1_score_metric(Language.FRENCH),
        ),
    )
]

# TQuAD v2: Turkish Question Answering Dataset version 2.
tquad_v2_tasks = [
    LightevalTaskConfig(
        name=f"tquadv2_{Language.TURKISH.value}",
        prompt_function=get_qa_prompt_function(
            Language.TURKISH,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [a["text"] for a in line["answers"]],
            },
        ),
        suite=("lighteval",),
        hf_repo="erdometo/tquad2",
        hf_subset="default",
        evaluation_splits=("validation",),
        few_shots_split="train",
        generation_size=400,
        stop_sequence=("\n",),
        metric=(
            multilingual_quasi_exact_match_metric(Language.TURKISH, "prefix"),
            multilingual_quasi_f1_score_metric(Language.TURKISH),
        ),
    )
]

# Other QA tasks for RC

# TyDi QA: A benchmark for information-seeking question answering in typologically diverse languages.
# https://arxiv.org/abs/2003.05002
tydiqa_tasks = [
    LightevalTaskConfig(
        name=f"tydiqa_{language.value}",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="google-research-datasets/tydiqa",
        hf_subset="secondary_task",
        evaluation_splits=("validation",),
        few_shots_split="train",
        generation_size=400,
        stop_sequence=("\n",),
        metric=(
            multilingual_quasi_exact_match_metric(language, "prefix"),
            multilingual_quasi_f1_score_metric(language),
        ),
    )
    for language in [
        Language.ENGLISH,
        Language.ARABIC,
        Language.BENGALI,
        Language.FINNISH,
        Language.INDONESIAN,
        Language.JAPANESE,
        Language.KOREAN,
        Language.SWAHILI,
        Language.RUSSIAN,
        Language.TELUGU,
        Language.THAI,
    ]
]

# C3: A Chinese Challenge Corpus for Cross-lingual and Cross-modal Tasks
# Reading comprehension task part of clue
# Paper: https://arxiv.org/abs/2004.05986
c3_tasks = [
    LightevalTaskConfig(
        name=f"c3_{Language.CHINESE.value}_{formulation.name.lower()}",
        suite=("lighteval",),
        prompt_function=get_mcq_prompt_function(
            Language.CHINESE,
            lambda line: {
                "question": line["question"],
                "choices": line["choice"],
                "gold_idx": line["choice"].index(line["answer"]),
                "context": " ".join(line["context"]),
            },
        ),
        hf_repo="clue/clue",
        hf_subset="c3",
        evaluation_splits=("validation",),
        few_shots_split="train",
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# Other MCF tasks for RC
# RACE: Reading Comprehension from Examinations
# RACE is a large-scale reading comprehension dataset collected from English exams for middle and high school Chinese students.
# This Arabic version is a translation of the original RACE dataset, adapted for Arabic language evaluation.
# Paper: https://aclanthology.org/2023.arabicnlp-1.21/
race_ar_task = [
    LightevalTaskConfig(
        name=f"alghafa_race_{Language.ARABIC.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(Language.ARABIC, alghafa_adapter),
        suite=["lighteval"],
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
        hf_subset="race_ar",
        hf_revision="08663706ee7cab30c4b7dc1bb00042a3227ce1ff",
        hf_avail_splits=["test", "validation"],
        evaluation_splits=["test"],
        few_shots_split="validation",
        trust_dataset=True,
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
# SOQAL: A large-scale Arabic reading comprehension dataset.
# https://arxiv.org/abs/1906.05394
soqal_tasks = [
    LightevalTaskConfig(
        name=f"soqal_{Language.ARABIC.value}_{formulation.name.lower()}",
        hf_subset="multiple_choice_grounded_statement_soqal_task",
        prompt_function=get_mcq_prompt_function(Language.ARABIC, alghafa_adapter),
        evaluation_splits=["test"],
        few_shots_split="validation",
        suite=["lighteval"],
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Native",
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# Belebele: A large-scale reading comprehension dataset covering 122 languages.
# https://arxiv.org/abs/2308.16884
belebele_tasks = [
    LightevalTaskConfig(
        name=f"belebele_{language}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            iso_639_3_ind_to_iso_639_3_macro[LangCodeLanguage.get(language).to_alpha3()],
            lambda line: {
                "question": line["question"],
                "context": line["flores_passage"],
                "choices": [line[f"mc_answer{i}"] for i in range(1, 5)],
                "gold_idx": int(line["correct_answer_num"]) - 1,
            },
        ),
        suite=("lighteval",),
        hf_repo="facebook/belebele",
        hf_subset=language,
        evaluation_splits=("test",),
        metric=[
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
        ],
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
    for language in [
        "acm_Arab",
        "arz_Arab",
        "ceb_Latn",
        "fin_Latn",
        "hin_Deva",
        "ita_Latn",
        "khm_Khmr",
        "lvs_Latn",
        "npi_Deva",
        "pol_Latn",
        "slv_Latn",
        "swe_Latn",
        # "tso_Latn",
        # "xho_Latn",
        "afr_Latn",
        "asm_Beng",
        "ces_Latn",
        "fra_Latn",
        "hin_Latn",
        "jav_Latn",
        # "kin_Latn",
        "mal_Mlym",
        "npi_Latn",
        "por_Latn",
        # "sna_Latn",
        "swh_Latn",
        "tur_Latn",
        "yor_Latn",
        "als_Latn",
        "azj_Latn",
        "ckb_Arab",
        # "fuv_Latn",
        "hrv_Latn",
        "jpn_Jpan",
        "kir_Cyrl",
        "mar_Deva",
        # "nso_Latn",
        "snd_Arab",
        "tam_Taml",
        "ukr_Cyrl",
        "zho_Hans",
        "amh_Ethi",
        # "bam_Latn",
        "dan_Latn",
        # "gaz_Latn",
        "hun_Latn",
        # "kac_Latn",
        "kor_Hang",
        "mkd_Cyrl",
        # "nya_Latn",
        "ron_Latn",
        "som_Latn",
        "tel_Telu",
        "urd_Arab",
        "zho_Hant",
        "apc_Arab",
        "ben_Beng",
        "deu_Latn",
        # "grn_Latn",
        "hye_Armn",
        "kan_Knda",
        "lao_Laoo",
        "mlt_Latn",
        "ory_Orya",
        "rus_Cyrl",
        # "sot_Latn",
        "tgk_Cyrl",
        "urd_Latn",
        "zsm_Latn",
        "arb_Arab",
        "ben_Latn",
        "ell_Grek",
        "guj_Gujr",
        # "ibo_Latn",
        "kat_Geor",
        # "lin_Latn",
        # "mri_Latn",
        "pan_Guru",
        # "shn_Mymr",
        "spa_Latn",
        "tgl_Latn",
        "uzn_Latn",
        # "zul_Latn",
        "arb_Latn",
        # "bod_Tibt",
        "eng_Latn",
        # "hat_Latn",
        # "ilo_Latn",
        "kaz_Cyrl",
        "lit_Latn",
        "mya_Mymr",
        "pbt_Arab",
        "sin_Latn",
        "srp_Cyrl",
        "tha_Thai",
        "vie_Latn",
        "ars_Arab",
        "bul_Cyrl",
        "est_Latn",
        # "hau_Latn",
        "ind_Latn",
        # "kea_Latn",
        # "lug_Latn",
        "nld_Latn",
        "pes_Arab",
        "sin_Sinh",
        # "ssw_Latn",
        # "tir_Ethi",
        "war_Latn",
        "ary_Arab",
        "cat_Latn",
        "eus_Latn",
        "heb_Hebr",
        "isl_Latn",
        # "khk_Cyrl",
        # "luo_Latn",
        "nob_Latn",
        "plt_Latn",
        "slk_Latn",
        # "sun_Latn",
        # "tsn_Latn",
        # "wol_Latn",
    ]
]

TASKS_TABLE.extend(
    [
        *xquad_tasks,
        *thaiqa_tasks,
        *sber_squad_tasks,
        *arcd_tasks,
        *kenswquad_tasks,
        *chinese_squad_tasks,
        *cmrc2018_tasks,
        *indicqa_tasks,
        *fquad_v2_tasks,
        *tquad_v2_tasks,
        *tydiqa_tasks,
        *soqal_tasks,
        *race_ar_task,
        *belebele_tasks,
        *c3_tasks,
    ]
)

# ------------------------------- GK Tasks ------------------------------- #
# General Knowledge (GK) tasks evaluate a model's broad understanding across various domains.
# These tasks typically involve answering questions on diverse subjects, testing the model's ability to recall and apply general information.


# -------------------------------- MMLU -------------------------------- #
# MMLU (Massive Multitask Language Understanding)
# A comprehensive test of world knowledge, covering 57 subjects across STEM, humanities, social sciences, and more.
# Paper: https://arxiv.org/abs/2009.03300
MMLU_SUBSETS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

# Meta MMLU: A multilingual version of MMLU (using google translation)
# Paper: https://arxiv.org/abs/2407.21783
meta_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"meta_mmlu_{language.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["input_question"],
                "choices": [v for _, v in sorted(line["input_choice_list"].items(), key=lambda x: x[0])],
                "gold_idx": LETTER_INDICES.index(line["input_correct_responses"][0]),
            },
        ),
        suite=("lighteval",),
        hf_repo="meta-llama/Meta-Llama-3.1-8B-Instruct-evals",
        hf_subset=f"Meta-Llama-3.1-8B-Instruct-evals__multilingual_mmlu_{standardize_tag(language.value)}__details",
        hf_filter=partial(
            lambda language, subset, line: line["subtask_name"]
            == f"mmlu_{standardize_tag(language.value)}_chat.{subset}",
            language,
            subset,
        ),
        evaluation_splits=("latest",),
        metric=[
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
        ],
    )
    for subset in MMLU_SUBSETS
    for language in [
        Language.GERMAN,
        Language.SPANISH,
        Language.FRENCH,
        Language.HINDI,
        Language.ITALIAN,
        Language.PORTUGUESE,
        Language.THAI,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# MLMM MMLU: Another multilingual version of MMLU
# Paper: https://github.com/nlp-uoregon/mlmm-evaluation
mlmm_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"mlmm_mmlu_{language.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": LETTER_INDICES.index(line["answer"]),
            },
        ),
        suite=("lighteval",),
        hf_repo="jon-tow/okapi_mmlu",
        hf_subset=standardize_tag(language.value),
        hf_revision="refs/pr/1",
        hf_filter=partial(lambda subset, line: line["id"].split("/")[0] == subset, subset),
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=[
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
        ],
    )
    for subset in MMLU_SUBSETS
    for language in [
        Language.RUSSIAN,
        Language.GERMAN,
        Language.CHINESE,
        Language.FRENCH,
        Language.SPANISH,
        Language.ITALIAN,
        Language.DUTCH,
        Language.VIETNAMESE,
        Language.INDONESIAN,
        Language.ARABIC,
        Language.HUNGARIAN,
        Language.ROMANIAN,
        Language.DANISH,
        Language.SLOVAK,
        Language.UKRAINIAN,
        Language.CATALAN,
        Language.SERBIAN,
        Language.CROATIAN,
        Language.HINDI,
        Language.BENGALI,
        Language.TAMIL,
        Language.NEPALI,
        Language.MALAYALAM,
        Language.MARATHI,
        Language.TELUGU,
        Language.KANNADA,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# RUMMLU: Russian Massive Multitask Language Understanding
# Paper: https://arxiv.org/html/2401.04531v2
rummlu = [
    LightevalTaskConfig(
        name=f"rummlu_{Language.RUSSIAN.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.RUSSIAN,
            lambda line: {
                "question": line["inputs"]["text"],
                "choices": [line["inputs"][f"option_{i.lower()}"] for i in LETTER_INDICES[:4]],
                "gold_idx": LETTER_INDICES.index(line["outputs"]),
            },
        ),
        suite=("lighteval",),
        hf_repo="ai-forever/MERA",
        hf_subset="rummlu",
        hf_filter=lambda x: x["meta"]["domain"] == subset,
        evaluation_splits=("public_test",),
        metric=[
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
        ],
    )
    for subset in MMLU_SUBSETS
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# MMLU Turkish: Turkish version of MMLU
# Translated using openai GPT
mmlu_turkish = [
    LightevalTaskConfig(
        name=f"tur_leaderboard_mmlu_{Language.TURKISH.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.TURKISH,
            lambda line: {"question": line["question"], "choices": line["choices"], "gold_idx": int(line["answer"])},
        ),
        suite=("lighteval",),
        hf_repo="malhajar/mmlu_tr-v0.2",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for subset in MMLU_SUBSETS
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# CMMLU: Chinese Massive Multitask Language Understanding
# Native translation with some new categories
# Paper: https://arxiv.org/abs/2306.09212
CMMLU_SUBSETS = [
    "agronomy",
    "anatomy",
    "ancient_chinese",
    "arts",
    "astronomy",
    "business_ethics",
    "chinese_civil_service_exam",
    "chinese_driving_rule",
    "chinese_food_culture",
    "chinese_foreign_policy",
    "chinese_history",
    "chinese_literature",
    "chinese_teacher_qualification",
    "clinical_knowledge",
    "college_actuarial_science",
    "college_education",
    "college_engineering_hydrology",
    "college_law",
    "college_mathematics",
    "college_medical_statistics",
    "college_medicine",
    "computer_science",
    "computer_security",
    "conceptual_physics",
    "construction_project_management",
    "economics",
    "education",
    "electrical_engineering",
    "elementary_chinese",
    "elementary_commonsense",
    "elementary_information_and_technology",
    "elementary_mathematics",
    "ethnology",
    "food_science",
    "genetics",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_geography",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_politics",
    "human_sexuality",
    "international_law",
    "journalism",
    "jurisprudence",
    "legal_and_moral_basis",
    "logical",
    "machine_learning",
    "management",
    "marketing",
    "marxist_theory",
    "modern_chinese",
    "nutrition",
    "philosophy",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_study",
    "sociology",
    "sports_science",
    "traditional_chinese_medicine",
    "virology",
    "world_history",
    "world_religions",
]

cmmlu_tasks = [
    LightevalTaskConfig(
        name=f"cmmlu_{Language.CHINESE.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.CHINESE,
            lambda line: {
                "question": line["Question"],
                "choices": [line["A"], line["B"], line["C"], line["D"]],
                "gold_idx": LETTER_INDICES.index(line["Answer"]),
            },
        ),
        suite=("lighteval",),
        hf_repo="haonan-li/cmmlu",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for subset in CMMLU_SUBSETS
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# Arabic MMLU: Arabic version of MMLU
# Native translation with some new categories
# Paper: https://arxiv.org/html/2402.12840v1
ARABIC_MMLU_SUBSETS = [
    "Driving Test",
    "High Geography",
    "High History",
    "Islamic Studies",
    "Univ Accounting",
    "Primary General Knowledge",
    "Univ Political Science",
    "Primary Math",
    "Middle General Knowledge",
    "High Biology",
    "Primary Natural Science",
    "High Economics",
    "Middle Natural Science",
    "Middle Geography",
    "Primary Social Science",
    "Middle Computer Science",
    "Middle Islamic Studies",
    "Primary Computer Science",
    "High Physics",
    "Middle Social Science",
    "Middle Civics",
    "High Computer Science",
    "General Knowledge",
    "High Civics",
    "Prof Law",
    "High Islamic Studies",
    "Primary Arabic Language",
    "High Arabic Language",
    "Arabic Language (Grammar)",
    "Primary History",
    "Middle History",
    "Univ Economics",
    "Arabic Language (General)",
    "Univ Computer Science",
    "Primary Islamic Studies",
    "Primary Geography",
    "High Philosophy",
    "Middle Arabic Language",
    "Middle Economics",
    "Univ Management",
]

arabic_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_{Language.ARABIC.value}_{formulation.name.lower()}:{normalize_subset(subset)}",
        prompt_function=get_mcq_prompt_function(
            Language.ARABIC,
            lambda line: {
                "context": line["Context"],
                "question": line["Question"],
                "choices": [o for o in [line[f"Option {i}"] for i in range(1, 6)] if o],
                "gold_idx": LETTER_INDICES.index(line["Answer Key"]),
            },
        ),
        suite=("lighteval",),
        hf_repo="yazeed7/ArabicMMLU",
        hf_subset=subset,
        evaluation_splits=("test",),
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for subset in ARABIC_MMLU_SUBSETS
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# C-Eval: Chinese Evaluation suite
# Similar to MMLu but with different categories
# Paper: https://arxiv.org/abs/2305.08322
CEVAL_SUBSET = [
    "computer_network",
    "operating_system",
    "computer_architecture",
    "college_programming",
    "college_physics",
    "college_chemistry",
    "advanced_mathematics",
    "probability_and_statistics",
    "discrete_mathematics",
    "electrical_engineer",
    "metrology_engineer",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_chemistry",
    "high_school_biology",
    "middle_school_mathematics",
    "middle_school_biology",
    "middle_school_physics",
    "middle_school_chemistry",
    "veterinary_medicine",
    "college_economics",
    "business_administration",
    "marxism",
    "mao_zedong_thought",
    "education_science",
    "teacher_qualification",
    "high_school_politics",
    "high_school_geography",
    "middle_school_politics",
    "middle_school_geography",
    "modern_chinese_history",
    "ideological_and_moral_cultivation",
    "logic",
    "law",
    "chinese_language_and_literature",
    "art_studies",
    "professional_tour_guide",
    "legal_professional",
    "high_school_chinese",
    "high_school_history",
    "middle_school_history",
    "civil_servant",
    "sports_science",
    "plant_protection",
    "basic_medicine",
    "clinical_medicine",
    "urban_and_rural_planner",
    "accountant",
    "fire_engineer",
    "environmental_impact_assessment_engineer",
    "tax_accountant",
    "physician",
]

ceval_tasks = [
    LightevalTaskConfig(
        name=f"ceval_{Language.CHINESE.value}_{formulation.name.lower()}:{subset}",
        # for CF the new line has the best results, however it's not really compatible with options presentation
        prompt_function=get_mcq_prompt_function(
            Language.CHINESE,
            partial(
                ceval_adapter,
                Language.CHINESE,
                "NEW_LINE" if isinstance(formulation, CFFormulation) else "COMMA",
            ),
        ),
        suite=("lighteval",),
        hf_repo="ceval/ceval-exam",
        hf_subset=subset,
        evaluation_splits=("val",),
        few_shots_split="dev",
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for subset in CEVAL_SUBSET
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(
    [
        *meta_mmlu_tasks,
        *mlmm_mmlu_tasks,
        *rummlu,
        *mmlu_turkish,
        *cmmlu_tasks,
        *arabic_mmlu_tasks,
        *ceval_tasks,
    ]
)


# ---------------------------- ARC ---------------------------- #
# ARC (AI2 Reasoning Challenge) is a dataset for question answering that requires reasoning.
# It consists of multiple-choice science questions from 3rd to 9th grade exams.
# The dataset is split into two parts: ARC-Easy and ARC-Challenge.
# ARC-Easy contains questions that can be answered correctly by both humans and simple baseline models.
# ARC-Challenge contains questions that are difficult for both humans and current AI systems.


# github: https://github.com/nlp-uoregon/mlmm-evaluation
mlmm_arc_challenge_tasks = [
    LightevalTaskConfig(
        name=f"mlmm_arc_{language.value}_{formulation.name.lower()}:challenge",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else LETTER_INDICES.index(line["answerKey"]),
            },
        ),
        suite=("lighteval",),
        hf_repo="jon-tow/okapi_arc_challenge",
        hf_subset=standardize_tag(language.value),
        hf_revision="823d5d7bfaf8974a3ab52a825b6cf4903b35dbc4",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="train",
        metric=(loglikelihood_acc_metric(normalization=LogProbPMINorm()),),
    )
    for language in [
        Language.RUSSIAN,
        Language.GERMAN,
        Language.CHINESE,
        Language.FRENCH,
        Language.SPANISH,
        Language.ITALIAN,
        Language.DUTCH,
        Language.VIETNAMESE,
        Language.INDONESIAN,
        Language.ARABIC,
        Language.HUNGARIAN,
        Language.ROMANIAN,
        Language.DANISH,
        Language.SLOVAK,
        Language.UKRAINIAN,
        Language.CATALAN,
        Language.SERBIAN,
        Language.CROATIAN,
        Language.HINDI,
        Language.BENGALI,
        Language.TAMIL,
        Language.NEPALI,
        Language.MALAYALAM,
        Language.MARATHI,
        Language.TELUGU,
        Language.KANNADA,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# Arabic ARC Easy
# It's based on the community arabic leaderboard task but uses
# the multilingual template
# Paper: https://aclanthology.org/2023.arabicnlp-1.21/
arabic_ledarboard_arc_easy = [
    LightevalTaskConfig(
        name=f"alghafa_arc_{Language.ARABIC.value}_{formulation.name.lower()}:easy",
        prompt_function=get_mcq_prompt_function(Language.ARABIC, alghafa_adapter),
        suite=["lighteval"],
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
        hf_subset="arc_easy_ar",
        hf_revision="08663706ee7cab30c4b7dc1bb00042a3227ce1ff",
        trust_dataset=True,
        evaluation_splits=["test"],
        few_shots_split="validation",
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# Turkish ARC
# Comes from the Turkish leaderboard
turkish_arc = [
    LightevalTaskConfig(
        name=f"community_arc_{Language.TURKISH.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.TURKISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else LETTER_INDICES.index(line["answerKey"]),
            },
        ),
        suite=("lighteval",),
        hf_repo="malhajar/arc-tr",
        hf_subset=f"ARC-{subset.capitalize()}",
        evaluation_splits=("test",),
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for subset in ["easy", "challenge"]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]


TASKS_TABLE.extend(
    [
        *mlmm_arc_challenge_tasks,
        *arabic_ledarboard_arc_easy,
        *turkish_arc,
    ]
)

# ---------------------------- TruthfulQA ---------------------------- #
# TruthfulQA: Measuring How Models Mimic Human Falsehoods
# Paper: https://arxiv.org/abs/2109.07958
# TruthfulQA is a benchmark dataset designed to measure the truthfulness of language models.
# It consists of questions that humans might answer incorrectly due to false beliefs or misconceptions.
# The task evaluates a model's ability to provide truthful answers and avoid common human biases.

# github: https://github.com/nlp-uoregon/mlmm-evaluation
mlmm_truthfulqa_tasks = [
    LightevalTaskConfig(
        name=f"mlmm_truthfulqa_{language.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            language,
            partial(
                lambda subset, line: {
                    "question": line["question"],
                    "choices": line[f"{subset}_targets"]["choices"],
                    "gold_idx": [ix for ix, label in enumerate(line[f"{subset}_targets"]["labels"]) if label == 1],  # type: ignore
                },
                subset,
            ),
        ),
        suite=("lighteval",),
        hf_repo="jon-tow/okapi_truthfulqa",
        hf_subset=standardize_tag(language.value),
        hf_revision="cdd5db1a66fd04105622109d1c2a5cbc8cde7586",
        trust_dataset=True,
        evaluation_splits=("validation",),
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for subset in ["mc1", "mc2"]
    for language in [
        Language.ARABIC,
        Language.BENGALI,
        Language.CATALAN,
        Language.DANISH,
        Language.GERMAN,
        Language.SPANISH,
        Language.BASQUE,
        Language.FRENCH,
        Language.GUJARATI,
        Language.HINDI,
        Language.CROATIAN,
        Language.HUNGARIAN,
        Language.ARMENIAN,
        Language.INDONESIAN,
        Language.ICELANDIC,
        Language.ITALIAN,
        Language.KANNADA,
        Language.MALAYALAM,
        Language.MARATHI,
        Language.NORWEGIAN,
        Language.NEPALI,
        Language.DUTCH,
        Language.PORTUGUESE,
        Language.ROMANIAN,
        Language.RUSSIAN,
        Language.SLOVAK,
        Language.SERBIAN,
        Language.SWEDISH,
        Language.TAMIL,
        Language.TELUGU,
        Language.UKRAINIAN,
        Language.VIETNAMESE,
        Language.CHINESE,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# Turkish TruthfulQA
# Based on turkish leaderboard
turkish_truthfulqa = [
    LightevalTaskConfig(
        name=f"community_truthfulqa_{Language.TURKISH.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.TURKISH,
            partial(
                lambda subset, line: {
                    "question": line["question"],
                    "choices": line[f"{subset}_targets"]["choices"],
                    "gold_idx": [ix for ix, label in enumerate(line[f"{subset}_targets"]["labels"]) if label == 1],  # type: ignore
                },
                subset,
            ),
        ),
        suite=("lighteval",),
        hf_repo="malhajar/truthful_qa-tr-v0.2",
        hf_subset="default",
        evaluation_splits=("validation",),
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for subset in ["mc1", "mc2"]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(
    [
        *mlmm_truthfulqa_tasks,
        *turkish_truthfulqa,
    ]
)

# ---------------------------- Exams like tasks ---------------------------- #

# Exams: A collection of exam questions from various countries and subjects
# Paper: https://arxiv.org/abs/2011.03080
exams_subjects_by_lang: dict[Language, set[str]] = {
    Language.ARABIC: {"Biology", "Islamic Studies", "Physics", "Science", "Social"},
    Language.BULGARIAN: {"Biology", "Chemistry", "Geography", "History", "Philosophy", "Physics"},
    Language.CROATIAN: {
        "Biology",
        "Chemistry",
        "Ethics",
        "Fine Arts",
        "Geography",
        "Geology",
        "History",
        "Informatics",
        "Philosophy",
        "Physics",
        "Politics",
        "Psychology",
        "Religion",
        "Sociology",
    },
    Language.HUNGARIAN: {
        "Agriculture",
        "Agriculture (Mechanical knowledge)",
        "Biology",
        "Chemistry",
        "Economics",
        "Economics & Marketing",
        "Economics Basics (Business)",
        "Economics Basics (Theoretical)",
        "Forestry",
        "Geography",
        "Landscaping",
        "Physics",
        "Politics",
        "Tourism",
    },
    Language.ITALIAN: {
        "Biology",
        "Chemistry",
        "Ethics",
        "Geography",
        "Geology",
        "History",
        "Informatics",
        "Philosophy",
        "Physics",
        "Politics",
        "Psychology",
        "Sociology",
    },
    Language.SERBIAN: {
        "Biology",
        "Chemistry",
        "Ethics",
        "Geography",
        "Geology",
        "History",
        "Informatics",
        "Philosophy",
        "Physics",
        "Politics",
        "Psychology",
        "Religion",
        "Sociology",
    },
    Language.FRENCH: {"Economics", "Economics & Marketing", "Economics Basics (Theoretical)", "Geography", "Physics"},
    Language.GERMAN: {
        "Chemistry",
        "Economics",
        "Economics & Marketing",
        "Economics Basics (Theoretical)",
        "Geography",
        "Physics",
        "Tourism",
    },
    Language.SPANISH: {"Geography", "Physics"},
    Language.LITHUANIAN: {"Geology", "History"},
    Language.ALBANIAN: {
        "Biology",
        "Business",
        "Chemistry",
        "Fine Arts",
        "History",
        "Philosophy",
        "Physics",
        "Sociology",
    },
    Language.MACEDONIAN: {
        "Biology",
        "Business",
        "Chemistry",
        "Fine Arts",
        "History",
        "Philosophy",
        "Physics",
        "Sociology",
    },
    Language.TURKISH: {
        "Biology",
        "Business",
        "Chemistry",
        "Geography",
        "History",
        "Philosophy",
        "Physics",
        "Sociology",
    },
    Language.POLISH: {"Professional"},
    Language.PORTUGUESE: {"Biology", "Economics", "Geology", "Philosophy"},
    Language.VIETNAMESE: {"Biology", "Chemistry", "Citizenship", "Geography", "History", "Physics"},
}

exams_tasks = [
    LightevalTaskConfig(
        name=f"exams_{language.value}_{formulation.name.lower()}:{normalize_subset(subject)}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"]["stem"],
                "choices": line["question"]["choices"]["text"],
                "gold_idx": line["question"]["choices"]["label"].index(line["answerKey"]),
            },
        ),
        suite=("lighteval",),
        hf_repo="mhardalov/exams",
        hf_subset="multilingual",
        # Weird bug in dataset
        hf_filter=partial(
            lambda language, subject, line: line["answerKey"] != "@"
            and line["info"]["language"] == LangCodeLanguage(standardize_tag(language.value)).language_name()
            and line["info"]["subject"] == subject,
            language,
            subject,
        ),
        evaluation_splits=("test",),
        few_shots_split="train",
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for language in exams_subjects_by_lang.keys()
    for subject in exams_subjects_by_lang[language]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# M3Exam: Multitask Multilingual Multimodal Evaluation Benchmark
# It also contains a multimodal version but we don't support that
# Paper: https://arxiv.org/abs/2306.05179
m3exams_tasks = [
    LightevalTaskConfig(
        name=f"m3exams_{language.value}_{formulation.name.lower()}",
        suite=("lighteval",),
        prompt_function=get_mcq_prompt_function(
            language,
            partial(get_m3exam_adapter, language),
        ),
        hf_repo="chiayewken/m3exam",
        hf_subset=LangCodeLanguage(standardize_tag(language.value)).language_name().lower(),
        evaluation_splits=("test",),
        few_shots_split="dev",
        generation_size=-1,
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for language in [
        Language.AFRIKAANS,
        Language.CHINESE,
        Language.ENGLISH,
        Language.ITALIAN,
        Language.JAVANESE,
        Language.PORTUGUESE,
        Language.SWAHILI,
        Language.THAI,
        Language.VIETNAMESE,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# Thai Exams
# We noticed very bad performance of models on this dataset
# However, it may just be because quality of the models themselves
# Paper: https://arxiv.org/abs/2312.13951

THAI_EXAMS_SUBSETS = ["a_level", "ic", "onet", "tgat", "tpat1"]

# If too hard we can add help with para
thai_exams_tasks = [
    LightevalTaskConfig(
        name=f"thai_exams_{Language.THAI.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(Language.THAI, thai_exams_adapter),
        suite=("lighteval",),
        hf_repo="scb10x/thai_exam",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="train",
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for subset in THAI_EXAMS_SUBSETS
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(
    [
        *exams_tasks,
        *m3exams_tasks,
        *thai_exams_tasks,
    ]
)

# ------------------------------- XCSQA ------------------------------- #
# XCSQA (Cross-lingual Commonsense QA) is part of the XCSR (Cross-lingual Commonsense Reasoning) benchmark
# It is a multilingual extension of the CommonsenseQA dataset, covering 16 languages
# The task involves answering multiple-choice questions that require commonsense reasoning
# Paper: https://arxiv.org/abs/2110.08462
xcsqa_tasks = [
    LightevalTaskConfig(
        name=f"xcsqa_{language.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"]["stem"],
                "choices": line["question"]["choices"]["text"],
                "gold_idx": line["question"]["choices"]["label"].index(line["answerKey"]),
            },
        ),
        suite=("lighteval",),
        hf_repo="INK-USC/xcsr",
        hf_subset=f"X-CSQA-{standardize_tag(language.value)}",
        hf_filter=lambda x: all(
            len(x["question"]["choices"]["text"][i].strip()) > 0 for i in range(len(x["question"]["choices"]["text"]))
        ),
        evaluation_splits=("validation",),
        few_shots_split="train",
        metric=[
            loglikelihood_acc_metric(normalization=LogProbPMINorm()),
        ],
    )
    for language in [
        Language.ARABIC,
        Language.GERMAN,
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.HINDI,
        Language.ITALIAN,
        Language.JAPANESE,
        Language.DUTCH,
        Language.POLISH,
        Language.PORTUGUESE,
        Language.RUSSIAN,
        Language.SWAHILI,
        Language.URDU,
        Language.VIETNAMESE,
        Language.CHINESE,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(
    [
        *xcsqa_tasks,
    ]
)

# ------------------------------- PIQA ------------------------------- #
# PIQA: Physical Interaction Question Answering
# PIQA is a benchmark for testing physical commonsense reasoning.
# This Arabic version is a translation of the original PIQA dataset, adapted for Arabic language evaluation.
# It tests the ability to reason about physical interactions in everyday situations.
# Paper: https://arxiv.org/abs/1911.11641
# Arabic version: https://aclanthology.org/2023.arabicnlp-1.21/
piqa_ar_tasks = [
    LightevalTaskConfig(
        name=f"alghafa_piqa_{Language.ARABIC.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(Language.ARABIC, alghafa_adapter),
        suite=["lighteval"],
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
        hf_revision="08663706ee7cab30c4b7dc1bb00042a3227ce1ff",
        hf_subset="piqa_ar",
        hf_avail_splits=["test", "validation"],
        evaluation_splits=["test"],
        few_shots_split="validation",
        trust_dataset=True,
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(
    [
        *piqa_ar_tasks,
    ]
)

# ------------------------------- OpenBookQA ------------------------------- #
# OpenBookQA: A Question-Answering Dataset for Open-Book Exams
# OpenBookQA is a question-answering dataset modeled after open-book exams for assessing human understanding of a subject.
# It consists of multiple-choice questions that require combining facts from a given open book with broad common knowledge.
# The task tests language models' ability to leverage provided information and apply common sense reasoning.
# Original paper: https://arxiv.org/abs/1809.02789
# Arabic version: https://aclanthology.org/2023.arabicnlp-1.21/
openbook_ara_tasks = [
    LightevalTaskConfig(
        name=f"alghafa_openbookqa_{Language.ARABIC.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(Language.ARABIC, alghafa_adapter),
        suite=["lighteval"],
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
        hf_subset="openbook_qa_ext_ar",
        hf_revision="08663706ee7cab30c4b7dc1bb00042a3227ce1ff",
        trust_dataset=True,
        evaluation_splits=["test"],
        few_shots_split="validation",
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# The Russian version is part of the MERA (Multilingual Enhanced Russian NLP Architectures) project.
# Paper: https://arxiv.org/abs/2401.04531
openbook_rus_tasks = [
    LightevalTaskConfig(
        name=f"mera_openbookqa_{Language.RUSSIAN.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.RUSSIAN,
            lambda line: {
                "question": line["inputs"]["question"],
                "choices": [line["inputs"][f"option_{i.lower()}"] for i in LETTER_INDICES[:4]],
                "gold_idx": LETTER_INDICES.index(line["outputs"]),
            },
        ),
        suite=["lighteval"],
        hf_repo="ai-forever/MERA",
        hf_subset="ruopenbookqa",
        evaluation_splits=("train",),
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(
    [
        *openbook_rus_tasks,
        *openbook_ara_tasks,
    ]
)

# ------------------------------- SciQ ------------------------------- #
# SciQ: Science Question Answering
# SciQ is a question-answering dataset designed to evaluate the ability of language models to answer science questions.
# It consists of multiple-choice questions that require scientific reasoning and factual knowledge.

# The Arabic version is part of the AlGhafa Arabic LLM Benchmark, a translation and adaptation of various English datasets.
# Paper: https://aclanthology.org/2023.arabicnlp-1.21/
sciqa_ar_task = [
    LightevalTaskConfig(
        name=f"alghafa_sciqa_{Language.ARABIC.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ARABIC,
            sciqa_adapter,
        ),
        suite=["lighteval"],
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
        hf_subset="sciq_ar",
        hf_revision="08663706ee7cab30c4b7dc1bb00042a3227ce1ff",
        hf_avail_splits=["test", "validation"],
        evaluation_splits=["test"],
        few_shots_split="validation",
        few_shots_select="sequential",
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
        trust_dataset=True,
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(
    [
        *sciqa_ar_task,
    ]
)

# ------------------------------- Math Tasks ------------------------------- #

# MathLogicQA is a dataset for evaluating mathematical reasoning in language models.
# It consists of multiple-choice questions that require logical reasoning and mathematical problem-solving.
# This Russian version is part of the MERA (Multilingual Evaluation of Reasoning Abilities) benchmark.
# MERA: https://github.com/ai-forever/MERA
mathlogicqa_rus_tasks = [
    LightevalTaskConfig(
        name=f"mathlogic_qa_{Language.RUSSIAN.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.RUSSIAN,
            lambda line: {
                "question": line["inputs"]["text"],
                "choices": [line["inputs"][f"option_{i.lower()}"] for i in LETTER_INDICES[:4]],
                "gold_idx": LETTER_INDICES.index(line["outputs"]),
            },
        ),
        suite=("lighteval",),
        hf_repo="ai-forever/MERA",
        hf_subset="mathlogicqa",
        evaluation_splits=("train",),
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(
    [
        *mathlogicqa_rus_tasks,
    ]
)

# ------------------------------- Misc ------------------------------- #

# AGIEval: Chinese AGI Evaluation suite (Excluding the english subsets)
# Paper: https://arxiv.org/abs/2304.06364
CHINESE_AGIEVAL_SUBSET = [
    "gaokao-biology",
    "gaokao-chinese",
    "gaokao-chemistry",
    "gaokao-geography",
    "gaokao-history",
    "gaokao-mathqa",
    "gaokao-physics",
    "logiqa-zh",
    "jec-qa-kd",
    "jec-qa-ca",
]

agieval_tasks_zh = [
    LightevalTaskConfig(
        name=f"agieval_{Language.CHINESE.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.CHINESE,
            partial(
                agieval_prompt,
                Language.CHINESE,
                "NEW_LINE" if isinstance(formulation, CFFormulation) else "COMMA",
            ),
        ),
        suite=("lighteval",),
        hf_repo=f"hails/agieval-{subset}",
        hf_subset="default",
        evaluation_splits=("test",),
        few_shots_split=None,
        metric=(loglikelihood_acc_metric(normalization=LogProbPMINorm()),),
    )
    for subset in CHINESE_AGIEVAL_SUBSET
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# WorldTree is a dataset for multi-hop inference in science question answering.
# It provides explanations for elementary science questions by combining facts from a semi-structured knowledge base.
# This Russian version is part of the MERA (Multilingual Evaluation of Reasoning Abilities) benchmark.
# MERA: https://github.com/ai-forever/MERA
worldtree_rus_tasks = [
    LightevalTaskConfig(
        name=f"mera_worldtree_{Language.RUSSIAN.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.RUSSIAN,
            lambda line: {
                "question": line["inputs"]["question"],
                "choices": [line["inputs"][f"option_{i.lower()}"] for i in LETTER_INDICES[:4]],
                "gold_idx": LETTER_INDICES.index(line["outputs"]),
            },
        ),
        suite=("lighteval",),
        hf_repo="ai-forever/MERA",
        hf_subset="ruworldtree",
        evaluation_splits=("train",),
        metric=(loglikelihood_acc_metric(normalization=LogProbTokenNorm()),),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(
    [
        *agieval_tasks_zh,
        *worldtree_rus_tasks,
    ]
)
