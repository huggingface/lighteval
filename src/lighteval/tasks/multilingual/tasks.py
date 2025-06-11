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
from itertools import combinations

from langcodes import Language as LangCodeLanguage
from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    loglikelihood_acc_metric,
    multilingual_quasi_exact_match_metric,
    multilingual_quasi_f1_score_metric,
)
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import (
    agieval_adapter,
    alghafa_adapter,
    ceval_adapter,
    enem_adapter,
    get_m3exam_adapter,
    get_mkqa_adapter,
    sciqa_adapter,
    thai_exams_adapter,
    winogrand_adapter,
    xcodah_adapter,
)
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation, normalize_subset
from lighteval.tasks.templates.boolq import get_boolq_prompt_function
from lighteval.tasks.templates.continuation import get_continuation_prompt_function
from lighteval.tasks.templates.copa import get_copa_prompt_function
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.nli import get_nli_prompt_function
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.tasks.templates.translation import get_translation_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language, iso_639_3_ind_to_iso_639_3_macro, manage_duplicate_language_codes


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
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=None),
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=None),
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
        hf_filter=lambda line: line["label"] in [0, 2]
        and line["premise"] is not None
        and line["hypothesis"] is not None,
        hf_repo=f"Harsit/xnli2.0_train_{LangCodeLanguage(standardize_tag(language.value)).language_name().lower()}",
        hf_subset="default",
        evaluation_splits=["train"],
        hf_avail_splits=["train"],
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
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=None),
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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

# African XNLI: African XNLI
# From https://arxiv.org/abs/2406.03368. Human translated MMLU.
afri_xnli_tasks = [
    LightevalTaskConfig(
        name=f"afri_xnli_{language.value}_{formulation.name.lower()}",
        suite=("lighteval",),
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
        hf_repo="masakhane/afrixnli",
        hf_subset=language.value,
        hf_filter=lambda x: int(x["label"]) in [0, 2],
        evaluation_splits=("test",),
        few_shots_split="validation",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=None),
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for language in [
        Language.AMHARIC,
        # Language.EWE,
        Language.FRENCH,
        # Language.HAUSA,
        # Language.IGBO,
        # Language.KINYARWANDA,
        # Language.LINGALA,
        # Language.LUGANDA,
        # Language.OROMO,
        # Language.SHONA,
        # Language.SOTHO,
        Language.SWAHILI,
        # Language.TWI,
        # Language.WOLOF,
        # Language.XHOSA,
        Language.YORUBA,
        # Language.ZULU,
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
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=None),
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
        evaluation_splits=("train",),
        few_shots_split="validation",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=None),
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=None),
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=None),
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

TASKS_TABLE.extend(
    [
        *xnli_tasks,
        *xnli2_tasks,
        *xnli_indic_tasks,
        *paws_x_tasks,
        *rcb_tasks,
        *ocnli_tasks,
        *cmnli_tasks,
        *afri_xnli_tasks,
    ]
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
        evaluation_splits=["test"],
        few_shots_split="validation",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for language in [
        Language.ARABIC,
        Language.ESTONIAN,
        Language.INDONESIAN,
        Language.ITALIAN,
        Language.SWAHILI,
        Language.TAMIL,
        Language.THAI,
        Language.TURKISH,
        Language.VIETNAMESE,
        Language.CHINESE,
        Language.HAITIAN,
        Language.QUECHUA,
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
        # Since we use trust_dataset, we have to be careful about what is inside the dataset
        # script. We thus lock the revision to ensure that the script doesn't change
        hf_revision="d356ef19a4eb287e88a51d07a56b73ba88c7f188",
        evaluation_splits=["test"],
        hf_avail_splits=["test"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
        name=f"mlmm_hellaswag_{lang.value}_{formulation.name.lower()}",
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
        # Since we use trust_dataset, we have to be careful about what is inside the dataset
        # script. We thus lock the revision to ensure that the script doesn't change
        hf_revision="96ed8e0dfc6172dad1d3df338d7b8ba6c1ff9d83",
        evaluation_splits=["validation"],
        hf_avail_splits=["validation"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
        trust_dataset=True,
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
        name=f"community_hellaswag_{Language.TURKISH.value}_{formulation.name.lower()}",
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
            wikihow_artifacts=[" [title]", " [başlık]", " [adım]", " [header]"],
        ),
        hf_repo="malhajar/hellaswag_tr-v0.2",
        hf_subset="default",
        evaluation_splits=["validation"],
        hf_avail_splits=["validation"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

# Hellaswag Thai
# This is a Thai adaptation of the Hellaswag task.
# Similar to the Turkish version, there's no specific paper, but it has been found to be effective
# for evaluating Thai language models on commonsense reasoning tasks.
hellaswag_tha_tasks = [
    LightevalTaskConfig(
        name=f"community_hellaswag_{Language.THAI.value}_{formulation.name.lower()}",
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
            wikihow_artifacts=[" [ชื่อ]", " [ส่วนหัว]", " [ขั้นตอน]", " [header]", " [Header]"],
        ),
        hf_repo="lighteval/hellaswag_thai",
        hf_subset="default",
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

hellaswag_hin_tasks = [
    LightevalTaskConfig(
        name=f"community_hellaswag_{Language.HINDI.value}_{formulation.name.lower()}",
        suite=["lighteval"],
        prompt_function=get_hellaswag_prompt_function(
            language=Language.HINDI,
            adapter=lambda line: {
                "ctx_a": line["ctx_a"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo="ai4bharat/hellaswag-hi",
        hf_filter=lambda line: all(len(choice.strip()) > 0 for choice in line["endings"]),
        hf_subset="hi",
        evaluation_splits=("validation",),
        few_shots_split="validation",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
            ],
        ),
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

hellaswag_tel_tasks = [
    LightevalTaskConfig(
        name=f"community_hellaswag_{Language.TELUGU.value}_{formulation.name.lower()}",
        suite=["lighteval"],
        prompt_function=get_hellaswag_prompt_function(
            language=Language.TELUGU,
            adapter=lambda line: {
                "ctx_a": line["ctx_a"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo="LightFury9/hellaswag-telugu",
        hf_subset="default",
        evaluation_splits=("valid",),
        few_shots_split="train",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

TASKS_TABLE.extend(
    [
        *mlmm_hellaswag_tasks,
        *hellaswag_tur_tasks,
        *hellaswag_tha_tasks,
        *hellaswag_hin_tasks,
        *hellaswag_tel_tasks,
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

# GermanQuAD: High-quality German QA dataset with 13,722 questions
# https://arxiv.org/abs/2104.12741
germanquad_tasks = [
    LightevalTaskConfig(
        name=f"germanquad_{Language.GERMAN.value}",
        prompt_function=get_qa_prompt_function(
            Language.GERMAN,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="deepset/germanquad",
        hf_subset="plain_text",
        trust_dataset=True,
        hf_revision="fff05ceaf2ffbe5b65c7e0c57e678f7b7e1a0581",
        hf_filter=lambda line: any(len(ans) > 0 for ans in line["answers"]["text"]),
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=400,
        stop_sequence=("\n",),
        metric=(
            multilingual_quasi_exact_match_metric(Language.GERMAN, "prefix"),
            multilingual_quasi_f1_score_metric(Language.GERMAN),
        ),
    )
]


# SQuAD-it: Italian translation of the SQuAD dataset
# https://github.com/crux82/squad-it
squad_it_tasks = [
    LightevalTaskConfig(
        name=f"squad_{Language.ITALIAN.value}",
        prompt_function=get_qa_prompt_function(
            Language.ITALIAN,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="crux82/squad_it",
        hf_subset="default",
        hf_filter=lambda line: any(len(ans) > 0 for ans in line["answers"]["text"]),
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=400,
        stop_sequence=("\n",),
        metric=(
            multilingual_quasi_exact_match_metric(Language.ITALIAN, "prefix"),
            multilingual_quasi_f1_score_metric(Language.ITALIAN),
        ),
    )
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
        hf_repo="lighteval/thaiqa_squad_fixed",
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

# FaQuAD: A Portuguese Reading Comprehension Dataset
# https://arxiv.org/abs/2007.15671
faquad_tasks = [
    LightevalTaskConfig(
        name=f"faquad_{Language.PORTUGUESE.value}",
        prompt_function=get_qa_prompt_function(
            Language.PORTUGUESE,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="eraldoluis/faquad",
        hf_subset="plain_text",
        trust_dataset=True,
        hf_revision="205ba826a2282a4a5aa9bd3651e55ee4f2da1546",
        hf_filter=lambda line: any(len(ans) > 0 for ans in line["answers"]["text"]),
        evaluation_splits=("validation",),
        few_shots_split="train",
        metric=(
            multilingual_quasi_exact_match_metric(Language.PORTUGUESE, "prefix"),
            multilingual_quasi_f1_score_metric(Language.PORTUGUESE),
        ),
        generation_size=400,
        stop_sequence=("\n",),
    )
]


# SQuAD-es: Spanish translation of the Stanford Question Answering Dataset
# https://huggingface.co/datasets/ccasimiro/squad_es
squad_es_tasks = [
    LightevalTaskConfig(
        name=f"squad_{Language.SPANISH.value}",
        prompt_function=get_qa_prompt_function(
            Language.SPANISH,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="ccasimiro/squad_es",
        hf_subset="v2.0.0",
        hf_filter=lambda line: any(len(ans) > 0 for ans in line["answers"]["text"]),
        evaluation_splits=("validation",),
        few_shots_split="train",
        metric=(
            multilingual_quasi_exact_match_metric(Language.SPANISH, "prefix"),
            multilingual_quasi_f1_score_metric(Language.SPANISH),
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
        evaluation_splits=("validation",),
        few_shots_split="train",
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
        hf_repo="lighteval/KenSwQuAD",
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
        hf_repo="lighteval/ChineseSquad",
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
        hf_filter=lambda line: any(len(ans) > 0 for ans in line["answers"]["text"]),
        # Since we use trust_dataset, we have to be careful about what is inside the dataset
        # script. We thus lock the revision to ensure that the script doesn't change
        hf_revision="92d96092ae229950973dac3b9998f8b3a8949b0a",
        trust_dataset=True,
        evaluation_splits=("test",),
        hf_avail_splits=("test",),
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
            formulation=formulation,
        ),
        hf_repo="clue/clue",
        hf_subset="c3",
        evaluation_splits=("validation",),
        few_shots_split="train",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
        prompt_function=get_mcq_prompt_function(Language.ARABIC, alghafa_adapter, formulation=formulation),
        suite=["lighteval"],
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
        hf_subset="race_ar",
        # Since we use trust_dataset, we have to be careful about what is inside the dataset
        # script. We thus lock the revision to ensure that the script doesn't change
        hf_revision="08663706ee7cab30c4b7dc1bb00042a3227ce1ff",
        hf_avail_splits=["test", "validation"],
        evaluation_splits=["test"],
        few_shots_split="validation",
        trust_dataset=True,
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
        prompt_function=get_mcq_prompt_function(Language.ARABIC, alghafa_adapter, formulation=formulation),
        evaluation_splits=["test"],
        few_shots_split="validation",
        suite=["lighteval"],
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Native",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance.
# It consists of QA instances in 7 languages: English, Arabic, German, Spanish, Hindi, Vietnamese, and Chinese.
# The dataset is derived from the SQuAD v1.1 dataset, with questions and contexts translated by professional translators.
# Paper: https://arxiv.org/abs/1910.07475
mlqa_tasks = [
    LightevalTaskConfig(
        name=f"mlqa_{lang.value}",
        prompt_function=get_qa_prompt_function(
            lang,
            lambda line: {
                "context": line["context"],
                "question": line["question"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="facebook/mlqa",
        hf_subset=f"mlqa.{standardize_tag(lang.value)}.{standardize_tag(lang.value)}",
        hf_revision="397ed406c1a7902140303e7faf60fff35b58d285",
        trust_dataset=True,
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        generation_size=400,
        stop_sequence=("\n",),
        metric=[
            multilingual_quasi_exact_match_metric(lang, "prefix"),
            multilingual_quasi_f1_score_metric(lang),
        ],
    )
    for lang in [
        Language.ARABIC,
        Language.GERMAN,
        Language.SPANISH,
        Language.CHINESE,
        Language.HINDI,
        Language.VIETNAMESE,
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
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="facebook/belebele",
        hf_subset=language,
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
        *squad_it_tasks,
        *squad_es_tasks,
        *faquad_tasks,
        *germanquad_tasks,
    ]
)

# ------------------------------- GK Tasks ------------------------------- #
# General Knowledge (GK) tasks evaluate a model's broad understanding across various domains.
# These tasks typically involve answering questions on diverse subjects, testing the model's ability to recall and apply general information.


# -------------------------------- MMLU -------------------------------- #
# MMLU (Massive Multitask Language Understanding)
# A comprehensive test of world knowledge, covering 57 subjects across STEM, humanities, social sciences, and more.
# Note that all MMLU tasks uses PMI normalization, this makes the computation 2x slower, however we found this metric to be less noisy and yield better results than the others.
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
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="meta-llama/Meta-Llama-3.1-8B-Instruct-evals",
        hf_subset=f"Llama-3.1-8B-Instruct-evals__multilingual_mmlu_{standardize_tag(language.value)}__details",
        hf_filter=partial(
            lambda language, subset, line: line["subtask_name"]
            == f"mmlu_{standardize_tag(language.value)}_chat.{subset}",
            language,
            subset,
        ),
        evaluation_splits=("latest",),
        hf_avail_splits=["latest"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
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
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="jon-tow/okapi_mmlu",
        hf_subset=standardize_tag(language.value),
        hf_revision="refs/pr/1",
        hf_filter=partial(lambda subset, line: line["id"].split("/")[0] == subset, subset),
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
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

openai_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"openai_mmlu_{language[0].value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            language[0],
            lambda line: {
                "question": line["Question"],
                "choices": [line["A"], line["B"], line["C"], line["D"]],
                "gold_idx": LETTER_INDICES.index(line["Answer"]),
            },
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="openai/MMMLU",
        hf_subset=language[1],
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        hf_filter=partial(lambda subset, x: x["Subject"].lower() == subset, subset),
        hf_revision="038c7808122969ead7456361af05cb8f47d247f8",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in MMLU_SUBSETS
    for language in [
        (Language.ARABIC, "AR_XY"),
        (Language.BENGALI, "BN_BD"),
        (Language.GERMAN, "DE_DE"),
        (Language.SPANISH, "ES_LA"),
        (Language.FRENCH, "FR_FR"),
        (Language.HINDI, "HI_IN"),
        (Language.INDONESIAN, "ID_ID"),
        (Language.ITALIAN, "IT_IT"),
        (Language.JAPANESE, "JA_JP"),
        (Language.KOREAN, "KO_KR"),
        (Language.PORTUGUESE, "PT_BR"),
        (Language.SWAHILI, "SW_KE"),
        (Language.YORUBA, "YO_NG"),
        (Language.CHINESE, "ZH_CN"),
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# Translated MMLU using both professional and non-professional translators. Contains tags for cultural sensitivity.
# CA: Cultural Agnostic
# CS: Cultural Specific
# UNK: Not annotated
# ALL: All of the above
# https://huggingface.co/papers/2412.03304
global_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"global_mmlu_{sensitivity_label.lower()}_{language.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": [line["option_a"], line["option_b"], line["option_c"], line["option_d"]],
                "gold_idx": LETTER_INDICES.index(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="CohereForAI/Global-MMLU",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="dev",
        hf_filter=partial(
            lambda subset, sensitivity_label, x: x["subject"].lower() == subset
            and (
                sensitivity_label == "ALL" or sensitivity_label in x["cultural_sensitivity_label"].replace("-", "UNK")
            )
            and all(x[f"option_{opt}"] is not None and x[f"option_{opt}"].strip() for opt in "abcd"),
            subset,
            sensitivity_label,
        ),
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in MMLU_SUBSETS
    for language in [
        Language.AMHARIC,
        Language.ARABIC,
        Language.BENGALI,
        Language.CHINESE,
        Language.CZECH,
        Language.GERMAN,
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.HEBREW,
        Language.HINDI,
        Language.INDONESIAN,
        Language.ITALIAN,
        Language.JAPANESE,
        Language.KOREAN,
        Language.MALAY,
        Language.DUTCH,
        Language.NORWEGIAN,
        Language.POLISH,
        Language.PORTUGUESE,
        Language.ROMANIAN,
        Language.RUSSIAN,
        Language.SERBIAN,
        Language.SWEDISH,
        Language.SWAHILI,
        Language.TAMIL,
        Language.TELUGU,
        Language.THAI,
        Language.TURKISH,
        Language.UKRAINIAN,
        Language.URDU,
        Language.VIETNAMESE,
        Language.YORUBA,
        Language.ZULU,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
    for sensitivity_label in ["ALL", "CA", "CS", "UNK"]
]


# There are only these subsets in the African MMLU
AFRI_MMLU_SUBSETS = [
    "elementary_mathematics",
    "high_school_mathematics",
    "high_school_geography",
    "high_school_microeconomics",
    "international_law",
    "global_facts",
]
# African MMLU: African Massive Multitask Language Understanding
# From https://arxiv.org/abs/2406.03368. Human translated MMLU.
afri_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"afri_mmlu_{language.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": LETTER_INDICES.index(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="masakhane/afrimmlu",
        # Temporary until the pr is merged
        hf_revision="refs/pr/1",
        hf_subset=language.value,
        hf_filter=partial(lambda subset, line: line["subject"] == subset, subset),
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in AFRI_MMLU_SUBSETS
    for language in [
        Language.AMHARIC,
        # Language.EWE,
        Language.FRENCH,
        # Language.HAUSA,
        # Language.IGBO,
        # Language.KINYARWANDA,
        # Language.LINGALA,
        # Language.LUGANDA,
        # Language.OROMO,
        # Language.SHONA,
        # Language.SOTHO,
        Language.SWAHILI,
        # Language.TWI,
        # Language.WOLOF,
        # Language.XHOSA,
        Language.YORUBA,
        # Language.ZULU,
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
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="ai-forever/MERA",
        hf_subset="rummlu",
        hf_filter=lambda x: x["meta"]["domain"] == subset,
        evaluation_splits=("public_test",),
        hf_avail_splits=["public_test"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
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
        name=f"community_mmlu_{Language.TURKISH.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.TURKISH,
            lambda line: {"question": line["question"], "choices": line["choices"], "gold_idx": int(line["answer"])},
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="malhajar/mmlu_tr-v0.2",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
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
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="haonan-li/cmmlu",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
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
    "Islamic Studies",
    "Islamic Studies (Middle School)",
    "Islamic Studies (Primary School)",
    "Islamic Studies (High School)",
    "Driving Test",
    "Natural Science (Middle School)",
    "Natural Science (Primary School)",
    "History (Middle School)",
    "History (Primary School)",
    "History (High School)",
    "General Knowledge",
    "General Knowledge (Middle School)",
    "General Knowledge (Primary School)",
    "Law (Professional)",
    "Physics (High School)",
    "Social Science (Middle School)",
    "Social Science (Primary School)",
    "Management (University)",
    "Arabic Language (Middle School)",
    "Arabic Language (Primary School)",
    "Arabic Language (High School)",
    "Political Science (University)",
    "Philosophy (High School)",
    "Accounting (University)",
    "Computer Science (Middle School)",
    "Computer Science (Primary School)",
    "Computer Science (High School)",
    "Computer Science (University)",
    "Geography (Middle School)",
    "Geography (Primary School)",
    "Geography (High School)",
    "Math (Primary School)",
    "Biology (High School)",
    "Economics (Middle School)",
    "Economics (High School)",
    "Economics (University)",
    "Arabic Language (General)",
    "Arabic Language (Grammar)",
    "Civics (Middle School)",
    "Civics (High School)",
]

arabic_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_{Language.ARABIC.value}_{formulation.name.lower()}:{normalize_subset(subset)}",
        prompt_function=get_mcq_prompt_function(
            Language.ARABIC,
            lambda line: {
                "context": line["Context"],
                "question": line["Question"],
                "choices": [str(o) for o in [line[f"Option {i}"] for i in range(1, 6)] if o],
                "gold_idx": LETTER_INDICES.index(line["Answer Key"]),
            },
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="MBZUAI/ArabicMMLU",
        hf_subset=subset,
        evaluation_splits=("test",),
        hf_avail_splits=["dev"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in ARABIC_MMLU_SUBSETS
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]


TURKISH_MMLU_SUBSET = [
    "Biology",
    "Chemistry",
    "Geography",
    "History",
    "Mathematics",
    "Philosophy",
    "Physics",
    "Religion_and_Ethics",
    "Turkish_Language_and_Literature",
]

turkish_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_{Language.TURKISH.value}_{formulation.name.lower()}:{normalize_subset(subset)}",
        prompt_function=get_mcq_prompt_function(
            Language.TURKISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": LETTER_INDICES.index(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="AYueksel/TurkishMMLU",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in TURKISH_MMLU_SUBSET
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
        *openai_mmlu_tasks,
        *arabic_mmlu_tasks,
        *turkish_mmlu_tasks,
        *afri_mmlu_tasks,
        *global_mmlu_tasks,
    ]
)


# ---------------------------- ARC ---------------------------- #
# ARC (AI2 Reasoning Challenge) is a dataset for question answering that requires reasoning.
# It consists of multiple-choice science questions from 3rd to 9th grade exams.
# The dataset is split into two parts: ARC-Easy and ARC-Challenge.
# ARC-Easy contains questions that can be answered correctly by both humans and simple baseline models.
# ARC-Challenge contains questions that are difficult for both humans and current AI systems.

# Similar to MMLU, ARC tasks uses PMI normalization by default but only for the challenge set.


# github: https://github.com/nlp-uoregon/mlmm-evaluation
mlmm_arc_challenge_tasks = [
    LightevalTaskConfig(
        name=f"mlmm_arc_{language.value}_{formulation.name.lower()}:challenge",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="jon-tow/okapi_arc_challenge",
        hf_subset=standardize_tag(language.value),
        hf_revision="823d5d7bfaf8974a3ab52a825b6cf4903b35dbc4",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="train",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
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
        prompt_function=get_mcq_prompt_function(Language.ARABIC, alghafa_adapter, formulation=formulation),
        suite=["lighteval"],
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
        hf_subset="arc_easy_ar",
        hf_revision="08663706ee7cab30c4b7dc1bb00042a3227ce1ff",
        trust_dataset=True,
        evaluation_splits=["test"],
        few_shots_split="validation",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]


lumi_arc = [
    LightevalTaskConfig(
        name=f"lumi_arc_{language.value}_{formulation.name.lower()}:challenge",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=["lighteval"],
        hf_repo="LumiOpen/arc_challenge_mt",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=["test"],
        few_shots_split="validation",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
    for language in [
        Language.DANISH,
        Language.GERMAN,
        Language.GREEK,
        Language.SPANISH,
        Language.FINNISH,
        Language.HUNGARIAN,
        Language.ITALIAN,
        # Language.NORWEGIAN_BOKMAL,
        Language.POLISH,
        Language.PORTUGUESE,
        Language.SWEDISH,
    ]
]

# Turkish ARC
# Comes from the Turkish leaderboard
turkish_arc_tasks = [
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
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="malhajar/arc-tr",
        hf_subset=f"ARC-{subset.capitalize()}",
        evaluation_splits=("test",),
        hf_avail_splits=["train"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ]
            + ([loglikelihood_acc_metric(normalization=LogProbPMINorm())] if subset == "challenge" else []),  # type: ignore
        ),
    )
    for subset in ["easy", "challenge"]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

hindi_arc_tasks = [
    LightevalTaskConfig(
        name=f"community_arc_{Language.HINDI.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.HINDI,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="ai4bharat/ai2_arc-hi",
        hf_subset=f"ARC-{subset.capitalize()}",
        evaluation_splits=("test",),
        few_shots_split="validation",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ]
            + ([loglikelihood_acc_metric(normalization=LogProbPMINorm())] if subset == "challenge" else []),  # type: ignore
        ),
    )
    for subset in ["easy", "challenge"]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

arabic_arc_tasks = [
    LightevalTaskConfig(
        name=f"alghafa_arc_{Language.ARABIC.value}_{formulation.name.lower()}:easy",
        prompt_function=get_mcq_prompt_function(Language.ARABIC, alghafa_adapter, formulation=formulation),
        suite=["lighteval"],
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
        hf_revision="08663706ee7cab30c4b7dc1bb00042a3227ce1ff",
        hf_subset="arc_easy_ar",
        evaluation_splits=["test"],
        few_shots_split="validation",
        few_shots_select="sequential",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
        trust_dataset=True,
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

swahili_arc_tasks = [
    LightevalTaskConfig(
        name=f"community_arc_{Language.SWAHILI.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.SWAHILI,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo=f"Mollel/ARC_{subset.capitalize()}_SWH",
        hf_subset="default",
        hf_revision="5347439d3193c8a0dabaab3819914bf076dc94d4"
        if subset == "easy"
        else "dc1df9df632d14c251594d9129fb833d2ca4429c",
        evaluation_splits=("test",),
        few_shots_split="train",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ]
            + ([loglikelihood_acc_metric(normalization=LogProbPMINorm())] if subset == "challenge" else []),  # type: ignore
        ),
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
        *lumi_arc,
        *turkish_arc_tasks,
        *hindi_arc_tasks,
        *swahili_arc_tasks,
        *arabic_arc_tasks,
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
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="jon-tow/okapi_truthfulqa",
        hf_subset=standardize_tag(language.value),
        hf_revision="cdd5db1a66fd04105622109d1c2a5cbc8cde7586",
        trust_dataset=True,
        evaluation_splits=("validation",),
        hf_avail_splits=["validation"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="malhajar/truthful_qa-tr-v0.2",
        hf_subset="default",
        evaluation_splits=("validation",),
        hf_avail_splits=["validation"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
            formulation=formulation,
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
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
            formulation=formulation,
        ),
        hf_repo="chiayewken/m3exam",
        hf_subset=LangCodeLanguage(standardize_tag(language.value)).language_name().lower(),
        evaluation_splits=("test",),
        few_shots_split="dev",
        generation_size=-1,
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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

thai_exams_tasks = [
    LightevalTaskConfig(
        name=f"thai_exams_{Language.THAI.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(Language.THAI, thai_exams_adapter, formulation=formulation),
        suite=("lighteval",),
        hf_repo="scb10x/thai_exam",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="train",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
# Uses PMI normalization
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
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="INK-USC/xcsr",
        hf_subset=f"X-CSQA-{standardize_tag(language.value) if language != Language.JAPANESE else 'jap'}",
        hf_filter=lambda x: all(
            len(x["question"]["choices"]["text"][i].strip()) > 0 for i in range(len(x["question"]["choices"]["text"]))
        ),
        evaluation_splits=("validation",),
        hf_avail_splits=["validation"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
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
        prompt_function=get_mcq_prompt_function(Language.ARABIC, alghafa_adapter, formulation=formulation),
        suite=["lighteval"],
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
        hf_revision="08663706ee7cab30c4b7dc1bb00042a3227ce1ff",
        hf_subset="piqa_ar",
        hf_avail_splits=["test", "validation"],
        evaluation_splits=["test"],
        few_shots_split="validation",
        trust_dataset=True,
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
        prompt_function=get_mcq_prompt_function(Language.ARABIC, alghafa_adapter, formulation=formulation),
        suite=["lighteval"],
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
        hf_subset="openbook_qa_ext_ar",
        hf_revision="08663706ee7cab30c4b7dc1bb00042a3227ce1ff",
        trust_dataset=True,
        evaluation_splits=["test"],
        few_shots_split="validation",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# Spanish version of OpenBookQA from BSC Language Technology group
# Dataset: https://huggingface.co/datasets/BSC-LT/openbookqa-es
openbook_es_tasks = [
    LightevalTaskConfig(
        name=f"openbookqa_{Language.SPANISH.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.SPANISH,
            lambda line: {
                "question": line["question_stem"],
                "choices": line["choices"]["text"],
                "gold_idx": LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=["lighteval"],
        hf_repo="BSC-LT/openbookqa-es",
        hf_subset="default",
        evaluation_splits=("test",),
        few_shots_split="validation",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
            formulation=formulation,
        ),
        suite=["lighteval"],
        hf_repo="ai-forever/MERA",
        hf_subset="ruopenbookqa",
        evaluation_splits=("train",),
        hf_avail_splits=["train"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
        *openbook_es_tasks,
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
            formulation=formulation,
        ),
        suite=["lighteval"],
        hf_repo="OALL/AlGhafa-Arabic-LLM-Benchmark-Translated",
        hf_subset="sciq_ar",
        hf_revision="08663706ee7cab30c4b7dc1bb00042a3227ce1ff",
        hf_avail_splits=["test", "validation"],
        evaluation_splits=["test"],
        few_shots_split="validation",
        few_shots_select="sequential",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="ai-forever/MERA",
        hf_subset="mathlogicqa",
        evaluation_splits=("train",),
        hf_avail_splits=["train"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [
        CFFormulation(),
        MCFFormulation(),
        HybridFormulation(),
    ]
]

cmath_tasks = [
    LightevalTaskConfig(
        name=f"cmath_{Language.CHINESE.value}",
        prompt_function=get_qa_prompt_function(
            Language.CHINESE,
            lambda line: {
                "question": line["question"],
                "choices": [line["golden"]],
            },
        ),
        suite=("lighteval",),
        hf_repo="weitianwen/cmath",
        hf_subset="default",
        evaluation_splits=("test",),
        few_shots_split="validation",
        generation_size=25,
        metric=[
            multilingual_quasi_exact_match_metric(Language.CHINESE, "full"),
        ],
        stop_sequence=("\n",),
    )
]

mgsm_tasks = [
    LightevalTaskConfig(
        name=f"mgsm_{language.value}",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                # The cot is available but we have no use:
                # line["answer"]
                "choices": [str(line["answer_number"])],
            },
        ),
        suite=("lighteval",),
        hf_repo="juletxara/mgsm",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=25,
        metric=[
            multilingual_quasi_exact_match_metric(language, "full"),
        ],
        stop_sequence=("\n",),
    )
    for language in [
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.GERMAN,
        Language.RUSSIAN,
        Language.CHINESE,
        Language.JAPANESE,
        Language.THAI,
        Language.SWAHILI,
        Language.BENGALI,
        Language.TELUGU,
    ]
]
# African MGSM: MGSM for African Languages
# From https://arxiv.org/abs/2406.03368. Human translated MGSM.
afri_mgsm_tasks = [
    LightevalTaskConfig(
        name=f"afri_mgsm_{language.value}",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                # The cot is available but we have no use:
                # line["answer"]
                "choices": [str(line["answer_number"])],
            },
        ),
        suite=("lighteval",),
        hf_repo="masakhane/afrimgsm",
        hf_subset=language.value,
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=25,
        metric=[
            multilingual_quasi_exact_match_metric(language, "full"),
        ],
        stop_sequence=("\n",),
    )
    for language in [
        Language.AMHARIC,
        # Language.EWE,
        Language.FRENCH,
        # Language.HAUSA,
        # Language.IGBO,
        # Language.KINYARWANDA,
        # Language.LINGALA,
        # Language.LUGANDA,
        # Language.OROMO,
        # Language.SHONA,
        # Language.SOTHO,
        Language.SWAHILI,
        # Language.TWI,
        # Language.WOLOF,
        # Language.XHOSA,
        Language.YORUBA,
        # Language.ZULU,
    ]
]
TASKS_TABLE.extend(
    [
        *cmath_tasks,
        *mathlogicqa_rus_tasks,
        *mgsm_tasks,
        *afri_mgsm_tasks,
    ]
)

# ------------------------------- Misc ------------------------------- #

# AGIEval: Chinese AGI Evaluation suite (Excluding the english subsets)
# Uses PMI normalization
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
                agieval_adapter,
                Language.CHINESE,
                formulation,
            ),
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo=f"hails/agieval-{subset}",
        hf_subset="default",
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        few_shots_split=None,
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in CHINESE_AGIEVAL_SUBSET
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
        prompt_function=get_mcq_prompt_function(
            Language.CHINESE,
            partial(
                ceval_adapter,
                Language.CHINESE,
                formulation,
            ),
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="ceval/ceval-exam",
        hf_subset=subset,
        evaluation_splits=("val",),
        few_shots_split="dev",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for subset in CEVAL_SUBSET
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]


# OAB Exams: A collection of questions from the Brazilian Bar Association exam
# The exam is required for anyone who wants to practice law in Brazil
# Dataset: https://huggingface.co/datasets/eduagarcia/oab_exams
oab_exams_tasks = [
    LightevalTaskConfig(
        name=f"oab_exams_{Language.PORTUGUESE.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.PORTUGUESE,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="eduagarcia/oab_exams",
        hf_subset="default",
        evaluation_splits=("train",),
        hf_avail_splits=["train"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

# ENEM (Exame Nacional do Ensino Médio) is a standardized Brazilian national secondary
# education examination. The exam is used both as a university admission test and as a
# high school evaluation test.
# Dataset: https://huggingface.co/datasets/maritaca-ai/enem
enem_tasks = [
    LightevalTaskConfig(
        name=f"enem_{Language.PORTUGUESE.value}_{formulation.name.lower()}:{year}",
        prompt_function=get_mcq_prompt_function(
            Language.PORTUGUESE,
            partial(
                enem_adapter,
                Language.PORTUGUESE,
            ),
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="maritaca-ai/enem",
        hf_subset=year,
        evaluation_splits=("train",),
        hf_avail_splits=["train"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for year in ["2022", "2023", "2024"]
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
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="ai-forever/MERA",
        hf_subset="ruworldtree",
        evaluation_splits=("train",),
        hf_avail_splits=["train"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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
        *ceval_tasks,
        *oab_exams_tasks,
        *enem_tasks,
    ]
)


# ------------------------------- Continuation Tasks ------------------------------- #
xcodah_tasks = [
    LightevalTaskConfig(
        name=f"xcodah_{language.value}_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(language, partial(xcodah_adapter, language), formulation=formulation),
        suite=("lighteval",),
        hf_repo="INK-USC/xcsr",
        hf_subset=f"X-CODAH-{standardize_tag(language.value) if language != Language.JAPANESE else 'jap'}",
        evaluation_splits=("validation",),
        hf_avail_splits=["validation"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
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

xstory_tasks = [
    LightevalTaskConfig(
        name=f"xstory_cloze_{lang.value}_{formulation.name.lower()}",
        prompt_function=get_continuation_prompt_function(
            lang,
            partial(
                lambda lang, line: {
                    "context": TRANSLATION_LITERALS[lang].sentence_space.join(
                        [
                            line["input_sentence_1"],
                            line["input_sentence_2"],
                            line["input_sentence_3"],
                            line["input_sentence_4"],
                        ]
                    ),
                    "continuations": [line["sentence_quiz1"], line["sentence_quiz2"]],
                    "gold_idx": int(line["answer_right_ending"]) - 1,  # type: ignore
                },
                lang,
            ),
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="juletxara/xstory_cloze",
        hf_subset=standardize_tag(lang.value),
        evaluation_splits=["eval"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for lang in [
        Language.RUSSIAN,
        Language.CHINESE,
        Language.SPANISH,
        Language.ARABIC,
        Language.HINDI,
        Language.INDONESIAN,
        Language.TELUGU,
        Language.SWAHILI,
        Language.BASQUE,
        Language.BURMESE,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(
    [
        *xcodah_tasks,
        *xstory_tasks,
    ]
)

# ------------------------------- Winogrande Tasks ------------------------------- #

xwinograd_tasks = [
    LightevalTaskConfig(
        name=f"xwinograd_{language.value}_{formulation.name.lower()}",
        suite=("lighteval",),
        prompt_function=get_continuation_prompt_function(
            language, partial(winogrand_adapter, language), formulation=formulation
        ),
        hf_repo="Muennighoff/xwinograd",
        hf_subset=standardize_tag(language.value) if language != Language.JAPANESE else "jp",
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        metric=[
            loglikelihood_acc_metric(normalization=None),
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
            loglikelihood_acc_metric(normalization=LogProbCharNorm()),
        ],
    )
    for language in [
        Language.ENGLISH,
        Language.FRENCH,
        Language.JAPANESE,
        Language.PORTUGUESE,
        Language.RUSSIAN,
        Language.CHINESE,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

winograd_turkish_task = [
    LightevalTaskConfig(
        name=f"community_xwinograd_{Language.TURKISH.value}_{formulation.name.lower()}",
        suite=("lighteval",),
        prompt_function=get_continuation_prompt_function(
            Language.TURKISH, partial(winogrand_adapter, Language.TURKISH), formulation=formulation
        ),
        hf_repo="malhajar/winogrande-tr-v0.2",
        hf_subset="default",
        evaluation_splits=("validation",),
        few_shots_split="train",
        metric=[
            loglikelihood_acc_metric(normalization=None),
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
            loglikelihood_acc_metric(normalization=LogProbCharNorm()),
        ],
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(
    [
        *xwinograd_tasks,
        *winograd_turkish_task,
    ]
)

# ------------------------------- General QA tasks ------------------------------- #

MKQA_TASK_TO_ID = {
    "entity": 0,
    "long_answer": 1,
    # "unanswerable": 2,
    "date": 3,
    "number": 4,
    "number_with_unit": 5,
    "short_phrase": 6,
    "binary": 7,
}

mkqa_tasks = [
    LightevalTaskConfig(
        name=f"mkqa_{language.value}:{subset}",
        prompt_function=get_qa_prompt_function(language, partial(get_mkqa_adapter, language)),
        suite=("lighteval",),
        hf_repo="apple/mkqa",
        hf_subset="mkqa",
        hf_revision="325131889721ae0ed885b76ecb8011369d75abad",
        hf_filter=partial(
            lambda language, subset, line: line["answers"][
                "zh_cn" if language == Language.CHINESE else standardize_tag(language.value)
            ][0]["type"]
            == MKQA_TASK_TO_ID[subset],
            language,
            subset,
        ),
        trust_dataset=True,
        evaluation_splits=("train",),
        hf_avail_splits=["train"],
        stop_sequence=("\n",),
        metric=[
            multilingual_quasi_exact_match_metric(language, "prefix"),
            multilingual_quasi_f1_score_metric(language),
        ]
        if subset in ["entity", "long_answer", "short_phrase"]
        else [
            multilingual_quasi_exact_match_metric(language, "full"),
        ],
    )
    for subset in MKQA_TASK_TO_ID.keys()
    for language in [
        Language.ARABIC,
        Language.DANISH,
        Language.GERMAN,
        Language.ENGLISH,
        Language.SPANISH,
        Language.FINNISH,
        Language.FRENCH,
        Language.HEBREW,
        Language.HUNGARIAN,
        Language.ITALIAN,
        Language.JAPANESE,
        Language.KOREAN,
        Language.KHMER,
        Language.MALAY,
        Language.DUTCH,
        Language.NORWEGIAN,
        Language.POLISH,
        Language.PORTUGUESE,
        Language.RUSSIAN,
        Language.SWEDISH,
        Language.THAI,
        Language.TURKISH,
        Language.VIETNAMESE,
        Language.CHINESE,  # Simplified
        # Language.CHINESE_HONG_KONG,
        # Language.CHINESE_TRADITIONAL,
    ]
]

mintaka_tasks = [
    LightevalTaskConfig(
        name=f"mintaka_{lang.value}",
        prompt_function=get_qa_prompt_function(
            lang,
            lambda line: {
                "question": line["question"],
                "choices": [line["answerText"]],
            },
        ),
        suite=("lighteval",),
        hf_repo="AmazonScience/mintaka",
        hf_subset=standardize_tag(lang.value),
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=400,
        stop_sequence=("\n",),
        metric=[
            multilingual_quasi_exact_match_metric(lang, "prefix"),
            multilingual_quasi_f1_score_metric(lang),
        ],
    )
    for lang in [
        Language.ARABIC,
        Language.GERMAN,
        Language.ENGLISH,
        Language.SPANISH,
        Language.FRENCH,
        Language.HINDI,
        Language.ITALIAN,
        Language.JAPANESE,
        Language.PORTUGUESE,
    ]
]

french_triviqa_tasks = [
    LightevalTaskConfig(
        name=f"community_triviaqa_{Language.FRENCH.value}",
        prompt_function=get_qa_prompt_function(
            Language.FRENCH,
            lambda line: {
                "question": line["Question"],
                "choices": [line["Answer"]],
            },
        ),
        suite=("lighteval",),
        hf_repo="manu/french-trivia",
        hf_subset="default",
        evaluation_splits=("train",),
        hf_avail_splits=["train"],
        generation_size=400,
        stop_sequence=("\n",),
        metric=[
            multilingual_quasi_exact_match_metric(Language.FRENCH, "prefix"),
            multilingual_quasi_f1_score_metric(Language.FRENCH),
        ],
    )
]


chegeka_tasks = [
    LightevalTaskConfig(
        name=f"chegeka_{Language.RUSSIAN.value}",
        prompt_function=get_qa_prompt_function(
            Language.RUSSIAN,
            lambda line: {
                "question": line["inputs"]["text"],
                "choices": [line["outputs"]],
            },
        ),
        suite=("lighteval",),
        hf_repo="ai-forever/MERA",
        hf_subset="chegeka",
        evaluation_splits=("train",),
        hf_avail_splits=["train"],
        generation_size=400,
        stop_sequence=("\n",),
        metric=[
            multilingual_quasi_exact_match_metric(Language.RUSSIAN, "prefix"),
            multilingual_quasi_f1_score_metric(Language.RUSSIAN),
        ],
    )
]

TASKS_TABLE.extend(
    [
        *mkqa_tasks,
        *mlqa_tasks,
        *chegeka_tasks,
        *mintaka_tasks,
        *french_triviqa_tasks,
    ]
)


# ------------------------------- BoolQ Tasks (yes/no) ------------------------------- #
ACVA_SUBSET = [
    "Algeria",
    "Ancient_Egypt",
    "Arab_Empire",
    "Arabic_Architecture",
    "Arabic_Art",
    "Arabic_Astronomy",
    "Arabic_Calligraphy",
    "Arabic_Ceremony",
    "Arabic_Clothing",
    "Arabic_Culture",
    "Arabic_Food",
    "Arabic_Funeral",
    "Arabic_Geography",
    "Arabic_History",
    "Arabic_Language_Origin",
    "Arabic_Literature",
    "Arabic_Math",
    "Arabic_Medicine",
    "Arabic_Music",
    "Arabic_Ornament",
    "Arabic_Philosophy",
    "Arabic_Physics_and_Chemistry",
    "Arabic_Wedding",
    "Bahrain",
    "Comoros",
    "Egypt_modern",
    "InfluenceFromAncientEgypt",
    "InfluenceFromByzantium",
    "InfluenceFromChina",
    "InfluenceFromGreece",
    "InfluenceFromIslam",
    "InfluenceFromPersia",
    "InfluenceFromRome",
    "Iraq",
    "Islam_Education",
    "Islam_branches_and_schools",
    "Islamic_law_system",
    "Jordan",
    "Kuwait",
    "Lebanon",
    "Libya",
    "Mauritania",
    "Mesopotamia_civilization",
    "Morocco",
    "Oman",
    "Palestine",
    "Qatar",
    "Saudi_Arabia",
    "Somalia",
    "Sudan",
    "Syria",
    "Tunisia",
    "United_Arab_Emirates",
    "Yemen",
    "communication",
    "computer_and_phone",
    "daily_life",
    "entertainment",
]

acva_tasks = [
    LightevalTaskConfig(
        name=f"acva_{Language.ARABIC.value}:{subset}",
        prompt_function=get_boolq_prompt_function(
            Language.ARABIC,
            lambda line: {
                "question": line["question"],
                "answer": line["answer"] == "صح",
            },
            formulation=CFFormulation(),
        ),
        suite=("lighteval",),
        hf_repo="OALL/ACVA",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="validation",
        metric=[multilingual_quasi_exact_match_metric(Language.ARABIC, "full"), loglikelihood_acc_metric()],
        generation_size=5,
        stop_sequence=("\n",),
    )
    for subset in ACVA_SUBSET
]


french_boolq_tasks = [
    LightevalTaskConfig(
        name=f"community_boolq_{Language.FRENCH.value}",
        prompt_function=get_boolq_prompt_function(
            Language.FRENCH,
            lambda line: {
                "question": line["question"],
                "answer": line["label"] == 1,
                "context": line["passage"],
            },
            formulation=CFFormulation(),
        ),
        suite=("lighteval",),
        hf_repo="manu/french_boolq",
        hf_subset="default",
        evaluation_splits=("test",),
        few_shots_split="valid",
        generation_size=5,
        stop_sequence=["\n"],
        metric=[multilingual_quasi_exact_match_metric(Language.FRENCH, "full"), loglikelihood_acc_metric()],
    )
]

hindi_boolq_tasks = [
    LightevalTaskConfig(
        name=f"community_boolq_{language.value}",
        prompt_function=get_boolq_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "answer": line["answer"],
                "context": line["passage"],
            },
            formulation=CFFormulation(),
        ),
        suite=("lighteval",),
        hf_repo="ai4bharat/boolq-hi",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("validation",),
        few_shots_split="train",
        generation_size=5,
        stop_sequence=["\n"],
        metric=[multilingual_quasi_exact_match_metric(language, "full"), loglikelihood_acc_metric()],
    )
    for language in [
        Language.HINDI,
        Language.GUJARATI,
        Language.MALAYALAM,
        Language.MARATHI,
        Language.TAMIL,
    ]
]


TASKS_TABLE.extend(
    [
        *acva_tasks,
        *french_boolq_tasks,
        *hindi_boolq_tasks,
    ]
)

# ------------------------------- Translation Tasks ------------------------------- #
flores_200_languages = [
    # "ace_Arab",
    "ace_Latn",
    "acm_Arab",
    "acq_Arab",
    "aeb_Arab",
    "afr_Latn",
    "ajp_Arab",
    "aka_Latn",
    "amh_Ethi",
    "apc_Arab",
    "arb_Arab",
    # "arb_Latn",
    "ars_Arab",
    "ary_Arab",
    "arz_Arab",
    "asm_Beng",
    "ast_Latn",
    "awa_Deva",
    "ayr_Latn",
    "azb_Arab",
    "azj_Latn",
    "bak_Cyrl",
    "bam_Latn",
    "ban_Latn",
    "bel_Cyrl",
    "bem_Latn",
    "ben_Beng",
    "bho_Deva",
    # "bjn_Arab",
    "bjn_Latn",
    "bod_Tibt",
    "bos_Latn",
    "bug_Latn",
    "bul_Cyrl",
    "cat_Latn",
    "ceb_Latn",
    "ces_Latn",
    "cjk_Latn",
    "ckb_Arab",
    "crh_Latn",
    "cym_Latn",
    "dan_Latn",
    "deu_Latn",
    "dik_Latn",
    "dyu_Latn",
    "dzo_Tibt",
    "ell_Grek",
    "eng_Latn",
    "epo_Latn",
    "est_Latn",
    "eus_Latn",
    "ewe_Latn",
    "fao_Latn",
    "fij_Latn",
    "fin_Latn",
    "fon_Latn",
    "fra_Latn",
    "fur_Latn",
    "fuv_Latn",
    "gla_Latn",
    "gle_Latn",
    "glg_Latn",
    "grn_Latn",
    "guj_Gujr",
    "hat_Latn",
    "hau_Latn",
    "heb_Hebr",
    "hin_Deva",
    "hne_Deva",
    "hrv_Latn",
    "hun_Latn",
    "hye_Armn",
    "ibo_Latn",
    "ilo_Latn",
    "ind_Latn",
    "isl_Latn",
    "ita_Latn",
    "jav_Latn",
    "jpn_Jpan",
    "kab_Latn",
    "kac_Latn",
    "kam_Latn",
    "kan_Knda",
    # "kas_Arab",
    "kas_Deva",
    "kat_Geor",
    # "knc_Arab",
    "knc_Latn",
    "kaz_Cyrl",
    "kbp_Latn",
    "kea_Latn",
    "khm_Khmr",
    "kik_Latn",
    "kin_Latn",
    "kir_Cyrl",
    "kmb_Latn",
    "kmr_Latn",
    "kon_Latn",
    "kor_Hang",
    "lao_Laoo",
    "lij_Latn",
    "lim_Latn",
    "lin_Latn",
    "lit_Latn",
    "lmo_Latn",
    "ltg_Latn",
    "ltz_Latn",
    "lua_Latn",
    "lug_Latn",
    "luo_Latn",
    "lus_Latn",
    "lvs_Latn",
    "mag_Deva",
    "mai_Deva",
    "mal_Mlym",
    "mar_Deva",
    # "min_Arab",
    "min_Latn",
    "mkd_Cyrl",
    "plt_Latn",
    "mlt_Latn",
    "mni_Beng",
    "khk_Cyrl",
    "mos_Latn",
    "mri_Latn",
    "mya_Mymr",
    "nld_Latn",
    "nno_Latn",
    "nob_Latn",
    "npi_Deva",
    "nso_Latn",
    "nus_Latn",
    "nya_Latn",
    "oci_Latn",
    "gaz_Latn",
    "ory_Orya",
    "pag_Latn",
    "pan_Guru",
    "pap_Latn",
    "pes_Arab",
    "pol_Latn",
    "por_Latn",
    "prs_Arab",
    "pbt_Arab",
    "quy_Latn",
    "ron_Latn",
    "run_Latn",
    "rus_Cyrl",
    "sag_Latn",
    "san_Deva",
    "sat_Olck",
    "scn_Latn",
    "shn_Mymr",
    "sin_Sinh",
    "slk_Latn",
    "slv_Latn",
    "smo_Latn",
    "sna_Latn",
    "snd_Arab",
    "som_Latn",
    "sot_Latn",
    "spa_Latn",
    "als_Latn",
    "srd_Latn",
    "srp_Cyrl",
    "ssw_Latn",
    "sun_Latn",
    "swe_Latn",
    "swh_Latn",
    "szl_Latn",
    "tam_Taml",
    "tat_Cyrl",
    "tel_Telu",
    "tgk_Cyrl",
    "tgl_Latn",
    "tha_Thai",
    "tir_Ethi",
    "taq_Latn",
    "taq_Tfng",
    "tpi_Latn",
    "tsn_Latn",
    "tso_Latn",
    "tuk_Latn",
    "tum_Latn",
    "tur_Latn",
    "twi_Latn",
    "tzm_Tfng",
    "uig_Arab",
    "ukr_Cyrl",
    "umb_Latn",
    "urd_Arab",
    "uzn_Latn",
    "vec_Latn",
    "vie_Latn",
    "war_Latn",
    "wol_Latn",
    "xho_Latn",
    "ydd_Hebr",
    "yor_Latn",
    "yue_Hant",
    "zho_Hans",
    # "zho_Hant",
    "zsm_Latn",
    "zul_Latn",
]


def flores_adapter(lang1, lang2):
    return lambda line: {
        "source_text": line[f"sentence_{lang1}"],
        "target_text": line[f"sentence_{lang2}"],
    }


flores200_tasks = [
    LightevalTaskConfig(
        name=f"flores200:{lang1}-{lang2}",
        prompt_function=get_translation_prompt_function(
            source_language=Language(manage_duplicate_language_codes(lang1.split("_")[0])),
            target_language=Language(manage_duplicate_language_codes(lang2.split("_")[0])),
            adapter=flores_adapter(lang1, lang2),
            formulation=CFFormulation(),
        ),
        suite=("lighteval",),
        hf_repo="facebook/flores",
        hf_subset=f"{lang1}-{lang2}",
        hf_avail_splits=["dev", "devtest"],
        evaluation_splits=["devtest"],
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=300,
        metric=[Metrics.chrf_plus, Metrics.bleu, Metrics.bleu_1, Metrics.bleu_4],
        stop_sequence=["\n"],
        trust_dataset=True,
        version=0,
    )
    for (lang1, lang2) in combinations(flores_200_languages, 2)
]

TASKS_TABLE.extend(
    [
        *flores200_tasks,
    ]
)
