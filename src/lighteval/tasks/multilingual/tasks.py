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

from langcodes import Language as LangCodeLanguage
from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    loglikelihood_acc_metric,
    multilingual_quasi_exact_match_metric,
    multilingual_quasi_f1_score_metric,
)
from lighteval.metrics.normalizations import LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
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

# Other MCF tasks for RC

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
        *belebele_tasks,
    ]
)
