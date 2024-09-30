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

from lighteval.metrics.dynamic_metrics import loglikelihood_acc_metric
from lighteval.metrics.normalizations import LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.nli import get_nli_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


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


TASKS_TABLE = [*xnli_tasks, *xnli2_tasks, *xnli_indic_tasks, *paws_x_tasks, *rcb_tasks, *ocnli_tasks, *cmnli_tasks]
