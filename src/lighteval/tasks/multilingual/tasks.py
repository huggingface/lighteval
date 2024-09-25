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

xnli_tasks = [
    LightevalTaskConfig(
        name=f"xnli_{language.value}_{formulation.name.lower()}",
        suite=["custom"],
        metric=[loglikelihood_acc_metric(normalization=LogProbTokenNorm())],
        prompt_function=get_nli_prompt_function(
            language=language,
            adapter=lambda line: {
                "premise": line["premise"],
                "hypothesis": line["hypothesis"],
                # Since we ignore the neural label
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

xnli2_tasks = [
    LightevalTaskConfig(
        name=f"xnli2.0_{language.value}_{formulation.name.lower()}",
        suite=["custom"],
        metric=[loglikelihood_acc_metric(normalization=LogProbTokenNorm())],
        prompt_function=get_nli_prompt_function(
            language=language,
            adapter=lambda line: {
                "premise": line["premise"],
                "hypothesis": line["hypothesis"],
                # Since we ignore the neural label
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

xnli_indic_tasks = [
    LightevalTaskConfig(
        name=f"indicnxnli_{language.value}_{formulation.name.lower()}",
        suite=["custom"],
        prompt_function=get_nli_prompt_function(
            language=language,
            adapter=lambda line: {
                "premise": line["premise"],
                "hypothesis": line["hypothesis"],
                # Since we ignore the neural label
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


TASKS_TABLE = [*xnli_tasks, *xnli2_tasks, *xnli_indic_tasks]
