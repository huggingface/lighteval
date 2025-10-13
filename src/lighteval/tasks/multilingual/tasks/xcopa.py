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


from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.copa import get_copa_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


# ------------------------------- Copa Tasks ------------------------------- #
# COPA (Choice of Plausible Alternatives) tasks involve determining the most plausible cause or effect
# for a given premise. These tasks test common sense reasoning and causal inference abilities.
# XCOPA: Cross-lingual Choice of Plausible Alternatives
# Paper: https://aclanthology.org/2020.emnlp-main.185/
# XCOPA extends the original English COPA task to 11 typologically diverse languages.

TASKS_TABLE = []


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
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
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
