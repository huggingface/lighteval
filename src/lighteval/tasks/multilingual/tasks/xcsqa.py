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
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


# ------------------------------- XCSQA ------------------------------- #
# XCSQA (Cross-lingual Commonsense QA) is part of the XCSR (Cross-lingual Commonsense Reasoning) benchmark
# It is a multilingual extension of the CommonsenseQA dataset, covering 16 languages
# The task involves answering multiple-choice questions that require commonsense reasoning
# Uses PMI normalization
# Paper: https://arxiv.org/abs/2110.08462

TASKS_TABLE = []


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
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                LogLikelihoodAccMetric(normalization=LogProbPMINorm()),
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
