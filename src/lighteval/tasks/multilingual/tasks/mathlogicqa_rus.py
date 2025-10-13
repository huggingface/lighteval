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


from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


# ------------------------------- Math Tasks ------------------------------- #
# MathLogicQA is a dataset for evaluating mathematical reasoning in language models.
# It consists of multiple-choice questions that require logical reasoning and mathematical problem-solving.
# This Russian version is part of the MERA (Multilingual Evaluation of Reasoning Abilities) benchmark.
# MERA: https://github.com/ai-forever/MERA

TASKS_TABLE = []


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
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [
        CFFormulation(),
        MCFFormulation(),
        HybridFormulation(),
    ]
]
