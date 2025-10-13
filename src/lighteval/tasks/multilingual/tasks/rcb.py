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
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.nli import get_nli_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


# Russian Commitment Bank (RCB) is a large-scale NLI dataset with Russian sentences,
# collected from the web and crowdsourcing.
# https://arxiv.org/abs/2401.04531

TASKS_TABLE = []


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
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=None),
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]
