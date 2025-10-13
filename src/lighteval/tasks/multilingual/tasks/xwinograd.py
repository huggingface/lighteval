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

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import (
    winogrand_adapter,
)
from lighteval.tasks.templates.continuation import get_continuation_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


# ------------------------------- Winogrande Tasks ------------------------------- #

TASKS_TABLE = []


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
        metrics=[
            LogLikelihoodAccMetric(normalization=None),
            LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
            LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
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
