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

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
import os

ruler_model = os.environ.get("RULER_MODEL")
ruler_org = os.environ.get("RULER_ORG")
if ruler_model is None:
    raise ValueError("RULER_MODEL environment variable is not set, set it to the model you want to evaluate")
if ruler_org is None:
    raise ValueError("RULER_ORG environment variable is not set, set it to the organization you want to evaluate")
subsets = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]

lengths = [262144,131072, 65536, 32768, 16384, 8192, 4096]

task_configs = []

for subset in subsets:
    for length in lengths:
        task_configs.append(
            LightevalTaskConfig(
                name=f"ruler_{length}:{subset}",
                suite=["lighteval"],
                prompt_function=prompt.ruler,
                hf_repo=f"{ruler_org}/RULER-{length}-{ruler_model}",
                hf_subset="default",
                hf_avail_splits=[subset],
                evaluation_splits=[subset],
                few_shots_split=None,
                few_shots_select=None,
                generation_size=128 if "niah" in subset else 30 if subset == "vt" else 120 if subset == "cwe" else 50,
                metric=[Metrics.ruler_match_any] if subset in ["qa_1", "qa_2"] else [Metrics.ruler_match_all],
                stop_sequence=None,
                trust_dataset=False,
                version=0,
            )
        )

TASKS_TABLE = task_configs
