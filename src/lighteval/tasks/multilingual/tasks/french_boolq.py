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
    MultilingualQuasiExactMatchMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.boolq import get_boolq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
)
from lighteval.utils.language import Language


TASKS_TABLE = []


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
        metrics=[MultilingualQuasiExactMatchMetric(Language.FRENCH, "full"), LogLikelihoodAccMetric()],
    )
]
