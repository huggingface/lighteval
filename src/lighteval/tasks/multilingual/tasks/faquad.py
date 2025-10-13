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
    MultilingualQuasiExactMatchMetric,
    MultilingualQuasiF1ScoreMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


# FaQuAD: A Portuguese Reading Comprehension Dataset
# https://arxiv.org/abs/2007.15671

TASKS_TABLE = []


faquad_tasks = [
    LightevalTaskConfig(
        name=f"faquad_{Language.PORTUGUESE.value}",
        prompt_function=get_qa_prompt_function(
            Language.PORTUGUESE,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="eraldoluis/faquad",
        hf_subset="plain_text",
        hf_revision="205ba826a2282a4a5aa9bd3651e55ee4f2da1546",
        hf_filter=lambda line: any(len(ans) > 0 for ans in line["answers"]["text"]),
        evaluation_splits=("validation",),
        few_shots_split="train",
        metrics=(
            MultilingualQuasiExactMatchMetric(Language.PORTUGUESE, "prefix"),
            MultilingualQuasiF1ScoreMetric(Language.PORTUGUESE),
        ),
        generation_size=400,
        stop_sequence=("\n",),
    )
]
