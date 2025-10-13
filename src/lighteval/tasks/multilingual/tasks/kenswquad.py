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


# KenSwQuAD: A question answering dataset for Kenyan Swahili.
# https://arxiv.org/abs/2205.02364

TASKS_TABLE = []


kenswquad_tasks = [
    LightevalTaskConfig(
        name=f"kenswquad_{Language.SWAHILI.value}",
        prompt_function=get_qa_prompt_function(
            Language.SWAHILI,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [line["answer"]],
            },
        ),
        suite=("lighteval",),
        hf_repo="lighteval/KenSwQuAD",
        hf_subset="default",
        evaluation_splits=("test",),
        few_shots_split="validation",
        metrics=(
            MultilingualQuasiExactMatchMetric(Language.SWAHILI, "prefix"),
            MultilingualQuasiF1ScoreMetric(Language.SWAHILI),
        ),
        generation_size=400,
        stop_sequence=("\n",),
    )
]
