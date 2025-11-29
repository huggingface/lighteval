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

import json
import logging
import tempfile
from functools import partial
from pathlib import Path

from custom_yourbench_task_mcq import yourbench_prompt
from datasets import Dataset, DatasetDict

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig


logger = logging.getLogger(__name__)

save_dir = str(tempfile.mkdtemp())

ds = DatasetDict(
    {
        "train": Dataset.from_dict(
            {
                "question": ["What is 2+2?", "Capital of France?"],
                "choices": [["1", "2", "3", "4"], ["Paris", "Berlin", "Rome", "Madrid"]],
                "gold": [[3], [0]],
            }
        )
    }
)


CustomTaskConfig = partial(
    LightevalTaskConfig,
    prompt_function=yourbench_prompt,
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=16,
    metrics=[Metrics.gpqa_instruct_metric],
    version=0,
)

# Example 1: save to disk (huggingface format) ####

ds.save_to_disk(save_dir)

yourbench_mcq = CustomTaskConfig(
    name="tiny_mcqa_dataset",
    hf_repo="arrow",
    hf_subset="default",
    hf_data_files=f"{save_dir}/**/*.arrow",
)

task = LightevalTask(yourbench_mcq)
eval_docs = task.eval_docs()

print("\n>>READING TASK FROM ARROW<<")
for doc in eval_docs:
    print(doc)


# Example 2: jsonlines format ####

jsonl_path = Path(save_dir) / "train.jsonl"
with open(jsonl_path, "w") as f:
    for row in ds["train"]:
        f.write(json.dumps(row) + "\n")

yourbench_mcq = CustomTaskConfig(
    name="tiny_mcqa_dataset",
    hf_repo="json",
    hf_subset="default",
    hf_data_files=str(jsonl_path),
)

task = LightevalTask(yourbench_mcq)
eval_docs = task.eval_docs()

print("\n>>READING TASK FROM JSONLINES<<")
for doc in eval_docs:
    print(doc)

# TASKS_TABLE = [yourbench_mcq]
