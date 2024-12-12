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

from lighteval.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


def mmlu_signs(line, topic):
    prompt = f"The following are multiple choice questions (with answers) about  {topic.replace('_', ' ')}.\n\n"
    prompt += line["question"] + "\n"
    prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(["+", "*", "=", "#"], line["choices"])])
    prompt += "Answer:"

    gold_ix = ["+", "*", "=", "#"].index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]
    is_few_shots = line.get("__few_shots", False)  # We are adding few shots

    return {
        "query": prompt,
        "choices": [" +", " *", " =", " #"] if is_few_shots else ["+", "*", "=", "#"],
        "fewshot_sorting_class": [" +", " *", " =", " #"][gold_ix],
        "gold_index": gold_ix,
        "instruction": f"The following are multiple choice questions (with answers) about  {topic.replace('_', ' ')}.\n\n",
    }


def mmlu_anatomy_signs(line):
    return mmlu_signs(line, "anatomy")


def mmlu_numbers(line, topic):
    prompt = f"The following are multiple choice questions (with answers) about  {topic.replace('_', ' ')}.\n\n"
    prompt += line["question"] + "\n"
    prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(["1", "2", "3", "4"], line["choices"])])
    prompt += "Answer:"

    gold_ix = ["1", "2", "3", "4"].index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]
    is_few_shots = line.get("__few_shots", False)  # We are adding few shots

    return {
        "query": prompt,
        "choices": [" 1", " 2", " 3", " 4"] if is_few_shots else ["1", "2", "3", "4"],
        "fewshot_sorting_class": [" 1", " 2", " 3", " 4"][gold_ix],
        "gold_index": gold_ix,
        "instruction": f"The following are multiple choice questions (with answers) about  {topic.replace('_', ' ')}.\n\n",
    }


def mmlu_anatomy(line):
    return mmlu_numbers(line, "anatomy")


TASKS_TABLE = [
    LightevalTaskConfig(
        name="mmlu:anatomy",
        suite=["custom"],
        prompt_function=mmlu_anatomy,
        hf_repo="lighteval/mmlu",
        hf_subset="anatomy",
        hf_avail_splits=["auxiliary_train", "test", "validation", "dev"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select="sequential",
        generation_size=5,
        metric=[Metrics.loglikelihood_acc_single_token],
        stop_sequence=["\n"],
    ),
    LightevalTaskConfig(
        name="mmlu:anatomy_signs",
        suite=["custom"],
        prompt_function=mmlu_anatomy_signs,
        hf_repo="lighteval/mmlu",
        hf_subset="anatomy",
        hf_avail_splits=["auxiliary_train", "test", "validation", "dev"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select="sequential",
        generation_size=5,
        metric=[Metrics.loglikelihood_acc_single_token],
        stop_sequence=["\n"],
    ),
]
