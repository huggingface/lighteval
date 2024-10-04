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

# ruff: noqa: F405, F403, F401

"""
This module contains task configurations and prompt functions for evaluating
LLM models on Serbian datasets.
Each task is defined using the `LightevalTaskConfig` class with its respective
prompt function.
The tasks cover a variety of benchmarks, including: ARC[E][C], BoolQ, Hellaswag,
OpenBookQA,PIQA, Winogrande and a custom OZ Eval.
"""

from enum import Enum
from typing import List, Optional

from lighteval.logging.hierarchical_logger import hlog_warn
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


class HFSubsets(Enum):
    """Enum for all available Hugging Face dataset subsets in Serbian evaluation tasks."""

    ARC_EASY = "arc_easy_serbian"
    ARC_CHALLENGE = "arc_challenge_serbian"
    BOOLQ = "boolq_serbian"
    HELLASWAG = "hellaswag_serbian"
    OPENBOOK = "openbookq_serbian"
    PIQA = "piqa_serbian"
    OZ_EVAL = "oz_eval_serbian"
    WINOGRANDE = "winogrande_serbian"


def serbian_eval_prompt(line: dict, task_name: Optional[str] = None) -> Doc:
    """
    Creates a prompt for a multiple-choice task in Serbian.

    Args:
        line: A dictionary containing the query and choices.
        task_name: The name of the task (optional).

    Returns:
        A `Doc` object containing the formatted prompt, choices, and correct answer.
    """
    question = line["query"]
    answer_index = int(line["answer"])
    choices = line["choices"]

    instruction = "Na osnovu sledećeg pitanja, izaberite tačanu opciju iz ponuđenih odgovora.\n"

    query = instruction
    query += f"Pitanje: {question}\n\n"
    query += "Ponuđeni odgovori:\n"

    for index, choice in enumerate(choices):
        query += f"{index}. {choice}\n"

    query += "\n\nKrajnji odgovor:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer_index,
        instruction=instruction,
        target_for_fewshot_sorting=choices[answer_index],
    )


def create_task_config(
    task_name: str,
    prompt_function,
    hf_repo: str,
    hf_subset: str,
    metric: List,
    evaluation_splits: List[str] = ["test"],
    suite: List[str] = ["community"],
    hf_avail_splits: List[str] = ["test", "validation"],
    few_shots_split: str = "validation",
) -> LightevalTaskConfig:
    """
    Creates a task configuration using dependency injection for flexible task creation.

    Args:
        task_name: The name of the task.
        prompt_function: The function to generate task prompts.
        hf_repo: Hugging Face repository.
        hf_subset: Subset of the dataset.
        metric: The metric(s) to use for the task.
        evaluation_splits: The evaluation splits to use (default is "test").
        suite: The suite of tasks.
        hf_avail_splits: Available splits (default is "test", "validation").
        few_shots_split: Split used for few-shot examples.

    Returns:
        A `LightevalTaskConfig` object for the task configuration.
    """
    return LightevalTaskConfig(
        name=task_name,
        prompt_function=prompt_function,
        suite=suite,
        hf_repo=hf_repo,
        hf_subset=hf_subset,
        hf_avail_splits=hf_avail_splits,
        evaluation_splits=evaluation_splits,
        few_shots_split=few_shots_split,
        few_shots_select="sequential",
        metric=metric,
        # Since we use trust_dataset, we have to be careful about what is inside the dataset
        # script. We thus lock the revision to ensure that the script doesn't change
        hf_revision="d356ef19a4eb287e88a51d07a56b73ba88c7f188",
        trust_dataset=True,
        version=0,
    )


arc_easy_serbian_task = create_task_config(
    task_name="serbian_evals:arc_easy",
    prompt_function=serbian_eval_prompt,
    hf_repo="datatab/serbian-llm-benchmark",
    hf_subset=HFSubsets.ARC_EASY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

arc_challenge_serbian_task = create_task_config(
    task_name="serbian_evals:arc_challenge",
    prompt_function=serbian_eval_prompt,
    hf_repo="datatab/serbian-llm-benchmark",
    hf_subset=HFSubsets.ARC_CHALLENGE.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

boolq_serbian_task = create_task_config(
    task_name="serbian_evals:boolq",
    prompt_function=serbian_eval_prompt,
    hf_repo="datatab/serbian-llm-benchmark",
    hf_subset=HFSubsets.BOOLQ.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

hellaswag_serbian_task = create_task_config(
    task_name="serbian_evals:hellaswag",
    prompt_function=serbian_eval_prompt,
    hf_repo="datatab/serbian-llm-benchmark",
    hf_subset=HFSubsets.HELLASWAG.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

openbook_qa_serbian_task = create_task_config(
    task_name="serbian_evals:openbook",
    prompt_function=serbian_eval_prompt,
    hf_repo="datatab/serbian-llm-benchmark",
    hf_subset=HFSubsets.OPENBOOK.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

piqa_serbian_task = create_task_config(
    task_name="serbian_evals:piqa",
    prompt_function=serbian_eval_prompt,
    hf_repo="datatab/serbian-llm-benchmark",
    hf_subset=HFSubsets.PIQA.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

oz_eval_task = create_task_config(
    task_name="serbian_evals:oz_task",
    prompt_function=serbian_eval_prompt,
    hf_repo="datatab/serbian-llm-benchmark",
    hf_subset=HFSubsets.OZ_EVAL.value,
    metric=[Metrics.loglikelihood_acc],
)

winogrande_task = create_task_config(
    task_name="serbian_evals:winogrande",
    prompt_function=serbian_eval_prompt,
    hf_repo="datatab/serbian-llm-benchmark",
    hf_subset=HFSubsets.WINOGRANDE.value,
    metric=[Metrics.loglikelihood_acc_norm],
)


TASKS_TABLE = [
    arc_easy_serbian_task,
    arc_challenge_serbian_task,
    boolq_serbian_task,
    hellaswag_serbian_task,
    openbook_qa_serbian_task,
    piqa_serbian_task,
    oz_eval_task,
    winogrande_task,
]


if __name__ == "__main__":
    hello_message = """
    -----------------------------------
    ------ Serbian LLM benchmark ------
    -----------------------------------

    Available tasks:
    -----------------
    {}
    """
    task_names = "\n".join([t.name for t in TASKS_TABLE])
    hlog_warn(f"{hello_message.format(task_names)}")
