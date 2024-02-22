# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval. Copy this file and complete it with the info for your task.

This file generally create just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Author:
"""
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks_prompt_formatting import LETTER_INDICES


## EVAL WITH SUBSET ##

# fmt: off
SAMPLE_SUBSETS = [] # list of all the subsets to use for this eval
# fmt: on


class CustomSubsetTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function="prompt_fn",  # must be defined in the file
            hf_repo="",
            metric=[""],
            hf_avail_splits=[],
            evaluation_splits=[],
            few_shots_split="",
            few_shots_select="",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            output_regex=None,
            frozen=False,
        )


SUBSET_TASKS = [CustomSubsetTask(name=f"mytask:{subset}", hf_subset=subset) for subset in SAMPLE_SUBSETS]


# Follow examples in src/lighteval/tasks/tasks_prompt_formatting.py
# Note that the input line is a line of your dataset
def prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query="",
        choices="",
        gold_index=0,
        instruction="",
    )


## EVAL WITH NO SUBSET ##
task = LightevalTaskConfig(
    name="mytask",
    prompt_function="prompt_fn",
    suite=["community"],
    hf_repo="",
    hf_subset="default",
    hf_avail_splits=[],
    evaluation_splits=[],
    few_shots_split="",
    few_shots_select="",
    metric=[""],
)


_TASKS = SUBSET_TASKS + [task]

# Convert to dict for lighteval
TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
