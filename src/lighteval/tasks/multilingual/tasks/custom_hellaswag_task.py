from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def hellaswag_prompt(line, task_name: str = None):
    query = "The following are multiple choice questions (with answers) about common sense.\n\n"
    query += f"Question: {line['activity_label']}: {line['ctx_a']} {line['ctx_b'].capitalize()}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(ascii_uppercase, line["endings"])])
    query += "Answer:"

    gold_ix = int(line["label"]) if line["label"] != "" else -1
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + i for i in ascii_uppercase[: len(line["endings"])]],
        gold_index=gold_ix,
        instruction="The following are multiple choice questions (with answers) about common sense.\n\n",
    )


myswag = LightevalTaskConfig(
    name="myswag",
    prompt_function=hellaswag_prompt,
    hf_repo="Rowan/hellaswag",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.CUSTOM_TTC,
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    myswag,
]
