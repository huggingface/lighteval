from typing import get_args

from lighteval.tasks.templates.tasks import MMLU_SUBSET, MetaMMLUTask
from lighteval.utils.language import Language


TASKS = [
    *[MetaMMLUTask(lang=Language.french, task=task) for task in get_args(MMLU_SUBSET)],
]

# Dynamically create variables for each task
for task in TASKS:
    globals()[task.name.replace("-", "_").replace(":", "_")] = task

# This approach ensures that when vars() is called on this module,
# it will include all the dynamically created task variables.
