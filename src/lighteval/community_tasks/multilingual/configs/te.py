from typing import get_args
from ..tasks.mqa.custom_hellaswags import CustomHellaswagTeluguTask
from ..tasks.utils.tasks_helpers import tasks_to_string
from ..tasks.qa.Indicqa import IndicQATask
from ..tasks.nli.indicnxnli import XNLIIndicTask
from ..tasks.mqa.indicxcopa import XCopaIndicTask
from ..tasks.mqa.mlmm import MMLU_SUBSET, get_mlmm_tasks, M_MMLUTask
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.qa.tydiqa import TydiqaTask


_GENERATIVE_TASKS = [
    TydiqaTask(lang="te"),
    IndicQATask(lang="te"),
]

_MC_TASKS = [
    XNLIIndicTask(lang="te", version=1),
    XNLIIndicTask(lang="te", version=2),
    XCopaIndicTask(lang="te"),
    BelebeleTask(lang="te"),
    XStoryClozeTask(lang="te"),
    CustomHellaswagTeluguTask(),
    *get_mlmm_tasks("te")
]

_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLIIndicTask(lang="te", version=2)]),
    "custom_hellaswag": tasks_to_string([CustomHellaswagTeluguTask()]),
    "early-signals": tasks_to_string([
        "belebele-te",
        "custom_hellaswag-te",
        "indicqa.te",
        *[M_MMLUTask("te", subset) for subset in get_args(MMLU_SUBSET)],
        "xcopa-te",
        "xstory_cloze-te",
    ])
        
}

TASKS_TABLE = [task.as_dict() for task in _GENERATIVE_TASKS + _MC_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
