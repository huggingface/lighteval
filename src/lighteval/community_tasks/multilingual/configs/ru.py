from typing import get_args

from ..tasks.qa.custom_squad import SberSquadTask
from ..tasks.qa.mkqa import MkqaTask, TaskType
from ..tasks.utils.tasks_helpers import tasks_to_string

from ..tasks.suites.mera import GENERATIVE_TASKS as _MERA_GENERATIVE_TASKS, MC_TASKS as _MERA_MC_TASKS, RUMMLU_SUBSET, RCBTask, RuMMLUTask
from ..tasks.mqa.mlmm import get_mlmm_tasks
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.m3exam import M3ExamTask
from ..tasks.mqa_with_context.xquad import XquadTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLITask, XNLI2Task
from ..tasks.nli.xwinograd import XWinogradeTask
from ..tasks.qa.tydiqa import TydiqaTask

_GENERATIVE_TASKS = [
    TydiqaTask(lang="ru"),
    XquadTask(lang="ru"),
    SberSquadTask(),
    *_MERA_GENERATIVE_TASKS,
    *[MkqaTask(lang="ru", type=task_type) for task_type in get_args(TaskType)]
]

_MC_TASKS = [
    BelebeleTask(lang="ru"),
    XCODAHTask(lang="ru"),
    XCSQATask(lang="ru"),
    XNLITask(lang="ru", version=1),
    XNLITask(lang="ru", version=2),
    XNLI2Task(lang="ru", version=1),
    XNLI2Task(lang="ru", version=2),
    XStoryClozeTask(lang="ru"),
    XWinogradeTask(lang="ru"),
    *get_mlmm_tasks("ru"),
    *_MERA_MC_TASKS,
]

_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([RCBTask(version=version) for version in (1, 2)] + [XNLITask(lang="ru", version=version) for version in (1, 2)] + [XNLI2Task(lang="ru", version=version) for version in (1, 2)]),
    "xnli2": tasks_to_string([XNLI2Task(lang="ru", version=2)]),
    "mkqa": tasks_to_string([MkqaTask(lang="ru", type=task_type) for task_type in get_args(TaskType)]),
    "sber_squad": tasks_to_string([SberSquadTask()]),
    "xcodah": tasks_to_string([XCODAHTask("ru")]),
    "winograde": tasks_to_string([XWinogradeTask("ru")]),
    "early-signals": tasks_to_string([
        "arc-ru",
        "belebele-ru",
        "hellaswag-ru",
        "parus",
        *[RuMMLUTask(subset) for subset in get_args(RUMMLU_SUBSET)],
        "ruopenbookqa",
        "tydiqa-ru",
        "x-codah-ru",
        "x-csqa-ru",
        "xnli-2.0-bool-v2-ru",
        "sber_squad",
        "xstory_cloze-ru",
    ]),
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))