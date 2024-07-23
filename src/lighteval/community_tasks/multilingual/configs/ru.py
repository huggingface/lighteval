from typing import get_args
from ..tasks.utils.tasks_helpers import tasks_to_string

from ..tasks.suites.mera import GENERATIVE_TASKS as _MERA_GENERATIVE_TASKS, MC_TASKS as _MERA_MC_TASKS, RCBTask
from ..tasks.mqa.mlmm import get_mlmm_tasks
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.m3exam import M3ExamTask
from ..tasks.mqa_with_context.xquad import XquadTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLITask
from ..tasks.nli.xwinograd import XWinogradeTask
from ..tasks.qa.tydiqa import TydiqaTask

_GENERATIVE_TASKS = [
    TydiqaTask(lang="ru"),
    XquadTask(lang="ru"),
    *_MERA_GENERATIVE_TASKS,
]

_MC_TASKS = [
    BelebeleTask(lang="ru"),
    XCODAHTask(lang="ru"),
    XCSQATask(lang="ru"),
    XNLITask(lang="ru", version=1),
    XNLITask(lang="ru", version=2),
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
    "xnli": tasks_to_string([RCBTask(version=version) for version in (1, 2)] + [XNLITask(lang="ru", version=version) for version in (1, 2)]),
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))