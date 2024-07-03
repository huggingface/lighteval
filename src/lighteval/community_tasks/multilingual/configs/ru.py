from typing import get_args

from ..tasks.suites.mera import _TASKS as _MERA_TASKS
from ..tasks.mqa.mlmm import M_ARCTask, M_HellaSwagTask, M_MMLUTask, M_TruthfulQATask
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.m3exam import M3ExamTask
from ..tasks.mqa_with_context.xquad import XquadTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLITask
from ..tasks.nli.xwinograd import XWinogradeTask
from ..tasks.qa.tydiqa import TydiqaTask

_TASKS = [
    BelebeleTask(lang="ru"),
    TydiqaTask(lang="ru"),
    XquadTask(lang="ru"),
    XCODAHTask(lang="ru"),
    XCSQATask(lang="ru"),
    XNLITask(lang="ru"),
    XStoryClozeTask(lang="ru"),
    XWinogradeTask(lang="ru"),
]


_MMLM_TASKS = [
    M_HellaSwagTask(lang="ru"),
    M_MMLUTask(lang="ru"),
    M_ARCTask(lang="ru"),
    M_TruthfulQATask(lang="ru", type="mc1"),
    M_TruthfulQATask(lang="ru", type="mc2"),
]


_TASKS += _MMLM_TASKS + _MERA_TASKS
_TASKS_STRINGS = ",".join([f"custom|{t.name}|0|1" for t in _TASKS])
TASKS_GROUPS = {
    "all": _TASKS_STRINGS,
}


TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
