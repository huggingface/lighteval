from typing import get_args

from ..tasks.qa.mintaka import MintakaTask
from ..tasks.mqa.exams import ExamsTask
from ..tasks.nli.lambada import LambadaTask
from ..tasks.mqa.mlmm import get_mlmm_tasks
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.nli.pawns import PawnsXTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLITask
from ..tasks.nli.xwinograd import XWinogradeTask
from ..tasks.qa.mlqa import MlqaTask
from ..tasks.suites.frenchbench import _TASKS as _FRENCH_BENCH_TASKS

_TASKS = [
    LambadaTask(lang="fr"),
    BelebeleTask(lang="fr"),
    MintakaTask(lang="fr"),
    PawnsXTask(lang="fr"),
    XCODAHTask(lang="fr"),
    XCSQATask(lang="fr"),
    XNLITask(lang="fr"),
    XWinogradeTask(lang="fr"),
]


_MMLM_TASKS = get_mlmm_tasks("fr")

_TASKS += _MMLM_TASKS + _FRENCH_BENCH_TASKS
_TASKS_STRINGS = ",".join([f"custom|{t.name}|0|1" for t in _TASKS])
TASKS_GROUPS = {
    "all": _TASKS_STRINGS,
}


TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
