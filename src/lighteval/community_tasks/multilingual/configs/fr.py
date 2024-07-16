from typing import get_args
from ..tasks.utils.tasks_helpers import tasks_to_string

from ..tasks.qa.mintaka import MintakaTask
from ..tasks.nli.lambada import LambadaTask
from ..tasks.mqa.mlmm import get_mlmm_tasks
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.nli.pawns import PawnsXTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLITask
from ..tasks.nli.xwinograd import XWinogradeTask
from ..tasks.suites.frenchbench import _GENERATIVE_TASKS as _FRENCH_BENCH_GENERATIVE_TASKS, _MC_TASKS as _FRENCH_BENCH_MC_TASKS

_GENERATIVE_TASKS = [
    MintakaTask(lang="fr"),
    *_FRENCH_BENCH_GENERATIVE_TASKS,
]

_MC_TASKS = [
    LambadaTask(lang="fr"),
    BelebeleTask(lang="fr"),
    PawnsXTask(lang="fr"),
    XCODAHTask(lang="fr"),
    XCSQATask(lang="fr"),
    XNLITask(lang="fr"),
    XWinogradeTask(lang="fr"),
    *get_mlmm_tasks("fr"),
    *_FRENCH_BENCH_MC_TASKS
]

_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))