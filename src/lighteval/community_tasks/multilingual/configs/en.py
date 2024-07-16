from ..tasks.utils.tasks_helpers import tasks_to_string
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.m3exam import M3ExamTask
from ..tasks.mqa_with_context.xquad import XquadTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.nli.lambada import LambadaTask
from ..tasks.nli.pawns import PawnsXTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLITask
from ..tasks.nli.xwinograd import XWinogradeTask
from ..tasks.qa.mintaka import MintakaTask
from ..tasks.qa.mlqa import MlqaTask
from ..tasks.qa.tydiqa import TydiqaTask


_GENERATIVE_TASKS = [
    MintakaTask(lang="en"),
    MlqaTask(lang="en"),
    TydiqaTask(lang="en"),
    XquadTask(lang="en"),
]

_MC_TASKS = [
    BelebeleTask(lang="en"),
    LambadaTask(lang="en"),
    PawnsXTask(lang="en"),
    XCODAHTask(lang="en"),
    XCSQATask(lang="en"),
    XNLITask(lang="en"),
    XStoryClozeTask(lang="en"),
    XWinogradeTask(lang="en"),
    M3ExamTask(lang="en"),
]

_ALL_TASKS = _GENERATIVE_TASKS + _MC_TASKS

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))