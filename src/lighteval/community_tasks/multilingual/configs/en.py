from ..tasks.mqa.mlmm import M_ARCTask, M_HellaSwagTask, M_MMLUTask, M_TruthfulQATask
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


_TASKS = [
    BelebeleTask(lang="en"),
    LambadaTask(lang="en"),
    MintakaTask(lang="en"),
    MlqaTask(lang="en"),
    PawnsXTask(lang="en"),
    TydiqaTask(lang="en"),
    XCODAHTask(lang="en"),
    XCSQATask(lang="en"),
    XNLITask(lang="en"),
    XquadTask(lang="en"),
    XStoryClozeTask(lang="en"),
    XWinogradeTask(lang="en"),
    M3ExamTask(lang="en"),
]
_TASKS_STRINGS = ",".join([f"custom|{t.name}|0|1" for t in _TASKS])
TASKS_GROUPS = {
    "all": _TASKS_STRINGS,
}



TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
