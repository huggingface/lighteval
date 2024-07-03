from typing import get_args
from ..tasks.mqa.agieval import CHINESE_AGIEVAL_TASK_TYPE, ChineseAgievalTask
from ..tasks.mqa.ceval import CEVAL_TASK_TYPE, CEvalTask
from ..tasks.mqa.cmmlu import CMMLU_TASK_TYPE, CMMLUTask
from ..tasks.mqa.mlmm import M_ARCTask, M_HellaSwagTask, M_MMLUTask, M_TruthfulQATask
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.m3exam import M3ExamTask
from ..tasks.mqa_with_context.xquad import XquadTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.nli.clue import C3Task, CMRC2018Task
from ..tasks.nli.pawns import PawnsXTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLITask
from ..tasks.nli.xwinograd import XWinogradeTask
from ..tasks.qa.cmath import CMathTask
from ..tasks.qa.mlqa import MlqaTask
from ..tasks.qa.tydiqa import TydiqaTask
from ..tasks.mqa.xcopa import XCopaTask


_TASKS = [
    BelebeleTask(lang="zh"),
    MlqaTask(lang="zh"),
    TydiqaTask(lang="zh"),
    XquadTask(lang="zh"),
    PawnsXTask(lang="zh"),
    XCODAHTask(lang="zh"),
    XCSQATask(lang="zh"),
    XNLITask(lang="zh"),
    XStoryClozeTask(lang="zh"),
    XCopaTask(lang="zh"),
    XWinogradeTask(lang="zh"),
    M3ExamTask(lang="zh"),
    CMathTask(),
]

_CMMLU_TASKS = [
    CMMLUTask(task) for task in get_args(CMMLU_TASK_TYPE)
]

_CEVAL_TASKS = [
    CEvalTask(task) for task in get_args(CEVAL_TASK_TYPE)
]

_AGIEVAL_TASKS = [
    ChineseAgievalTask(task) for task in get_args(CHINESE_AGIEVAL_TASK_TYPE)
]

_CLUE_TASKS = [
    CMRC2018Task(),
    C3Task(),
]

_MMLM_TASKS = [
    M_HellaSwagTask(lang="zh"),
    M_MMLUTask(lang="zh"),
    M_ARCTask(lang="zh"),
    M_TruthfulQATask(lang="zh", type="mc1"),
    M_TruthfulQATask(lang="zh", type="mc2"),
]

_TASKS += _CMMLU_TASKS + _CEVAL_TASKS + _AGIEVAL_TASKS + _CLUE_TASKS  + _MMLM_TASKS
_TASKS_STRINGS = ",".join([f"custom|{t.name}|0|1" for t in _TASKS])
TASKS_GROUPS = {
    "all": _TASKS_STRINGS,
}


TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
