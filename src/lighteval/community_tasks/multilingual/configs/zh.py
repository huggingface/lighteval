from typing import get_args

from ..tasks.qa.custom_squad import ChineseSQuADTask
from ..tasks.utils.tasks_helpers import tasks_to_string
from ..tasks.mqa.agieval import CHINESE_AGIEVAL_TASK_TYPE, ChineseAgievalTask, MULTICHOICE_JOIN_VARIANT
from ..tasks.mqa.ceval import CEVAL_TASK_TYPE, CEvalTask
from ..tasks.mqa.cmmlu import CMMLU_TASK_TYPE, CMMLUTask
from ..tasks.mqa.mlmm import M_HellaSwagTask, get_mlmm_tasks
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.m3exam import M3ExamTask
from ..tasks.mqa_with_context.xquad import XquadTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.nli.clue import CMNLITask, OCNLI, C3Task, CMRC2018Task
from ..tasks.nli.pawns import PawnsXTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLITask, XNLI2Task
from ..tasks.nli.xwinograd import XWinogradeTask
from ..tasks.qa.cmath import CMathTask
from ..tasks.qa.mlqa import MlqaTask
from ..tasks.qa.tydiqa import TydiqaTask
from ..tasks.mqa.xcopa import XCopaTask
from ..tasks.qa.mkqa import MkqaTask, TaskType



_GENERATIVE_TASKS = [
    *[MkqaTask(lang="zh", type=task_type) for task_type in get_args(TaskType)],
    MlqaTask(lang="zh"),
    TydiqaTask(lang="zh"),
    XquadTask(lang="zh"),
    CMathTask(),
    CMRC2018Task(),
    ChineseSQuADTask(),
]

_MC_TASKS = [
    BelebeleTask(lang="zh"),
    PawnsXTask(lang="zh", version=2),
    XCODAHTask(lang="zh"),
    XCSQATask(lang="zh"),
    XNLITask(lang="zh", version=2),
    XNLI2Task(lang="zh", version=2),
    OCNLI(version=2),
    XStoryClozeTask(lang="zh"),
    XCopaTask(lang="zh"),
    XWinogradeTask(lang="zh"),
    CMNLITask(version=2),
    M3ExamTask(lang="zh", version=1),
    M3ExamTask(lang="zh", version=2),
    C3Task(),
    *[CMMLUTask(task) for task in get_args(CMMLU_TASK_TYPE)],
    *[CEvalTask(task, show_options=False, join_variant=join_variant) for task in get_args(CEVAL_TASK_TYPE) for join_variant in get_args(MULTICHOICE_JOIN_VARIANT)],
    *[ChineseAgievalTask(task, show_options=False, join_variant=join_variant) for task in get_args(CHINESE_AGIEVAL_TASK_TYPE) for join_variant in get_args(MULTICHOICE_JOIN_VARIANT)],
    *get_mlmm_tasks("zh")
]

_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))
TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLITask(lang="zh", version=version) for version in (1, 2)] +
                            [XNLI2Task(lang="zh", version=version) for version in (1, 2)] +
                            [OCNLI(version=version) for version in (1, 2)] +
                            [PawnsXTask(lang="zh", version=version) for version in (1, 2)]),
    "xnli2": tasks_to_string([XNLI2Task(lang="zh", version=version) for version in (1, 2)]),
    "ceval": tasks_to_string([CEvalTask(task, show_options=False, join_variant=join_variant) for task in get_args(CEVAL_TASK_TYPE) for join_variant in get_args(MULTICHOICE_JOIN_VARIANT)]),
    "agieval": tasks_to_string([ChineseAgievalTask(task, show_options=False, join_variant=join_variant) for task in get_args(CHINESE_AGIEVAL_TASK_TYPE) for join_variant in get_args(MULTICHOICE_JOIN_VARIANT)]),
    "ocnli": tasks_to_string([OCNLI(version=version) for version in (1, 2)]),
    "mkqa": tasks_to_string([MkqaTask(lang="zh", type=task_type) for task_type in get_args(TaskType)]),
    "xcodah": tasks_to_string([XCODAHTask(lang="zh")]),
    "m3exam": tasks_to_string([M3ExamTask(lang="zh", version=version) for version in (2,)]),
    "cmnli": tasks_to_string([CMNLITask(version=2)]),
    "squad-zh": tasks_to_string([ChineseSQuADTask()]),
    "early-signals": tasks_to_string([
            *[ChineseAgievalTask(task, show_options=False, join_variant="NEW_LINE") for task in get_args(CHINESE_AGIEVAL_TASK_TYPE) if task != "gaokao-mathqa"],
            "belebele-zh",
            "c3",
            *[CEvalTask(task, show_options=False, join_variant="NEW_LINE") for task in get_args(CEVAL_TASK_TYPE)],
            *[CMMLUTask(subset) for subset in get_args(CMMLU_TASK_TYPE)],
            "cmrc",
            "hellaswag-zh",
            "m3exam-zh",
            "mlqa-zh",
            "x-codah-zh",
            "x-csqa-zh",
            "xcopa-zh",
            "xquad-zh",
            "xstory_cloze-zh",
            "xwinograd-zh",
    ])

}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))