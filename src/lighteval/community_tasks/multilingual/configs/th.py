from typing import get_args

from ..tasks.mqa.thai_exam import ThaiExamsTask, ThaiExamSubset
from ..tasks.mqa.xcopa import XCopaTask
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.m3exam import M3ExamTask
from ..tasks.mqa_with_context.xquad import XquadTask
from ..tasks.nli.xnli import XNLITask
from ..tasks.qa.tydiqa import TydiqaTask

_TASKS = [
    TydiqaTask(lang="th"),
    XquadTask(lang="th"),
    XNLITask(lang="th"),
    XCopaTask(lang="th"),
    M3ExamTask(lang="th"),
    BelebeleTask(lang="th"),
]

_THAI_EXAM = [
    ThaiExamsTask(subset=sb) for sb in get_args(ThaiExamSubset)
]

_TASKS += _THAI_EXAM

_TASKS_STRINGS = ",".join([f"custom|{t.name}|0|1" for t in _TASKS])
TASKS_GROUPS = {
    "all": _TASKS_STRINGS,
}


TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
