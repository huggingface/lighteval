from tasks.mqa.xcopa import XCopaTask
from tasks.mqa_with_context.m3exam import M3ExamTask
from tasks.nli.xcsr import XCODAHTask, XCSQATask
from tasks.nli.xnli import XNLITask
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.suites.swahili_leaderboard import TASKS as SW_TASKS
from ..tasks.qa.tydiqa import TydiqaTask


_TASKS = [
    BelebeleTask(lang="sw"),
    TydiqaTask(lang="sw"),
    XStoryClozeTask(lang="sw"),
    XCopaTask(lang="sw"),
    XNLITask(lang="sw"),
    M3ExamTask(lang="sw"),
    XCSQATask(lang="sw"),
    XCODAHTask(lang="sw"),
]

_TASKS += SW_TASKS
_TASKS_STRINGS = ",".join([f"custom|{t.name}|0|1" for t in _TASKS])
TASKS_GROUPS = {
    "all": _TASKS_STRINGS,
}


TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
