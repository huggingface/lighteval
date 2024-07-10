from ..tasks.qa.Indicqa import IndicQATask
from ..tasks.nli.indicnxnli import XNLIIndicTask
from ..tasks.mqa.indicxcopa import XCopaIndicTask
from ..tasks.mqa.mlmm import get_mlmm_tasks
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.qa.tydiqa import TydiqaTask


_TASKS = [
    BelebeleTask(lang="te"),
    TydiqaTask(lang="te"),
    XStoryClozeTask(lang="te"),
    XCopaIndicTask(lang="te"),
    XNLIIndicTask(lang="te"),
    IndicQATask(lang="te"),
]


_MMLM_TASKS = get_mlmm_tasks("te")

_TASKS += _MMLM_TASKS
_TASKS_STRINGS = ",".join([f"custom|{t.name}|0|1" for t in _TASKS])
TASKS_GROUPS = {
    "all": _TASKS_STRINGS,
}


TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
