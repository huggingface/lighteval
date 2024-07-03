from ..tasks.qa.Indicqa import IndicQATask
from ..tasks.nli.indicnxnli import XNLIIndicTask
from ..tasks.mqa.indicxcopa import XCopaIndicTask
from ..tasks.mqa.mlmm import M_ARCTask, M_HellaSwagTask, M_MMLUTask, M_TruthfulQATask
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


_MMLM_TASKS = [
    M_HellaSwagTask(lang="te"),
    M_MMLUTask(lang="te"),
    M_ARCTask(lang="te"),
    M_TruthfulQATask(lang="te", type="mc1"),
    M_TruthfulQATask(lang="te", type="mc2"),
]

_TASKS += _MMLM_TASKS
_TASKS_STRINGS = ",".join([f"custom|{t.name}|0|1" for t in _TASKS])
TASKS_GROUPS = {
    "all": _TASKS_STRINGS,
}


TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
