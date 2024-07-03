from typing import get_args

from ..tasks.suites.indic_evals import ARCIndTask, BoolQIndTask, HellaSwagIndTask
from ..tasks.qa.Indicqa import IndicQATask
from ..tasks.mqa.indicxcopa import XCopaIndicTask
from ..tasks.nli.indicnxnli import XNLIIndicTask
from ..tasks.qa.mintaka import MintakaTask
from ..tasks.mqa.mlmm import M_ARCTask, M_HellaSwagTask, M_MMLUTask, M_TruthfulQATask
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.xquad import XquadTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLITask
from ..tasks.qa.mlqa import MlqaTask
from ..tasks.qa.tydiqa import TydiqaTask


_TASKS = [
    MlqaTask(lang="hi"),
    XquadTask(lang="hi"),
    BelebeleTask(lang="hi"),
    TydiqaTask(lang="hi"),
    MintakaTask(lang="hi"),
    XCODAHTask(lang="hi"),
    XCSQATask(lang="hi"),
    XNLITask(lang="hi"),
    XStoryClozeTask(lang="hi"),
    XNLIIndicTask(lang="hi"),
    XCopaIndicTask(lang="hi"),
    IndicQATask(lang="hi"),
]

_MMLM_TASKS = [
    M_HellaSwagTask(lang="hi"),
    M_MMLUTask(lang="hi"),
    M_ARCTask(lang="hi"),
    M_TruthfulQATask(lang="hi", type="mc1"),
    M_TruthfulQATask(lang="hi", type="mc2"),
]

_INDIC_EVAL_TASKS = [
    ARCIndTask(subset="easy"),
    ARCIndTask(subset="challenge"),
    HellaSwagIndTask(),
    BoolQIndTask(),
]

_TASKS += _INDIC_EVAL_TASKS + _MMLM_TASKS
_TASKS_STRINGS = ",".join([f"custom|{t.name}|0|1" for t in _TASKS])
TASKS_GROUPS = {
    "all": _TASKS_STRINGS,
}


TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
