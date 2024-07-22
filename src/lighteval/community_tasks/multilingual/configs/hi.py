from ..tasks.utils.tasks_helpers import tasks_to_string
from ..tasks.suites.indic_evals import ARCIndTask, BoolQIndTask, HellaSwagIndTask
from ..tasks.qa.Indicqa import IndicQATask
from ..tasks.mqa.indicxcopa import XCopaIndicTask
from ..tasks.nli.indicnxnli import XNLIIndicTask
from ..tasks.qa.mintaka import MintakaTask
from ..tasks.mqa.mlmm import get_mlmm_tasks
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.xquad import XquadTask
from ..tasks.nli.indicnxnli import XNLIIndicTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLITask
from ..tasks.qa.mlqa import MlqaTask
from ..tasks.qa.tydiqa import TydiqaTask


_GENERATIVE_TASKS = [
    MlqaTask(lang="hi"),
    XquadTask(lang="hi"),
    TydiqaTask(lang="hi"),
    MintakaTask(lang="hi"),
    IndicQATask(lang="hi"),
    BoolQIndTask(),
]

_MC_TASKS = [
    BelebeleTask(lang="hi"),
    XCODAHTask(lang="hi"),
    XCSQATask(lang="hi"),
    XNLITask(lang="hi"),
    XStoryClozeTask(lang="hi"),
    XNLIIndicTask(lang="hi"),
    XCopaIndicTask(lang="hi"),
    ARCIndTask(subset="easy"),
    ARCIndTask(subset="challenge"),
    HellaSwagIndTask(),
    BoolQIndTask(),
    *get_mlmm_tasks("hi")
]


_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLITask(lang="hi"), XNLIIndicTask(lang="hi")]),
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))