from ..tasks.utils.tasks_helpers import tasks_to_string
from ..tasks.mqa.xcopa import XCopaTask
from ..tasks.mqa.mlmm import get_mlmm_tasks
from ..tasks.mqa_with_context.xquad import XquadTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLITask
from ..tasks.qa.mlqa import MlqaTask
from ..tasks.qa.tydiqa import TydiqaTask
from ..tasks.suites.arabic_evals import GENERATIVE_TASKS as ARABIC_EVALS_GENERATIVE_TASKS, MC_TASKS as ARABIC_EVALS_MC_TASKS


_GENERATIVE_TASKS = [
    MlqaTask(lang="ar"),
    TydiqaTask(lang="ar"),
    XquadTask(lang="ar"),
    *ARABIC_EVALS_GENERATIVE_TASKS,
]

_MC_TASKS = [
    XCODAHTask(lang="ar"),
    XCopaTask(lang="ar"),
    XCSQATask(lang="ar"),
    XNLITask(lang="ar", version=1),
    XNLITask(lang="ar", version=2),
    XStoryClozeTask(lang="ar"),
    *get_mlmm_tasks("ar"),
    *ARABIC_EVALS_MC_TASKS,
]

_ALL_TASKS = _GENERATIVE_TASKS + _MC_TASKS

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLITask(lang="ar", version=version) for version in (1, 2)]),
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))