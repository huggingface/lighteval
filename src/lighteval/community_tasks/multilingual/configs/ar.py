from ..tasks.mqa.xcopa import XCopaTask
from ..tasks.mqa.mlmm import get_mlmm_tasks
from ..tasks.mqa_with_context.xquad import XquadTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLITask
from ..tasks.qa.mlqa import MlqaTask
from ..tasks.qa.tydiqa import TydiqaTask
from ..tasks.suites.arabic_evals import TASKS as ARABIC_EVALS_TASKS


_TASKS = [
    MlqaTask(lang="ar"),
    TydiqaTask(lang="ar"),
    XquadTask(lang="ar"),
    XCODAHTask(lang="ar"),
    XCopaTask(lang="ar"),
    XCSQATask(lang="ar"),
    XNLITask(lang="ar"),
    XStoryClozeTask(lang="ar"),
]


_MMLM_TASKS = get_mlmm_tasks("ar")
_TASKS = _MMLM_TASKS + _TASKS + ARABIC_EVALS_TASKS
_TASKS_STRINGS = ",".join([f"custom|{t.name}|0|1" for t in _TASKS])
TASKS_GROUPS = {
    "all": _TASKS_STRINGS,
}


TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
