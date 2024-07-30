from typing import get_args

from ..tasks.mqa.meta_mmlu import MetaMMLUTask, MMLU_SUBSET

from ..tasks.utils.tasks_helpers import tasks_to_string

from ..tasks.nli.wsci import WSCITask
from ..tasks.mqa.thai_exam import ThaiExamsTask, ThaiExamSubset
from ..tasks.mqa.xcopa import XCopaTask
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.m3exam import M3ExamTask
from ..tasks.mqa_with_context.xquad import XquadTask
from ..tasks.nli.xnli import XNLITask
from ..tasks.qa.tydiqa import TydiqaTask
from ..tasks.mqa.xcopa import XCopaTask

_GENERATIVE_TASKS = [
    TydiqaTask(lang="th"),
    XquadTask(lang="th"),
]

_MC_TASKS = [
    XNLITask(lang="th", version=1),
    XNLITask(lang="th", version=2),
    XCopaTask(lang="th"),
    M3ExamTask(lang="th"),
    BelebeleTask(lang="th"),
    WSCITask(lang="th"),
    *[ThaiExamsTask(subset=sb) for sb in get_args(ThaiExamSubset)],
    *[MetaMMLUTask("th", subset) for subset in get_args(MMLU_SUBSET)],
]
_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLITask(lang="th", version=version) for version in (1, 2)]),
    "meta_mmlu": tasks_to_string([MetaMMLUTask("th", subset) for subset in get_args(MMLU_SUBSET)])
}


TASKS_TABLE = [task.as_dict() for task in _GENERATIVE_TASKS + _MC_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
