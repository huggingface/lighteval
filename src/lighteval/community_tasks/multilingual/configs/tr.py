from typing import get_args
from ..tasks.mqa.exams import ExamsTask
from ..tasks.qa.tquad import Tquad2Task
from ..tasks.suites.turkish_leaderboard import ARCEasyTrTask, HellaSwagTrTask, MMLUTaskTr, TruthfulQATrTask, WinogradeTrTask, MMLU_SUBSETS
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.xquad import XquadTask
from ..tasks.nli.xnli import XNLITask
from ..tasks.mqa.xcopa import XCopaTask


_TASKS = [
    BelebeleTask(lang="tr"),
    XquadTask(lang="tr"),
    XNLITask(lang="tr"),
    XCopaTask(lang="tr"),
    ExamsTask(lang="tr"),
    Tquad2Task(),
]

_TURKISH_LEADEBOARD = [
    HellaSwagTrTask(),
    TruthfulQATrTask("mc1"),
    TruthfulQATrTask("mc2"),
    ARCEasyTrTask(),
    WinogradeTrTask(),
] + [MMLUTaskTr(subset) for subset in get_args(MMLU_SUBSETS)]

_TASKS = _TASKS + _TURKISH_LEADEBOARD

_TASKS_STRINGS = ",".join([f"custom|{t.name}|0|1" for t in _TASKS])
TASKS_GROUPS = {
    "all": _TASKS_STRINGS,
}


TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
