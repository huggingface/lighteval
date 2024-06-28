from community_tasks.multilingual.tasks.mqa.mlmm import M_ARCTask, M_MMLUTask
from community_tasks.multilingual.tasks.mqa_with_context.belebele import BelebeleTask
from community_tasks.multilingual.tasks.mqa_with_context.m3exam import M3ExamTask
from community_tasks.multilingual.tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from community_tasks.multilingual.tasks.nli.pawns import PawnsXTask
from community_tasks.multilingual.tasks.nli.xnli import XNLITask
from community_tasks.multilingual.tasks.nli.xwinograd import XWinogradeTask


_TASKS = [
    BelebeleTask(lang="zh"),
    # Generate metric disabled for now
    # MlqaTask(lang="zh"),
    # TydiqaTask(lang="zh"),
    # XquadTask(lang="zh"),
    # TODO: Add alternative
    # M_HellaSwagTask(lang="zh"),
    M_MMLUTask(lang="zh"),
    M_ARCTask(lang="zh"),
    PawnsXTask(lang="zh"),
    # HF currently doesn't have answer keys for test dataset
    # XCODAHTask(lang="zh"),
    # XCSQATask(lang="zh"),
    XNLITask(lang="zh"),
    XStoryClozeTask(lang="zh"),
    XWinogradeTask(lang="zh"),
    M3ExamTask(lang="zh"),
]
_TASKS_STRINGS = ",".join([f"custom|{t.name}|0|1" for t in _TASKS])
TASKS_GROUPS = {
    "all": _TASKS_STRINGS,
}


TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
