from community_tasks.multilingual.tasks.qa.mintaka import MintakaTask


_TASKS = [
    # BelebeleTask(lang="en"),
    # LambadaTask(lang="en"),
    MintakaTask(lang="en"),
    # MlqaTask(lang="en"),
    # M_HellaSwagTask(lang="en"),
    # M_MMLUTask(lang="en"),
    # M_ARCTask(lang="en"),
    # PawnsXTask(lang="en"),
    # TydiqaTask(lang="en"),
    # # HF currently doesn't have answer keys for test dataset
    # # XCODAHTask(lang="en"),
    # # XCSQATask(lang="en"),
    # XNLITask(lang="en"),
    # XquadTask(lang="en"),
    # XStoryClozeTask(lang="en"),
    # XWinogradeTask(lang="en"),
    # M3ExamTask(lang="en"),
]
_TASKS_STRINGS = ",".join([f"custom|{t.name}|0|1" for t in _TASKS])
TASKS_GROUPS = {
    "all": _TASKS_STRINGS,
}


TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
