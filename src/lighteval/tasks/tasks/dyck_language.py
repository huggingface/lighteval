"""
name:
Dyck Language

dataset:
lighteval/DyckLanguage

abstract:
Scenario testing hierarchical reasoning through the Dyck formal languages.

languages:
english

tags:
reasoning

paper:
https://aclanthology.org/W19-3905/
"""

from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


PROMPT = "Please complete the rest of the following Dyck sequences, making sure that the parentheses are closed properly.\n Input: {prompt}"


def record_to_sample(record):
    return Sample(input=PROMPT.format(prompt=record["input"]), target=record["output"])


@scorer(metrics=[accuracy(), stderr()])
def dyck_exact() -> Scorer:
    """Compares the closing bracket sequence to the target, ignoring whitespace only.

    The inspect_ai `exact()` and `match()` scorers drop punctuation while normalizing,
    which reduces any bracket sequence to the empty string and so marks every answer
    correct. Bracket types and their order are the property under test here, so they
    must survive normalization.
    """

    def normalize(text: str) -> str:
        return "".join(text.split())

    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion
        correct = any(normalize(answer) == normalize(t) for t in target if t.strip())
        return Score(value=CORRECT if correct else INCORRECT, answer=answer)

    return score


def dyck_language_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Please complete the rest of the following Dyck sequences, making sure that the parentheses are closed properly.\n Input: {line['input']}",
        choices=[line["output"]],
        gold_index=0,
        instruction="Please complete the rest of the following Dyck sequences, making sure that the parentheses are closed properly.\n ",
    )


dyck_language_2 = LightevalTaskConfig(
    name="dyck_language:2",
    prompt_function=dyck_language_prompt,
    hf_repo="lighteval/DyckLanguage",
    hf_subset="2",
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=dyck_exact(),
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)


dyck_language_3 = LightevalTaskConfig(
    name="dyck_language:3",
    prompt_function=dyck_language_prompt,
    hf_repo="lighteval/DyckLanguage",
    hf_subset="3",
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=dyck_exact(),
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)


dyck_language_4 = LightevalTaskConfig(
    name="dyck_language:4",
    prompt_function=dyck_language_prompt,
    hf_repo="lighteval/DyckLanguage",
    hf_subset="4",
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=dyck_exact(),
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    dyck_language_2,
    dyck_language_3,
    dyck_language_4,
]
