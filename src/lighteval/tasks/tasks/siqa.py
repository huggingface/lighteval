"""
name:
Siqa

dataset:
allenai/social_i_qa

abstract:
We introduce Social IQa: Social Interaction QA, a new question-answering
benchmark for testing social commonsense intelligence. Contrary to many prior
benchmarks that focus on physical or taxonomic knowledge, Social IQa focuses on
reasoning about people's actions and their social implications. For example,
given an action like "Jesse saw a concert" and a question like "Why did Jesse do
this?", humans can easily infer that Jesse wanted "to see their favorite
performer" or "to enjoy the music", and not "to see what's happening inside" or
"to see if it works". The actions in Social IQa span a wide variety of social
situations, and answer candidates contain both human-curated answers and
adversarially-filtered machine-generated candidates. Social IQa contains over
37,000 QA pairs for evaluating models' abilities to reason about the social
implications of everyday events and situations.

languages:
english

tags:
commonsense, multiple-choice, qa

paper:
"""

from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def siqa_prompt(line, task_name: str = None):
    query = "The following are multiple choice questions (with answers) about common sense.\n"
    query += f"Question: {line['context']} {line['question']}\n"
    query += "".join(
        [
            f"{key}. {choice}\n"
            for key, choice in zip(list(ascii_uppercase)[:3], [line["answerA"], line["answerB"], line["answerC"]])
        ]
    )
    query += "Answer: "

    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C"],
        gold_index=int(line["label"]) - 1,
        instruction="The following are multiple choice questions (with answers) about common sense.\n",
    )


siqa = LightevalTaskConfig(
    name="siqa",
    prompt_function=siqa_prompt,
    hf_repo="allenai/social_i_qa",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    siqa,
]
