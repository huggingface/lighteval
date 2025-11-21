"""
name:
Math 500

dataset:
HuggingFaceH4/MATH-500

abstract:
This dataset contains a subset of 500 problems from the MATH benchmark that
OpenAI created in their Let's Verify Step by Step paper.

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2305.20050

starred:
true
"""

from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


MATH_QUERY_TEMPLATE = """
Solve the following problem. The final line of your response MUST be of the following format:
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the final answer. Think step by step before answering.

{prompt}
""".strip()


def math_500_prompt(line, task_name: str = None):
    query = MATH_QUERY_TEMPLATE.format(prompt=line["problem"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f"ANSWER: {line['solution']}"],
        gold_index=0,
    )


def record_to_sample(record):
    query = record["problem"]
    target = record["answer"]
    return Sample(input=query, target=target)


math_500 = LightevalTaskConfig(
    name="math_500",
    prompt_function=math_500_prompt,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
    ],
    version=2,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=model_graded_fact(),
)

TASKS_TABLE = [
    math_500,
]
