"""
name:
Aime

dataset:
HuggingFaceH4/aime_2024, yentinglin/aime_2025

abstract:
The American Invitational Mathematics Examination (AIME) is a prestigious,
invite-only mathematics competition for high-school students who perform in the
top 5% of the AMC 12 mathematics exam. It involves 15 questions of increasing
difficulty, with the answer to every question being a single integer from 0 to
999. The median score is historically between 4 and 6 questions correct (out of
the 15 possible). Two versions of the test are given every year (thirty
questions total).

languages:
english

tags:
math, reasoning

paper:
https://maa.org/aime-thresholds-are-available/

starred:
true
"""

from textwrap import dedent

from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics, math_scorer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Prompt template adapted from
# - simple-evals: https://github.com/openai/simple-evals/blob/6e84f4e2aed6b60f6a0c7b8f06bbbf4bfde72e58/math_eval.py#L17
# - Llama 3: https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals/viewer/Llama-3.2-1B-Instruct-evals__math__details?views%5B%5D=llama_32_1b_instruct_evals__math__details
# Note that it is important to have the final answer in a box for math-verify to work correctly
MATH_PROMPT_TEMPLATE = dedent("""
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{prompt}
""")


def record_to_sample(record):
    return Sample(input=record["problem"], target=record["answer"])


def aime_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_PROMPT_TEMPLATE.format(prompt=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


aime24 = LightevalTaskConfig(
    name="aime24",
    prompt_function=aime_prompt,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.pass_at_k_math(sample_params={"k": 1}), Metrics.avg_at_n_math(sample_params={"n": 1})],
    version=2,
)

aime24_avg = LightevalTaskConfig(
    name="aime24_avg",
    prompt_function=aime_prompt,
    sample_fields=record_to_sample,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=2,
)

aime24_gpassk = LightevalTaskConfig(
    name="aime24_gpassk",
    prompt_function=aime_prompt,
    sample_fields=record_to_sample,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

aime25 = LightevalTaskConfig(
    name="aime25",
    prompt_function=aime_prompt,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}), Metrics.avg_at_n_math(sample_params={"n": 1})],
    version=2,
)

aime25_avg = LightevalTaskConfig(
    name="aime25_avg",
    prompt_function=aime_prompt,
    sample_fields=record_to_sample,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=2,
)

aime25_gpassk = LightevalTaskConfig(
    name="aime25_gpassk",
    prompt_function=aime_prompt,
    sample_fields=record_to_sample,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

TASKS_TABLE = [
    aime24,
    aime24_avg,
    aime24_gpassk,
    aime25,
    aime25_avg,
    aime25_gpassk,
]
