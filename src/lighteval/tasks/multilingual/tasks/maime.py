"""
name:
mAIME2025

dataset:
LumiOpen/mAIME2025

abstract:
The Multilingual AIME 2025 (mAIME2025) is a multilingual version of the
2025 AIME (American Invitational Mathematics Examination) problems,
professionally translated into European languages. This dataset contains
all 30 problems from AIME I and AIME II 2025, translated and
human-reviewed by native speakers to preserve mathematical accuracy and
LaTeX formatting.

languages:
danish, finnish

tags:
math, multilingual, reasoning

paper:
https://maa.org/aime-thresholds-are-available/
"""

from textwrap import dedent

from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics, math_scorer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Prompt template adapted from AIME task
# Note: Uses English instructions for consistency with AIME
MATH_PROMPT_TEMPLATE = dedent("""
Solve the following math problem efficiently and clearly.  
The last line of your response should be of the following format: 
'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' 
(without quotes) where ANSWER is just the final number or expression 
that solves the problem. Think step by step before answering.

{prompt}
""")



def record_to_sample(record):
    return Sample(input=record["question"], target=record["solution"])


def maime_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_PROMPT_TEMPLATE.format(prompt=line["question"]),
        choices=[line["solution"]],
        gold_index=0,
    )


# Danish tasks
maime25_da = LightevalTaskConfig(
    name="maime25:da",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="da_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
        Metrics.avg_at_n_math(sample_params={"n": 1}),
    ],
    version=1,
)

maime25_da_avg = LightevalTaskConfig(
    name="maime25_avg:da",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="da_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=1,
)

maime25_da_gpassk = LightevalTaskConfig(
    name="maime25_gpassk:da",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="da_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

# Finnish tasks
maime25_fi = LightevalTaskConfig(
    name="maime25:fi",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="fi_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
        Metrics.avg_at_n_math(sample_params={"n": 1}),
    ],
    version=1,
)

maime25_fi_avg = LightevalTaskConfig(
    name="maime25_avg:fi",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="fi_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=1,
)

maime25_fi_gpassk = LightevalTaskConfig(
    name="maime25_gpassk:fi",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="fi_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

TASKS_TABLE = [
    maime25_da,
    maime25_da_avg,
    maime25_da_gpassk,
    maime25_fi,
    maime25_fi_avg,
    maime25_fi_gpassk,
]
