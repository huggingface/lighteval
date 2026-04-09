"""
name:
Omni-MATH

dataset:
KbsdJames/Omni-MATH

abstract:
Omni-MATH is a benchmark of 4,428 olympiad-level math problems sourced from
international competitions (IMO, USAMO, China National Olympiad, etc.) across
33+ mathematical domains with difficulty levels from 1 to 9.5.

languages:
english

tags:
math, reasoning, olympiad

paper:
https://arxiv.org/abs/2410.07985

starred:
true
"""

from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


OMNI_MATH_QUERY_TEMPLATE = """
Solve the following math problem. The final line of your response MUST be of the following format:
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the final answer. Think step by step before answering.

{prompt}
""".strip()


def omni_math_prompt(line, task_name: str = None):
    query = OMNI_MATH_QUERY_TEMPLATE.format(prompt=line["problem"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f"ANSWER: {line['answer']}"],
        gold_index=0,
    )


def record_to_sample(record):
    return Sample(input=record["problem"], target=record["answer"])


# Domain categories matching the paper's leaderboard
DOMAINS = [
    "Algebra",
    "Precalculus",
    "Calculus",
    "Geometry",
    "Discrete Mathematics",
    "Number Theory",
    "Applied Mathematics",
]

# Difficulty ranges matching the paper's leaderboard
DIFFICULTY_RANGES = [
    ("1_3", 1, 3),
    ("3_5", 3, 5),
    ("5_8", 5, 8),
    ("8_10", 8, 10),
]

_COMMON_CONFIG = dict(
    prompt_function=omni_math_prompt,
    hf_repo="KbsdJames/Omni-MATH",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
    ],
    version=1,
    sample_fields=record_to_sample,
    solver=[prompt_template(OMNI_MATH_QUERY_TEMPLATE), generate(cache=True)],
    scorer=model_graded_fact(),
)

# Main task (all problems)
omni_math = LightevalTaskConfig(
    name="omni_math",
    **_COMMON_CONFIG,
)

# Tasks by domain
omni_math_by_domain = [
    LightevalTaskConfig(
        name=f"omni_math:domain_{domain.lower().replace(' ', '_')}",
        hf_filter=lambda x, d=domain: d in x.get("domain", ""),
        **_COMMON_CONFIG,
    )
    for domain in DOMAINS
]

# Tasks by difficulty range
omni_math_by_difficulty = [
    LightevalTaskConfig(
        name=f"omni_math:difficulty_{label}",
        hf_filter=lambda x, lo=lo, hi=hi: lo <= x.get("difficulty", 0) < hi,
        **_COMMON_CONFIG,
    )
    for label, lo, hi in DIFFICULTY_RANGES
]

TASKS_TABLE = [
    omni_math,
    *omni_math_by_domain,
    *omni_math_by_difficulty,
]
