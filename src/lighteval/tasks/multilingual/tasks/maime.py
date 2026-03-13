"""
name:
mAIME2025, mAIME2026

dataset:
LumiOpen/mAIME2025, LumiOpen/mAIME2026

abstract:
The Multilingual AIME 2025 and 2026 (mAIME2025, mAIME2026) is a multilingual version of the
2025 and 2026 AIME (American Invitational Mathematics Examination) problems,
machine translated by Claude Opus 4.5 and manually reviewed by native speakers. 
This dataset contains all 30 problems from AIME I and AIME II 2025.

All files are:
- translated by Claude Opus 4.5 
- human-reviewed by native speakers
- LaTeX formatting is automatically validated

2025 languages:
Czech, Danish, Finnish, German, Slovak, Swedish

2026 languages:
Danish, Finnish

tags:
math, multilingual, reasoning

Reference:

@misc{maa2025aime,
  title={American Invitational Mathematics Examination (AIME)},
  author={{Mathematical Association of America}},
  year={2025, 2026},
  url={https://maa.org/math-competitions/american-invitational-mathematics-examination-aime}
}

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


# Sampling parameters for stochastic evaluation (avg@n, g_pass@k).
# Following "Are Your LLMs Capable of Stable Reasoning?" (arXiv:2412.13147)
# which uses temperature=1.0, top_p=0.8 with n=48, k=16 for g_pass@k.
SAMPLING_SOLVER = [
    prompt_template(MATH_PROMPT_TEMPLATE),
    generate(temperature=1.0, top_p=0.8),
]


def record_to_sample(record):
    return Sample(input=record["question"], target=record["solution"])


def maime_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_PROMPT_TEMPLATE.format(prompt=line["question"]),
        choices=[line["solution"]],
        gold_index=0,
    )

# 2025
## Czech tasks
maime25_cs = LightevalTaskConfig(
    name="maime25:cs",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="cs_combined",
    hf_avail_splits=["test"],

    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}), Metrics.avg_at_n_math(sample_params={"n": 1})],
    version=1,
)

maime25_cs_avg = LightevalTaskConfig(
    name="maime25_avg:cs",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=SAMPLING_SOLVER,
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="cs_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=1,
)

maime25_cs_gpassk = LightevalTaskConfig(
    name="maime25_gpassk:cs",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=SAMPLING_SOLVER,
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="cs_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,   
    stop_sequence=[],
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

## Danish tasks
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
    stop_sequence=[],
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
    solver=SAMPLING_SOLVER,
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="da_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=1,
)

maime25_da_gpassk = LightevalTaskConfig(
    name="maime25_gpassk:da",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=SAMPLING_SOLVER,
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="da_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

## German tasks
maime25_de = LightevalTaskConfig(
    name="maime25:de",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="de_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}), Metrics.avg_at_n_math(sample_params={"n": 1})],
    version=1,
)

maime25_de_avg = LightevalTaskConfig(
    name="maime25_avg:de",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=SAMPLING_SOLVER,
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="de_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=1,
)

maime25_de_gpassk = LightevalTaskConfig(
    name="maime25_gpassk:de",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=SAMPLING_SOLVER,
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="de_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

## Finnish tasks
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
    stop_sequence=[],
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
    solver=SAMPLING_SOLVER,
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="fi_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=1,
)

maime25_fi_gpassk = LightevalTaskConfig(
    name="maime25_gpassk:fi",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=SAMPLING_SOLVER,
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="fi_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

## Swedish tasks
maime25_se = LightevalTaskConfig(
    name="maime25:se",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="se_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}), Metrics.avg_at_n_math(sample_params={"n": 1})],
    version=1,
)

maime25_se_avg = LightevalTaskConfig(
    name="maime25_avg:se",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=SAMPLING_SOLVER,
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="se_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=1,
)

maime25_se_gpassk = LightevalTaskConfig(
    name="maime25_gpassk:se",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=SAMPLING_SOLVER,
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="se_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

## Slovak tasks
maime25_sk = LightevalTaskConfig(
    name="maime25:sk",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="sk_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[
        Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}),
        Metrics.avg_at_n_math(sample_params={"n": 1}),
    ],
    version=1,
)

maime25_sk_avg = LightevalTaskConfig(
    name="maime25_avg:sk",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=SAMPLING_SOLVER,
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="sk_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=1,
)

maime25_sk_gpassk = LightevalTaskConfig(
    name="maime25_gpassk:sk",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=SAMPLING_SOLVER,
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2025",
    hf_subset="sk_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

# 2026
## Danish tasks
maime26_da = LightevalTaskConfig(
    name="maime26:da",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2026",
    hf_subset="da_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}), Metrics.avg_at_n_math(sample_params={"n": 1})],
    version=1,
)

maime26_da_avg = LightevalTaskConfig(
    name="maime26_avg:da",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=SAMPLING_SOLVER,
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2026",
    hf_subset="da_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=1,
)

maime26_da_gpassk = LightevalTaskConfig(
    name="maime26_gpassk:da",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=SAMPLING_SOLVER,
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2026",
    hf_subset="da_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

## Finnish
maime26_fi = LightevalTaskConfig(
    name="maime26:fi",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2026",
    hf_subset="fi_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.pass_at_k_math(sample_params={"k": 1, "n": 1}), Metrics.avg_at_n_math(sample_params={"n": 1})],
    version=1,
)

maime26_fi_avg = LightevalTaskConfig(
    name="maime26_avg:fi",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=SAMPLING_SOLVER,
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2026",
    hf_subset="fi_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.avg_at_n_math(sample_params={"n": 64})],
    version=1,
)

maime26_fi_gpassk = LightevalTaskConfig(
    name="maime26_gpassk:fi",
    prompt_function=maime_prompt,
    sample_fields=record_to_sample,
    solver=SAMPLING_SOLVER,
    scorer=math_scorer(),
    hf_repo="LumiOpen/mAIME2026",
    hf_subset="fi_combined",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],
    metrics=[Metrics.g_pass_at_k_math(sample_params={"k": 16, "n": 48})],
    version=1,
)

TASKS_TABLE = [
    maime25_cs,
    maime25_cs_avg,
    maime25_cs_gpassk,
    maime25_da,
    maime25_da_avg,
    maime25_da_gpassk,
    maime25_de,
    maime25_de_avg,
    maime25_de_gpassk,
    maime25_fi,
    maime25_fi_avg,
    maime25_fi_gpassk,
    maime25_se,
    maime25_se_avg,
    maime25_se_gpassk,
    maime25_sk,
    maime25_sk_avg,
    maime25_sk_gpassk,
    maime26_da,
    maime26_da_avg,
    maime26_da_gpassk,
    maime26_fi,
    maime26_fi_avg,
    maime26_fi_gpassk,
]
