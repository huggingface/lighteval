"""
name:
Gsm Plus

dataset:
qintongli/GSM-Plus

abstract:
GSM-Plus is an adversarial extension of GSM8K that tests the robustness of LLMs'
mathematical reasoning by introducing varied perturbations to grade-school math
problems.

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2402.19255
"""

from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, prompt_template

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics, math_scorer
from lighteval.tasks.lighteval_task import LightevalTaskConfig


# setup for problem + instructions for providing answer
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your
response should be of the form "ANSWER: $ANSWER" (without quotes)
where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to
the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def record_to_sample(record):
    return Sample(
        input=record["question"],
        target=record["answer"],
        metadata={"reasoning": record["solution"]},
    )


def sample_to_fewshot(sample):
    return f"{sample.input}\n\nReasoning:\n" + f"{sample.metadata['reasoning']}\n\n" + f"ANSWER: {sample.target}"


gsm_plus = LightevalTaskConfig(
    name="gsm_plus",
    prompt_function=prompt.gsm_plus,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="qintongli/GSM-Plus",
    hf_subset="default",
    hf_avail_splits=["test", "testmini"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.expr_gold_metric],
    stop_sequence=None,
    version=0,
)

TASKS_TABLE = [
    gsm_plus,
]
