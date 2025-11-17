"""
name:
Gsm8K

dataset:
openai/gsm8k

abstract:
GSM8K is a dataset of 8,000+ high-quality, single-step arithmetic word problems.

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2110.14168
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
    DELIM = "####"
    input = record["question"]
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    reasoning = DELIM.join(answer)
    return Sample(input=input, target=target, metadata={"reasoning": reasoning.strip()})


def sample_to_fewshot(sample):
    return f"{sample.input}\n\nReasoning:\n" + f"{sample.metadata['reasoning']}\n\n" + f"ANSWER: {sample.target}"


gsm8k = LightevalTaskConfig(
    name="gsm8k",
    prompt_function=prompt.gsm8k,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="openai/gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=256,
    metrics=[
        Metrics.expr_gold_metric,
    ],
    stop_sequence=["Question:"],
    version=0,
)

TASKS_TABLE = [
    gsm8k,
]
