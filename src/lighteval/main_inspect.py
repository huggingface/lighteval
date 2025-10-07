from dataclasses import dataclass
from typing import Callable

from inspect_ai import Epochs, Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message

from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    extract_target_from_pred,
    get_extraction_regexes,
)
from lighteval.tasks.default_prompts import aime_prompt_fn, gsm8k
from lighteval.tasks.extended.ifeval.main import ifeval_prompt, ifeval_scorer
from lighteval.utils.language import Language


@dataclass
class TaskConfig:
    name: str
    prompt_function: Callable[[dict], Sample]
    hf_repo: str
    hf_subset: str
    split: str
    metrics: list
    system_prompt: str 
    epochs: int = 1
    generation_size: int | None = None
    num_samples: list[int] | None = None
    epochs_reducer: str | None = None


MATH_SYSTEM_PROMPT = """Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering."""
IFEVAL_SYSTEM_PROMPT = """FOLLOW THE INSTRUCTIONS STRICTLY."""


@scorer(metrics=[accuracy(), stderr()])
def extractive_math_scorer():
    gold_extraction_target = (ExprExtractionConfig(),)
    pred_extraction_target = (ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0))
    language = Language.ENGLISH
    fallback_mode = "first_match"
    extraction_mode = "first_match"
    timeout_seconds = 5

    gold_extraction_regexes = get_extraction_regexes(gold_extraction_target, language)
    pred_extraction_regexes = get_extraction_regexes(pred_extraction_target, language)

    async def score(state: TaskState, target: Target):
        extracted_predictions = extract_target_from_pred(
            state.output.completion, pred_extraction_regexes, fallback_mode, extraction_mode, timeout_seconds
        )
        extracted_gold = extract_target_from_pred(
            target.text, gold_extraction_regexes, fallback_mode, extraction_mode, timeout_seconds
        )
        return Score(
            value="C" if extracted_predictions == extracted_gold else "I",
            explanation=state.output.completion,
            answer=str(extracted_predictions),
        )

    return score


@task
def get_task(lighteval_task_config: TaskConfig):
    name = lighteval_task_config.name
    sample_fields = lighteval_task_config.prompt_function
    split = lighteval_task_config.split
    system_prompt = lighteval_task_config.system_prompt
    metrics = lighteval_task_config.metrics
    hf_repo = lighteval_task_config.hf_repo
    hf_subset = lighteval_task_config.hf_subset

    dataset = hf_dataset(
        hf_repo, name=hf_subset, split=split, sample_fields=sample_fields
    )
    solver = [
        system_message(system_prompt),
        generate(cache=True),
    ]
    scorer = metrics
    epochs = lighteval_task_config.epochs
    epochs_reducer = lighteval_task_config.epochs_reducer

    return Task(dataset=dataset, solver=solver, scorer=scorer, name=name, epochs=Epochs(epochs, epochs_reducer))


gsm8k_task_config = TaskConfig(
    name="gsm8k",
    prompt_function=gsm8k,
    hf_repo="openai/gsm8k",
    hf_subset="main",
    split="train",
    metrics=[extractive_math_scorer()],
    system_prompt=MATH_SYSTEM_PROMPT,
    epochs=4,
)
aime25_task_config = TaskConfig(
    name="aime25",
    prompt_function=aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    split="train",
    metrics=[extractive_math_scorer()],
    system_prompt=MATH_SYSTEM_PROMPT,
    epochs=4,
)
ifeval_task_config = TaskConfig(
    name="ifeval",
    prompt_function=ifeval_prompt,
    hf_repo="google/IFEval",
    split="train",
    hf_subset="default",
    metrics=[ifeval_scorer()],
    system_prompt=IFEVAL_SYSTEM_PROMPT,
)


def main():
    MODEL = "openai/gpt-4o"

    eval(get_task(gsm8k_task_config), model=MODEL, display="rich", limit=10)


if __name__ == "__main__":
    main()
