from inspect_ai import Task, eval, task
from inspect_ai.dataset import hf_dataset
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, self_critique, system_message

from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    extract_target_from_pred,
    get_extraction_regexes,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.registry import Registry
from lighteval.utils.language import Language


MATH_SYSTEM_PROMPT = """Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering."""

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
def get_task(lighteval_task_config: LightevalTaskConfig):
    name = lighteval_task_config.name
    sample_fields = lighteval_task_config.prompt_function
    split = lighteval_task_config.evaluation_splits[0]
    dataset = hf_dataset(
        lighteval_task_config.hf_repo, name=lighteval_task_config.hf_subset, split=split, sample_fields=sample_fields
    )
    solver = [
        system_message(MATH_SYSTEM_PROMPT),
        generate(),
    ]
    scorer = [extractive_math_scorer()]

    return Task(dataset=dataset, solver=solver, scorer=scorer, name=name)


def main():
    TASK = "lighteval|aime25|0"
    MODEL = "openai/gpt-4o"

    registry = Registry(tasks=TASK)
    config = registry._update_task_configs()[TASK.rsplit("|", 1)[0]][0]

    eval(get_task(config), model=MODEL)


if __name__ == "__main__":
    main()
