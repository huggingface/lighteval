from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, accuracy, scorer
from inspect_ai.solver import generate

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


QUESTION_PROMPT_YES_NO = """Response:

{response}

Evaluate whether the response above satisfies this criterion: {criterion_description}. Only answer Yes or No."""

weight_to_scale = {"Critical": 4, "Major": 3, "Minor": 2, "Additional": 1}


async def evaluate_criterion_with_judge(response, criterion_description, domain, criterion_type):
    """Evaluate a single criterion using LLM judge."""
    prompt = QUESTION_PROMPT_YES_NO.format(response=response, criterion_description=criterion_description)

    model = get_model()
    result = await model.generate(prompt)

    judge_rating = result.completion.strip()
    return judge_rating.startswith("Yes"), judge_rating


@scorer(metrics=[accuracy()])
def profbench_weighted_scorer():
    """Scorer that evaluates all criteria and computes weighted ProfBench scores."""

    async def score(state, target):
        rubrics = state.metadata.get("rubrics", [])
        response = state.output.completion
        task_id = state.metadata.get("task_id", "")
        domain = state.metadata.get("domain", "")

        # Evaluate each criterion
        criterion_results = []
        total_weight = 0
        achieved_weight = 0

        for rubric in rubrics:
            criterion_description = rubric["criterion_description"]
            criterion_weight = rubric["criterion_weight"]
            criterion_type = rubric["criterion_type"]

            weight_scale = weight_to_scale.get(criterion_weight, 1)
            total_weight += weight_scale

            # Evaluate criterion
            fulfilled, judge_rating = await evaluate_criterion_with_judge(
                response, criterion_description, domain, criterion_type
            )

            if fulfilled:
                achieved_weight += weight_scale

            criterion_results.append(
                {
                    "criterion_description": criterion_description,
                    "criterion_weight": criterion_weight,
                    "criterion_type": criterion_type,
                    "fulfilled": fulfilled,
                    "judge_rating": judge_rating,
                }
            )

        # Calculate score for this task
        task_score = (achieved_weight / total_weight) if total_weight > 0 else 0.0

        return Score(
            value=task_score,
            metadata={
                "task_id": task_id,
                "domain": domain,
                "task_score": task_score,
                "achieved_weight": achieved_weight,
                "total_weight": total_weight,
                "criterion_results": criterion_results,
                "response": response,
            },
        )

    return score


def record_to_sample(record):
    """Convert ProfBench dataset record to Inspect Sample."""
    return Sample(
        input=record["prompt"],
        target="",  # No target for generation tasks
        metadata={
            "task_id": record["task_id"],
            "domain": record["domain"],
            "rubrics": record["rubrics"],
            "filepaths": record.get("filepaths", []),
        },
    )


@task
def _profbench():
    """
    ProfBench report generation task.
    """
    # Load dataset
    dataset_obj = hf_dataset(
        path="nvidia/ProfBench",
        split="test",
        sample_fields=record_to_sample,
    )

    return Task(
        dataset=dataset_obj,
        solver=[generate()],
        scorer=profbench_weighted_scorer(),
    )


profbench = LightevalTaskConfig(
    name="profbench",
    prompt_function=lambda line, task_name: line["prompt"],
    hf_repo="nvidia/ProfBench",
    hf_subset="default",
    evaluation_splits=["test"],
    metrics=[Metrics.exact_match],
    version=0,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=profbench_weighted_scorer(),
)

TASKS_TABLE = [profbench]
