"""
name:
SLR-Bench

dataset:
AIML-TUDA/SLR-Bench

abstract:
SLR-Bench is a large-scale benchmark for scalable logical reasoning with
language models, comprising 19,000 prompts organized into 20 curriculum levels.

languages:
english

tags:
reasoning, symbolic

paper:
https://arxiv.org/abs/2506.15787
"""

import logging

import numpy as np

from lighteval.metrics.utils.metric_utils import SampleLevelComputation, SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.imports import is_package_available, requires


if is_package_available("evaluate"):
    from evaluate import load
else:
    load = None

logger = logging.getLogger(__name__)


@requires("evaluate")
def prompt_fn(line: dict, task_name: str):
    """Defines how to go from a dataset line to a doc object."""
    # Check for SWI-Prolog installation
    import shutil

    if shutil.which("swipl") is None:
        raise ImportError(
            "SWI-Prolog (swipl) is not installed or not in PATH. Please install SWI-Prolog to use this task. "
        )

    return Doc(
        task_name=task_name, query=line["prompt"], choices=[str(line.get("validation program", ""))], gold_index=0
    )


class VerifiableRewardMetric(SampleLevelComputation):
    # Load the symbolic judge for evaluating Prolog programs

    def compute(self, doc, model_response, **kwargs):
        symbolic_judge = load("AIML-TUDA/VerifiableRewardsForScalableLogicalReasoning")
        try:
            prediction = model_response.final_text[0]
            validation_program = doc.choices[0] if doc.choices else ""
            ref_format = [
                {
                    "validation_program": validation_program,
                    "evaluation_config": {"positive_predicate": "eastbound", "negative_predicate": "westbound"},
                }
            ]

            results = symbolic_judge.compute(predictions=[prediction], references=ref_format)
            return results["accuracy"]

        except Exception as e:
            logger.error("Error during the computation of the metric")
            raise RuntimeError(f"Failed to compute verifiable reward metric: {e}")


custom_metric = SampleLevelMetric(
    metric_name="verifiable_reward",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=VerifiableRewardMetric(),
    corpus_level_fn=np.mean,
)

# Define the subsets available in the SLR-Bench dataset
CONFIGURATIONS = ["All", "Basic", "Easy", "Medium", "Hard"]


class SLRBenchTask(LightevalTaskConfig):
    """Task configuration for SLR-Bench evaluation."""

    def __init__(
        self,
        config: str,
    ):
        name = f"slr_bench_{config.lower()}"
        super().__init__(
            name=name,
            hf_subset=f"v1-{config}",
            prompt_function=prompt_fn,
            hf_repo="AIML-TUDA/SLR-Bench",
            metrics=[custom_metric],
            hf_avail_splits=["train", "validation", "test"],
            evaluation_splits=["test"],
            few_shots_split="train",
            few_shots_select="random_sampling_from_train",
            generation_size=4096,
            stop_sequence=None,
            version=1,
        )


# Create a single task instance for each configuration
TASKS = [SLRBenchTask(config) for config in CONFIGURATIONS]

# Export tasks table
TASKS_TABLE = TASKS
