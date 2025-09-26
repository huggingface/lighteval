# MIT License

# Copyright (c) 2025 Lukas Helff

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
SLR-Bench is a large-scale benchmark for scalable logical reasoning with language models, comprising 19,000 prompts organized into 20 curriculum levels.
The tasks progressively increase in relational, arithmetic, and recursive complexity, requiring models to synthesize Prolog rules that classify train compositions.
For more details see: https://huggingface.co/datasets/AIML-TUDA/SLR-Bench
The paper can be found here: https://arxiv.org/abs/2506.15787
Before using this task, please ensure that SWI-Prolog and evaluate are installed on your system, as they are required for symbolic verification of the generated Prolog programs.
"""

import logging
import shutil

import numpy as np
from evaluate import load

from lighteval.metrics.utils.metric_utils import SampleLevelComputation, SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


logger = logging.getLogger(__name__)


# Check for SWI-Prolog installation
if shutil.which("swipl") is None:
    raise ImportError(
        "SWI-Prolog (swipl) is not installed or not in PATH. "
        "Please install SWI-Prolog to use this task. "
        "You can install required dependencies with: pip install -r community_tasks/slr_bench_requirements.txt"
    )

# Load the symbolic judge for evaluating Prolog programs
symbolic_judge = load("AIML-TUDA/VerifiableRewardsForScalableLogicalReasoning")


def prompt_fn(line: dict, task_name: str):
    """Defines how to go from a dataset line to a doc object."""
    return Doc(
        task_name=task_name, query=line["prompt"], choices=[str(line.get("validation program", ""))], gold_index=0
    )


class VerifiableRewardMetric(SampleLevelComputation):
    def compute(self, doc, model_response, **kwargs):
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
            suite=["community"],
            generation_size=4096,
            stop_sequence=None,
            version=1,
        )


# Create a single task instance for each configuration
TASKS = [SLRBenchTask(config) for config in CONFIGURATIONS]

# Export tasks table
TASKS_TABLE = TASKS
