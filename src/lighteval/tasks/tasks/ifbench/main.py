"""
name:
IFBench

dataset:
allenai/IFBench_test, allenai/IFBench_multi-turn

abstract:
Challenging benchmark for precise instruction following.

languages:
english

tags:
instruction-following

paper:
https://arxiv.org/abs/2507.02833

starred:
true
"""

import numpy as np
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import (
    SampleLevelMetricGrouping,
)
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks.ifbench import evaluation_lib
from lighteval.utils.imports import requires


def ifbench_prompt(line, task_name: str = ""):
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[""],
        gold_index=0,
        instruction="",
        specific={"instruction_id_list": line["instruction_id_list"], "kwargs": line["kwargs"]},
    )


submetric_names = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]


class IFBench(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> dict:
        response = model_response.final_text[0]

        # Create InputExample from the doc data
        inp = evaluation_lib.InputExample(
            key=0,  # Not used in evaluation
            instruction_id_list=doc.specific["instruction_id_list"],
            prompt=doc.query,
            kwargs=doc.specific["kwargs"],
        )

        # Create prompt_to_response mapping for evaluation_lib functions
        prompt_to_response = {doc.query: response}

        # Use existing evaluation_lib functions
        strict_result = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
        loose_result = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)

        return {
            "prompt_level_strict_acc": int(strict_result.follow_all_instructions),
            "inst_level_strict_acc": strict_result.follow_instruction_list,
            "prompt_level_loose_acc": int(loose_result.follow_all_instructions),
            "inst_level_loose_acc": loose_result.follow_instruction_list,
        }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc


@requires("langdetect")
def record_to_sample(record):
    metadata = {"instruction_id_list": record["instruction_id_list"], "kwargs": record["kwargs"]}
    return Sample(
        input=record["prompt"],
        metadata=metadata,
    )


@scorer(
    metrics={
        "prompt_level_strict_acc": [accuracy(), stderr()],
        "prompt_level_loose_acc": [accuracy(), stderr()],
    }
)
def ifbench_scorer():
    async def score(state: TaskState, target: Target):
        response = state.output.completion
        # Create InputExample from the doc data
        inp = evaluation_lib.InputExample(
            key=0,  # Not used in evaluation
            instruction_id_list=state.metadata["instruction_id_list"],
            prompt=state.input,
            kwargs=state.metadata["kwargs"],
        )

        # Create prompt_to_response mapping for evaluation_lib functions
        prompt_to_response = {state.input: response}

        # Use existing evaluation_lib functions
        strict_result = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
        loose_result = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
        return Score(
            value={
                "prompt_level_strict_acc": int(strict_result.follow_all_instructions),
                "prompt_level_loose_acc": int(loose_result.follow_all_instructions),
            },
            explanation=str(state.metadata["instruction_id_list"]),
        )

    return score


ifbench_metrics = SampleLevelMetricGrouping(
    metric_name=submetric_names,
    higher_is_better=dict.fromkeys(submetric_names, True),
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=IFBench(),
    corpus_level_fn={
        "prompt_level_strict_acc": np.mean,
        "inst_level_strict_acc": agg_inst_level_acc,
        "prompt_level_loose_acc": np.mean,
        "inst_level_loose_acc": agg_inst_level_acc,
    },
)

# Main IFBench test dataset task config
ifbench_test = LightevalTaskConfig(
    name="ifbench_test",
    prompt_function=ifbench_prompt,
    hf_repo="allenai/IFBench_test",
    hf_subset="default",
    metrics=[ifbench_metrics],
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=1280,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=ifbench_scorer(),
)

# Multi-turn IFBench task config
ifbench_multiturn = LightevalTaskConfig(
    name="ifbench_multiturn",
    prompt_function=ifbench_prompt,
    hf_repo="allenai/IFBench_multi-turn",
    hf_subset="ifbench_constraints",
    metrics=[ifbench_metrics],
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="test",
    few_shots_select="random_sampling",
    generation_size=1280,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=ifbench_scorer(),
)

TASKS_TABLE = [ifbench_test, ifbench_multiturn]
