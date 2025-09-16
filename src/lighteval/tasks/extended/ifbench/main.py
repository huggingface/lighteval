# MIT License

# Copyright (c) 2024 The HuggingFace Team

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


import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import (
    SampleLevelMetricGrouping,
)
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.extended.ifbench import evaluation_lib
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


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
    suite=["extended"],
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
)

# Multi-turn IFBench task config
ifbench_multiturn = LightevalTaskConfig(
    name="ifbench_multiturn",
    prompt_function=ifbench_prompt,
    suite=["extended"],
    hf_repo="allenai/IFBench_multi-turn",
    hf_subset="default",
    metrics=[ifbench_metrics],
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="test",
    few_shots_select="random_sampling",
    generation_size=1280,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
)

TASKS_TABLE = [ifbench_test, ifbench_multiturn]

extend_enum(Metrics, "ifbench_metric", ifbench_metrics)
