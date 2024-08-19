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

import lighteval.tasks.extended.ifeval.instructions_registry as instructions_registry
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetricGrouping,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Very specific task where there are no precise outputs but instead we test if the format obeys rules
def ifeval_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[""],
        gold_index=0,
        instruction="",
        specific={"instructions_id_list": line["instruction_id_list"], "kwargs": line["kwargs"]},
    )


submetric_names = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]


def ifeval_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> dict:
    response = predictions[0]

    # Strict instructions
    instruction_list = formatted_doc.specific["instructions_id_list"]
    all_kwargs = formatted_doc.specific["kwargs"]
    prompt = formatted_doc.query

    # Loose instructions
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]

    is_following_list_strict = []
    is_following_list_loose = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        task_kwargs = {k: v for k, v in all_kwargs[index].items() if v}
        instruction.build_description(**task_kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)

        # Strict
        if response.strip() and instruction.check_following(response):
            is_following_list_strict.append(True)
        else:
            is_following_list_strict.append(False)

        # Loose
        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list_loose.append(is_following)

    return {
        "prompt_level_strict_acc": int(all(is_following_list_strict)),
        "inst_level_strict_acc": is_following_list_strict,
        "prompt_level_loose_acc": int(all(is_following_list_loose)),
        "inst_level_loose_acc": is_following_list_loose,
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc


ifeval_metrics = SampleLevelMetricGrouping(
    metric_name=submetric_names,
    higher_is_better={n: True for n in submetric_names},
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=ifeval_metric,
    corpus_level_fn={
        "prompt_level_strict_acc": np.mean,
        "inst_level_strict_acc": agg_inst_level_acc,
        "prompt_level_loose_acc": np.mean,
        "inst_level_loose_acc": agg_inst_level_acc,
    },
)

# We create the task config
ifeval = LightevalTaskConfig(
    name="ifeval",
    prompt_function=ifeval_prompt,
    suite=["extended"],
    hf_repo="google/IFEval",
    hf_subset="default",
    metric=[ifeval_metrics],
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=1280,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
)


TASKS_TABLE = [ifeval]

extend_enum(Metrics, "ifeval_metric", ifeval_metrics)

if __name__ == "__main__":
    # Adds the metric to the metric list!
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
