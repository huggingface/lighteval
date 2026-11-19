"""
name:
IFEval-Ko

dataset:
allganize/IFEval-Ko

abstract:
Korean adaptation of IFEval for evaluating verifiable instruction following.
Uses Allganize's Korean reference scoring rules.

languages:
korean

tags:
instruction-following

paper:
https://arxiv.org/abs/2311.07911
"""

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks.ifeval_ko.scoring import score_response
from lighteval.utils.imports import requires


@requires("langdetect", "immutabledict")
def ifeval_ko_prompt(line: dict, task_name: str = "") -> Doc:
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[""],
        gold_index=0,
        instruction="",
        specific={
            "instructions_id_list": line["instruction_id_list"],
            "kwargs": line["kwargs"],
            "ifeval_ko": {
                "key": line["key"],
            },
        },
    )


class IFEvalKoMetrics(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> dict:
        return score_response(
            prompt=doc.query,
            instruction_id_list=doc.specific["instructions_id_list"],
            kwargs=doc.specific["kwargs"],
            response=model_response.final_text[0],
        )


def agg_inst_level_acc(items):
    flags = [flag for instruction_flags in items for flag in instruction_flags]
    return sum(flags) / len(flags)


METRIC_NAMES = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]


ifeval_ko_metrics = SampleLevelMetricGrouping(
    metric_name=METRIC_NAMES,
    higher_is_better=dict.fromkeys(METRIC_NAMES, True),
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=IFEvalKoMetrics(),
    corpus_level_fn={
        "prompt_level_strict_acc": np.mean,
        "inst_level_strict_acc": agg_inst_level_acc,
        "prompt_level_loose_acc": np.mean,
        "inst_level_loose_acc": agg_inst_level_acc,
    },
)
KO_DATA_REVISION = "54199e3801116897697babf341865741dcd06fc8"

ifeval_ko = LightevalTaskConfig(
    name="ifeval_ko",
    prompt_function=ifeval_ko_prompt,
    hf_repo="allganize/IFEval-Ko",
    hf_subset="default",
    hf_revision=KO_DATA_REVISION,
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    num_fewshots=0,
    metrics=[ifeval_ko_metrics],
    generation_size=1280,
    stop_sequence=["<end_of_turn>", "<|im_end|>"],
    version=0,
)

TASKS_TABLE = [ifeval_ko]
