"""
name:
LICA-Bench

dataset:
purvanshi/lica-bench-eval

abstract:
LICA-Bench is a structured evaluation suite for vision-language models on graphic
design artifacts, comprising 39 tasks across 7 domains: layout, typography, SVG,
templates, temporal, Lottie, and category. Tasks cover both understanding
(e.g. classify a design, identify fonts) and generation (e.g. produce SVG code,
describe layouts).

languages:
english

tags:
multimodal, vision-language, graphic-design, evaluation

paper:
https://github.com/purvanshi/lica-bench

starred:
false
"""

import json

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


HF_REPO = "purvanshi/lica-bench-eval"

TASK_IDS = [
    "category-1", "category-2",
    "layout-1", "layout-2", "layout-3", "layout-4",
    "layout-5", "layout-6", "layout-7", "layout-8",
    "svg-1", "svg-2", "svg-3", "svg-4",
    "svg-5", "svg-6", "svg-7", "svg-8",
    "template-1", "template-2", "template-3", "template-4", "template-5",
    "temporal-1", "temporal-2", "temporal-3",
    "temporal-4", "temporal-5", "temporal-6",
    "typography-1", "typography-2", "typography-3", "typography-4",
    "typography-5", "typography-6", "typography-7", "typography-8",
    "lottie-1", "lottie-2",
]


def lica_bench_prompt(line, task_name: str = None):
    """Convert a dataset row into a Doc for lighteval.

    Expected dataset columns: question, answer, domain, image (optional PIL).
    """
    question = line.get("question", "")

    answer = line.get("answer", "")
    if isinstance(answer, (dict, list)):
        answer = json.dumps(answer, ensure_ascii=False, sort_keys=True, default=str)
    else:
        answer = str(answer)

    images = []
    if line.get("image") is not None:
        img = line["image"]
        try:
            images.append(img.convert("RGB"))
        except Exception:
            pass

    return Doc(
        task_name=task_name,
        query=question,
        choices=[answer],
        gold_index=0,
        images=images if images else None,
        specific={
            "domain": line.get("domain", ""),
            "task_id": line.get("task_id", ""),
        },
    )


def _make_config(task_id: str) -> LightevalTaskConfig:
    subset = task_id
    name = f"lica_bench:{task_id}"
    return LightevalTaskConfig(
        name=name,
        prompt_function=lica_bench_prompt,
        hf_repo=HF_REPO,
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=1024,
        metrics=[Metrics.exact_match],
        stop_sequence=None,
        version=0,
    )


TASKS_TABLE = [_make_config(tid) for tid in TASK_IDS]
