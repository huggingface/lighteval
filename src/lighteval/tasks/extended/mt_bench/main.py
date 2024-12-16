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

# ruff: noqa: F405, F403, F401, I001
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics_sample import JudgeLLMMTBench
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping, MetricCategory, MetricUseCase
from lighteval.tasks.extended.mt_bench.judge_prompt_templates import (
    flow_judge_prompt_mt_bench_with_ref,
    flow_judge_prompt_mt_bench_without_ref,
)
import re
import numpy as np


def mt_bench_prompt(line, task_name: str = ""):
    return Doc(
        task_name=task_name,
        query=f"{line['turns'][0]}",
        choices=[],
        instruction=None,
        gold_index=[],
        specific={
            "reference": line["reference"],
            "category": line["category"],
            "multi_turn_queries": line["turns"],
            "id": line["question_id"],
        },
    )


def process_judge_response(x):
    search = re.search(r"<score>\s(\d)\s</score>", x)
    return int(search.group(1)) if search else 0


def flow_judge_mt_bench_prompt(question, answer, options, gold):
    if gold is not None and len(gold) > 0:
        return flow_judge_prompt_mt_bench_with_ref(question, options, answer, gold)

    return flow_judge_prompt_mt_bench_without_ref(question, options, answer, gold)


llm_judge_mt_bench = SampleLevelMetricGrouping(
    metric_name=["judge_score_turn_1", "judge_score_turn_2"],
    higher_is_better={"judge_score_turn_1": True, "judge_score_turn_2": True},
    category=MetricCategory.LLM_AS_JUDGE_MULTI_TURN,
    use_case=MetricUseCase.SUMMARIZATION,
    sample_level_fn=JudgeLLMMTBench(
        judge_model_name="flowaicom/Flow-Judge-v0.1",
        template=flow_judge_mt_bench_prompt,
        process_judge_response=process_judge_response,
        judge_backend="vllm",
    ).compute,
    corpus_level_fn={
        "judge_score_turn_1": np.mean,
        "judge_score_turn_2": np.mean,
    },
)

task = LightevalTaskConfig(
    name="mt_bench",
    prompt_function=mt_bench_prompt,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["extended"],
    hf_repo="lighteval/mt-bench",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="",
    few_shots_select="random",
    metric=[llm_judge_mt_bench],
    generation_size=1024,
    stop_sequence=[],
)


TASKS_TABLE = [task]
