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

import logging
import re

import numpy as np

from lighteval.metrics.metrics_sample import JudgeLLMMixEval
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase, SampleLevelMetricGrouping
from lighteval.tasks.extended.mix_eval.judge_prompts import (
    flow_judge_for_freeform_template,
    flow_judge_for_multichoice_template,
    gpt_judge_for_closeended_freeform,
    gpt_judge_for_closeended_multiplechoice,
)
from lighteval.tasks.extended.mix_eval.prompts import construct_prompt_freeform, construct_prompt_multichoice
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


logger = logging.getLogger(__name__)


def mixeval_freeform_prompt(line, task_name: str = ""):
    prompt = construct_prompt_freeform(line)
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=line["target"],
        gold_index=list(range(len(line["target"]))),
        instruction="",
        specific={
            "problem-type": line["problem_type"],
            "benchmark-name": line["benchmark_name"],
            "question": line["prompt"],
        },
    )


# Very specific task where there are no precise outputs but instead we test if the format obeys rules
def mixeval_multichoice_prompt(line, task_name: str = ""):
    prompt = construct_prompt_multichoice(line)
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=line["options"],
        gold_index=[int(target) for target in line["target"]],
        instruction="",
        specific={
            "problem-type": line["problem_type"],
            "benchmark-name": line["benchmark_name"],
            "question": line["prompt"],
        },
    )


def process_judge_response(x):
    try:
        search = re.search(r"<score>\s(\d)\s</score>", x)
        return int(search.group(1)) if search else 0
    except Exception as e:
        logger.warning(f"Error processing judge response for flow: {e}")
        return 0


def process_judge_response_multichoice_gpt(x):
    try:
        search = re.search(r"\[\[([01])\]\]", x)
        return int(search.group(1)) if search else 0
    except Exception as e:
        logger.warning(f"Error processing judge response for multichoice GPT: {e}")
        return 0


def process_judge_response_freeform_gpt(x):
    try:
        search = re.search(r"\[\[(\d.\d)\]\]", x)
        return float(search.group(1)) if search else 0
    except Exception as e:
        logger.warning(f"Error processing judge response for freeform GPT: {e}")
        return 0


llm_judge_mixeval_multichoice_flow_judge = SampleLevelMetricGrouping(
    metric_name=["llm_judge_mixeval_flow"],
    higher_is_better={"judge_score_flow": True},
    category=MetricCategory.LLM_AS_JUDGE,
    use_case=MetricUseCase.SUMMARIZATION,
    sample_level_fn=JudgeLLMMixEval(
        judge_model_name="flowaicom/Flow-Judge-v0.1",
        template=flow_judge_for_multichoice_template,
        process_judge_response=process_judge_response,
        judge_backend="vllm",
        short_judge_name="flow",
    ).compute,
    corpus_level_fn={
        "judge_score_flow": np.mean,
    },
)

llm_judge_mixeval_multichoice_gpt_judge = SampleLevelMetricGrouping(
    metric_name=["llm_judge_mixeval_gpt3"],
    higher_is_better={"judge_score_gpt-3.5": True},
    category=MetricCategory.LLM_AS_JUDGE,
    use_case=MetricUseCase.SUMMARIZATION,
    sample_level_fn=JudgeLLMMixEval(
        judge_model_name="gpt-3.5-turbo",
        template=gpt_judge_for_closeended_multiplechoice,
        process_judge_response=process_judge_response_multichoice_gpt,
        judge_backend="openai",
        short_judge_name="gpt-3.5",
    ).compute,
    corpus_level_fn={
        "judge_score_gpt-3.5": np.mean,
    },
)


def mean_dv_5(x):
    return np.mean(x) / 5


llm_judge_mixeval_freeform_flow_judge = SampleLevelMetricGrouping(
    metric_name=["llm_judge_mixeval_flow"],
    higher_is_better={"judge_score": True},
    category=MetricCategory.LLM_AS_JUDGE,
    use_case=MetricUseCase.SUMMARIZATION,
    sample_level_fn=JudgeLLMMixEval(
        judge_model_name="flowaicom/Flow-Judge-v0.1",
        template=flow_judge_for_freeform_template,
        process_judge_response=process_judge_response,
        judge_backend="vllm",
        short_judge_name="flow",
    ).compute,
    corpus_level_fn={
        "judge_score_flow": mean_dv_5,
    },
)

llm_judge_mixeval_freeform_gpt_judge = SampleLevelMetricGrouping(
    metric_name=["llm_judge_mixeval_gpt3"],
    higher_is_better={"judge_score_gpt-3.5": True},
    category=MetricCategory.LLM_AS_JUDGE,
    use_case=MetricUseCase.SUMMARIZATION,
    sample_level_fn=JudgeLLMMixEval(
        judge_model_name="gpt-3.5-turbo",
        template=gpt_judge_for_closeended_freeform,
        process_judge_response=process_judge_response_freeform_gpt,
        judge_backend="openai",
        short_judge_name="gpt-3.5",
    ).compute,
    corpus_level_fn={
        "judge_score_gpt-3.5": np.mean,
    },
)


mixeval_freeform_easy = LightevalTaskConfig(
    name="mixeval_easy:freeform",
    prompt_function=mixeval_freeform_prompt,
    suite=["extended"],
    hf_repo="MixEval/MixEval",
    hf_subset="MixEval",
    metric=[llm_judge_mixeval_freeform_flow_judge, llm_judge_mixeval_freeform_gpt_judge],
    hf_avail_splits=["free_form"],
    evaluation_splits=["free_form"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=100,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
)


mixeval_multichoice_easy = LightevalTaskConfig(
    name="mixeval_easy:multichoice",
    prompt_function=mixeval_multichoice_prompt,
    suite=["extended"],
    hf_repo="MixEval/MixEval",
    hf_subset="MixEval",
    metric=[llm_judge_mixeval_multichoice_flow_judge, llm_judge_mixeval_multichoice_gpt_judge],
    hf_avail_splits=["multiple_choice"],
    evaluation_splits=["multiple_choice"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=100,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
)

mixeval_freeform_hard = LightevalTaskConfig(
    name="mixeval_hard:freeform",
    prompt_function=mixeval_freeform_prompt,
    suite=["extended"],
    hf_repo="MixEval/MixEval",
    hf_subset="MixEval_Hard",
    metric=[llm_judge_mixeval_freeform_flow_judge, llm_judge_mixeval_freeform_gpt_judge],
    hf_avail_splits=["free_form"],
    evaluation_splits=["free_form"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=100,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
)


mixeval_multichoice_hard = LightevalTaskConfig(
    name="mixeval_hard:multichoice",
    prompt_function=mixeval_multichoice_prompt,
    suite=["extended"],
    hf_repo="MixEval/MixEval",
    hf_subset="MixEval_Hard",
    metric=[llm_judge_mixeval_multichoice_flow_judge, llm_judge_mixeval_multichoice_gpt_judge],
    hf_avail_splits=["multiple_choice"],
    evaluation_splits=["multiple_choice"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=100,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
)


TASKS_TABLE = [mixeval_multichoice_easy, mixeval_freeform_easy, mixeval_multichoice_hard, mixeval_freeform_hard]
