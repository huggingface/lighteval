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

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


# gpqa_diamond = LightevalTaskConfig_inspect(
#     name="gpqa:diamond",
#     prompt_function=prompt.gpqa_instruct,
#     dataset_repo="Idavidrein/gpqa",
#     dataset_subset="gpqa_diamond",
#     dataset_split="train",
#     scorers=[multichoice_scorer(), choice()],
#     solvers=[multiple_choice()],
#     system_prompt="Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.",
# )


# gpqa_extended = LightevalTaskConfig_inspect(
#     name="gpqa:extended",
#     prompt_function=prompt.gpqa_instruct,
#     dataset_repo="Idavidrein/gpqa",
#     dataset_subset="gpqa_extended",
#     dataset_split="train",
#     scorers=[multichoice_scorer(), choice()],
#     solvers=[multiple_choice()],
#     system_prompt="Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.",
# )


# gpqa_main = LightevalTaskConfig_inspect(
#     name="gpqa:main",
#     prompt_function=prompt.gpqa_instruct,
#     dataset_repo="Idavidrein/gpqa",
#     dataset_subset="gpqa_main",
#     dataset_split="train",
#     scorers=[multichoice_scorer(), choice()],
#     solvers=[multiple_choice()],
#     system_prompt="Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.",
# )

gpqa_lighteval = LightevalTaskConfig(
    name="gpqa:mc",
    suite=["lighteval"],
    prompt_function=prompt.gpqa,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_main",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select="random_sampling",
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)
gpqa_diamond_instruct_lighteval = LightevalTaskConfig(
    name="gpqa:diamond",
    suite=["lighteval"],
    prompt_function=prompt.gpqa_instruct,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metrics=[Metrics.gpqa_instruct_pass_at_k(sample_params={"k": 1})],
    stop_sequence=[],  # no stop sequence, will use eos token
    version=1,
)
gpqa_extended_instruct_lighteval = LightevalTaskConfig(
    name="gpqa:extended",
    suite=["lighteval"],
    prompt_function=prompt.gpqa_instruct,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_extended",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    version=0,
)
gpqa_main_instruct_lighteval = LightevalTaskConfig(
    name="gpqa:main",
    suite=["lighteval"],
    prompt_function=prompt.gpqa_instruct,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_main",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metrics=[Metrics.gpqa_instruct_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    version=0,
)