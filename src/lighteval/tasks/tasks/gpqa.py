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

from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import multichoice_scorer
from lighteval.tasks.lighteval_task import LightevalTaskConfig_inspect


gpqa_diamond = LightevalTaskConfig_inspect(
    name="gpqa:diamond",
    prompt_function=prompt.gpqa_instruct,
    dataset_repo="Idavidrein/gpqa",
    dataset_subset="gpqa_diamond",
    dataset_split="train",
    scorers=[multichoice_scorer(), choice()],
    solvers=[multiple_choice()],
    system_prompt="Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.",
)


gpqa_extended = LightevalTaskConfig_inspect(
    name="gpqa:extended",
    prompt_function=prompt.gpqa_instruct,
    dataset_repo="Idavidrein/gpqa",
    dataset_subset="gpqa_extended",
    dataset_split="train",
    scorers=[multichoice_scorer(), choice()],
    solvers=[multiple_choice()],
    system_prompt="Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.",
)


gpqa_main = LightevalTaskConfig_inspect(
    name="gpqa:main",
    prompt_function=prompt.gpqa_instruct,
    dataset_repo="Idavidrein/gpqa",
    dataset_subset="gpqa_main",
    dataset_split="train",
    scorers=[multichoice_scorer(), choice()],
    solvers=[multiple_choice()],
    system_prompt="Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.",
)
