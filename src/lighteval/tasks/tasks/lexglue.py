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


"""
LexGLUE: A Benchmark Dataset for Legal Language Understanding in English

https://arxiv.org/abs/2110.00976
"""

lexglue_case_hold = LightevalTaskConfig(
    name="lexglue:case_hold",
    suite=["lighteval"],
    prompt_function=prompt.lex_glue_case_hold,
    hf_repo="lighteval/lexglue",
    hf_subset="case_hold",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lexglue_ecthr_a = LightevalTaskConfig(
    name="lexglue:ecthr_a",
    suite=["lighteval"],
    prompt_function=prompt.lex_glue_ecthr_a,
    hf_repo="lighteval/lexglue",
    hf_subset="ecthr_a",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lexglue_ecthr_b = LightevalTaskConfig(
    name="lexglue:ecthr_b",
    suite=["lighteval"],
    prompt_function=prompt.lex_glue_ecthr_b,
    hf_repo="lighteval/lexglue",
    hf_subset="ecthr_b",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lexglue_eurlex = LightevalTaskConfig(
    name="lexglue:eurlex",
    suite=["lighteval"],
    prompt_function=prompt.lex_glue_eurlex,
    hf_repo="lighteval/lexglue",
    hf_subset="eurlex",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lexglue_ledgar = LightevalTaskConfig(
    name="lexglue:ledgar",
    suite=["lighteval"],
    prompt_function=prompt.lex_glue_ledgar,
    hf_repo="lighteval/lexglue",
    hf_subset="ledgar",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lexglue_scotus = LightevalTaskConfig(
    name="lexglue:scotus",
    suite=["lighteval"],
    prompt_function=prompt.lex_glue_scotus,
    hf_repo="lighteval/lexglue",
    hf_subset="scotus",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

lexglue_unfair_tos = LightevalTaskConfig(
    name="lexglue:unfair_tos",
    suite=["lighteval"],
    prompt_function=prompt.lex_glue_unfair_tos,
    hf_repo="lighteval/lexglue",
    hf_subset="unfair_tos",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)
