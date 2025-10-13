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
Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school
math problems.
The same 250 problems from GSM8K are each translated via human annotators in 10
languages.
language list: en, es, fr, de, ru, zh, ja, th, sw, bn, te

https://arxiv.org/abs/2210.03057
"""

mgsm_en = LightevalTaskConfig(
    name="mgsm:en",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_en,
    hf_repo="juletxara/mgsm",
    hf_subset="en",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n", "=", "Question="],
    version=0,
)

mgsm_es = LightevalTaskConfig(
    name="mgsm:es",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_es,
    hf_repo="juletxara/mgsm",
    hf_subset="es",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n", "=", "Pregunta="],
    version=0,
)

mgsm_fr = LightevalTaskConfig(
    name="mgsm:fr",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_fr,
    hf_repo="juletxara/mgsm",
    hf_subset="fr",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n", "=", "Question="],
    version=0,
)

mgsm_de = LightevalTaskConfig(
    name="mgsm:de",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_de,
    hf_repo="juletxara/mgsm",
    hf_subset="de",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n", "=", "Frage="],
    version=0,
)

mgsm_ru = LightevalTaskConfig(
    name="mgsm:ru",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_ru,
    hf_repo="juletxara/mgsm",
    hf_subset="ru",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n", "=", "\u0417\u0430\u0434\u0430\u0447\u0430="],
    version=0,
)

mgsm_zh = LightevalTaskConfig(
    name="mgsm:zh",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_zh,
    hf_repo="juletxara/mgsm",
    hf_subset="zh",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n", "=", "\u95ee\u9898="],
    version=0,
)

mgsm_ja = LightevalTaskConfig(
    name="mgsm:ja",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_ja,
    hf_repo="juletxara/mgsm",
    hf_subset="ja",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n", "=", "\u554f\u984c="],
    version=0,
)

mgsm_th = LightevalTaskConfig(
    name="mgsm:th",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_th,
    hf_repo="juletxara/mgsm",
    hf_subset="th",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n", "=", "\u0e42\u0e08\u0e17\u0e22\u0e4c="],
    version=0,
)

mgsm_sw = LightevalTaskConfig(
    name="mgsm:sw",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_sw,
    hf_repo="juletxara/mgsm",
    hf_subset="sw",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n", "=", "Swali="],
    version=0,
)

mgsm_bn = LightevalTaskConfig(
    name="mgsm:bn",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_bn,
    hf_repo="juletxara/mgsm",
    hf_subset="bn",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n", "=", "\u09aa\u09cd\u09b0\u09b6\u09cd\u09a8="],
    version=0,
)

mgsm_te = LightevalTaskConfig(
    name="mgsm:te",
    suite=["lighteval"],
    prompt_function=prompt.mgsm_te,
    hf_repo="juletxara/mgsm",
    hf_subset="te",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n", "=", "\u0c2a\u0c4d\u0c30\u0c36\u0c4d\u0c28="],
    version=0,
)
