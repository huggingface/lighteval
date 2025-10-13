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


squad_v2 = LightevalTaskConfig(
    name="squad_v2",
    prompt_function=get_qa_prompt_function(
        Language.ENGLISH,
        lambda line: {
            "question": line["question"],
            "context": line["context"],
            "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
        },
    ),
    suite=("lighteval",),
    hf_repo="rajpurkar/squad_v2",
    hf_subset="squad_v2",
    hf_filter=lambda line: any(ans for ans in line["answers"]["text"] if len(ans) > 0),
    evaluation_splits=("validation",),
    few_shots_split="train",
    stop_sequence=["\n", "Question:", "question:"],
    generation_size=200,
    metrics=(
        Metrics.exact_match(
            sample_params={
                "normalize_gold": helm_normalizer,
                "normalize_pred": helm_normalizer,
                "type_exact_match": "prefix",
            }
        ),
        Metrics.f1_score(sample_params={"normalize_gold": helm_normalizer, "normalize_pred": helm_normalizer}),
    ),
)
