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
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


"""
abstract:
The DROP dataset is a new question-answering dataset designed to evaluate the
ability of language models to answer complex questions that require reasoning
over multiple sentences.

languages:
en

paper:
https://arxiv.org/abs/1810.00505
"""

drop_qa = LightevalTaskConfig(
    name="drop",
    prompt_function=get_qa_prompt_function(
        Language.ENGLISH,
        lambda line: {
            "context": line["passage"],
            "question": line["question"],
            "choices": list(
                filter(
                    lambda x: x,
                    [line["answer"].get("number")]
                    + line["answer"]["spans"]
                    + [prompt.get_drop_date(line["answer"].get("date"))],
                )
            ),
        },
    ),
    suite=("lighteval",),
    hf_repo="lighteval/drop_harness",
    hf_subset="default",
    hf_filter=lambda line: list(
        filter(
            lambda x: x,
            [line["answer"].get("number")]
            + line["answer"]["spans"]
            + [prompt.get_drop_date(line["answer"].get("date"))],
        )
    ),
    evaluation_splits=("validation",),
    few_shots_split="train",
    generation_size=250,
    stop_sequence=["Question:", "question:", "\n"],
    metrics=[Metrics.exact_match],
    version=1,
)
