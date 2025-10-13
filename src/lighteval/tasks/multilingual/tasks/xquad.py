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


from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    MultilingualQuasiExactMatchMetric,
    MultilingualQuasiF1ScoreMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


# ------------------------------- RC Tasks ------------------------------- #
# Reading Comprehension (RC) tasks evaluate a model's ability to understand and extract information from text passages.
# These tasks typically involve answering questions based on given contexts, spanning multiple languages and formats.
# Add RC tasks supporting about 130 unique languages/scripts.
# SQuAD - like
# XQuAD: Cross-lingual Question Answering Dataset, extending SQuAD to 11 languages.
# https://arxiv.org/abs/1910.11856

TASKS_TABLE = []


xquad_tasks = [
    LightevalTaskConfig(
        name=f"xquad_{language.value}",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="google/xquad",
        hf_subset=f"xquad.{standardize_tag(language.value)}",
        evaluation_splits=("validation",),
        few_shots_split="validation",
        generation_size=400,
        stop_sequence=("\n",),
        metrics=(
            MultilingualQuasiExactMatchMetric(language, "prefix"),
            MultilingualQuasiF1ScoreMetric(language),
        ),
    )
    for language in [
        Language.ARABIC,
        Language.GERMAN,
        Language.GREEK,
        Language.ENGLISH,
        Language.SPANISH,
        Language.HINDI,
        Language.ROMANIAN,
        Language.RUSSIAN,
        Language.THAI,
        Language.TURKISH,
        Language.VIETNAMESE,
        Language.CHINESE,
    ]
]
