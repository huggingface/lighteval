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


from langcodes import Language as LangCodeLanguage

from lighteval.metrics.dynamic_metrics import (
    MultilingualQuasiExactMatchMetric,
    MultilingualQuasiF1ScoreMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


# IndicQA: A reading comprehension dataset for 11 Indian languages.
# https://arxiv.org/abs/2407.13522

TASKS_TABLE = []


indicqa_tasks = [
    LightevalTaskConfig(
        name=f"indicqa_{language.value}",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
            },
        ),
        suite=("lighteval",),
        hf_repo="ai4bharat/IndicQA",
        hf_subset=f"indicqa.{LangCodeLanguage.get(language.value).language}",
        hf_filter=lambda line: any(len(ans) > 0 for ans in line["answers"]["text"]),
        hf_revision="92d96092ae229950973dac3b9998f8b3a8949b0a",
        evaluation_splits=("test",),
        hf_avail_splits=("test",),
        generation_size=400,
        metrics=(
            MultilingualQuasiExactMatchMetric(language, "prefix"),
            MultilingualQuasiF1ScoreMetric(language),
        ),
        stop_sequence=("\n",),
    )
    for language in [
        Language.ASSAMESE,
        Language.BENGALI,
        Language.GUJARATI,
        Language.HINDI,
        Language.KANNADA,
        Language.MALAYALAM,
        Language.MARATHI,
        Language.ORIYA,
        Language.PUNJABI,
        Language.TAMIL,
        Language.TELUGU,
    ]
]
