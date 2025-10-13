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

from functools import partial

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    MultilingualQuasiExactMatchMetric,
    MultilingualQuasiF1ScoreMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import (
    get_mkqa_adapter,
)
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


TASKS_TABLE = []


MKQA_TASK_TO_ID = {
    "entity": 0,
    "long_answer": 1,
    # "unanswerable": 2,
    "date": 3,
    "number": 4,
    "number_with_unit": 5,
    "short_phrase": 6,
    "binary": 7,
}


mkqa_tasks = [
    LightevalTaskConfig(
        name=f"mkqa_{language.value}:{subset}",
        prompt_function=get_qa_prompt_function(language, partial(get_mkqa_adapter, language)),
        suite=("lighteval",),
        hf_repo="apple/mkqa",
        hf_subset="mkqa",
        hf_revision="325131889721ae0ed885b76ecb8011369d75abad",
        hf_filter=partial(
            lambda language, subset, line: line["answers"][
                "zh_cn" if language == Language.CHINESE else standardize_tag(language.value)
            ][0]["type"]
            == MKQA_TASK_TO_ID[subset],
            language,
            subset,
        ),
        evaluation_splits=("train",),
        hf_avail_splits=["train"],
        stop_sequence=("\n",),
        metrics=[
            MultilingualQuasiExactMatchMetric(language, "prefix"),
            MultilingualQuasiF1ScoreMetric(language),
        ]
        if subset in ["entity", "long_answer", "short_phrase"]
        else [
            MultilingualQuasiExactMatchMetric(language, "full"),
        ],
    )
    for subset in MKQA_TASK_TO_ID.keys()
    for language in [
        Language.ARABIC,
        Language.DANISH,
        Language.GERMAN,
        Language.ENGLISH,
        Language.SPANISH,
        Language.FINNISH,
        Language.FRENCH,
        Language.HEBREW,
        Language.HUNGARIAN,
        Language.ITALIAN,
        Language.JAPANESE,
        Language.KOREAN,
        Language.KHMER,
        Language.MALAY,
        Language.DUTCH,
        Language.NORWEGIAN,
        Language.POLISH,
        Language.PORTUGUESE,
        Language.RUSSIAN,
        Language.SWEDISH,
        Language.THAI,
        Language.TURKISH,
        Language.VIETNAMESE,
        Language.CHINESE,  # Simplified
        # Language.CHINESE_HONG_KONG,
        # Language.CHINESE_TRADITIONAL,
    ]
]
