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


import numpy as np

from lighteval.metrics.dynamic_metrics import (
    IndicesExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.metrics.metrics import MetricCategory, MetricUseCase, SampleLevelMetric
from lighteval.metrics.metrics_sample import (
    PassAtK,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


TASKS_TABLE = []


def mgsm_prompt(line, task_name: str = None):
    query_template = (
        "Given the following question, output the value corresponding to the correct answer. The last line of your response should be of the following format: 'Answer: $NUMBER' (without quotes) where NUMBER is a number. Think step by step before answering.\n\n###\nQuery:\n{Question}",
    )

    query = query_template.format(
        Question=line["question"],
    )
    instruction = query_template.split("###\n")[0]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[str(line["answer_number"])],
        gold_index=0,
        instruction=instruction,
    )


MGSM_TASKS = [
    LightevalTaskConfig(
        name=f"mgsm_instruct_{language.value}",
        prompt_function=mgsm_prompt,
        suite=("lighteval",),
        hf_repo="juletxara/mgsm",
        hf_subset=lang,
        evaluation_splits=("test",),
        few_shots_split="train",
        metric=[
            SampleLevelMetric(
                metric_name="pass@1:1_samples",
                sample_level_fn=PassAtK(
                    k=1,
                    n=1,
                    sample_scoring_function=lambda pred, ref, doc: multilingual_extractive_match_metric(
                        language=language,
                        gold_extraction_target=[
                            IndicesExtractionConfig(
                                prefix_for_extraction="NativeLetters", try_extract_without_anchor=False
                            )
                        ],
                        pred_extraction_target=[
                            IndicesExtractionConfig(
                                prefix_for_extraction="NativeLetters", try_extract_without_anchor=False
                            )
                        ],
                        precision=6,
                    ).sample_level_fn([ref], [pred], doc),
                ).compute,
                category=MetricCategory.GENERATIVE_SAMPLING,
                use_case=MetricUseCase.REASONING,
                corpus_level_fn=np.mean,
                higher_is_better=True,
            )
        ],
        generation_size=32768,  # needed for reasoning models like R1
        stop_sequence=[],  # no stop sequence, will use eos token
    )
    for lang, language in [
        ("bn", Language.BENGALI),
        ("de", Language.GERMAN),
        ("en", Language.ENGLISH),
        ("es", Language.SPANISH),
        ("fr", Language.FRENCH),
        ("ja", Language.JAPANESE),
        ("ru", Language.RUSSIAN),
        ("sw", Language.SWAHILI),
        ("te", Language.TELUGU),
        ("th", Language.THAI),
        ("zh", Language.CHINESE),
    ]
]
TASKS_TABLE.extend(MGSM_TASKS)
