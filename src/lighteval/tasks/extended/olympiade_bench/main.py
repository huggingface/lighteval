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


from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


# Very specific task where there are no precise outputs but instead we test if the format obeys rules
def olympiad_bench_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[line["final_answer"]],
        gold_index=0,
        instruction="",
        specific={},
    )


# * OE: Open-ended questions
# * TP: Theorem proof problems
# * MM: Multimodal
# * TO: Text-only
# * physics: Physics problems
# * maths: Math problems
# * en: English
# * zh: Chinese
# * COMP: Competition problems
# * CEE: Chinese College Entrance Exam problems

question_type = ["OE", "TP"]
multimodality = ["TO"]  # MM
subject = ["physics", "maths"]
language = ["en"]  # "zh"]
source = ["COMP", "CEE"]

olympiad_bench_subsets = []

for qt in question_type:
    for mm in multimodality:
        for sub in subject:
            for lang in language:
                for src in source:
                    olympiad_bench_subsets.append(f"{qt}_{mm}_{sub}_{lang}_{src}")

extraction_targets = [ExprExtractionConfig(), LatexExtractionConfig()]

metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=extraction_targets,
    pred_extraction_target=extraction_targets,
    precision=6,
)
# We create the task config
olympiad_bench = LightevalTaskConfig(
    name="olympiad_bench",
    prompt_function=olympiad_bench_prompt,
    suite=["extended"],
    hf_repo="Hothan/OlympiadBench",
    hf_subset=olympiad_bench_subsets[0],
    metric=[metric],
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=2048,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="1.0",
)

# print(olympiad_bench)

TASKS_TABLE = [olympiad_bench]
