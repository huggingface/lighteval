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

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def filipino_global_mmlu_pfn(line, task_name: str = None) -> Doc:
    instruction = "Ang sumusunod na tanong ay isang multiple-choice na tanong. Piliin ang tamang sagot:\n\n"
    choices = []
    valid_keys = []

    for idx, key in enumerate(["a", "b", "c", "d", "e"]):
        option = line.get(f"option_{idx + 1}")
        if option:
            choices.append(option)
            valid_keys.append(key)

    answer_index = valid_keys.index(str(line["answer"]).upper())
    query = f"{instruction}{line['question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(valid_keys, choices)])
    query += "Sagot:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=valid_keys,
        gold_index=answer_index,
        instruction=instruction,
    )


# fmt: off
MMLU_SUBSETS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology", "college_chemistry", "college_computer_science",
    "college_mathematics", "college_medicine", "college_physics", "computer_security", "conceptual_physics", "econometrics", "electrical_engineering",
    "elementary_mathematics", "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics", "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]
# fmt: on


def create_global_mmlu_task(name: str, hf_subset: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=name,
        hf_subset=hf_subset,
        prompt_function=filipino_global_mmlu_pfn,
        hf_repo="UD-Filipino/Global-MMLU-Filipino",
        metric=[Metrics.loglikelihood_acc_norm],
        hf_avail_splits=["test"],
        evaluation_splits=["dev"],
        few_shots_split=["dev"],
        few_shots_select="sequential",
        suite=["filbench"],
        generation_size=-1,
        stop_sequence=None,
        trust_dataset=True,
        version=0,
    )


FILIPINO_GLOBAL_MMLU_TASKS = [
    create_global_mmlu_task(name=f"filipino_mmlu_mt:{subset}", hf_subset=subset) for subset in MMLU_SUBSETS
]
