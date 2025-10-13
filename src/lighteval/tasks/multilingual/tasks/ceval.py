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

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import (
    ceval_adapter,
)
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


TASKS_TABLE = []


CEVAL_SUBSET = [
    "computer_network",
    "operating_system",
    "computer_architecture",
    "college_programming",
    "college_physics",
    "college_chemistry",
    "advanced_mathematics",
    "probability_and_statistics",
    "discrete_mathematics",
    "electrical_engineer",
    "metrology_engineer",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_chemistry",
    "high_school_biology",
    "middle_school_mathematics",
    "middle_school_biology",
    "middle_school_physics",
    "middle_school_chemistry",
    "veterinary_medicine",
    "college_economics",
    "business_administration",
    "marxism",
    "mao_zedong_thought",
    "education_science",
    "teacher_qualification",
    "high_school_politics",
    "high_school_geography",
    "middle_school_politics",
    "middle_school_geography",
    "modern_chinese_history",
    "ideological_and_moral_cultivation",
    "logic",
    "law",
    "chinese_language_and_literature",
    "art_studies",
    "professional_tour_guide",
    "legal_professional",
    "high_school_chinese",
    "high_school_history",
    "middle_school_history",
    "civil_servant",
    "sports_science",
    "plant_protection",
    "basic_medicine",
    "clinical_medicine",
    "urban_and_rural_planner",
    "accountant",
    "fire_engineer",
    "environmental_impact_assessment_engineer",
    "tax_accountant",
    "physician",
]


ceval_tasks = [
    LightevalTaskConfig(
        name=f"ceval_{Language.CHINESE.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.CHINESE,
            partial(
                ceval_adapter,
                Language.CHINESE,
                formulation,
            ),
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="ceval/ceval-exam",
        hf_subset=subset,
        evaluation_splits=("val",),
        few_shots_split="dev",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for subset in CEVAL_SUBSET
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
