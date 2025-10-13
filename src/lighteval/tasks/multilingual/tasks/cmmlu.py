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
from itertools import permutations

from langcodes import Language as LangCodeLanguage
from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
    MultilingualQuasiExactMatchMetric,
    MultilingualQuasiF1ScoreMetric,
)
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import (
    agieval_adapter,
    alghafa_adapter,
    ceval_adapter,
    enem_adapter,
    get_m3exam_adapter,
    get_mkqa_adapter,
    sciqa_adapter,
    thai_exams_adapter,
    winogrand_adapter,
    xcodah_adapter,
)
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation, normalize_subset
from lighteval.tasks.templates.boolq import get_boolq_prompt_function
from lighteval.tasks.templates.continuation import get_continuation_prompt_function
from lighteval.tasks.templates.copa import get_copa_prompt_function
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.nli import get_nli_prompt_function
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.tasks.templates.translation import get_translation_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.language import Language, iso_639_3_ind_to_iso_639_3_macro, manage_duplicate_language_codes


TASKS_TABLE = []


CMMLU_SUBSETS = [
    "agronomy",
    "anatomy",
    "ancient_chinese",
    "arts",
    "astronomy",
    "business_ethics",
    "chinese_civil_service_exam",
    "chinese_driving_rule",
    "chinese_food_culture",
    "chinese_foreign_policy",
    "chinese_history",
    "chinese_literature",
    "chinese_teacher_qualification",
    "clinical_knowledge",
    "college_actuarial_science",
    "college_education",
    "college_engineering_hydrology",
    "college_law",
    "college_mathematics",
    "college_medical_statistics",
    "college_medicine",
    "computer_science",
    "computer_security",
    "conceptual_physics",
    "construction_project_management",
    "economics",
    "education",
    "electrical_engineering",
    "elementary_chinese",
    "elementary_commonsense",
    "elementary_information_and_technology",
    "elementary_mathematics",
    "ethnology",
    "food_science",
    "genetics",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_geography",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_politics",
    "human_sexuality",
    "international_law",
    "journalism",
    "jurisprudence",
    "legal_and_moral_basis",
    "logical",
    "machine_learning",
    "management",
    "marketing",
    "marxist_theory",
    "modern_chinese",
    "nutrition",
    "philosophy",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_study",
    "sociology",
    "sports_science",
    "traditional_chinese_medicine",
    "virology",
    "world_history",
    "world_religions",
]


cmmlu_tasks = [
    LightevalTaskConfig(
        name=f"cmmlu_{Language.CHINESE.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.CHINESE,
            lambda line: {
                "question": line["Question"],
                "choices": [line["A"], line["B"], line["C"], line["D"]],
                "gold_idx": LETTER_INDICES.index(line["Answer"]),
            },
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="haonan-li/cmmlu",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                LogLikelihoodAccMetric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in CMMLU_SUBSETS
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
