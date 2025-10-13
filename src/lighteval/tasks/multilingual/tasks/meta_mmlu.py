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


# Meta MMLU: A multilingual version of MMLU (using google translation)
# Paper: https://arxiv.org/abs/2407.21783

TASKS_TABLE = []


MMLU_SUBSETS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


meta_mmlu_tasks = [
    LightevalTaskConfig(
        name=f"meta_mmlu_{language.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["input_question"],
                "choices": [v for _, v in sorted(line["input_choice_list"].items(), key=lambda x: x[0])],
                "gold_idx": LETTER_INDICES.index(line["input_correct_responses"][0]),
            },
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="meta-llama/Meta-Llama-3.1-8B-Instruct-evals",
        hf_subset=f"Llama-3.1-8B-Instruct-evals__multilingual_mmlu_{standardize_tag(language.value)}__details",
        hf_filter=partial(
            lambda language, subset, line: line["subtask_name"]
            == f"mmlu_{standardize_tag(language.value)}_chat.{subset}",
            language,
            subset,
        ),
        evaluation_splits=("latest",),
        hf_avail_splits=["latest"],
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                LogLikelihoodAccMetric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in MMLU_SUBSETS
    for language in [
        Language.GERMAN,
        Language.SPANISH,
        Language.FRENCH,
        Language.HINDI,
        Language.ITALIAN,
        Language.PORTUGUESE,
        Language.THAI,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
