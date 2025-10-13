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


ACVA_SUBSET = [
    "Algeria",
    "Ancient_Egypt",
    "Arab_Empire",
    "Arabic_Architecture",
    "Arabic_Art",
    "Arabic_Astronomy",
    "Arabic_Calligraphy",
    "Arabic_Ceremony",
    "Arabic_Clothing",
    "Arabic_Culture",
    "Arabic_Food",
    "Arabic_Funeral",
    "Arabic_Geography",
    "Arabic_History",
    "Arabic_Language_Origin",
    "Arabic_Literature",
    "Arabic_Math",
    "Arabic_Medicine",
    "Arabic_Music",
    "Arabic_Ornament",
    "Arabic_Philosophy",
    "Arabic_Physics_and_Chemistry",
    "Arabic_Wedding",
    "Bahrain",
    "Comoros",
    "Egypt_modern",
    "InfluenceFromAncientEgypt",
    "InfluenceFromByzantium",
    "InfluenceFromChina",
    "InfluenceFromGreece",
    "InfluenceFromIslam",
    "InfluenceFromPersia",
    "InfluenceFromRome",
    "Iraq",
    "Islam_Education",
    "Islam_branches_and_schools",
    "Islamic_law_system",
    "Jordan",
    "Kuwait",
    "Lebanon",
    "Libya",
    "Mauritania",
    "Mesopotamia_civilization",
    "Morocco",
    "Oman",
    "Palestine",
    "Qatar",
    "Saudi_Arabia",
    "Somalia",
    "Sudan",
    "Syria",
    "Tunisia",
    "United_Arab_Emirates",
    "Yemen",
    "communication",
    "computer_and_phone",
    "daily_life",
    "entertainment",
]


acva_tasks = [
    LightevalTaskConfig(
        name=f"acva_{Language.ARABIC.value}:{subset}",
        prompt_function=get_boolq_prompt_function(
            Language.ARABIC,
            lambda line: {
                "question": line["question"],
                "answer": line["answer"] == "صح",
            },
            formulation=CFFormulation(),
        ),
        suite=("lighteval",),
        hf_repo="OALL/ACVA",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="validation",
        metrics=[MultilingualQuasiExactMatchMetric(Language.ARABIC, "full"), LogLikelihoodAccMetric()],
        generation_size=5,
        stop_sequence=("\n",),
    )
    for subset in ACVA_SUBSET
]
