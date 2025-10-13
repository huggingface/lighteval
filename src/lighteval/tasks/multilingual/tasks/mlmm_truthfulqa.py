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


# ---------------------------- TruthfulQA ---------------------------- #
# TruthfulQA: Measuring How Models Mimic Human Falsehoods
# Paper: https://arxiv.org/abs/2109.07958
# TruthfulQA is a benchmark dataset designed to measure the truthfulness of language models.
# It consists of questions that humans might answer incorrectly due to false beliefs or misconceptions.
# The task evaluates a model's ability to provide truthful answers and avoid common human biases.
# github: https://github.com/nlp-uoregon/mlmm-evaluation

TASKS_TABLE = []


mlmm_truthfulqa_tasks = [
    LightevalTaskConfig(
        name=f"mlmm_truthfulqa_{language.value}_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            language,
            partial(
                lambda subset, line: {
                    "question": line["question"],
                    "choices": line[f"{subset}_targets"]["choices"],
                    "gold_idx": [ix for ix, label in enumerate(line[f"{subset}_targets"]["labels"]) if label == 1],  # type: ignore
                },
                subset,
            ),
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="jon-tow/okapi_truthfulqa",
        hf_subset=standardize_tag(language.value),
        hf_revision="cdd5db1a66fd04105622109d1c2a5cbc8cde7586",
        evaluation_splits=("validation",),
        hf_avail_splits=["validation"],
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for subset in ["mc1", "mc2"]
    for language in [
        Language.ARABIC,
        Language.BENGALI,
        Language.CATALAN,
        Language.DANISH,
        Language.GERMAN,
        Language.SPANISH,
        Language.BASQUE,
        Language.FRENCH,
        Language.GUJARATI,
        Language.HINDI,
        Language.CROATIAN,
        Language.HUNGARIAN,
        Language.ARMENIAN,
        Language.INDONESIAN,
        Language.ICELANDIC,
        Language.ITALIAN,
        Language.KANNADA,
        Language.MALAYALAM,
        Language.MARATHI,
        Language.NORWEGIAN,
        Language.NEPALI,
        Language.DUTCH,
        Language.PORTUGUESE,
        Language.ROMANIAN,
        Language.RUSSIAN,
        Language.SLOVAK,
        Language.SERBIAN,
        Language.SWEDISH,
        Language.TAMIL,
        Language.TELUGU,
        Language.UKRAINIAN,
        Language.VIETNAMESE,
        Language.CHINESE,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
