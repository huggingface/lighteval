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


from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


# ---------------------------- ARC ---------------------------- #
# ARC (AI2 Reasoning Challenge) is a dataset for question answering that requires reasoning.
# It consists of multiple-choice science questions from 3rd to 9th grade exams.
# The dataset is split into two parts: ARC-Easy and ARC-Challenge.
# ARC-Easy contains questions that can be answered correctly by both humans and simple baseline models.
# ARC-Challenge contains questions that are difficult for both humans and current AI systems.
# Similar to MMLU, ARC tasks uses PMI normalization by default but only for the challenge set.
# github: https://github.com/nlp-uoregon/mlmm-evaluation

TASKS_TABLE = []


mlmm_arc_challenge_tasks = [
    LightevalTaskConfig(
        name=f"mlmm_arc_{language.value}_{formulation.name.lower()}:challenge",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=("lighteval",),
        hf_repo="jon-tow/okapi_arc_challenge",
        hf_subset=standardize_tag(language.value),
        hf_revision="823d5d7bfaf8974a3ab52a825b6cf4903b35dbc4",
        evaluation_splits=("test",),
        few_shots_split="train",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
                LogLikelihoodAccMetric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for language in [
        Language.RUSSIAN,
        Language.GERMAN,
        Language.CHINESE,
        Language.FRENCH,
        Language.SPANISH,
        Language.ITALIAN,
        Language.DUTCH,
        Language.VIETNAMESE,
        Language.INDONESIAN,
        Language.ARABIC,
        Language.HUNGARIAN,
        Language.ROMANIAN,
        Language.DANISH,
        Language.SLOVAK,
        Language.UKRAINIAN,
        Language.CATALAN,
        Language.SERBIAN,
        Language.CROATIAN,
        Language.HINDI,
        Language.BENGALI,
        Language.TAMIL,
        Language.NEPALI,
        Language.MALAYALAM,
        Language.MARATHI,
        Language.TELUGU,
        Language.KANNADA,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
