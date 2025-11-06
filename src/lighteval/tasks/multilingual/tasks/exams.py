"""
name:
Exams

dataset:
mhardalov/exams

abstract:
Exams multilingual benchmark.

languages:
albanian, arabic, bulgarian, croatian, french, german, hungarian, italian,
lithuanian, macedonian, polish, portuguese, serbian, spanish, turkish,
vietnamese

tags:
knowledge, multilingual, multiple-choice

paper:
"""

from functools import partial

from langcodes import Language as LangCodeLanguage
from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation, normalize_subset
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language


exams_subjects_by_lang: dict[Language, set[str]] = {
    Language.ARABIC: {"Biology", "Islamic Studies", "Physics", "Science", "Social"},
    Language.BULGARIAN: {"Biology", "Chemistry", "Geography", "History", "Philosophy", "Physics"},
    Language.CROATIAN: {
        "Biology",
        "Chemistry",
        "Ethics",
        "Fine Arts",
        "Geography",
        "Geology",
        "History",
        "Informatics",
        "Philosophy",
        "Physics",
        "Politics",
        "Psychology",
        "Religion",
        "Sociology",
    },
    Language.HUNGARIAN: {
        "Agriculture",
        "Agriculture (Mechanical knowledge)",
        "Biology",
        "Chemistry",
        "Economics",
        "Economics & Marketing",
        "Economics Basics (Business)",
        "Economics Basics (Theoretical)",
        "Forestry",
        "Geography",
        "Landscaping",
        "Physics",
        "Politics",
        "Tourism",
    },
    Language.ITALIAN: {
        "Biology",
        "Chemistry",
        "Ethics",
        "Geography",
        "Geology",
        "History",
        "Informatics",
        "Philosophy",
        "Physics",
        "Politics",
        "Psychology",
        "Sociology",
    },
    Language.SERBIAN: {
        "Biology",
        "Chemistry",
        "Ethics",
        "Geography",
        "Geology",
        "History",
        "Informatics",
        "Philosophy",
        "Physics",
        "Politics",
        "Psychology",
        "Religion",
        "Sociology",
    },
    Language.FRENCH: {"Economics", "Economics & Marketing", "Economics Basics (Theoretical)", "Geography", "Physics"},
    Language.GERMAN: {
        "Chemistry",
        "Economics",
        "Economics & Marketing",
        "Economics Basics (Theoretical)",
        "Geography",
        "Physics",
        "Tourism",
    },
    Language.SPANISH: {"Geography", "Physics"},
    Language.LITHUANIAN: {"Geology", "History"},
    Language.ALBANIAN: {
        "Biology",
        "Business",
        "Chemistry",
        "Fine Arts",
        "History",
        "Philosophy",
        "Physics",
        "Sociology",
    },
    Language.MACEDONIAN: {
        "Biology",
        "Business",
        "Chemistry",
        "Fine Arts",
        "History",
        "Philosophy",
        "Physics",
        "Sociology",
    },
    Language.TURKISH: {
        "Biology",
        "Business",
        "Chemistry",
        "Geography",
        "History",
        "Philosophy",
        "Physics",
        "Sociology",
    },
    Language.POLISH: {"Professional"},
    Language.PORTUGUESE: {"Biology", "Economics", "Geology", "Philosophy"},
    Language.VIETNAMESE: {"Biology", "Chemistry", "Citizenship", "Geography", "History", "Physics"},
}


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"exams_{language.value}_{formulation.name.lower()}:{normalize_subset(subject)}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"]["stem"],
                "choices": line["question"]["choices"]["text"],
                "gold_idx": line["question"]["choices"]["label"].index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        hf_repo="mhardalov/exams",
        hf_subset="multilingual",
        # Weird bug in dataset
        hf_filter=partial(
            lambda language, subject, line: line["answerKey"] != "@"
            and line["info"]["language"] == LangCodeLanguage(standardize_tag(language.value)).language_name()
            and line["info"]["subject"] == subject,
            language,
            subject,
        ),
        evaluation_splits=("test",),
        few_shots_split="train",
        metrics=get_metrics_for_formulation(
            formulation,
            [
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
                LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for language in exams_subjects_by_lang.keys()
    for subject in exams_subjects_by_lang[language]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]
