from typing import Literal, get_args

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.templates.multichoice_config import DatasetConfig, MCQInput, MultiChoiceTaskConfig, SplitConfig
from lighteval.utils.language import Language


MMLU_SUBSET = Literal[
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


class MMLU(MultiChoiceTaskConfig):
    def __init__(
        self, language: Language, task: MMLU_SUBSET, metric=[Metrics.loglikelihood_acc], instruction: str | None = None
    ):
        super().__init__(
            name=f"meta_mmlu-{language}:{task}",
            adapter=lambda line: MCQInput(
                instruction=instruction,
                question=line["input_question"],
                choices=[v for _, v in sorted(line["input_choice_list"].items())],
                gold_idxs=LETTER_INDICES.index(line["input_correct_responses"][0]),
            ),
            dataset_config=DatasetConfig(
                hf_repo="meta-llama/Meta-Llama-3.1-8B-Instruct-evals",
                hf_subset=f"Meta-Llama-3.1-8B-Instruct-evals__multilingual_mmlu_{language}__details",
                filter=lambda line: line["subtask_name"] == f"mmlu_{language}_chat.{task}",
            ),
            split_config=SplitConfig(
                evaluation_split=("latest",),
            ),
            language=language,
            metrics=metric,
        )


mmlu_no_instruct_tasks = [MMLU(language=Language.english, task=task) for task in get_args(MMLU_SUBSET)]
mmlu_instruct_tasks = [
    MMLU(
        language=Language.english,
        task=task,
        instruction="The following are the multiple choice questions (with answers):",
    )
    for task in get_args(MMLU_SUBSET)
]

mmlu_olmes_instruct_tasks = [
    MMLU(
        language=Language.english,
        task=task,
        instruction=f"The following are multiple choice questions (with answers) about {task}.",
    )
    for task in get_args(MMLU_SUBSET)
]
