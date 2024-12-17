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


"""
This module contains task configurations and prompt functions for evaluating
LLM models on Serbian datasets.
Each task is defined using the `LightevalTaskConfig` class with its respective
prompt function.
The tasks cover a variety of benchmarks, including: standard task like ARC[E][C],
BoolQ, Hellaswag, OpenBookQA,PIQA, Winogrande and a custom OZ Eval.
MMLU is separated by subject and also all in one.
"""

from enum import Enum
from typing import List, Optional

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


class HFSubsets(Enum):
    """Enum for all available Hugging Face dataset subsets in Serbian evaluation tasks."""

    HF_BASE_REPO = "datatab/serbian-llm-benchmark"
    HF_REVISION = "209c5b5f999cae5c02eef5735eb817ead18ac214"

    # ARC (AI2 Reasoning Challenge)
    ARC_EASY = "arc_easy_serbian"
    ARC_CHALLENGE = "arc_challenge_serbian"
    # Question Answering and Knowledge
    BOOLQ = "boolq_serbian"
    OPENBOOK = "openbookq_serbian"
    # Commonsense Reasoning
    HELLASWAG = "hellaswag_serbian"
    PIQA = "piqa_serbian"
    WINOGRANDE = "winogrande_serbian"
    # Custom/Other Task
    OZ_EVAL = "oz_eval_serbian"
    # MMLU (Miscellaneous)
    MMLU_ANATOMY = "mmlu_anatomija_serbian"
    MMLU_ASTRONOMY = "mmlu_astronomija_serbian"
    MMLU_BUSINESS_ETHICS = "mmlu_poslovna_etika_serbian"
    MMLU_CLINICAL_KNOWLEDGE = "mmlu_kliničko_znanje_serbian"
    MMLU_MISCELLANEOUS = "mmlu_miscellaneous_serbian"
    MMLU_ELECTRONIC_ENGINEERING = "mmlu_electrical_engineering_serbian"
    # MMLU (Business Professional)
    MMLU_MARKETING = "mmlu_marketing_serbian"
    MMLU_MANAGEMENT = "mmlu_management_serbian"
    # MMLU (College Level Tasks)
    MMLU_COLLEGE_BIOLOGY = "mmlu_college_biology_serbian"
    MMLU_COLLEGE_CHEMISTRY = "mmlu_college_chemistry_serbian"
    MMLU_COLLEGE_COMPUTER_SCIENCE = "mmlu_college_computer_science_serbian"
    MMLU_COLLEGE_MATHEMATICS = "mmlu_college_mathematics_serbian"
    MMLU_COLLEGE_MEDICINE = "mmlu_college_medicine_serbian"
    MMLU_COLLEGE_PHYSICS = "mmlu_college_physics_serbian"
    MMLU_COLLEGE_COMPUTER_SECURITY = "mmlu_computer_security_serbian"
    # MMLU (Ethics, Philosophy)
    MMLU_MORAL_DISPUTES = "mmlu_moral_disputes_serbian"
    MMLU_MORAL_SCENARIOS = "mmlu_moral_scenarios_serbian"
    MMLU_PHILOSOPHY = "mmlu_philosophy_serbian"
    MMLU_WORLD_RELIGIONS = "mmlu_world_religions_serbian"
    # MMLU (High School Level Tasks)
    MMLU_HIGH_SCHOOL_BIOLOGY = "mmlu_high_school_biology_serbian"
    MMLU_HIGH_SCHOOL_CHEMISTRY = "mmlu_high_school_chemistry_serbian"
    MMLU_HIGH_SCHOOL_COMPUTER_SCIENCE = "mmlu_high_school_computer_science_serbian"
    MMLU_HIGH_SCHOOL_EURO_HISTORY = "mmlu_high_school_european_history_serbian"
    MMLU_HIGH_SCHOOL_GEOGRAPHY = "mmlu_high_school_geography_serbian"
    MMLU_HIGH_SCHOOL_MATHEMATICS = "mmlu_high_school_mathematics_serbian"
    MMLU_HIGH_SCHOOL_MICROECONOMICS = "mmlu_high_school_microeconomics_serbian"
    MMLU_HIGH_SCHOOL_PHYSICS = "mmlu_high_school_physics_serbian"
    MMLU_HIGH_SCHOOL_PSYCHOLOGY = "mmlu_high_school_psychology_serbian"
    MMLU_HIGH_SCHOOL_STATISTICS = "mmlu_high_school_statistics_serbian"
    MMLU_HIGH_SCHOOL_WORLD_HISTORY = "mmlu_high_school_world_history"
    # MMLU (Math, Logic)
    MMLU_ABSTRACT_ALGEBRA = "mmlu_abstract_algebra_serbian"
    MMLU_ELEMENTARY_MATHEMATICS = "mmlu_osnovna_matematika_serbian"
    MMLU_FORMAL_LOGIC = "mmlu_formalna_logika_serbian"
    MMLU_CONCEPTUAL_PHYSICS = "mmlu_conceptual_physics_serbian"
    MMLU_ECONOMETRICS = "mmlu_econometrics_serbian"
    MMLU_MACHINE_LEARNING = "mmlu_machine_learning_serbian"
    # MMLU (Social Sciences)
    MMLU_GLOBAL_FACT = "mmlu_global_facts_serbian"
    MMLU_LOGICAL_FALLACIES = "mmlu_logicke_zablude_serbian"
    MMLU_SOCIOLOGY = "mmlu_sociology_serbian"
    MMLU_HUMAN_AGING = "mmlu_human_aging_serbian"
    # MMLU (All-inclusive Task Entry)
    MMLU_SERBIAN_ALL = "mmlu_all_serbian"


def prompt_fn_oz_eval_task(line, task_name: str = None):
    """
    Prepares a question and answer set in Serbian from the OZ Eval (Opšte Znanje Evaluacija) dataset
    for use in a LightEval task. This dataset, specifically designed for evaluating general knowledge
    in Serbian, contains questions derived from entrance exams at the University of Belgrade's Faculty
    of Philosophy and Faculty of Organizational Sciences, covering enrollment periods from 2003 to 2024.

    The function accepts a dictionary with a question, five answer choices, and a correct answer
    designation, returning a structured `Doc` object formatted for LightEval's TASKS_TABLE or TASKS_GROUPS.

    Args:
        line (dict): A dictionary with required keys:
            - 'query' (str): The main question string.
            - 'choices' (list of str): A list containing exactly five answer options.
            - 'answer_str' (str): A single character from "A" to "E" representing the correct answer.
        task_name (str, optional): An optional string specifying the evaluation task name.

    Returns:
        Doc: A structured object for LightEval containing:
            - task_name (str): The task name, if provided.
            - query (str): Formatted question with embedded answer choices.
            - choices (list of str): List of option identifiers ["A", "B", "C", "D", "E"].
            - gold_index (int): Index of the correct answer within the 'choices' list.

    Raises:
        ValueError: If the 'choices' list does not contain exactly five items,
                    or if 'answer_str' is not one of ["A", "B", "C", "D", "E"].

    Note:
        This function is part of the LightEval setup, specifically for loading OZ Eval dataset questions
        into the evaluation environment. For consistent evaluation results, run the task with
        `--use_chat_template`. The OZ Eval dataset is available at https://huggingface.co/datasets/DjMel/oz-eval.

    """
    query_template = """Pitanje: {question}\n
    Ponuđeni odgovori:
    A. {choice_a}
    B. {choice_b}
    C. {choice_c}
    D. {choice_d}
    E. {choice_e}

    Krajnji odgovor:"""

    options = line["choices"]

    query = query_template.format(
        question=line["query"],
        choice_a=options[0],
        choice_b=options[1],
        choice_c=options[2],
        choice_d=options[3],
        choice_e=options[4],
    )

    choices = ["A", "B", "C", "D", "E"]
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=choices.index(line["answer_str"]),
    )


def serbian_eval_prompt(line: dict, task_name: Optional[str] = None) -> Doc:
    """
    Creates a prompt for a multiple-choice task in Serbian. This function formats the prompt
    based on the provided query and choices, handling both standard tasks and MMLU-specific
    tasks (if "mmlu" is part of the task name).

    The prompt includes an instruction in Serbian, followed by the query, available choices,
    and finally the correct answer. The function determines how to compute the correct answer
    based on whether the task name contains "mmlu".

    Args:
        line (dict): A dictionary containing the following keys:
            - "query" (str): The question or query to present to the user.
            - "choices" (list of str): A list of possible answer choices.
            - "answer" (int or str): The correct answer, either as an index (for regular tasks)
               or as a string (for MMLU tasks).
        task_name (Optional[str]): The name of the task. If "mmlu" is in the task name, the
            function treats the task as an MMLU task and searches for the correct answer
            by matching the string value of the answer.

    Returns:
        Doc: A `Doc` object containing the formatted prompt, choices, and the correct answer index.
        The `Doc` object includes the following fields:
            - task_name (Optional[str]): The name of the task.
            - query (str): The formatted query prompt in Serbian, including instructions and choices.
            - choices (list of str): The list of available answer choices.
            - gold_index (int): The index of the correct answer.
            - instruction (str): The instruction shown to the user in Serbian.
    """

    question = line["query"]
    choices = line["choices"]
    instruction = "Na osnovu sledećeg pitanja, izaberite tačanu opciju iz ponuđenih odgovora.\n"

    # Build the query and determine the gold_index in a single pass
    query = f"{instruction}Pitanje: {question}\n\nPonuđeni odgovori:\n"

    gold_index = None

    # ARC is <int> base gold index, but MMLU we handle gold index as <str>
    if task_name and "mmlu" in task_name:
        correct_answer = str(line["answer"])
        gold_index = next((i for i, choice in enumerate(choices) if correct_answer in choice), None)
    else:
        gold_index = int(line["answer"])

    # Show all choises
    for i, choice in enumerate(choices):
        query += f"{i}. {choice}\n"

    query += "\n\nKrajnji odgovor:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=instruction,
    )


def boolq_serbian(line, task_name: str = None):
    # remove extra `?`
    question = line["question"][:-1] if line["question"][-2:] == "??" else line["question"]
    return Doc(
        task_name=task_name,
        query=f"Passage: {line['passage']}\nQuestion: {question}\nAnswer:",
        choices=[" Da", " Ne"],
        gold_index=["Da", "Ne"].index(line["answer"]),
    )


def create_task_config(
    task_name: str,
    prompt_function,
    hf_repo: str,
    hf_subset: str,
    metric: List,
    evaluation_splits: List[str] = ["test"],
    suite: List[str] = ["community"],
    hf_avail_splits: List[str] = ["test", "validation"],
    few_shots_split: str = "validation",
    generation_size=5,
) -> LightevalTaskConfig:
    """
    Creates a task configuration using dependency injection for flexible task creation.

    Args:
        task_name: The name of the task.
        prompt_function: The function to generate task prompts.
        hf_repo: Hugging Face repository.
        hf_subset: Subset of the dataset.
        metric: The metric(s) to use for the task.
        evaluation_splits: The evaluation splits to use (default is "test").
        suite: The suite of tasks.
        hf_avail_splits: Available splits (default is "test", "validation").
        few_shots_split: Split used for few-shot examples.

    Returns:
        A `LightevalTaskConfig` object for the task configuration.
    """
    return LightevalTaskConfig(
        name=task_name,
        prompt_function=prompt_function,
        suite=suite,
        hf_repo=hf_repo,
        hf_subset=hf_subset,
        hf_avail_splits=hf_avail_splits,
        evaluation_splits=evaluation_splits,
        few_shots_split=few_shots_split,
        few_shots_select="sequential",
        metric=metric,
        generation_size=generation_size,
        # Since we use trust_dataset, we have to be careful about what is inside the dataset
        # script. We thus lock the revision to ensure that the script doesn't change
        hf_revision=HFSubsets.HF_REVISION.value,
        trust_dataset=True,
        version=0,
    )


# ============================================
# ===== ARC (AI2 Reasoning Challenge)=========
# ============================================

arc_easy = create_task_config(
    task_name="serbian_evals:arc_easy",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.ARC_EASY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

arc_challenge = create_task_config(
    task_name="serbian_evals:arc_challenge",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.ARC_CHALLENGE.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

# ============================================
# ========= Commonsense Reasoning ============
# ============================================

hellaswag = create_task_config(
    task_name="serbian_evals:hellaswag",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.HELLASWAG.value,
    metric=[Metrics.loglikelihood_acc_norm],
)
piqa = create_task_config(
    task_name="serbian_evals:piqa",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.PIQA.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

winogrande = create_task_config(
    task_name="serbian_evals:winogrande",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.WINOGRANDE.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

# ============================================
# =========== Custom/Other Task ==============
# ============================================

oz_eval = create_task_config(
    task_name="serbian_evals:oz_eval",
    prompt_function=prompt_fn_oz_eval_task,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.OZ_EVAL.value,
    metric=[Metrics.loglikelihood_acc],
)

# ============================================
# ========== MMLU (Miscellaneous) ============
# ============================================

mmlu_anatomy = create_task_config(
    task_name="serbian_evals:mmlu_anatomija",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_ANATOMY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_astronomy = create_task_config(
    task_name="serbian_evals:mmlu_astronomija",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_ASTRONOMY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_business_ethics = create_task_config(
    task_name="serbian_evals:mmlu_poslovna_etika",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_BUSINESS_ETHICS.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_clinical_knowledge = create_task_config(
    task_name="serbian_evals:mmlu_kliničko_znanje",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_CLINICAL_KNOWLEDGE.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_miscellaneous = create_task_config(
    task_name="serbian_evals:mmlu_razno",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_MISCELLANEOUS.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_electrical_engineering = create_task_config(
    task_name="serbian_evals:mmlu_elektrotehnika",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_ELECTRONIC_ENGINEERING.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

# ============================================
# ====== MMLU (All-inclusive Task Entry) =====
# ============================================

mmlu_all = create_task_config(
    task_name="serbian_evals:mmlu",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_SERBIAN_ALL.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

# ============================================
# ======= MMLU (Business Professional) =======
# ============================================

mmlu_marketing = create_task_config(
    task_name="serbian_evals:mmlu_marketing",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_MARKETING.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_management = create_task_config(
    task_name="serbian_evals:mmlu_manadzment",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_MANAGEMENT.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

# ============================================
# ======== MMLU (College Level Tasks) ========
# ============================================

mmlu_college_biology = create_task_config(
    task_name="serbian_evals:mmlu_fakultet_biologija",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_COLLEGE_BIOLOGY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_college_chemistry = create_task_config(
    task_name="serbian_evals:mmlu_fakultet_hemija",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_COLLEGE_CHEMISTRY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_college_computer_science = create_task_config(
    task_name="serbian_evals:mmlu_fakultet_racunari",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_COLLEGE_COMPUTER_SCIENCE.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_college_mathematics = create_task_config(
    task_name="serbian_evals:mmlu_fakultet_matematika",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_COLLEGE_MATHEMATICS.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_college_medicine = create_task_config(
    task_name="serbian_evals:mmlu_fakultet_medicina",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_COLLEGE_MEDICINE.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_college_physics = create_task_config(
    task_name="serbian_evals:mmlu_fakultet_fizika",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_COLLEGE_PHYSICS.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_computer_security = create_task_config(
    task_name="serbian_evals:mmlu_sigurnost_racunara",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_COLLEGE_COMPUTER_SECURITY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

# ============================================
# ======== MMLU (Ethics, Philosophy) =========
# ============================================

mmlu_moral_disputes = create_task_config(
    task_name="serbian_evals:mmlu_moralni_sporovi",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_MORAL_DISPUTES.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_moral_scenarios = create_task_config(
    task_name="serbian_evals:mmlu_moralne_dileme",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_MORAL_SCENARIOS.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_philosophy = create_task_config(
    task_name="serbian_evals:mmlu_filozofija",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_PHILOSOPHY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_world_religions = create_task_config(
    task_name="serbian_evals:mmlu_svetska_religija",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_WORLD_RELIGIONS.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

# ============================================
# ====== MMLU (High School Level Tasks) ======
# ============================================

mmlu_high_school_biology = create_task_config(
    task_name="serbian_evals:mmlu_srednja_skola_biologija",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_HIGH_SCHOOL_BIOLOGY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_high_school_chemistry = create_task_config(
    task_name="serbian_evals:mmlu_srednja_skola_hemija",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_HIGH_SCHOOL_CHEMISTRY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_high_school_computer_science = create_task_config(
    task_name="serbian_evals:mmlu_srednja_skola_racunari",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_HIGH_SCHOOL_COMPUTER_SCIENCE.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_high_school_european_history = create_task_config(
    task_name="serbian_evals:mmlu_srednja_skola_istorija_evrope",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_HIGH_SCHOOL_EURO_HISTORY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_high_school_geography = create_task_config(
    task_name="serbian_evals:mmlu_srednja_skola_geografija",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_HIGH_SCHOOL_GEOGRAPHY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_high_school_mathematics = create_task_config(
    task_name="serbian_evals:mmlu_srednja_skola_matematika",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_HIGH_SCHOOL_MATHEMATICS.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_high_school_microeconomics = create_task_config(
    task_name="serbian_evals:mmlu_srednja_skola_mikroekonomija",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_HIGH_SCHOOL_MICROECONOMICS.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_high_school_physics = create_task_config(
    task_name="serbian_evals:mmlu_srednja_skola_fizika",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_HIGH_SCHOOL_PHYSICS.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_high_school_psychology = create_task_config(
    task_name="serbian_evals:mmlu_srednja_skola_psihologija",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_HIGH_SCHOOL_PSYCHOLOGY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_high_school_statistics = create_task_config(
    task_name="serbian_evals:mmlu_srednja_skola_statistika",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_HIGH_SCHOOL_STATISTICS.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_high_school_world_history = create_task_config(
    task_name="serbian_evals:mmlu_srednja_skola_svetska_istorija",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_HIGH_SCHOOL_WORLD_HISTORY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

# ============================================
# ============ MMLU (Math, Logic) ============
# ============================================

mmlu_abstract_algebra = create_task_config(
    task_name="serbian_evals:mmlu_abstract_algebra",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_ABSTRACT_ALGEBRA.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_elementary_mathematics = create_task_config(
    task_name="serbian_evals:mmlu_osnovna_matematika",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_ELEMENTARY_MATHEMATICS.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_formal_logic = create_task_config(
    task_name="serbian_evals:mmlu_formalna_logika",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_FORMAL_LOGIC.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_conceptual_physics = create_task_config(
    task_name="serbian_evals:mmlu_konceptualna_fizika",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_CONCEPTUAL_PHYSICS.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_econometrics = create_task_config(
    task_name="serbian_evals:mmlu_metrika_ekonomije",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_ECONOMETRICS.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_machine_learning = create_task_config(
    task_name="serbian_evals:mmlu_masinsko_ucenje",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_MACHINE_LEARNING.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

# ============================================
# ========== MMLU (Social Sciences) ==========
# ============================================

mmlu_global_facts = create_task_config(
    task_name="serbian_evals:mmlu_globalne_cinjenice",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_GLOBAL_FACT.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_logical_fallacies = create_task_config(
    task_name="serbian_evals:mmlu_logicke_zablude",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_LOGICAL_FALLACIES.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_sociology = create_task_config(
    task_name="serbian_evals:mmlu_sociologija",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_SOCIOLOGY.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

mmlu_human_aging = create_task_config(
    task_name="serbian_evals:mmlu_human_aging",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.MMLU_HUMAN_AGING.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

# ============================================
# ===== Question Answering and Knowledge =====
# ============================================

boolq = create_task_config(
    task_name="serbian_evals:boolq",
    prompt_function=boolq_serbian,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.BOOLQ.value,
    metric=[Metrics.loglikelihood_acc_norm],
)

openbook_qa = create_task_config(
    task_name="serbian_evals:openbook",
    prompt_function=serbian_eval_prompt,
    hf_repo=HFSubsets.HF_BASE_REPO.value,
    hf_subset=HFSubsets.OPENBOOK.value,
    metric=[Metrics.loglikelihood_acc_norm],
)


TASKS_TABLE = [
    arc_easy,
    arc_challenge,
    boolq,
    hellaswag,
    openbook_qa,
    piqa,
    oz_eval,
    winogrande,
    mmlu_abstract_algebra,
    mmlu_anatomy,
    mmlu_astronomy,
    mmlu_business_ethics,
    mmlu_clinical_knowledge,
    mmlu_college_biology,
    mmlu_college_chemistry,
    mmlu_college_computer_science,
    mmlu_college_mathematics,
    mmlu_college_medicine,
    mmlu_college_physics,
    mmlu_computer_security,
    mmlu_conceptual_physics,
    mmlu_econometrics,
    mmlu_electrical_engineering,
    mmlu_elementary_mathematics,
    mmlu_formal_logic,
    mmlu_global_facts,
    mmlu_high_school_biology,
    mmlu_high_school_chemistry,
    mmlu_high_school_computer_science,
    mmlu_high_school_european_history,
    mmlu_high_school_geography,
    mmlu_high_school_mathematics,
    mmlu_high_school_microeconomics,
    mmlu_high_school_physics,
    mmlu_high_school_psychology,
    mmlu_high_school_statistics,
    mmlu_high_school_world_history,
    mmlu_human_aging,
    mmlu_logical_fallacies,
    mmlu_marketing,
    mmlu_machine_learning,
    mmlu_management,
    mmlu_moral_disputes,
    mmlu_miscellaneous,
    mmlu_moral_scenarios,
    mmlu_sociology,
    mmlu_philosophy,
    mmlu_world_religions,
    mmlu_all,
]
