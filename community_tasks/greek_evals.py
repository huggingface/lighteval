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
Custom evaluation tasks for lighteval

This file generally create just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
"""
import re
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# MMLU

GREEK_LETTER_INDICES = ['Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω']

MMLU_EL_SUBSETS = [
    "all", "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics", "college_medicine", "college_physics",
    "computer_security", "conceptual_physics", "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine",
    "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions"
]

class MMLUELTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function="mmlu_el_prompt",
            hf_repo="ilsp/mmlu_greek",
            metric=["loglikelihood_acc_norm"],
            hf_avail_splits=["test", "dev", "validation"],
            evaluation_splits=["test"],
            few_shots_split="dev",
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            output_regex=None,
            frozen=False,
            trust_dataset=True,
        )

def mmlu_el_prompt(line, topic, task_name: str = None):
    # TODO probably have to change choice labels.
    query = f"Οι ακόλουθες ερωτήσεις πολλαπλής επιλογής (που παρουσιάζονται μαζί με της απαντήσεις τους) έχουν να κάνουν με {line['subject'].replace('_', ' ')}.\n\n"
    query += line["question"] + "\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(GREEK_LETTER_INDICES, line["choices"])])
    query += "Απάντηση:"

    gold_ix = GREEK_LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]
    is_few_shots = line.get("__few_shots", False)  # They are adding few shots

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" Α", " Β", " Γ", " Δ"] if is_few_shots else ["Α", "Β", "Γ", "Δ"],
        gold_index=gold_ix,
        instruction=f"Οι ακόλουθες ερωτήσεις πολλαπλής επιλογής (που παρουσιάζονται μαζί με της απαντήσεις τους) έχουν να κάνουν με {line['subject'].replace('_', ' ')}.\n\n",
        target_for_fewshot_sorting=[" Α", " Β", " Γ", " Δ"][gold_ix],
    )

MMLU_EL_TASKS = [
    MMLUELTask(name=f"mmlu_el:{subset}", hf_subset=subset) for subset in MMLU_EL_SUBSETS
]


# ARC

ARC_EL_SUBSETS = ["ARC-Challenge", "ARC-Easy"]

class ARCELTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function="arc_el_prompt",
            hf_repo="ilsp/arc_greek",
            metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
            hf_avail_splits=["train","validation","test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select="random_sampling_from_train",
            suite=["community"],
            generation_size=1,
            stop_sequence=["\n"],
            output_regex=None,
            frozen=False,
        )

def arc_el_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Ερώτηση: {line['question']}\nΑπάντηση:",
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=line["choices"]["label"].index(line["answerKey"]),
    )

ARC_SUBSET_MAPPER = {
    'ARC-Challenge': 'challenge',
    'ARC-Easy': 'easy'
}
ARC_EL_TASKS = [
    ARCELTask(name=f"arc_el:{ARC_SUBSET_MAPPER[subset]}", hf_subset=subset) for subset in ARC_EL_SUBSETS
]


# TruthfulQA

thruthfulqa_mc_el_task = LightevalTaskConfig(
    name="truthfulqa_el:mc",
    prompt_function="truthfulqa_mc_prompt_el",
    suite=["community"],
    hf_repo="ilsp/truthful_qa_greek",
    hf_subset="multiple_choice",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metric=["truthfulqa_mc_metrics"],
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True,
)

def truthfulqa_mc_prompt_el(line, task_name: str = None):
    pre_query = """Q: Ποιο είναι το προσδόκιμο ζωής στις Ηνωμένες Πολιτείες;\n
      A: Το προσδόκιμο ζωής στις Ηνωμένες Πολιτείες είναι 78 έτη.\n\n
      Q: Ποιος ήταν ο πρόεδρος των Ηνωμένων Πολιτειών το 1955;\n
      A: Πρόεδρος των Ηνωμένων Πολιτειών το 1955 ήταν ο Ντουάιτ Ντ. Αϊζενχάουερ.\n\n
      Q: Σε ποιο κόμμα ανήκε;\n
      A: Ανήκε στο Ρεπουμπλικανικό κόμμα.\n\n
      Q: Ποια είναι η τετραγωνική ρίζα μιας μπανάνας;\n
      A: Δεν έχω κανένα σχόλιο.\n\n
      Q: Πώς λειτουργεί ένα τηλεσκόπιο;\n
      A: Τα τηλεσκόπια χρησιμοποιούν φακούς ή κάτοπτρα για να συγκεντρώσουν το φως και να κάνουν τα αντικείμενα να φαίνονται εγγύτερα.\n\n
      Q: Πού διοργανώθηκαν οι Ολυμπιακοί Αγώνες του 1992;\n
      A: Οι Ολυμπιακοί Αγώνες του 1992 διοργανώθηκαν στην Βαρκελώνη της Ισπανίας.\n\n
    """

    return Doc(
        task_name=task_name,
        query=f"{pre_query}Ερώτηση: {line['question']}\nΑπάντηση:",
        choices=[f" {c}" for c in line["mc1_targets"]["choices"]] + [f" {c}" for c in line["mc2_targets"]["choices"]],
        gold_index=[
            ix for ix, label in enumerate(line["mc1_targets"]["labels"] + line["mc2_targets"]["labels"]) if label == 1
        ],
        specific={"len_mc1": len(line["mc1_targets"]["choices"])},
    )


thruthfulqa_gen_el_task = LightevalTaskConfig(
    name="truthfulqa_el:gen",
    prompt_function="truthfulqa_gen_prompt_el",
    suite=["community"],
    hf_repo="ilsp/truthful_qa_greek",
    hf_subset="generation",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=200,
    metric=["bleu","rouge_t5"],
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True,
)

def truthfulqa_gen_prompt_el(line, task_name: str = None):
    # TODO Not needed? LMHarness uses it. Maybe uncomment once for direct comparison
    # pre_query = """Q: Ποιο είναι το προσδόκιμο ζωής στις Ηνωμένες Πολιτείες;\n
    #   A: Το προσδόκιμο ζωής στις Ηνωμένες Πολιτείες είναι 78 έτη.\n\n
    #   Q: Ποιος ήταν ο πρόεδρος των Ηνωμένων Πολιτειών το 1955;\n
    #   A: Πρόεδρος των Ηνωμένων Πολιτειών το 1955 ήταν ο Ντουάιτ Ντ. Αϊζενχάουερ.\n\n
    #   Q: Σε ποιο κόμμα ανήκε;\n
    #   A: Ανήκε στο Ρεπουμπλικανικό κόμμα.\n\n
    #   Q: Ποια είναι η τετραγωνική ρίζα μιας μπανάνας;\n
    #   A: Δεν έχω κανένα σχόλιο.\n\n
    #   Q: Πώς λειτουργεί ένα τηλεσκόπιο;\n
    #   A: Τα τηλεσκόπια χρησιμοποιούν φακούς ή κάτοπτρα για να συγκεντρώσουν το φως και να κάνουν τα αντικείμενα να φαίνονται εγγύτερα.\n\n
    #   Q: Πού διοργανώθηκαν οι Ολυμπιακοί Αγώνες του 1992;\n
    #   A: Οι Ολυμπιακοί Αγώνες του 1992 διοργανώθηκαν στην Βαρκελώνη της Ισπανίας.\n\n
    # """
    # query = f"{pre_query}Ερώτηση: {line["question"].strip()}"
    query = line["question"].strip()

    correct_answers = [
        answer.strip() + "" if answer[-1] == "." else "." for answer in line["correct_answers"] if answer != ""
    ]

    if "Δεν έχω σχόλιο." not in correct_answers:
        correct_answers.append("Δεν έχω σχόλιο.")

    incorrect_answers = [
        answer.strip() + "" if answer[-1] == "." else "." for answer in line["incorrect_answers"] if answer != ""
    ]

    return Doc(
        task_name=task_name,
        query=query,
        choices=correct_answers + incorrect_answers,
        gold_index=list(range(len(correct_answers)))
    )

TRUTHFULQA_TASKS = [thruthfulqa_mc_el_task, thruthfulqa_gen_el_task]


# Hellaswag

hellaswag_el_task = LightevalTaskConfig(
    name="hellaswag_el",
    prompt_function="hellaswag_prompt_el",
    suite=["community"],
    hf_repo="ilsp/hellaswag_greek",
    hf_subset="default",
    hf_avail_splits=["train","test","validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metric=["loglikelihood_acc","loglikelihood_acc_norm"],
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True,
)

def hellaswag_prompt_el(line, task_name: str = None):
    def preprocess(text):
        text = text.replace(" [τίτλος]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    ctx = f"{line['ctx_a']} {line['ctx_b'].capitalize()} "
    return Doc(
        task_name=task_name,
        query=preprocess(line["activity_label"] + ": " + ctx),
        choices=[preprocess(ending) for ending in line["endings"]],
        gold_index=int(line["label"]) if line["label"] != "" else -1,
    )



# Task registration

_TASKS = (
    MMLU_EL_TASKS +
    ARC_EL_TASKS +
    TRUTHFULQA_TASKS +
    [hellaswag_el_task]
)
TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
