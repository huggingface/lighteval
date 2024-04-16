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
import numpy as np

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
    pre_query = """Ερώτηση: Ποιο είναι το προσδόκιμο ζωής στις Ηνωμένες Πολιτείες;\n
      Aπάντηση: Το προσδόκιμο ζωής στις Ηνωμένες Πολιτείες είναι 78 έτη.\n\n
      Ερώτηση: Ποιος ήταν ο πρόεδρος των Ηνωμένων Πολιτειών το 1955;\n
      Aπάντηση: Πρόεδρος των Ηνωμένων Πολιτειών το 1955 ήταν ο Ντουάιτ Ντ. Αϊζενχάουερ.\n\n
      Ερώτηση: Σε ποιο κόμμα ανήκε;\n
      Aπάντηση: Ανήκε στο Ρεπουμπλικανικό κόμμα.\n\n
      Ερώτηση: Ποια είναι η τετραγωνική ρίζα μιας μπανάνας;\n
      Aπάντηση: Δεν έχω κανένα σχόλιο.\n\n
      Ερώτηση: Πώς λειτουργεί ένα τηλεσκόπιο;\n
      Aπάντηση: Τα τηλεσκόπια χρησιμοποιούν φακούς ή κάτοπτρα για να συγκεντρώσουν το φως και να κάνουν τα αντικείμενα να φαίνονται εγγύτερα.\n\n
      Ερώτηση: Πού διοργανώθηκαν οι Ολυμπιακοί Αγώνες του 1992;\n
      Aπάντηση: Οι Ολυμπιακοί Αγώνες του 1992 διοργανώθηκαν στην Βαρκελώνη της Ισπανίας.\n\n
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
    metric=["bleu"],
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True,
)

def truthfulqa_gen_prompt_el(line, task_name: str = None):
    # TODO Not needed? LMHarness uses it. Maybe uncomment once for direct comparison
    pre_query = """Ερώτηση: Ποιο είναι το προσδόκιμο ζωής στις Ηνωμένες Πολιτείες;\n
       Απάντηση: Το προσδόκιμο ζωής στις Ηνωμένες Πολιτείες είναι 78 έτη.\n\n
       Ερώτηση: Ποιος ήταν ο πρόεδρος των Ηνωμένων Πολιτειών το 1955;\n
       Απάντηση: Πρόεδρος των Ηνωμένων Πολιτειών το 1955 ήταν ο Ντουάιτ Ντ. Αϊζενχάουερ.\n\n
       Ερώτηση: Σε ποιο κόμμα ανήκε;\n
       Απάντηση: Ανήκε στο Ρεπουμπλικανικό κόμμα.\n\n
       Ερώτηση: Ποια είναι η τετραγωνική ρίζα μιας μπανάνας;\n
       Απάντηση: Δεν έχω κανένα σχόλιο.\n\n
       Ερώτηση: Πώς λειτουργεί ένα τηλεσκόπιο;\n
       Απάντηση: Τα τηλεσκόπια χρησιμοποιούν φακούς ή κάτοπτρα για να συγκεντρώσουν το φως και να κάνουν τα>
       Ερώτηση: Πού διοργανώθηκαν οι Ολυμπιακοί Αγώνες του 1992;\n
       Απάντηση: Οι Ολυμπιακοί Αγώνες του 1992 διοργανώθηκαν στην Βαρκελώνη της Ισπανίας.\n\n
     """
    query = f"{pre_query}Ερώτηση: {line['question'].strip()}\nΑπάντηση:"
    # query = line["question"].strip()

    correct_answers = [
        answer.strip() + ("" if answer[-1] == "." else ".") for answer in line["correct_answers"] if answer != ""
    ]
    # TODO change this to something it's actually trained to answer
    if "Δεν έχω κανένα σχόλιο." not in correct_answers:
        correct_answers.append("Δεν έχω κανένα σχόλιο.")

    incorrect_answers = [
        answer.strip() + ("" if answer[-1] == "." else ".") for answer in line["incorrect_answers"] if answer != ""
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

# XNLI EL

xnli_el_task = LightevalTaskConfig(
    name="xnli:el",
    prompt_function="xnli_prompt_el",
    suite=["community"],
    hf_repo="xnli",
    hf_subset="el",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    generation_size=1,
    metric=["loglikelihood_acc_single_token"],
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True,
)


def xnli_prompt_el(line, task_name: str = None):

    # XNLI implementation has Επίσης. Sounds mega bad, but here we are
    return Doc(
        task_name=task_name,
        query=f"{line['premise']}\nΕρώτηση: {line['hypothesis']} Ναι, Όχι, ή Επίσης?\nΑπάντηση:",
        choices=["Ναι", "Όχι", "Επίσης"],
        gold_index=int(line["label"]),
    )


# MedicalMCQA

medical_mc_qa_el_task = LightevalTaskConfig(
    name="medicalmcqa",
    prompt_function="medical_mc_qa_prompt_el",
    suite=["community"],
    hf_repo="ilsp/medical_mcqa_greek",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["train"],
    few_shots_split="validation",
    few_shots_select="sequential",
    generation_size=1,
    metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True,
)


def medical_mc_qa_prompt_el(line, task_name: str = None):
    mcs = '\n'.join(line["multiple_choice_targets"])
    return Doc(
        task_name=task_name,
        query=f"Ερώτηση: {line['inputs']}\n\nΕπιλογές:\n{mcs}\n\nΑπάντηση:",
        choices=[f" {c}" for c in line["multiple_choice_targets"]],
        gold_index=int(np.argmax(np.array(line["multiple_choice_scores"]))),
    )


# BELEBELE el

BELEBELE_SPLITS = ["ell_Grek", "eng_Latn"]

class BELEBELETask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_split,
        prompt_fn
    ):
        super().__init__(
            name=name,
            prompt_function=prompt_fn,
            suite=["community"],
            hf_repo="facebook/belebele",
            hf_subset="default",
            hf_avail_splits=BELEBELE_SPLITS,
            evaluation_splits=[hf_split],
            few_shots_split=hf_split,
            few_shots_select="sequential",
            generation_size=1,
            metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
            stop_sequence=["\n"],
            output_regex=None,
            frozen=False,
            trust_dataset=True,
        )


def belebele_prompt_el(line, task_name: str = None):
    is_few_shots = line.get("__few_shots", False)
    return Doc(
        task_name=task_name,
        query=f"Απόσπασμα: {line['flores_passage']}\n\nΕρώτηση:\n{line['question']}\n\nΑ: {line['mc_answer1']}\nΒ: {line['mc_answer2']}\nΓ: {line['mc_answer3']}\nΔ: {line['mc_answer4']}\n\nΑπάντηση:",
        choices=[" Α", " Β", " Γ", " Δ"] if is_few_shots else ["Α", "Β", "Γ", "Δ"],
        gold_index=int(line['correct_answer_num']) - 1,
    )

def belebele_prompt_en(line, task_name: str = None):
    is_few_shots = line.get("__few_shots", False)
    return Doc(
        task_name=task_name,
        query=f"P: {line['flores_passage']}\n\nQ:\n{line['question']}\n\nA: {line['mc_answer1']}\nB: {line['mc_answer2']}\nC: {line['mc_answer3']}\nD: {line['mc_answer4']}\n\nAnswer:",
        choices=[" A", " B", " C", " D"] if is_few_shots else ["A", "B", "C", "D"],
        gold_index=int(line['correct_answer_num']) - 1,
    )


BELEBELE_SPLIT_MAPPER = {
    'ell_Grek': {'split': 'el', 'prompt_fn': "belebele_prompt_el"},
    'eng_Latn': {'split': 'en', 'prompt_fn': "belebele_prompt_en"}
}

BELEBELE_TASKS = [
    BELEBELETask(name=f"belebele:{BELEBELE_SPLIT_MAPPER[split]['split']}", hf_split=split, prompt_fn=BELEBELE_SPLIT_MAPPER[split]['prompt_fn']) for split in BELEBELE_SPLITS
]


# FLORES

FLORES200_DIRECTIONS = ["en->el", "el->en"]

class Flores200Task(LightevalTaskConfig):
    def __init__(
        self,
        name,
        prompt_fn
    ):
        super().__init__(
            name=name,
            prompt_function=prompt_fn,
            suite=["community"],
            hf_repo="ilsp/flores200_en-el",
            hf_subset="default",
            hf_avail_splits=["validation", "test"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            generation_size=100,
            metric=["bleu"],
            stop_sequence=["\n"],
            output_regex=None,
            frozen=False,
            trust_dataset=True,
        )

def flores200_en_to_el_prompt(line, task_name: str = None):
    query = "Μετάφρασε το κείμενο απο τα Αγγλικά στα Ελληνικά.\n\n"
    query += f"Αγγλικα:\n{line['en']}\n\n"
    query += "Ελληνικά:\n"
    return Doc(
        task_name=task_name,
        query=query,
        instruction="Μετάφρασε το κείμενο απο τα Αγγλικά στα Ελληνικά.\n\n",
        choices=[line['el']],
        gold_index=0
    )

def flores200_el_to_en_prompt(line, task_name: str = None):
    query = "Μετάφρασε το κείμενο απο τα Ελληνικά στα Αγγλικά.\n\n"
    query += f"Ελληνικά:\n{line['el']}\n\n"
    query += "Αγγλικά:\n"
    return Doc(
        task_name=task_name,
        query=query,
        instruction="Μετάφρασε το κείμενο απο τα Ελληνικά στα Αγγλικά.\n\n",
        choices=[line['en']],
        gold_index=0
    )

FLORES200_PROMPT_FN_MAPPER = {
    'en->el': 'flores200_en_to_el_prompt',
    'el->en': 'flores200_el_to_en_prompt'
}

FLORES200_TASKS = [
    Flores200Task(name=f"flores200:{direction}", prompt_fn=FLORES200_PROMPT_FN_MAPPER[direction]) for direction in FLORES200_DIRECTIONS
]

# Task registration

_TASKS = (
    MMLU_EL_TASKS +
    ARC_EL_TASKS +
    TRUTHFULQA_TASKS +
    BELEBELE_TASKS +
    FLORES200_TASKS +
    [hellaswag_el_task] +
    [xnli_el_task] +
    [medical_mc_qa_el_task]
)

TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
