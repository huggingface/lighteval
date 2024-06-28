# MMML

import re
from typing import Literal

from community_tasks.multilingual.tasks.utils.translation_literals import (
    ANSWER,
    CONTRADICTION_LABELS,
    ENTAILMENT_LABELS,
    IMPOSSIBLE,
    LANGS,
    NEUTRAL_LABELS,
    NLI_QUESTION,
    QUESTION,
)
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks_prompt_formatting import LETTER_INDICES


# Notes:
# - For the context we can also put something in front (not implemented right now)

# QA-Tasks (multichoice)
MULTI_QA_TEMPLATE = "{context}{question_word}: {question}\n{answer_word}:"


def _get_multi_qa_prompt(lang: LANGS):
    def multi_qa_prompt(task_name: str, question: str, answers: list[str], gold_index, context: str | None = None):
        query = MULTI_QA_TEMPLATE.format(
            question=question,
            context=f"{context}\n" if context else "",
            question_word=QUESTION[lang],
            answer_word=ANSWER[lang],
        )
        return Doc(
            task_name=task_name,
            query=query,
            gold_index=gold_index,
            choices=[f" {c}" for c in answers if c],
            uncoditioned_prefix=f"{ANSWER[lang]}:",
        )

    return multi_qa_prompt


def get_m_arc_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)

    def adapter(line: dict, task_name: str):
        keys = [key for key in line.keys() if key.startswith("option_")]
        letters = [key.split("_")[-1].upper() for key in keys]
        return prompter(
            task_name, line["instruction"], [line[k] for k in keys if line[k]], letters.index(line["answer"])
        )

    return adapter


def get_french_arc_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)

    def adapter(line, task_name):
        return prompter(task_name, line["question"], line["choices"], LETTER_INDICES.index(line["answerKey"]))

    return adapter


def get_french_boolqa_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)

    def adapter(line, task_name):
        return prompter(
            task_name,
            line["question"],
            [ENTAILMENT_LABELS[lang], CONTRADICTION_LABELS[lang]],
            [1, 0].index(line["label"]),
            context=line["context"],
        )

    return adapter


def get_m_exams_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)

    def adapter(line, task_name):
        letters = line["question"]["choices"]["label"]
        texts = line["question"]["choices"]["text"]
        return prompter(
            task_name,
            line["question"]["stem"],
            texts,
            letters.index(line["label"]),
        )

    return adapter


def get_m_belebele_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)
    return lambda line, task_name: prompter(
        task_name,
        line["question"],
        [line[f"mc_answer{i}"] for i in range(1, 5)],
        int(line["correct_answer_num"]) - 1,
        line["flores_passage"],
    )


def get_m_xcsr_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)

    def adapter(line, task_name):
        letters = line["question"]["choices"]["label"]
        texts = line["question"]["choices"]["text"]
        return prompter(task_name, line["question"]["stem"], texts, letters.index(line["answerKey"]))

    return adapter


def get_m_m3exam_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)
    prefix_re = re.compile(r"^\([A-Da-d1-4]\)\s*|^[A-Da-d1-3]\.\s*")

    def adapter(line, task_name):
        is_letter_based = line["answer_text"].isalpha()
        clean_options = [prefix_re.sub("", c) for c in line["options"]]
        gold_idx = (
            LETTER_INDICES.index(line["answer_text"].upper()) if is_letter_based else int(line["answer_text"]) - 1
        )
        return prompter(task_name, line["question_text"], clean_options, gold_idx, context=line["background"])

    return adapter


# QA-Tasks (No multichoice)


QA_TEMPLATE = "{instruction}{context}{question_word}: {question}\n{answer_word}:"


def get_qa_prompt(lang: LANGS):
    def qa_prompt(
        task_name: str, question: str, answer: str, context: str | None = None, instruction: str | None = None
    ):
        query = QA_TEMPLATE.format(
            instruction=f"{instruction}\n" if instruction else "",
            question=question,
            context=f"{context}\n" if context else "",
            question_word=QUESTION[lang],
            answer_word=ANSWER[lang],
        )
        return Doc(
            task_name=task_name, query=query, gold_index=0, choices=[answer], uncoditioned_prefix=f"{ANSWER[lang]}:"
        )

    return qa_prompt


def get_mlqa_prompt(lang: LANGS):
    prompter = get_qa_prompt(lang)
    return lambda line, task_name: prompter(task_name, line["question"], line["answers"]["text"][0], line["context"])


def get_mintaka_prompt(lang: LANGS):
    prompter = get_qa_prompt(lang)
    return lambda line, task_name: prompter(task_name, line["question"], line["answerText"])


def get_french_fquadv2_prompt(lang):
    prompter = get_qa_prompt(lang)
    # Possibly fix to allow multilang
    instruct = "après l'information dans le contexte donné, donne la réponse à la question en citant quelques mots du contexte. Si il est impossible de répondre avec les informations du contexte, répond 'Impossible."

    def adapter(task_name, line):
        answer = line["answers"]["text"][0] if line["answers"]["text"] else IMPOSSIBLE[lang]
        return prompter(task_name, line["question"], answer, context=line["context"], instruction=instruct)

    return adapter


# NLI task
NLI_TEMPLATE = "{premise}, {question_word}? {label}, {hypothesis}"


def get_nli_prompt(lang: LANGS, pos_labels: list[Literal["entailment", "neutral", "contradiction"]]):
    labels = []
    if "entailment" in pos_labels:
        labels.append(ENTAILMENT_LABELS[lang])
    if "neutral" in pos_labels:
        labels.append(NEUTRAL_LABELS[lang])
    if "contradiction" in pos_labels:
        labels.append(CONTRADICTION_LABELS[lang])

    def nli_prompt(task_name: str, premise: str, hypothesis: str, label: int):
        return Doc(
            task_name=task_name,
            query="",
            choices=[
                NLI_TEMPLATE.format(
                    premise=premise,
                    question_word=NLI_QUESTION[lang],
                    label=label,
                    hypothesis=hypothesis,
                )
                for label in labels
            ],
            gold_index=label,
            uncoditioned_prefix="",
        )

    return nli_prompt


def get_xnli_prompt(lang: LANGS):
    prompter = get_nli_prompt(lang, ["entailment", "neutral", "contradiction"])
    return lambda line, task_name: prompter(task_name, line["premise"], line["hypothesis"], int(line["label"]))


def get_paws_x_prompt(lang: LANGS):
    prompter = get_nli_prompt(lang, ["entailment", "contradiction"])
    return lambda line, task_name: prompter(task_name, line["sentence1"], line["sentence2"], int(line["label"]))


# Misc
def preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def m_hellaswag_prompt(line, task_name: str = ""):
    def preprocess(text):
        """Comes from AiHarness"""
        # text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    ctx = f"{line['ctx_a']} {line['ctx_b'].capitalize()} "
    return Doc(
        task_name=task_name,
        query=preprocess(line["activity_label"] + ": " + ctx),
        choices=[" " + preprocess(ending) for ending in line["endings"]],
        gold_index=int(line["label"]) if line["label"] != "" else -1,  # -1 for test
        uncoditioned_prefix="",
    )
