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

import re

import numpy as np

from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.multilingual.utils.adapters_utils import (
    MULTICHOICE_JOIN_VARIANT,
    extract_answers_from_string,
    multichoice_compose,
    multichoice_join,
)
from lighteval.tasks.templates.multichoice import MCQInput
from lighteval.tasks.templates.utils.formatting_utils import PUNCT
from lighteval.tasks.templates.utils.translation_literals import TranslationLiterals
from lighteval.utils.language import Language


m3_exam_answer_prefix_re = re.compile(r"^\([A-Da-d1-5๑๒๓๔๕]\)\s*|^[A-Da-e1-5๑๒๓๔๕][.．।。]\s*")


def get_m3exam_adapter(lang: Language, line: dict) -> MCQInput | None:
    letter_indices = "๑๒๓๔๕" if lang == "th" else LETTER_INDICES
    is_number_based = line["answer_text"].isdigit()
    clean_options = [m3_exam_answer_prefix_re.sub("", c) for c in line["options"]]
    gold_idx = int(line["answer_text"]) - 1 if is_number_based else letter_indices.index(line["answer_text"].upper())

    if not all(len(c) > 0 for c in clean_options) or gold_idx >= len(clean_options):
        return None

    return {
        "choices": clean_options,
        "gold_idx": gold_idx,
        "question": line["question_text"],
        "context": line["background"],
    }


def thai_exams_adapter(line: dict) -> MCQInput | None:
    pos_letters = [letter.lower() for letter in LETTER_INDICES[:5]]

    choices = [line[letter] for letter in pos_letters if letter in line]
    if any(opt.strip() == "" for opt in choices):
        return None

    gold_index = choices.index(line["answer"])
    return {
        "question": line["question"],
        "choices": choices,
        "gold_idx": gold_index,
    }


def alghafa_adapter(line: dict) -> MCQInput | None:
    answer_index = int(line["label"])
    choices_keys = [key for key in line.keys() if key not in ["query", "label", "__few_shots"]]
    choices = [line[key] for key in choices_keys]
    return {
        "question": line["query"],
        "choices": choices,
        "gold_idx": answer_index,
    }


def sciq_adapter(line: dict) -> MCQInput | None:
    # Randomly generate the correct answer index
    gold_idx = np.random.randint(0, 4)
    incorrect_answers = [line["distractor1"], line["distractor2"], line["distractor3"]]
    choices = incorrect_answers[:gold_idx] + [line["correct_answer"]] + incorrect_answers[gold_idx:]
    return {
        "question": line["question"],
        "choices": choices,
        "gold_idx": gold_idx,
        "context": line["support"],
    }


def ceval_adapter(lang: Language, join_variant: MULTICHOICE_JOIN_VARIANT, line: dict) -> MCQInput | None:
    # All ceval tasks ends with ____(。?)
    # Some can follow with with possible answers in format
    # (①|②|③|④)text
    # We must thus remove ___ and extract the answers from the second part of input text

    translation_literals = TranslationLiterals(lang)
    choices = [line["A"], line["B"], line["C"], line["D"]]

    parts = line["question"].rsplit("____", maxsplit=1)
    cleaned_question = parts[0].rstrip(PUNCT).strip()
    possible_answers_part = parts[1].lstrip(PUNCT)
    gold_index = LETTER_INDICES.index(line["answer"])
    # Sometimes there can be choose one of the following from options.
    # In this case we want to extract the answer from the question to comply with CF format.

    answer_prefixes = [answer.replace("和", "").strip() for answer in choices]
    answer_prefixes = set("".join(answer_prefixes))

    # We only attempt to extract answers if the answers are numbers
    maybe_extracted_answers = (
        extract_answers_from_string(possible_answers_part, list(answer_prefixes))
        if answer_prefixes.issubset("①②③④⑤⑥")
        else None
    )
    if maybe_extracted_answers:
        start_index, multi_choices = maybe_extracted_answers
        # Here we don't expect anything to be in front of the answers from second part
        assert start_index == 0, f"Start index is not 0: {start_index}"
        choices = [multichoice_join(mc, join_variant, translation_literals) for mc in multi_choices]
    else:
        # If the second part is not list of answers we put it back
        cleaned_question = f"{cleaned_question} {possible_answers_part}" if possible_answers_part else cleaned_question

    # Lastly make it into question:
    cleaned_question = f"{cleaned_question.strip().rstrip(PUNCT)}{translation_literals.question_mark}"

    # If we still have only the numbers in the answers or we have just single choice we discard this sample
    if set("".join(choices).replace("和", "").strip()).issubset("①②③④⑤⑥") or len(choices) <= 1:
        return None

    return {
        "question": cleaned_question,
        "choices": choices,
        "gold_idx": gold_index,
    }


def agieval_prompt(lang: Language, join_variant: MULTICHOICE_JOIN_VARIANT, line: dict) -> MCQInput | None:
    translation_literals = TranslationLiterals(lang)

    # Remove the question at the start as it's added by template
    context, rest = line["query"].split("问题：", maxsplit=1)

    # Remove the options as we build them ourselves
    question, _ = rest.split(" 选项：", maxsplit=1)
    original_choices = line["choices"]
    cleaned_choices = [m3_exam_answer_prefix_re.sub("", c).strip() for c in original_choices]
    gold_index = line["gold"]

    # Here is the most tricky part. In some subsets (e.g. gaokai-history) the answers can be the chinese digits only.
    # This would break the CF formulation and we thus try to extract the full answers from the question.
    # Example
    # Quesiton: 问题：在中美关系的发展中，台湾问题是一大障碍，在扫除这一障碍的过程中，取得突破性进展的事件包括（　　）①中国恢复联合国席位 ②尼克松总统访华③中美两国正式建交 ④邓小平访问美国。 选项：(A)①② (B)①③ (C)②③ (D)③④ 答案：从A到D, 我们应选择
    # Answers: [ "(A)①②", "(B)①③", "(C)②③", "(D)③④" ]

    answer_prefixes = [answer.replace("和", "").strip() for answer in original_choices]
    answer_prefixes = set("".join(answer_prefixes))

    # We only attempt to extract answers if the answers are chinese numbers
    maybe_extracted_answers = (
        extract_answers_from_string(question.strip().rstrip(PUNCT), list(answer_prefixes))
        if answer_prefixes.issubset("①②③④⑤⑥")
        else None
    )
    if maybe_extracted_answers:
        start_index, multi_choices = maybe_extracted_answers
        question = question[:start_index]
        cleaned_choices = [multichoice_join(mc, join_variant, translation_literals) for mc in multi_choices]

    # Agi-eval is multi-choice but we convert it to single choice
    cleaned_choices, gold_index = multichoice_compose(cleaned_choices, gold_index, join_variant, translation_literals)
    question = question.strip()

    # If the answers are still only contian the chinese numbers or we have just single choice we discard this sample
    if set("".join(cleaned_choices).replace("和", "").strip()).issubset("①②③④⑤⑥") or len(cleaned_choices) <= 1:
        return None

    return {
        "question": question,
        "choices": cleaned_choices,
        "gold_idx": gold_index,
        "context": context,
    }
