# MMML

from functools import partial, reduce
import os
import re
from typing import Any, Literal, Optional, Callable

from lighteval.utils import as_list

from ..utils.translation_literals import (
    ANSWER,
    CAUSE_LABELS,
    COLON,
    COMMA,
    CONTRADICTION_LABELS,
    CORRECT_LABELS,
    EFFECT_LABELS,
    ENTAILMENT_LABELS,
    INCORRECT_LABELS,
    OPTIONS,
    LANGS,
    NEUTRAL_LABELS,
    NLI_QUESTION,
    NO_LABELS,
    QUESTION,
    QUESTION_MARK,
    SENTENCE_SPACE,
    WORD_SPACE,
    YES_LABELS,
    FULL_STOP,
    AND,
    OR
)
from lighteval.tasks.doc import Doc
from lighteval.tasks.tasks_prompt_formatting import LETTER_INDICES

PUNCT = "-.!?،؟‽,。，？؟،। "
MULTICHOICE_JOIN_VARIANT = Literal["AND", "OR", "NEW_LINE", "COMMA"]


def decapitalize(word: str):
    if len(word) == 0:
        return word
    return word[0].lower() + word[1:]

def capitalize(word: str):
    if len(word) == 0:
        return word
    return word[0].upper() + word[1:]


# Notes:
# - For the context we can also put something in front (not implemented right now)

# For thai we use space in front of completions no matter the fact they might not be used, this is to ensure correct tokeniztion


def fix_ending_punct(ctx: str, lang: LANGS):
    ctx = ctx.strip()
    if len(ctx) == 0:
        return ctx
    if ctx.endswith("?"):
        ctx = ctx[:-1] + QUESTION_MARK[lang]
    elif ctx.endswith("."):
        ctx = ctx[:-1] + FULL_STOP[lang]
    elif ctx.endswith(","):
        ctx = ctx[:-1] + COMMA[lang]
    elif ctx.endswith(":"):
        ctx = ctx[:-1] + COLON[lang]
    return ctx

def is_ended_sentence(text: str, lang: LANGS):
    return text.strip().endswith(f"{QUESTION_MARK[lang]}{FULL_STOP[lang]}{COLON[lang]}")

def should_follow_sentence_space(prefix: str, lang: LANGS):
    return prefix.strip().endswith(f"{QUESTION_MARK[lang]}{FULL_STOP[lang]}{COLON[lang]}{COMMA[lang]}")

def fix_capitalization(prefix: str, text: str, lang: LANGS):
    # TODO: Prob cache this
    return capitalize(text) if is_ended_sentence(prefix, lang) else decapitalize(text)


# QA-Tasks (multichoice no helpers)
MULTI_QA_SIMPLE_TEMPLATE = "{context}{question}"
def _get_multi_qa_simple_prompt(lang: LANGS):
    def multi_qa_prompt(
        task_name: str,
        question: str,
        answers: list[str],
        gold_index,
        context: str | None = None,
    ):
        context = capitalize(fix_ending_punct(context, lang)) if context else ""
        question = fix_capitalization(context, fix_ending_punct(question, lang), lang)
        answers = [fix_capitalization(context, fix_ending_punct(answer, lang), lang) for answer in answers]
        query = MULTI_QA_SIMPLE_TEMPLATE.format(
            context=f"{context}\n" if context else "",
            question=question,
        )
        return Doc(
            task_name=task_name,
            query=query,
            gold_index=gold_index,
            choices=[f"{SENTENCE_SPACE[lang]}{c}" for c in answers if c],
            uncoditioned_prefix=f"{ANSWER[lang]}{COLON[lang]}",
        )

    return multi_qa_prompt


answer_prefix_re = re.compile(rf"^\([A-Da-d1-5๑๒๓๔๕]\)\s*|^[A-Da-e1-5๑๒๓๔๕][.．।。]\s*")
def get_m_m3exam_prompt(lang: LANGS):
    prompter = _get_multi_qa_simple_prompt(lang)
    # TODO: Would be nice to have general solution for the letters
    letter_indices = "๑๒๓๔๕" if lang == "th" else LETTER_INDICES

    def adapter(line, task_name):
        is_number_based = line["answer_text"].isdigit()
        clean_options = [answer_prefix_re.sub("", c) for c in line["options"]]
        gold_idx = (
            int(line["answer_text"]) - 1
            if is_number_based
            else letter_indices.index(line["answer_text"].upper())
        )
        
        if not all(len(c) > 0 for c in clean_options) or gold_idx >= len(clean_options):
            return None

        return prompter(
            task_name,
            line["question_text"],
            clean_options,
            gold_idx,
            context=line["background"],
        )

    return adapter

# QA-Tasks (multichoice)
MULTI_QA_TEMPLATE = "{context}{question_word}{colon}{sentence_space}{question}\n{options}{answer_word}{colon}"


def _get_multi_qa_prompt(lang: LANGS):
    def build_options(answers: list[str]):
        options = "\n".join([f"{OPTIONS[lang]}{COLON[lang]}"] + [f"{LETTER_INDICES[i]}. {c}" for i, c in enumerate(answers)])
        return f"{options}\n"
    
    def multi_qa_prompt(
        task_name: str,
        question: str,
        answers: list[str],
        gold_index,
        context: str | None = None,
        show_options: bool = False,
    ):
        context = capitalize(fix_ending_punct(context, lang)) if context else ""
        question = fix_capitalization(context, fix_ending_punct(question, lang), lang)
        answers = [capitalize(fix_ending_punct(answer, lang)) for answer in answers]
        query = MULTI_QA_TEMPLATE.format(
            question=question,
            context=f"{context}\n" if context else "",
            question_word=QUESTION[lang],
            answer_word=ANSWER[lang],
            colon=COLON[lang],
            sentence_space=SENTENCE_SPACE[lang],
            options=build_options(answers) if show_options else "",
        )
        return Doc(
            task_name=task_name,
            query=query,
            gold_index=gold_index,
            choices=[f"{SENTENCE_SPACE[lang]}{c}" for c in answers if c],
            uncoditioned_prefix=f"{ANSWER[lang]}{COLON[lang]}",
        )

    return multi_qa_prompt


# TODO: Uggly
def get_mmlu_prompt(lang: LANGS, is_number_choice: bool = False, zero_based=True):
    prompter = _get_multi_qa_prompt(lang)

    def adapter(line, task_name):
        gold_index = (
            LETTER_INDICES.index(line["answer"])
            if not is_number_choice
            else int(line["answer"])
        )
        if not zero_based:
            gold_index -= 1
        return prompter(task_name, line["question"], line["choices"], gold_index)

    return adapter


def get_c3_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)
    return lambda line, task_name: prompter(
        task_name,
        line["question"],
        line["choice"],
        line["choice"].index(line["answer"]),
        context=" ".join(line["context"]),
    )


def get_arc_prompt(lang: LANGS, nested_choices=False):
    prompter = _get_multi_qa_prompt(lang)

    def adapter(line, task_name):
        choices = line["choices"]["text"] if nested_choices else line["choices"]
        is_number_choice = line["answerKey"].isdigit()
        gold_index = (
            LETTER_INDICES.index(line["answerKey"])
            if not is_number_choice
            else int(line["answerKey"]) - 1
        )
        return prompter(task_name, line["question"], choices, gold_index)

    return adapter


def get_cmllu_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)
    return lambda line, task_name: prompter(
        task_name,
        line["Question"],
        [line["A"], line["B"], line["C"], line["D"]],
        LETTER_INDICES.index(line["Answer"]),
    )


def get_thai_exams_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)
    pos_letters = [l.lower() for l in LETTER_INDICES[:5]]

    def adapter(line, task_name):
        letters = [letter for letter in pos_letters if letter in line]
        options = [str(line[letter]) for letter in letters]
        gold_index = letters.index(line["answer"])
        return prompter(
            task_name,
            line["question"],
            options,
            gold_index,
            show_options=True,
        )

    return adapter


def get_ar_mmlu_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)
    return lambda line, task_name: prompter(
        task_name,
        line["question"],
        [line["A"], line["B"], line["C"], line["D"]],
        LETTER_INDICES.index(line["answer"]),
    )
    
def get_meta_mmlu_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)
    return lambda line, task_name: prompter(
        task_name,
        line["input_question"],
        [v for _, v in sorted(line["input_choice_list"].items())],
        LETTER_INDICES.index(line["input_correct_responses"][0]),
    )
    
def get_arabic_mmlu_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)
    def adapter(line, task_name):
        question = line['Question']
        if line["Context"]:
            question = f"{question}\n{line['Context']}"

        gold_index = LETTER_INDICES.index(line["Answer Key"])
        options = [line[f"Option {i}"] for i in range(1, 6)]
        options = [o for o in options if o]
        return prompter(
            task_name,
            question,
            options,
            gold_index,
        )
    return adapter

def get_ceval_prompt(lang: LANGS, show_options: bool = False, join_variant: MULTICHOICE_JOIN_VARIANT="AND"):
    prompter = _get_multi_qa_prompt(lang)
    # All ceval tasks ends with ____(。?)
    # Some can follow this fill space with possible answers in format
    # (①|②|③|④)text
    # We must thus remove ___ and extract the answers from the question
    

    multichoice_joiner = partial(multichoice_join, lang=lang, variant=join_variant)

    def adapter(line, task_name):
        answers = [line["A"], line["B"], line["C"], line["D"]]
        parts = line['question'].rsplit('____', maxsplit=1)
        cleaned_question = parts[0].rstrip(PUNCT).strip()
        possible_answers_part = parts[1].lstrip(PUNCT)
        gold_index = LETTER_INDICES.index(line["answer"])
        # Sometimes there can be choose one of the following from options.
        # In this case we want to extract the answer from the question to comply with CF format.

        maybe_extracted_answers = _extract_answers_from_string(possible_answers_part, answers)
        if maybe_extracted_answers:
            start_index, multi_choices = maybe_extracted_answers
            # Here we don't expect anything to be before the answers from second part
            assert start_index == 0, f"Start index is not 0: {start_index}"
            answers = [multichoice_joiner(mc) for mc in multi_choices]
        else:
            # If the second part is not list of answers we return it back
            cleaned_question = f"{cleaned_question} {possible_answers_part}" if possible_answers_part else cleaned_question
            
        # Lastly make it into question:
        cleaned_question = f"{cleaned_question.rstrip(PUNCT).strip()}？"
        
        # If still have weird numbers only answers we discrd this sample or we have just single option
        if set("".join(answers).replace("和", "").strip()).issubset("①②③④⑤⑥") or len(answers) <= 1:
            return None

        return prompter(task_name, cleaned_question, answers, gold_index, show_options=show_options)

    return adapter


def get_alghafa_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)

    def adapter(line, task_name):
        answer_index = int(line["label"])
        # Dynamically determining the choices by excluding '__few_shots', 'query' and 'label'
        choices_keys = [
            key for key in line.keys() if key not in ["query", "label", "__few_shots"]
        ]
        choices = [line[key] for key in choices_keys]
        return prompter(task_name, line["query"], choices, answer_index)

    return adapter


def get_m_exams_prompt(lang: LANGS, show_options: bool = False):
    prompter = _get_multi_qa_prompt(lang)

    def adapter(line, task_name):
        letters = line["question"]["choices"]["label"]
        texts = line["question"]["choices"]["text"]
        return prompter(
            task_name,
            line["question"]["stem"],
            texts,
            letters.index(line["answerKey"]),
            show_options=show_options
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
        return prompter(
            task_name, line["question"]["stem"], texts, letters.index(line["answerKey"])
        )

    return adapter


def get_agieval_prompt(lang: Literal["zh"], show_options: bool = False, join_variant: MULTICHOICE_JOIN_VARIANT = "AND"):
    # Kinda meh as some questions require knowledge of possible answers
    prompter = _get_multi_qa_prompt(lang)
    multichoice_joiner = partial(multichoice_join, lang=lang, variant=join_variant)
    multichoice_composer = partial(multichoice_compose, lang=lang, variant=join_variant)

    def adapter(line, task_name):
        # Remove the question at the start to get consistency
        # Ensure there is exactly one '问题：' in the query
        context, rest = line["query"].split("问题：", maxsplit=1)
        question, _ = rest.split(" 选项：", maxsplit=1)
        original_choices = line["choices"]
        cleaned_choices = [answer_prefix_re.sub("", c).strip() for c in original_choices]
        gold_index = line["gold"]
        
        # Sometimes there can be choose one of the following from options.
        # In this case we want to extract the answer from the question to comply with CF format.
        maybe_extracted_answers = _extract_answers_from_string(question, cleaned_choices)
        if maybe_extracted_answers:
            start_index, multi_choices = maybe_extracted_answers
            question = question[:start_index]
            cleaned_choices = [multichoice_joiner(mc) for mc in multi_choices]
        
        cleaned_choices, gold_index = multichoice_composer(cleaned_choices, gold_index)
        question = question.strip()

        # If still have weird numbers only answers we discrd this sample
        if set("".join(cleaned_choices).replace("和", "").strip()).issubset("①②③④⑤⑥") or len(cleaned_choices) <= 1:
            return None
        
        return prompter(
            task_name, question, cleaned_choices, gold_index, context=context,
            show_options=show_options
        )

    return adapter


def get_m_truthfulqa_prompt(lang: LANGS, type: Literal["mc1", "mc2"]):
    prompter = _get_multi_qa_prompt(lang)

    def adapter(line, task_name):

        choices = line[f"{type}_targets"]["choices"]
        labels = line[f"{type}_targets"]["labels"]
        gold_index = [ix for ix, label in enumerate(labels) if label == 1]
        return prompter(task_name, line["question"], choices, gold_index)

    return adapter


def get_sciq_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)
    return lambda line, task_name: prompter(
        task_name,
        line["question"],
        [
            line["distractor1"],
            line["distractor2"],
            line["distractor3"],
            line["correct_answer"],
        ],
        3,
        context=line["support"],
    )



def get_mathqa_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)

    def adapter(line, task_name):
        options = [line["inputs"][f"option_{i.lower()}"] for i in LETTER_INDICES[:4]]
        return prompter(
            task_name,
            line["inputs"]["text"],
            options,
            LETTER_INDICES.index(line["outputs"]),
        )

    return adapter


def get_openbookqa_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)

    def adapter(line, task_name):
        options = [line["inputs"][f"option_{i.lower()}"] for i in LETTER_INDICES[:4]]
        return prompter(
            task_name,
            line["inputs"]["question"],
            options,
            LETTER_INDICES.index(line["outputs"]),
        )

    return adapter


# QA-Tasks (No multichoice)
QA_TEMPLATE = "{topic}{context}{question_word}{colon}{sentence_space}{question}\n{answer_word}{colon}{sentence_space}"


def _get_qa_prompt(lang: LANGS):
    def qa_prompt(
        task_name: str,
        question: str,
        answer: list[str],
        context: str | None = None,
        topic: str | None = None,
    ):
        context = capitalize(fix_ending_punct(context, lang)) if context else ""
        question = fix_capitalization(fix_ending_punct(context, lang), question, lang)
        assert isinstance(
            answer, list
        ), f"Answer is not a list: {answer} in task {task_name}"
        answer = [capitalize(ans.strip()) for ans in answer]
        query = QA_TEMPLATE.format(
            # topic=f"{topic}\n" if topic else "",
            topic="",
            question=question,
            context=f"{context}\n" if context else "",
            question_word=QUESTION[lang],
            answer_word=ANSWER[lang],
            colon=COLON[lang],
            sentence_space=SENTENCE_SPACE[lang],
        )
        return Doc(
            task_name=task_name,
            query=query,
            gold_index=list(range(len(answer))),
            choices=answer,
        )

    return qa_prompt


def get_mlqa_prompt(lang: LANGS, answer_key: str = "text"):
    prompter = _get_qa_prompt(lang)
    def adapter(line, task_name):
        # remove empty answers
        answers = [ans for ans in line["answers"][answer_key] if len(ans) > 0]
        return prompter(
            task_name, line["question"], answers, line["context"]
        )
    return adapter


def get_kenswquad_prompt(lang: LANGS):
    prompter = _get_qa_prompt(lang)
    return lambda line, task_name: prompter(
        task_name, line["question"], [line["answer"]], context=line["context"]
    )

def get_mkqa_prompt(lang: LANGS, lang_key: str):
    prompter = _get_qa_prompt(lang)
    def adapter(line, task_name):
        text = line["answers"][lang_key][0]["text"]
        if text is None:
            return None

        aliases = line["answers"][lang_key][0]["aliases"]
        answers = list(filter(lambda x: len(x.strip()) > 0, [text] + aliases))
        # Some samples are broken so this is heuristic
        # e. g   'text': '七月 20, 1969',
        #        'aliases': ['1', 'u', ',', '2', ' ', '6', 'l', 'y', '9', '0', 'j']}],
        if len(answers) == 0 or len(answers) > 5:
            return None

        return prompter(
            task_name,
            line["queries"][lang_key],
            answers
    )
    return adapter

def get_tquad_prompt(lang: LANGS):
    prompter = _get_qa_prompt(lang)
    return lambda line, task_name: prompter(
        task_name,
        line["question"],
        [a["text"] for a in line["answers"]],
        line["context"],
    )


def get_mintaka_prompt(lang: LANGS):
    prompter = _get_qa_prompt(lang)
    return lambda line, task_name: prompter(
        task_name, line["question"], [line["answerText"]]
    )


def get_cmath_prompt(lang: LANGS):
    prompter = _get_qa_prompt(lang)
    return lambda line, task_name: prompter(
        task_name, line["question"], [line["golden"]]
    )


def get_chekega_prompt(lang: LANGS):
    prompter = _get_qa_prompt(lang)
    return lambda line, task_name: prompter(
        task_name,
        line["inputs"]["text"],
        [line["outputs"]],
        topic=line["inputs"]["topic"],
    )


def get_french_trivia_prompt(lang: LANGS):
    prompter = _get_qa_prompt(lang)
    return lambda line, task_name: prompter(
        task_name, line["Question"], [line["Answer"]]
    )


# NLI premise/hypthesis
NLI_TEMPLATE = "{premise}{full_stop}{sentence_space}{hypothesis}{comma}{sentence_space}{question_word}{question_mark}"
NLI_CONT_TEMPLATE = "{sentence_space}{label}"

# NLI v2 long premise/hypthesis
NLI_2_TEMPLATE = "{premise}{comma}{sentence_space}{question_word}{question_mark}"
NLI_2_CONT_TEMPLATE = "{sentence_space}{label}{comma}{sentence_space}{hypothesis}"

def _get_nli_prompt(
    lang: LANGS,
    pos_labels: list[Literal["entailment", "contradiction"]],
    template_version: Literal[1,2] = 1
) -> Callable[[str, str, str, int], Doc]:
    """
    Create an NLI prompt function based on the specified language, labels, and template version.

    Args:
        lang: The language to use for the prompt.
        pos_labels: List of possible labels, order matters for gold_index.
        template_version: Which template version to use ("v1" or "v2").

    Returns:
        A function that generates an NLI prompt given task name, premise, hypothesis, and label.
    """
    def _get_pos_label(label: Literal["entailment", "contradiction"]) -> str | None:
        if label == "entailment":
            return ENTAILMENT_LABELS[lang]
        elif label == "contradiction":
            return CONTRADICTION_LABELS[lang]

    labels = [_get_pos_label(label) for label in pos_labels if _get_pos_label(label) is not None]

    def nli_prompt(task_name: str, premise: str, hypothesis: str, label: int) -> Doc:
        premise = capitalize(premise.rstrip(PUNCT))
        hypothesis = hypothesis.rstrip(PUNCT)
        if template_version == 1:
            hypothesis = capitalize(hypothesis)
        else:
            hypothesis = decapitalize(hypothesis)
        
        if template_version == 1:
            query_template = NLI_TEMPLATE
            cont_template = NLI_CONT_TEMPLATE
        else:  # v2
            query_template = NLI_2_TEMPLATE
            cont_template = NLI_2_CONT_TEMPLATE

        query = query_template.format(
            premise=premise,
            hypothesis=hypothesis,
            full_stop=FULL_STOP[lang],
            question_word=NLI_QUESTION[lang],
            question_mark=QUESTION_MARK[lang],
            comma=COMMA[lang],
            sentence_space=SENTENCE_SPACE[lang],
        )

        choices = [
            cont_template.format(label=label, hypothesis=hypothesis, comma=COMMA[lang], sentence_space=SENTENCE_SPACE[lang])
            for label in labels
        ]

        unconditioned_prefix = query_template.format(
            premise="",
            hypothesis="",
            question_word=NLI_QUESTION[lang],
            question_mark=QUESTION_MARK[lang],
            comma=COMMA[lang],
            sentence_space=SENTENCE_SPACE[lang],
            full_stop=FULL_STOP[lang],
        )

        return Doc(
            task_name=task_name,
            query=query,
            choices=choices,
            gold_index=label,
            uncoditioned_prefix=unconditioned_prefix,
        )

    return nli_prompt

def get_rcb_prompt(lang: LANGS, version: Literal[1,2]):
    prompter = _get_nli_prompt(lang, ["entailment", "contradiction"], version)
    return lambda line, task_name: prompter(
        task_name,
        line["inputs"]["premise"],
        line["inputs"]["hypothesis"],
        int(line["outputs"]) - 1,
    )


def get_xnli_prompt(lang: LANGS, version: Literal[1,2]):
    prompter = _get_nli_prompt(lang, ["entailment", "contradiction"], version)
    # 0 is entailment contradiction is 2
    label_remap = {
        0: 0,
        2: 1,
    }
    return lambda line, task_name: prompter(
        task_name, line["premise"], line["hypothesis"], label_remap[int(line["label"])],
    )

def get_ocnli_prompt(lang: LANGS, version: Literal[1,2]):
    prompter = _get_nli_prompt(lang, ["entailment", "contradiction"], version)
    label_remap = {
        1: 0,
        2: 1,
    }
    return lambda line, task_name: prompter(
        task_name, line["sentence1"], line["sentence2"], label_remap[int(line["label"])]
    )

def get_paws_x_prompt(lang: LANGS, version: Literal[1,2]):
    # Each label has two possible values: 0 indicates the pair has different meaning, while 1 indicates the pair is a paraphrase.
    prompter = _get_nli_prompt(lang, ["entailment", "contradiction"], version)
    return lambda line, task_name: prompter(
        task_name, line["sentence1"], line["sentence2"], int(line["label"])
    )


# NLI Cause/Effect (Copa)
COPA_TEMPLATE = "{premise}{comma}{sentence_space}{cause_or_effect}"
COPA_HYPOTHESES_TEMPLATE = "{sentence_space}{hypothesis}"


def _get_copa_prompt(lang: LANGS):
    def copa_prompt(
        task_name: str,
        premise: str,
        cause_or_effect: Literal["cause", "effect"],
        hypotheses: list[str],
        gold_index: int,
    ):
        # Convert it into He was nice (premise) thus he was nice (hypothesis).
        # We expecte hypotheses and premise to be ended by .
        premise = capitalize(premise.rstrip(PUNCT))
        hypotheses = [decapitalize(hyp) for hyp in hypotheses]
        cause_or_effect_trans = (
            CAUSE_LABELS[lang] if cause_or_effect == "cause" else EFFECT_LABELS[lang]
        )
        return Doc(
            task_name=task_name,
            query=COPA_TEMPLATE.format(
                premise=premise,
                cause_or_effect=cause_or_effect_trans,
                comma=COMMA[lang],
                sentence_space=SENTENCE_SPACE[lang],
            ),
            choices=[
                COPA_HYPOTHESES_TEMPLATE.format(hypothesis=hypothesis, sentence_space=SENTENCE_SPACE[lang])
                for hypothesis in hypotheses
            ],
            gold_index=gold_index,
            uncoditioned_prefix=COPA_TEMPLATE.format(
                premise="",
                cause=CAUSE_LABELS[lang],
                cause_or_effect=cause_or_effect_trans,
                comma=COMMA[lang],
                sentence_space=SENTENCE_SPACE[lang],
            ),
        )

    return copa_prompt


def get_copa_prompt(lang: LANGS):
    prompter = _get_copa_prompt(lang)
    return lambda line, task_name: prompter(
        task_name,
        line["premise"],
        line["question"],
        [line["choice1"], line["choice2"]],
        int(line["label"]),
    )


def get_parus_prompt(lang: LANGS):
    prompter = _get_copa_prompt(lang)
    return lambda line, task_name: prompter(
        task_name,
        line["inputs"]["premise"],
        line["meta"]["task"],
        [line["inputs"]["choice1"], line["inputs"]["choice2"]],
        int(line["outputs"]) - 1,
    )


# QA YES/NO
def _get_boolq_prompt(lang: LANGS):
    yes, no = YES_LABELS[lang], NO_LABELS[lang]
    prompter = _get_multi_qa_prompt(lang)

    def boolq_prompt(
        task_name: str, question: str, label: bool, context: str | None = None
    ):
        return prompter(task_name, question, [yes, no], 0 if label else 1, context)

    return boolq_prompt


def get_boolq_prompt(lang: LANGS):
    prompter = _get_boolq_prompt(lang)
    return lambda line, task_name: prompter(
        task_name, line["question"], line["answer"], context=line["passage"]
    )


def get_indic_boolq_prompt(lang: LANGS):
    prompter = _get_boolq_prompt(lang)
    return lambda line, task_name: prompter(
        task_name,
        line["itv2 hi question"],
        line["answer"],
        context=line["itv2 hi passage"],
    )


def get_french_boolqa_prompt(lang: LANGS):
    prompter = _get_boolq_prompt(lang)

    def adapter(line, task_name):
        return prompter(
            task_name,
            line["question"],
            line["label"] == 1,
            context=line["passage"],
        )

    return adapter

def get_acva_prompt(lang: LANGS):
    prompter = _get_boolq_prompt(lang)
    choices = [CORRECT_LABELS[lang], INCORRECT_LABELS[lang]]
    def adapter(line, task_name):
        question = f"{line['question'].rstrip(PUNCT)}{QUESTION_MARK[lang]}"
        return prompter(
            task_name, question, choices.index(line["answer"]) == 0
        )
    return adapter


# NLI Hellaswag
DEFAULT_DOT_REPLACEMENT = [" [title]"]
DOT_REPLACEMENTS: dict[LANGS, list[str]] = {
    # https://github.com/malhajar17/lm-evaluation-harness_turkish/blob/main/lm_eval/tasks/hellaswag_tr-v0.2/utils.py
    "tr": [" [title]", " [başlık]", " [adım]", " [header]"],
}

HELLASWAG_TEMPLATE = "{activity_label}{ctx}"

def _get_hellaswag_prompt(lang: LANGS):
    dot_replacment = DOT_REPLACEMENTS.get(lang, DEFAULT_DOT_REPLACEMENT)

    def preprocess(text):
        """Comes from AiHarness"""
        # text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        for dot_repl in dot_replacment:
            text = text.replace(dot_repl, ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        text = text.replace(r"\.+", r"\.")
        return text.strip()
    
    def process_context(ctx):
        if ctx == "":
            return ""
        return capitalize(fix_ending_punct(preprocess(ctx), lang))

    def hellaswag_prompt(
        task_name: str,
        ctx: tuple[str, str] | str,
        endings: list[str],
        label: int,
        activity_label: str | None = None,
    ):
        sentence_space = SENTENCE_SPACE[lang]
        ctx_list = list(ctx) if isinstance(ctx, tuple) else [ctx]
        # Last one should be left as is
        ctxs = [process_context(c) for c in ctx_list]
        ctxs = [c for c in ctxs if c != ""]
        context = sentence_space.join(ctxs)
        activity_label = f"{capitalize(activity_label)}:\n" if activity_label else ""
        # Removoal of the [header] can happen and we need the first letter to be capital afterwards
        full_context = HELLASWAG_TEMPLATE.format(activity_label=activity_label, ctx=context)
        separator_query = SENTENCE_SPACE[lang] if should_follow_sentence_space(full_context, lang) else WORD_SPACE[lang]
        choices = [f"{separator_query}{fix_capitalization(full_context, preprocess(ending), lang)}" for ending in endings]  
        if any(len(c.strip()) == 0 for c in choices):
            return None
        return Doc(
            task_name=task_name,
            query=full_context,
            choices=choices,
            gold_index=int(label) if label != "" else -1,  # -1 for test
            uncoditioned_prefix="",
        )

    return hellaswag_prompt


def get_hellaswag_prompt(lang: LANGS, use_activity_label: bool = True):
    prompter = _get_hellaswag_prompt(lang)
    return lambda line, task_name: prompter(
        task_name,
        (line["ctx_a"], line["ctx_b"]),
        line["endings"],
        line["label"],
        activity_label=line.get("activity_label") if use_activity_label else None,
    )


def get_hellaswag_prompt_full_ctx(lang: LANGS, use_activity_label: bool = True):
    prompter = _get_hellaswag_prompt(lang)
    return lambda line, task_name: prompter(
        task_name,
        line["ctx"],
        line["endings"],
        line["label"],
        activity_label=line.get("activity_label") if use_activity_label else None,
    )


def get_xcodah_prompt(lang: LANGS):
    def xcodah_prompt(line: dict[str, Any], task_name: str):
        gold_index = line["question"]["choices"]["label"].index(line["answerKey"])
        # All the choices have already common prefix "baken in" so we have to remove to get clearer signal
        # Extract common prefix from choices
        choices = line["question"]["choices"]["text"]
        common_prefix = os.path.commonprefix(choices)

        # Backtract to first space to get good tokenization
        first_word = common_prefix.rfind(WORD_SPACE[lang])
        
        # If there is no word_space we shouldn't remove the common prefix
        common_prefix = common_prefix[:first_word] if first_word != -1 else ""
        
        # Remove common prefix from each choice
        cleaned_choices = [choice[len(common_prefix):] for choice in choices]
        
        if any(len(c.strip()) == 0 for c in cleaned_choices):
            return None

        
        return Doc(
            task_name=task_name,
            query=common_prefix,
            choices=cleaned_choices,
            gold_index=gold_index,
            uncoditioned_prefix="",
        )
    return xcodah_prompt


# NLI (collocations)
def get_winogrande_prompt(lang: LANGS):

    def winogrande(line, task_name: str = None):
        # LL of query + choices
        query, end_of_target = line["sentence"].split("_")
        query = capitalize(fix_ending_punct(query, lang))
        options = [
            fix_capitalization(query, o, lang)
            for o in [line["option1"], line["option2"]]
        ]
        end_of_target = fix_ending_punct(end_of_target.strip(), lang)

        separator_query = SENTENCE_SPACE[lang] if should_follow_sentence_space(query, lang) else WORD_SPACE[lang]
        return Doc(
            task_name=task_name,
            query=query,
            choices=[f"{separator_query}{o}{WORD_SPACE[lang]}{end_of_target}" for o in options],
            gold_index=(
                int(line["answer"]) - 1 if line["answer"] != "" else -1
            ),  # managing unk test index
            uncoditioned_prefix="",
        )
    
    return winogrande


# WSCI (collocations), right now only for thai
def get_wsci_prompt(lang: Literal["th"]):

    def is_possessive(pronoun):
        # Check if the pronoun is a possessive form
        return pronoun.startswith("ของ")

    def add_possessive(pronoun):
        return f"ของ{pronoun}"
    
    def process_opt(option, pronoun):
        return add_possessive(option) if is_possessive(pronoun) else option
    

    def wsci(line, task_name: str):
        pronoun = line["pronoun"]
        quote, ending = line["text"][:line["pronoun_loc"]], line["text"][line["pronoun_loc"]+len(pronoun):]
        options = [process_opt(opt, pronoun) for opt in line["options"]]
        separator_query = SENTENCE_SPACE[lang] if should_follow_sentence_space(quote, lang) else WORD_SPACE[lang]
        return Doc(
            task_name=task_name,
            query=quote,
            # We have to use spacing, because of tokenization
            choices=[f"{separator_query}{option}{ending}" for option in options],
            gold_index=line["label"],
            uncoditioned_prefix="",
        )
    return wsci


def _extract_answers_from_string(answer_string: str, task_answers: list[str]) -> Optional[tuple[int, list[list[str]]]]:
    """
    Extracts answers from the question. The search is done from the end to the beginning.
    The goal is to extract multichoice answers from a question

    Example:
    This is a question. 
    ① Yes ② No ③ Yes ④ No
    Args:
        answer_string (str): String possibly containing answers.
        answers (list[str]): Task answers
        gold_idx (int): The index of the gold answer.
    Returns:
        Optional[tuple[int, list[list[str]]]]: A tuple containing the start index of the answers, list of list of answers (as answers can have multiple correct solutions).
    """
    def extract_answer(acc: tuple[str, int, list[str]], symbol: str) -> tuple[str, int, list[str]]:
        """
        Extracts an answer from the text until the next symbol is found.
        If the last index == -1 it means we failed to find the symbol.
        """
        text, last_index, answers = acc
        if last_index == -1:
            return text, last_index, answers
        start_index = last_index
        end_index = text.rfind(symbol[:last_index])
        if end_index == -1:
            return text, -1, answers
        return text, end_index, answers + [text[end_index:start_index]]
    
        
    # Remove and from answers
    task_answers = [answer.replace("和", "").strip() for answer in task_answers]
    answer_letters = set("".join(task_answers))
    
    if not answer_letters.issubset("①②③④⑤⑥"):
        return None
    

    # Try to extract answers from the possible_answers_part
    answer_string = answer_string.strip()
    
    # Split by ①②③④ to get the answers
    _, last_index, found_answers = reduce(extract_answer, sorted(list(answer_letters), reverse=True), (answer_string, len(answer_string), []))
    if last_index == -1:
        return None
    
    found_answers = [x for x in found_answers if x.strip() != ""]
    

    # Ensure we have extracted all answers
    if len(found_answers) != len(answer_letters):
        return None
    

    found_answers = [answer.rstrip(PUNCT + ";").strip() for answer in found_answers]
    letter_answer_dict = {answer[:1]: answer[1:].strip() for answer in found_answers}
    
    new_answer_list = [[letter_answer_dict.get(letter) for letter in answer] for answer in task_answers]
    
    # I can't figure out case where this wouldn't hold but just to be sure
    if any(any(a is None for a in l_ans) for l_ans in new_answer_list):
        return None
    
    return last_index, new_answer_list


def multichoice_join(choices: list[str], lang: str, variant: MULTICHOICE_JOIN_VARIANT):
    separator: str
    if variant == "AND":
        separator = f"{WORD_SPACE[lang]}{AND[lang]}{WORD_SPACE[lang]}"
    elif variant == "OR":
        separator = f"{WORD_SPACE[lang]}{OR[lang]}{WORD_SPACE[lang]}"
    elif variant == "COMMA":
        separator = f"{COMMA[lang]}{SENTENCE_SPACE[lang]}"
    elif variant == "NEW_LINE":
        # We keep space to get consistent tokenization
        separator = f"\n"
        
    return separator.join(choices)


def multichoice_compose(choices: list[str], gold_idx: list[int], lang: str, variant: MULTICHOICE_JOIN_VARIANT):
    if len(gold_idx) == 1:
        return choices, gold_idx
    
    multichoice_joiner = partial(multichoice_join, lang=lang, variant=variant)
    
    new_choices = [multichoice_joiner([choices[i] for i in gold_idx])] + [
        choice for i,choice in enumerate(choices) if not i in gold_idx
    ]
    
    # All correct choices are now at 0
    return new_choices, [0]
    
    

