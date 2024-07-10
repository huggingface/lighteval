# MMML

import re
from typing import Any, Literal

from ..utils.translation_literals import (
    ANSWER,
    CAUSE_LABELS,
    COMMA,
    CONTRADICTION_LABELS,
    CORRECT_LABELS,
    EFFECT_LABELS,
    ENTAILMENT_LABELS,
    IMPOSSIBLE,
    INCORRECT_LABELS,
    LANGS,
    NEUTRAL_LABELS,
    NLI_QUESTION,
    NO_LABELS,
    QUESTION,
    QUESTION_MARK,
    SPACE,
    YES_LABELS,
    FULL_STOP,
)
from lighteval.tasks.doc import Doc
from lighteval.tasks.tasks_prompt_formatting import LETTER_INDICES

PUNCT = "-.!?،؟‽,。，？؟،। "


def decapitalize(word: str):
    return word[0].lower() + word[1:]


# Notes:
# - For the context we can also put something in front (not implemented right now)

# For thai we use space in front of completions no matter the fact they might not be used, this is to ensure correct tokeniztion

# QA-Tasks (multichoice no helpers)
MULTI_QA_SIMPLE_TEMPLATE = "{context}{full_stop}{space}{question}"


def _get_multi_qa_simple_prompt(lang: LANGS):
    def multi_qa_prompt(
        task_name: str,
        question: str,
        answers: list[str],
        gold_index,
        context: str | None = None,
    ):
        question = question.strip()
        context = context.rstrip(PUNCT) if context else ""
        answers = [answer.strip() for answer in answers]
        query = MULTI_QA_SIMPLE_TEMPLATE.format(
            context=f"{context}\n" if context else "",
            question=question,
            full_stop=FULL_STOP[lang],
            space=SPACE[lang],
        )
        return Doc(
            task_name=task_name,
            query=query,
            gold_index=gold_index,
            choices=[f" {c}" for c in answers if c],
            uncoditioned_prefix=f"{ANSWER[lang]}:",
        )

    return multi_qa_prompt


def get_m_m3exam_prompt(lang: LANGS):
    prompter = _get_multi_qa_simple_prompt(lang)
    # Would be nice to have general solution for the letters

    dot = re.escape(FULL_STOP[lang])
    prefix_re = re.compile(rf"^\([A-Da-d1-5๑๒๓๔๕]\)\s*|^[A-Da-e1-5๑๒๓๔๕][{dot}．]\s*")

    def adapter(line, task_name):
        is_letter_based = line["answer_text"].isalpha()
        clean_options = [prefix_re.sub("", c) for c in line["options"]]
        gold_idx = (
            LETTER_INDICES.index(line["answer_text"].upper())
            if is_letter_based
            else int(line["answer_text"]) - 1
        )
        return prompter(
            task_name,
            line["question_text"],
            clean_options,
            gold_idx,
            context=line["background"],
        )

    return adapter


# QA-Tasks (multichoice)
MULTI_QA_TEMPLATE = "{context}{question_word}: {question}\n{answer_word}:"


def _get_multi_qa_prompt(lang: LANGS):
    def multi_qa_prompt(
        task_name: str,
        question: str,
        answers: list[str],
        gold_index,
        context: str | None = None,
    ):
        question = question.strip()
        context = context.strip() if context else ""
        answers = [answer.strip() for answer in answers]
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

    def adapter(line, task_name):
        letters = [letter.lower() for letter in LETTER_INDICES[:5]]
        options = [line[letter] for letter in letters]
        non_empty_options = [opt for opt in options if opt != ""]
        gold_index = letters.index(line["answer"])
        return prompter(
            task_name,
            line["question"],
            non_empty_options,
            gold_index,
        )

    return adapter


def get_ceval_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)
    return lambda line, task_name: prompter(
        task_name,
        line["question"],
        [line["A"], line["B"], line["C"], line["D"]],
        LETTER_INDICES.index(line["answer"]),
    )


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


def get_m_exams_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)

    def adapter(line, task_name):
        letters = line["question"]["choices"]["label"]
        texts = line["question"]["choices"]["text"]
        return prompter(
            task_name,
            line["question"]["stem"],
            texts,
            letters.index(line["answerKey"]),
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


def get_agieval_prompt(lang: Literal["zh"]):
    prefix_re = re.compile(r"^\([A-D]\)")
    prompter = _get_multi_qa_prompt(lang)

    def adapter(line, task_name):
        # Remove the question at the start to get consistency
        # Ensure there is exactly one '问题：' in the query
        context, rest = line["query"].split("问题：", maxsplit=1)
        question, _ = rest.split(" 选项：", maxsplit=1)
        original_choices = line["choices"]
        no_letter_choices = [prefix_re.sub("", c) for c in original_choices]
        return prompter(
            task_name, question, no_letter_choices, line["gold"], context=context
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


def get_acva_prompt(lang: LANGS):
    prompter = _get_multi_qa_prompt(lang)
    choices = [CORRECT_LABELS[lang], INCORRECT_LABELS[lang]]
    return lambda line, task_name: prompter(
        task_name, line["question"], choices, choices.index(line["answer"])
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
QA_TEMPLATE = "{topic}{context}{question_word}: {question}\n{answer_word}:"


def _get_qa_prompt(lang: LANGS):
    # TODO: I am not sure what gold it should have
    def qa_prompt(
        task_name: str,
        question: str,
        answer: list[str],
        context: str | None = None,
        topic: str | None = None,
    ):
        question = question.strip()
        context = context.strip() if context else ""
        assert isinstance(
            answer, list
        ), f"Answer is not a list: {answer} in task {task_name}"
        answer = [ans.strip() for ans in answer]
        query = QA_TEMPLATE.format(
            # topic=f"{topic}\n" if topic else "",
            topic="",
            question=question,
            context=f"{context}\n" if context else "",
            question_word=QUESTION[lang],
            answer_word=ANSWER[lang],
        )
        return Doc(
            task_name=task_name,
            query=query,
            gold_index=list(range(len(answer))),
            choices=answer,
            uncoditioned_prefix=f"{ANSWER[lang]}:",
        )

    return qa_prompt


def get_mlqa_prompt(lang: LANGS):
    prompter = _get_qa_prompt(lang)
    return lambda line, task_name: prompter(
        task_name, line["question"], line["answers"]["text"], line["context"]
    )


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
NLI_TEMPLATE = "{premise}{comma}{space}{question_word}{question_mark}"
NLI_CONT_TEMPLATE = " {label}{comma}{space}{hypothesis}{full_stop}"


def _get_nli_prompt(
    lang: LANGS, pos_labels: list[Literal["entailment", "neutral", "contradiction"]]
):
    """
    pos_labels: list[Literal["entailment", "neutral", "contradiction"]] what labels are possible, the ordering matters as
    the the gold_index should be the index of the label in the list.
    """

    def _get_pos_label(label: Literal["entailment", "neutral", "contradiction"]):
        if label == "entailment":
            return ENTAILMENT_LABELS[lang]
        elif label == "neutral":
            return NEUTRAL_LABELS[lang]
        elif label == "contradiction":
            return CONTRADICTION_LABELS[lang]

    labels = [_get_pos_label(label) for label in pos_labels]
    labels = [label for label in labels if label is not None]

    def nli_prompt(task_name: str, premise: str, hypothesis: str, label: int):
        premise = premise.rstrip(PUNCT)
        hypothesis = decapitalize(hypothesis.rstrip(PUNCT))
        return Doc(
            task_name=task_name,
            query=NLI_TEMPLATE.format(
                premise=premise,
                question_word=NLI_QUESTION[lang].capitalize(),
                question_mark=QUESTION_MARK[lang],
                comma=COMMA[lang],
                space=SPACE[lang],
            ),
            choices=[
                NLI_CONT_TEMPLATE.format(
                    comma=COMMA[lang],
                    label=label,
                    hypothesis=hypothesis,
                    full_stop=FULL_STOP[lang],
                    space=SPACE[lang],
                )
                for label in labels
            ],
            gold_index=label,
            uncoditioned_prefix=NLI_TEMPLATE.format(
                premise="",
                question_word=NLI_QUESTION[lang].capitalize(),
                question_mark=QUESTION_MARK[lang],
                comma=COMMA[lang],
                space=SPACE[lang],
            ),
        )

    return nli_prompt


def get_rcb_prompt(lang: LANGS):
    prompter = _get_nli_prompt(lang, ["entailment", "contradiction", "neutral"])
    return lambda line, task_name: prompter(
        task_name,
        line["inputs"]["premise"],
        line["inputs"]["hypothesis"],
        int(line["outputs"]) - 1,
    )


def get_xnli_prompt(lang: LANGS):
    prompter = _get_nli_prompt(lang, ["entailment", "neutral", "contradiction"])
    return lambda line, task_name: prompter(
        task_name, line["premise"], line["hypothesis"], int(line["label"])
    )


def get_paws_x_prompt(lang: LANGS):
    # Each label has two possible values: 0 indicates the pair has different meaning, while 1 indicates the pair is a paraphrase.
    prompter = _get_nli_prompt(lang, ["contradiction", "entailment"])
    return lambda line, task_name: prompter(
        task_name, line["sentence1"], line["sentence2"], int(line["label"])
    )


# NLI Cause/Effect (Copa)
COPA_TEMPLATE = "{premise}{comma}{space}{cause_or_effect}"
COPA_HYPOTHESES_TEMPLATE = " {hypothesis}"


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
        premise = premise.rstrip(PUNCT)
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
                space=SPACE[lang],
            ),
            choices=[
                COPA_HYPOTHESES_TEMPLATE.format(hypothesis=hypothesis)
                for hypothesis in hypotheses
            ],
            gold_index=gold_index,
            uncoditioned_prefix=COPA_TEMPLATE.format(
                premise="",
                cause=CAUSE_LABELS[lang],
                cause_or_effect=cause_or_effect_trans,
                comma=COMMA[lang],
                space=SPACE[lang],
            ),
        )

    return copa_prompt


def get_copa_prompt(lang: LANGS):
    # TODO: solve the punctuation issue
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
        task_name, line["question"], line["answer"] == "true", context=line["passage"]
    )


def get_indic_boolq_prompt(lang: LANGS):
    prompter = _get_boolq_prompt(lang)
    return lambda line, task_name: prompter(
        task_name,
        line["itv2 hi question"],
        line["answer"] == "true",
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


# NLI Hellaswag
DEFAULT_DOT_REPLACEMENT = [" [title]"]
DOT_REPLACEMENTS: dict[LANGS, list[str]] = {
    # https://github.com/malhajar17/lm-evaluation-harness_turkish/blob/main/lm_eval/tasks/hellaswag_tr-v0.2/utils.py
    "tr": [" [title]", " [başlık]", " [adım]", " [header]"],
}

HELLASWAG_TEMPLATE = "{activity_label}{ctx}"
CTX_TEMPLATE = "{ctx}{full_stop}"


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
        return text.strip()
    
    def process_context(ctx):
        if ctx == "":
            return ""
        return CTX_TEMPLATE.format(ctx=ctx.rstrip(PUNCT).capitalize(), full_stop=FULL_STOP[lang])

    def hellaswag_prompt(
        task_name: str,
        ctx: tuple[str, str] | str,
        endings: list[str],
        label: int,
        activity_label: str | None = None,
    ):
        space = SPACE[lang]
        ctx_list = list(ctx) if isinstance(ctx, tuple) else [ctx]
        ctxs = [process_context(c) for c in ctx_list]
        ctxs = [c for c in ctxs if c != ""]
        context = space.join(ctxs)
        activity_label = f"{activity_label}: " if activity_label else ""
        full_context = HELLASWAG_TEMPLATE.format(activity_label=activity_label, ctx=context)
        return Doc(
            task_name=task_name,
            query=preprocess(full_context),
            choices=[" " + preprocess(ending) for ending in endings],
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


def xcodah_prompt(line: dict[str, Any], task_name: str):
    gold_index = line["question"]["choices"]["label"].index(line["answerKey"])
    return Doc(
        task_name=task_name,
        query="",
        choices=line["question"]["choices"]["text"],
        gold_index=gold_index,
        uncoditioned_prefix=None,
    )


# NLI (collocations)


# TODO use trans literals for punct
def winogrande(line, task_name: str = None):
    # LL of query + choices
    query, end_of_target = line["sentence"].split("_")
    query = query.strip()
    assert len(query) > 0, f"Query must not be empty {line['sentence']}"
    query_ends_with_punct = query[-1] in PUNCT
    options = [
        o.capitalize() if query_ends_with_punct else decapitalize(o)
        for o in [line["option1"], line["option2"]]
    ]
    end_of_target = end_of_target.strip()
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {o} {end_of_target}" for o in options],
        gold_index=(
            int(line["answer"]) - 1 if line["answer"] != "" else -1
        ),  # managing unk test index
        uncoditioned_prefix="",
    )
