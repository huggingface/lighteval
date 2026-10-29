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

import json
import re

import numpy as np

from lighteval.metrics.armenian_metrics import (
    armenian_exam_metric,
    armenian_mcqa_metric,
    armenian_mmlu_pro_metric,
    bert_score_arm,
    ner_span_metric,
    pos_metric,
)
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.metric_utils import (
    SampleLevelComputation,
    SampleLevelMetric,
    SamplingMethod,
)
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import Doc, LightevalTaskConfig


prompt_language = "hy"
SIB200_LABEL_MAP = {
    "technology": "տեխնոլոգիա",
    "travel": "ճանապարհորդություն",
    "business": "բիզնես",
    "disasters": "աղետներ",
    "sports": "սպորտ",
    "science": "գիտություն",
    "nature": "բնություն",
    "politics": "քաղաքականություն",
    "religion": "կրոն",
    "education": "կրթություն",
    "crime": "հանցագործություն",
    "entertainment": "ժամանց",
    "geography": "աշխարհագրություն",
    "health": "առողջապահություն",
}
SIB200_LABELS = list(SIB200_LABEL_MAP.values())
SENTIMENT_LABELS = ["դրական", "բացասական", "չեզոք", "երկիմաստ"]
URGENCY_LABEL_MAP = {"high": "բարձր", "medium": "միջին", "low": "ցածր"}
URGENCY_LABELS = list(URGENCY_LABEL_MAP.values())

PROMPTS_SIB200 = {
    "instruction": {
        "hy": ("Տրված տեքստի համար ընտրիր այն թեման, որը լավագույնս նկարագրում է այն։\n"),
        "en": ("For the given text, choose the topic that best describes it.\n"),
    },
    "query": {
        "hy": ("Տեքստ\n{text}\n\nՀնարավոր թեմաներ՝ {labels}"),
        "en": ("Text: {text}\nPossible topics: {labels}"),
    },
}

PROMPTS_SENTIMENT = {
    "instruction": {
        "hy": ("Որոշիր տեքստի սենտիմենտը։\n"),
        "en": ("Determine the sentiment of the text.\n"),
    },
    "query": {
        "hy": ("Տեքստ\n{text}\n\nՀնարավոր տարբերակներ՝ դրական, բացասական, չեզոք, երկիմաստ"),
        "en": ("Text: {text}\nPossible options: positive, negative, neutral, ambiguous."),
    },
}
PROMPTS_URGENCY = {
    "instruction": {
        "hy": ("Որոշիր նամակի կարևորության մակարդակը։\n"),
        "en": ("Determine the urgency level of the email.\n"),
    },
    "query": {
        "hy": ("Նամակ\n{text}\n\nՏարբերակներ՝ բարձր, միջին, ցածր"),
        "en": ("Email: {text}\nOptions: high, medium, low"),
    },
}

PROMPTS_TEXT_TAGGING = {
    "instruction": {
        "hy": (
            "Տրված տեքստից առանձնացրեք և գրեք բանալի բառեր ու բառակապակցություններ։ Գրեք արդյունքը` բաժանված ստորակետերով։\n"
        ),
        "en": (
            "Extract and write key words and phrases from the given text. Write the result as a single sentence, separated by commas.\n"
        ),
    },
    "query": {"hy": ("{text}"), "en": ("{text}")},
}


PROMPTS_MMLU_PRO = {
    "system": 'The following are multiple choice questions (with answers) about {category}.\nThink step by step and then output the answer in the format of "The answer is (X)" at the end.',
    "input": "Question: {question}\nChoices:",
}


def prompt_mmlu_pro(line, task_name=None):
    system_prompt = PROMPTS_MMLU_PRO["system"]
    input_prompt = PROMPTS_MMLU_PRO["input"]

    system_prompt = system_prompt.format(category=line["category"])
    formated_input_prompt = input_prompt.format(question=line["question_arm"]) + "\n"

    choice_map = "ABCDEFGHIJ"
    options = line["options_arm"]

    for i, opt in enumerate(options):
        formated_input_prompt += "({}) {}\n".format(choice_map[i], opt)
    formated_input_prompt += "Output: "

    gold_index = line["answer_index"]

    return Doc(
        task_name=task_name,
        query=formated_input_prompt,
        instruction=system_prompt,
        choices=options,
        gold_index=gold_index,
        specific={"answer": line["answer"]},
    )


PROMPTS_EXAM_HISTORY = {
    1: {
        "system": """The following is multiple-choice question (with answers). Use the provided context or passage to guide your reasoning if the context exists.
Think step by step and then output the answer in the format of "The answer is (X)" at the end.""",
        "input": """Question: {question}
Context: {context}
Choices:
""",
    },
    2: {
        "system": """The following is multiple-answer question (with answers). Each question may have a different number of correct choices. Use the provided context or passage to guide your reasoning if the context exists.
Think step by step and then output the answer in the format of [X, Y, Z, ...] listing all applicable choices.""",
        "input": """Question: {question}
Context: {context}
Possible Choices:""",
    },
    3: {
        "system": """The following is multiple-option question. Each question has exactly 6 options. For each option, decide whether it is {Ճիշտ է, Սխալ է, Չգիտեմ}. Use the provided context or passage to guide your reasoning if the context exists.
Think step by step and then output the answer in the format of ['Ճիշտ է', 'Սխալ է', 'Չգիտեմ', 'Ճիշտ է', 'Սխալ է', 'Ճիշտ է'].""",
        "input": """Instruction: {question}
Context: {context}
Options:""",
    },
    4: {
        "system": """You are presented with a matching task. Match each item from Character Items to its corresponding item from Numeric Items. Character Items are: (Ա, Բ, Գ, ...). Numeric Items are: (1, 2, 3, ...).
Think step by step, considering each item and its possible matches.
Return your answer as a set of key-value pairs where the keys are the items from Character Items and the values are the corresponding items from Numeric Items.
Ensure that the output includes only the items listed in Character Items and that the number of keys in your output matches the number of items in Character Items. For example, if Character Items are (Ա, Բ, Գ), and Numeric Items are (1, 2, 3, 4, 5) the output should be {"Ա": 2, "Բ": 1, "Գ": 5}.""",
        "input": """Instruction: {question}
Character Items: {context}
Numeric Items:""",
    },
    5: {
        "system": """Arrange the following events in the modern history of Armenia in strict chronological order, from the earliest to the most recent. Provide the sequence of numbers corresponding to the events. Ensure the returned sequence has the same length as the number of events listed.
Think step by step and return the answer in the following format, where the sequence of numbers is presented as a list of strings e.g. ['5', '3', '1', '2', '4']""",
        "input": """Instruction: {question}
Context: {context}
Events:""",
    },
}


def prompt_exam_history(line, task_name=None):
    system_prompt = PROMPTS_EXAM_HISTORY[line["task_type"]]["system"]
    input_prompt = PROMPTS_EXAM_HISTORY[line["task_type"]]["input"]

    if not line["context"]:
        formatted_input_prompt = input_prompt.format(question=line["question"], context="")
        formatted_input_prompt = formatted_input_prompt.replace("\nContext: ", "")
    else:
        formatted_input_prompt = input_prompt.format(question=line["question"], context=line["context"])

    choice_map = "123456789"
    for i in range(len(line["choices"])):
        formatted_input_prompt += "{}. {}\n".format(choice_map[i], line["choices"][i])

    formatted_input_prompt += "Output: "

    return Doc(
        task_name=task_name,
        query=formatted_input_prompt,
        choices=line["choices"],
        gold_index=0,
        instruction=system_prompt,
        specific={"task_type": line["task_type"], "label": line["label"]},
    )


PROMPTS_EXAM_MATH = {
    6: {
        "system": """The following is a multiple-choice question (MCQ) with answer choices.
You will be given a mathematical task followed by a question based on that task. Your goal is to read the task carefully, understand the question, and select the correct answer choice from the given options.
Think step by step and provide the correct answer choice in the following format: "The answer is (X)\"""",
        "input": """Task: {task}
Question: {question}
Choices:""",
    },
    7: {
        "system": """You will be given initial parameters, data, or conditions, followed by a question that requires solving a problem based on them.
Think step by step and then output the answer in the format of "The answer is (X)" at the end.""",
        "input": """Given conditions: {task}
Question: {question}""",
    },
    3: {
        "system": """You will be given a task with initial conditions, followed by question and multiple-options. Each task has exactly 6 options. For each option, decide whether it is {ճիշտ է, սխալ է, Չգիտեմ}.

Think step by step and then output the answer in the format of:
['ճիշտ է', 'սխալ է', 'Չգիտեմ', 'ճիշտ է', 'սխալ է', 'ճիշտ է']""",
        "input": """Given conditions: {task}
Question: {question}
Options:""",
    },
}


def prompt_exam_math(line, task_name=None):
    if line["task_type"] not in PROMPTS_EXAM_MATH:
        return None
    system_prompt = PROMPTS_EXAM_MATH[line["task_type"]]["system"]
    input_prompt = PROMPTS_EXAM_MATH[line["task_type"]]["input"]

    formatted_input_prompt = input_prompt.format(task=line["task"], question=line["question"])
    choice_map = "ABCDEFGHIJ"
    if line["task_type"] != 7:
        for i in range(len(line["choices"])):
            formatted_input_prompt += "{}. {}\n".format(choice_map[i], line["choices"][i])

    formatted_input_prompt += "Output: "

    return Doc(
        task_name=task_name,
        query=formatted_input_prompt,
        choices=line["choices"],
        gold_index=0,
        instruction=system_prompt,
        specific={"task_type": line["task_type"], "label": line["label"]},
    )


PROMPTS_EXAM_LITERATURE = {
    1: {
        "system": """The following are multiple-choice questions (with answers). Use the provided context or passage to guide your reasoning if the context exists.
Think step by step and then output the answer in the format of "The answer is (X)" at the end.""",
        "input": """Question: {question}
Context: {context}
Choices:""",
    },
    2: {
        "system": """The following are multiple-answer questions (with answers). Each question may have a different number of correct choices. Use the provided context or passage to guide your reasoning if the context exists.
Think step by step and then output the answer in the format of [X, Y, Z, ...] listing all applicable choices.""",
        "input": """Question: {question}
Context: {context}
Possible Choices:""",
    },
    3: {
        "system": """The following are multiple-option questions (with answers). Each question has exactly 6 options. For each option, decide whether it is {Ճիշտ է, Սխալ է, Չգիտեմ}. Use the provided context or passage to guide your reasoning if the context exists.
Think step by step and then output the answer in the format of ['Ճիշտ է', 'Սխալ է', 'Չգիտեմ', 'Ճիշտ է', 'Սխալ է', 'Ճիշտ է'].""",
        "input": """Instruction: {question}
Context: {context}
Options:""",
    },
}


def prompt_exam_literature(line, task_name=None):
    system_prompt = PROMPTS_EXAM_LITERATURE[line["task_type"]]["system"]
    input_prompt = PROMPTS_EXAM_LITERATURE[line["task_type"]]["input"]

    if not line["context"]:
        formatted_input_prompt = input_prompt.format(question=line["question"], context="")
        formatted_input_prompt = formatted_input_prompt.replace("\nContext: ", "")
    else:
        formatted_input_prompt = input_prompt.format(question=line["question"], context=line["context"])

    choice_map = "123456789"
    for i in range(len(line["choices"])):
        formatted_input_prompt += "{}. {}\n".format(choice_map[i], line["choices"][i])

    formatted_input_prompt += "Output: "

    return Doc(
        task_name=task_name,
        query=formatted_input_prompt,
        choices=line["choices"],
        gold_index=0,
        instruction=system_prompt,
        specific={"task_type": line["task_type"], "label": line["label"]},
    )


def prompt_sib200(line, task_name=None):
    eng_label = line["category"]
    gold = SIB200_LABEL_MAP[eng_label]
    choices = SIB200_LABELS
    gold_index = choices.index(gold)

    choice_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    formatted_labels = "\n".join([f"({choice_map[i]}) {opt}" for i, opt in enumerate(choices)])

    query = PROMPTS_SIB200["query"][prompt_language].format(labels=formatted_labels, text=line["text"])
    instruction = PROMPTS_SIB200["instruction"][prompt_language]

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=instruction,
        specific={"answer": choice_map[gold_index]},
    )


def prompt_sentiment(line, task_name=None):
    gold = line["sentiment_categories"][0]
    choices = SENTIMENT_LABELS
    gold_index = choices.index(gold) if gold in choices else 0

    text = line["text"]
    choice_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, opt in enumerate(choices):
        text += f"\n({choice_map[i]}) {opt}"

    query = PROMPTS_SENTIMENT["query"][prompt_language].format(text=text)
    instruction = PROMPTS_SENTIMENT["instruction"][prompt_language]

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=instruction,
        specific={"answer": choice_map[gold_index]},
    )


PROMPTS_INSTRUCTION = {
    "instruction": {
        "hy": "Կարդա պահանջը և տրամադրված կոնտեքստը և պատասխանիր հարցին։\n",
        "en": "Read the instruction and the provided context and answer the question.\n",
    },
    "query": {
        "hy": "Պահանջ\n{instruction}\n\nԿոնտեքստ\n{context}",
        "en": "Instruction: {instruction}\nContext: {context}",
    },
}

PROMPTS_QA_CONTEXT = {
    "instruction": {
        "hy": "Օգտվելով տեքստից` պատասխանիր հարցին։\n",
        "en": "Answer the question using the context.\n",
    },
    "query": {
        "hy": "Տեքստ\n{context}\n\nՀարց\n{question}",
        "en": "Context: {context}\nQuestion: {question}",
    },
}


def prompt_ms_marco(line, task_name=None):  # noqa: C901
    raw = line["armenian"]

    question, context, answers = "", "", ""
    section = None
    buffer = []

    for raw_line in raw.splitlines():
        if raw_line.startswith("INPUT:"):
            if section == "ANSWERS":
                answers = "\n".join(buffer).strip()
            elif section == "CONTEXT":
                context = "\n".join(buffer).strip()
            elif section == "INPUT":
                question = "\n".join(buffer).strip()
            section, buffer = "INPUT", [raw_line.replace("INPUT:", "").strip()]
        elif raw_line.startswith("CONTEXT:"):
            if section == "INPUT":
                question = "\n".join(buffer).strip()
            elif section == "ANSWERS":
                answers = "\n".join(buffer).strip()
            section, buffer = "CONTEXT", [raw_line.replace("CONTEXT:", "").strip()]
        elif raw_line.startswith("ANSWERS:"):
            if section == "CONTEXT":
                context = "\n".join(buffer).strip()
            elif section == "INPUT":
                question = "\n".join(buffer).strip()
            section, buffer = "ANSWERS", [raw_line.replace("ANSWERS:", "").strip()]
        else:
            buffer.append(raw_line.strip())

    if section == "INPUT":
        question = "\n".join(buffer).strip()
    elif section == "CONTEXT":
        context = "\n".join(buffer).strip()
    elif section == "ANSWERS":
        answers = "\n".join(buffer).strip()

    query = PROMPTS_QA_CONTEXT["query"][prompt_language].format(context=context, question=question)
    return Doc(
        instruction=PROMPTS_QA_CONTEXT["instruction"][prompt_language],
        task_name=task_name,
        query=query,
        choices=[answers],
        gold_index=0,
        specific={"gold": [answers]},
    )


def prompt_squad(line, task_name=None):
    context = line["context"]
    question = line["question"]
    query = PROMPTS_QA_CONTEXT["query"][prompt_language].format(context=context, question=question)
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["answer"]],
        gold_index=0,
        instruction=PROMPTS_QA_CONTEXT["instruction"][prompt_language],
    )


PROMPTS_CONTEXT_MCQA = {
    "instruction": {
        "hy": "Ընտրիր ճիշտ պատասխանը տրված տարբերակներից` օգտվելով կոնտեքստից։\n",
        "en": "Choose the correct answer from the given options using the context.\n",
    },
    "query": {
        "hy": ("Կոնտեքստ\n{context}\n\nՀարց\n{question}\nՃիշտ տարբերակ:"),
        "en": ("Context: {context}\nQuestion: {question}\nCorrect Option:"),
    },
}

PROMPTS_DREAM = {
    "instruction": {
        "hy": "Պատասխանեք հարցին՝ հիմնվելով երկխոսության վրա։\n",
        "en": "Answer the question based on the dialogue.\n",
    },
    "query": {
        "hy": ("Երկխոսություն\n {dialogue}\n\nՀարց\n{question}\nՃիշտ տարբերակ:"),
        "en": ("Dialogue:\n{dialogue}\nQuestion: {question}\nCorrect Option:"),
    },
}


def prompt_belebele(line, task_name=None):
    passage = line["flores_passage"]
    question = line["question"]
    choices = [
        line["mc_answer1"],
        line["mc_answer2"],
        line["mc_answer3"],
        line["mc_answer4"],
    ]
    gold_index = line["correct_answer_num"] - 1

    choice_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, opt in enumerate(choices):
        question += f"\n({choice_map[i]}) {opt}"

    query = PROMPTS_CONTEXT_MCQA["query"][prompt_language].format(context=passage, question=question)
    instruction = PROMPTS_CONTEXT_MCQA["instruction"][prompt_language]

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=instruction,
        specific={"answer": choice_map[gold_index]},
    )


def prompt_scientific(line, task_name=None):
    context = line["context"]
    question = line["question"]
    choices = line["choices"]
    gold_index = line["gold_index"]

    choice_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, opt in enumerate(choices):
        question += f"\n({choice_map[i]}) {opt}"

    query = PROMPTS_CONTEXT_MCQA["query"][prompt_language].format(context=context, question=question)
    instruction = PROMPTS_CONTEXT_MCQA["instruction"][prompt_language]

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=instruction,
        specific={"answer": choice_map[gold_index]},
    )


def prompt_syndarin(line, task_name=None):
    context = line["paragraph"]
    question = line["question"]
    choices = [
        line["answer_candidate_1"],
        line["answer_candidate_2"],
        line["answer_candidate_3"],
        line["answer_candidate_4"],
    ]
    gold_index = choices.index(line["correct_answer"])

    choice_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, opt in enumerate(choices):
        question += f"\n({choice_map[i]}) {opt}"

    query = PROMPTS_CONTEXT_MCQA["query"][prompt_language].format(context=context, question=question)
    instruction = PROMPTS_CONTEXT_MCQA["instruction"][prompt_language]

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=instruction,
        specific={"answer": choice_map[gold_index]},
    )


def prompt_dream(line, task_name=None):
    dialogue = line["dialogue"]
    if isinstance(dialogue, list):
        dialogue = "\n".join(dialogue)

    question = line["question"]
    choices = line["choices"]
    gold_index = line["label"]

    choice_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, opt in enumerate(choices):
        question += f"\n({choice_map[i]}) {opt}"

    query = PROMPTS_DREAM["query"][prompt_language].format(dialogue=dialogue, question=question)
    instruction = PROMPTS_DREAM["instruction"][prompt_language]

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=instruction,
        specific={"answer": choice_map[gold_index]},
    )


PROMPTS_MCQA = {
    "instruction": {
        "hy": "Պատասխանիր հարցին՝ ընտրելով ճիշտ տարբերակը։\n",
        "en": "Answer the question by choosing the correct option.\n",
    },
    "query": {
        "hy": "{question}\nՃիշտ տարբերակ:",
        "en": "{question}\nCorrect Option:",
    },
}


def prompt_hartak_mcqa(line, task_name=None):
    choices_text = [line["answer"]] + line["distractors"]
    query = PROMPTS_MCQA["query"][prompt_language].format(question=line["question"])

    choice_map = "ABCD"
    for i, opt in enumerate(choices_text):
        query += f"\n({choice_map[i]}) {opt}"

    query += "\n"

    gold_index = 0
    choices = [f" {c}" for c in choices_text]
    instruction = PROMPTS_MCQA["instruction"][prompt_language]

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=instruction,
        specific={"answer": choice_map[gold_index]},
    )


def prompt_include(line, task_name=None):
    choices_text = [
        line["option_a"],
        line["option_b"],
        line["option_c"],
        line["option_d"],
    ]
    query = PROMPTS_MCQA["query"][prompt_language].format(question=line["question"])

    choice_map = "ABCD"
    for i, opt in enumerate(choices_text):
        query += f"\n({choice_map[i]}) {opt}"

    query += "\n"

    gold_index = line["answer"]
    choices = [f" {c}" for c in choices_text]
    instruction = PROMPTS_MCQA["instruction"][prompt_language]

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=instruction,
        specific={"answer": choice_map[gold_index]},
    )


UPOS_TAGS = [
    "Գոյական",
    "Ածական",
    "Բայ",
    "Մակբայ",
    "Թիվ",
    "Դերանուն",
    "Կապ / Կապակցիչ",
    "Մասնիկ",
    "Ձայնարկություն",
]

BIO_TAGS = ["PER", "ORG", "LOC"]

PROMPTS_NER = {
    "instruction": {
        "hy": (
            "Տեքստում գտիր անվանական սուբյեկտները (named entities) և նշիր դրանց "
            "տեսակը օգտվելով միայն առկա թեգերի ցանկից։ Պատասխանը վերադարձրու JSON array ֆորմատով, "
            "որտեղ յուրաքանչյուր օբյեկտ ունի 'tag' և 'entity' բանալիներ: Վերադարձրու միայն մեկ JSON array, "
            "առանց կրկնությունների: Եթե տեքստում չկան թեգեր, ապա վերադարձրու '[]':\n"
        ),
        "en": (
            "Find the named entities in the text and specify their type using only the provided tags list. "
            "Return the answer as a single JSON array where each object has 'tag' and 'entity' keys. "
            "Return only one JSON array without repetition. "
            "If there are no named entities in the text, return '[]':\n"
        ),
    },
    "query": {
        "hy": ("Տեքստ\n{text}\n\nՀնարավոր թեգեր՝ {tags}"),
        "en": ("Text: {text}\nPossible tags: {tags}\n"),
    },
}

PROMPTS_UPOS = {
    "instruction": {
        "hy": (
            "Տրված բառի համար որոշիր ճիշտ խոսքի մասը՝ օգտվելով միայն առկա խոսքի մասերի ցանկից։ "
            'Պատասխանը գրիր այս ձևաչափով՝ "Այս բառի խոսքի մասն է <խոսքի մաս>"\n'
        ),
        "en": (
            "For the given word, determine the correct part of speech using the provided list.\n"
            'Write the answer in this format: "The part of speech of this word is <part of speech>".\n'
        ),
    },
    "query": {
        "hy": ("Տրված բառը՝ '{form}'\nԽոսքի մասեր՝ {tags}\n"),
        "en": ("Given word: '{form}'\nParts of speech: {tags}\n"),
    },
}


def prompt_finer(line, task_name=None):
    gold = [(t, ty) for t, ty in line["gold_entities"]]
    tag_pool = sorted({ty for _, ty in gold})
    query = PROMPTS_NER["query"][prompt_language].format(text=line["text"], tags=", ".join(tag_pool))
    return Doc(
        instruction=PROMPTS_NER["instruction"][prompt_language],
        task_name=task_name,
        query=query,
        choices=None,
        gold_index=None,
        specific={"gold_entities": gold},
    )


def prompt_pioner(line, task_name=None):
    tokens, tags = line["tokens"], line["ner_tags"]
    gold, current, current_type = [], [], None

    for tok, tag in zip(tokens, tags):
        if tag != "O":
            if current_type == tag:
                current.append(tok)
            else:
                if current:
                    gold.append((" ".join(current), current_type))
                current, current_type = [tok], tag
        else:
            if current:
                gold.append((" ".join(current), current_type))
                current, current_type = [], None
    if current:
        gold.append((" ".join(current), current_type))

    query = PROMPTS_NER["query"][prompt_language].format(
        text=" ".join(tokens),
        tags=", ".join(BIO_TAGS),
    )
    return Doc(
        instruction=PROMPTS_NER["instruction"][prompt_language],
        task_name=task_name,
        query=query,
        choices=None,
        gold_index=None,
        specific={"gold_entities": gold},
    )


def prompt_ud_armtdp(line, task_name=None):
    word = line["form"]
    gold = [(word, str(line["upos_hy"]))] if line.get("upos_hy") else []
    query = PROMPTS_UPOS["query"][prompt_language].format(form=word, tags=", ".join(UPOS_TAGS))
    return Doc(
        instruction=PROMPTS_UPOS["instruction"][prompt_language],
        task_name=task_name,
        query=query,
        choices=None,
        gold_index=None,
        specific={"gold_entities": gold},
    )


def prompt_qa(line, task_name=None):
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[line["answer"]],
        gold_index=0,
        instruction="",
    )


PROMPTS_EMAIL_SUM = {
    "instruction": {
        "hy": "Ամփոփիր էլեկտրոնային նամակը {num_sents} նախադասությամբ։\n",
        "en": "Generate a summary for the email in {num_sents} sentences.\n",
    },
    "query": {"hy": "Էլեկտրոնային նամակ\n{email}", "en": "Email: {email}"},
}

PROMPTS_CONV_SUM = {
    "instruction": {
        "hy": "Ամփոփիր երկխոսությունը {num_sents} նախադասությամբ։\n",
        "en": "Generate a summary for the dialogue in {num_sents} sentences.\n",
    },
    "query": {"hy": "Երկխոսություն\n{dialogue}", "en": "Dialogue:\n{dialogue}"},
}

PROMPTS_PARAPHRASE = {
    "instruction": {
        "hy": "Վերաշարադրիր նախադասությունը և վերադարձու միայն փոփոխված տեքստը։\n",
        "en": "Paraphrase the sentence and return only the paraphrased text.\n",
    },
    "query": {"hy": "{text}", "en": "{text}"},
}

PROMPTS_TRANSLATION = {
    "instruction": {
        "hy": "Թարգմանիր տրված անգլերեն տեքստը հայերեն և վերադարձու միայն թարգմանված տարբերակը։\n",
        "en": "Translate the given English text into Armenian and return the translated text only.\n",
    },
    "query": {
        "hy": "Անգլերեն: {eng}\n\nՀայերեն:",
        "en": "English: {eng}\n\nArmenian:",
    },
}


def count_sentences(text: str) -> int:
    if not text:
        return 0
    normalized = text.replace(":", "։")
    parts = [p.strip() for p in normalized.split("։") if p.strip()]
    return len(parts)


def prompt_email_sum(line, task_name=None):
    num_sents = count_sentences(line["summary"])
    query = PROMPTS_EMAIL_SUM["query"][prompt_language].format(email=line["email"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["summary"]],
        gold_index=0,
        instruction=PROMPTS_EMAIL_SUM["instruction"][prompt_language].format(num_sents=num_sents),
    )


def prompt_conv_sum(line, task_name=None):
    num_sents = count_sentences(line["summary"])
    query = PROMPTS_CONV_SUM["query"][prompt_language].format(dialogue=line["dialogue"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["summary"]],
        gold_index=0,
        instruction=PROMPTS_CONV_SUM["instruction"][prompt_language].format(num_sents=num_sents),
    )


def prompt_paraphrase(line, task_name=None):
    query = PROMPTS_PARAPHRASE["query"][prompt_language].format(text=line["text"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=line["paraphrases"],
        gold_index=0,
        instruction=PROMPTS_PARAPHRASE["instruction"][prompt_language],
    )


def prompt_translation(line, task_name=None):
    query = PROMPTS_TRANSLATION["query"][prompt_language].format(eng=line["eng"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["hy"]],
        gold_index=0,
        instruction=PROMPTS_TRANSLATION["instruction"][prompt_language],
    )


PUNCTUATION_CHARS = {",", "՝", ":", "։", "`"}

PROMPTS_SPACE_FIX = {
    "instruction": {
        "hy": (
            "Ավելացրու տեքստում պակաս բացատները և վերադարձրու ուղղված տարբերակը "
            "առանց որևէ մեկնաբանության և բացատրության։\n"
        ),
        "en": (
            "Add the missing spaces in the text and return only the corrected version without any comments or explanations.\n"
        ),
    },
    "query": {
        "hy": ("Տեքստ: {corrupted}\nՊատասխան:"),
        "en": ("Text: {corrupted}\nAnswer:"),
    },
}

PROMPTS_PUNCTUATION = {
    "instruction": {
        "hy": (
            "Ուղղիր (ավելացրու) տեքստում բաց թողնված կետադրական նշանները և վերադարձրու "
            "միայն ուղղված տարբերակը առանց որևէ մեկնաբանության և բացատրության։\n"
        ),
        "en": (
            "Correct (add) the missing punctuation marks in the text and return only the corrected version without any comments or explanations.\n"
        ),
    },
    "query": {
        "hy": ("Տեքստ: {corrupted}\nՊատասխան:"),
        "en": ("Text: {corrupted}\nAnswer:"),
    },
}


def mean_corpus_level(items):
    return float(np.mean(items)) if items else 0.0


def extract_fixed_text(model_response):
    raw_text = (
        model_response.text_post_processed[0]
        if hasattr(model_response, "text_post_processed") and model_response.text_post_processed
        else (model_response.text[0] if hasattr(model_response, "text") and model_response.text else "")
    ).strip()

    match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                if "ուղղված տարբերակ" in parsed:
                    return parsed["ուղղված տարբերակ"].strip()
                elif "corrected version" in parsed:
                    return parsed["corrected version"].strip()
        except Exception:
            pass
    return raw_text


def prompt_space_fix(line, task_name=None):
    gold = line["gold"]
    corrupted = line["corrupted_spaces"]
    query = PROMPTS_SPACE_FIX["query"][prompt_language].format(corrupted=corrupted)
    return Doc(
        task_name=task_name,
        query=query,
        choices=None,
        gold_index=None,
        instruction=PROMPTS_SPACE_FIX["instruction"][prompt_language],
        specific={"gold": gold},
    )


def prompt_punctuation(line, task_name=None):
    gold = line["gold"]
    corrupted = line["corrupted_punctuation"]
    query = PROMPTS_PUNCTUATION["query"][prompt_language].format(corrupted=corrupted)
    return Doc(
        task_name=task_name,
        query=query,
        choices=[gold],
        gold_index=0,
        instruction=PROMPTS_PUNCTUATION["instruction"][prompt_language],
    )


class SpaceAccuracyComputation(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs):
        gold = doc.specific["gold"].strip()
        pred = extract_fixed_text(model_response)
        gold_words, pred_words = gold.split(), pred.split()
        matched = 0
        for g in gold_words:
            if not pred_words:
                break
            if g in pred_words:
                pred_words.remove(g)
                matched += 1
                continue
            found_idx = next((i for i, p in enumerate(pred_words) if p.endswith(g)), None)
            if found_idx is not None:
                matched += 1
                pred_words.pop(found_idx)
        return matched / len(gold_words) if gold_words else 0.0


class PunctuationAccuracyComputation(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs):
        gold = doc.choices[0].strip()
        pred = extract_fixed_text(model_response).strip()

        if gold == pred:
            return 1.0

        def remove_punct(text: str):
            return "".join(ch for ch in text if ch not in PUNCTUATION_CHARS)

        gold_nopunct = remove_punct(gold)
        pred_nopunct = remove_punct(pred)

        if gold_nopunct != pred_nopunct:
            return 0.0

        gold_words, pred_words = gold.split(), pred.split()

        if len(gold_words) != len(pred_words):
            return 0.0

        total_punctuated, correct = 0, 0
        for gw, pw in zip(gold_words, pred_words):
            if any(ch in PUNCTUATION_CHARS for ch in gw):
                total_punctuated += 1
                if gw == pw:
                    correct += 1

        return correct / total_punctuated if total_punctuated > 0 else 1.0


space_accuracy_metric = SampleLevelMetric(
    metric_name="space_accuracy",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=SpaceAccuracyComputation(),
    corpus_level_fn=mean_corpus_level,
)

punctuation_accuracy_metric = SampleLevelMetric(
    metric_name="punctuation_accuracy",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=PunctuationAccuracyComputation(),
    corpus_level_fn=mean_corpus_level,
)


class ArmenianEvalTask(LightevalTaskConfig):
    def __init__(
        self,
        short_name,
        hf_subset,
        prompt_function,
        metrics,
        generation=False,
        few_shots_split=None,
        few_shots_select=None,
        evaluation_splits=None,
        hf_avail_splits=None,
    ):
        default_eval_splits = evaluation_splits if evaluation_splits is not None else ["train"]
        default_hf_splits = hf_avail_splits if hf_avail_splits is not None else ["train"]

        super().__init__(
            name=f"armenian:{short_name}",
            prompt_function=prompt_function,
            hf_repo="Metric-AI/ArmBench-LLM-data",
            hf_subset=hf_subset,
            metrics=metrics,
            hf_avail_splits=default_hf_splits,
            evaluation_splits=default_eval_splits,
            few_shots_split=few_shots_split,
            few_shots_select=few_shots_select,
            generation_size=2048 if generation else -1,
            stop_sequence=None,
        )


TASKS_TABLE = [
    ArmenianEvalTask(
        "topic-14class",
        "topic-14class",
        prompt_sib200,
        [armenian_mcqa_metric],
        generation=True,
    ),
    ArmenianEvalTask(
        "sentiment",
        "sentiment",
        prompt_sentiment,
        [armenian_mcqa_metric],
        generation=True,
    ),
    ArmenianEvalTask(
        "space_fix",
        "space_fix",
        prompt_space_fix,
        [space_accuracy_metric],
        generation=True,
    ),
    ArmenianEvalTask(
        "punctuation",
        "punctuation",
        prompt_punctuation,
        [punctuation_accuracy_metric],
        generation=True,
    ),
    ArmenianEvalTask("finer", "finer", prompt_finer, [ner_span_metric], generation=True),
    ArmenianEvalTask("pioner", "pioner", prompt_pioner, [ner_span_metric], generation=True),
    ArmenianEvalTask("pos", "pos", prompt_ud_armtdp, [pos_metric], generation=True),
    ArmenianEvalTask("arak", "simpleqa", prompt_qa, [Metrics.bleu], generation=True),
    ArmenianEvalTask(
        "ms_marco",
        "ms-marco-in-context-qa",
        prompt_ms_marco,
        [Metrics.bleu],
        generation=True,
    ),
    ArmenianEvalTask("squad", "squad-in-context-qa", prompt_squad, [Metrics.bleu], generation=True),
    ArmenianEvalTask(
        "belebele",
        "belebele-in-context-mcqa",
        prompt_belebele,
        [armenian_mcqa_metric],
        generation=True,
    ),
    ArmenianEvalTask(
        "scientific",
        "scientific-in-context-mcqa",
        prompt_scientific,
        [armenian_mcqa_metric],
        generation=True,
    ),
    ArmenianEvalTask(
        "syndarin",
        "syndarin-in-context-mcqa",
        prompt_syndarin,
        [armenian_mcqa_metric],
        generation=True,
    ),
    ArmenianEvalTask(
        "dream",
        "conversation-in-context-qa",
        prompt_dream,
        [armenian_mcqa_metric],
        generation=True,
    ),
    ArmenianEvalTask(
        "include",
        "include-mcqa",
        prompt_include,
        [armenian_mcqa_metric],
        generation=True,
    ),
    ArmenianEvalTask(
        "hartak",
        "public-services-mcqa",
        prompt_hartak_mcqa,
        [armenian_mcqa_metric],
        generation=True,
    ),
    ArmenianEvalTask(
        "email",
        "email-sum",
        prompt_email_sum,
        [Metrics.bleu, bert_score_arm],
        generation=True,
    ),
    ArmenianEvalTask(
        "conversation",
        "conversational-sum",
        prompt_conv_sum,
        [Metrics.bleu, bert_score_arm],
        generation=True,
    ),
    ArmenianEvalTask(
        "paraphrase",
        "paraphrase",
        prompt_paraphrase,
        [Metrics.bleu, bert_score_arm],
        generation=True,
    ),
    ArmenianEvalTask(
        "short_sentences_translation",
        "translation_short_sentences",
        prompt_translation,
        [Metrics.bleu, bert_score_arm],
        generation=True,
    ),
    ArmenianEvalTask(
        "mmlu_pro",
        "mmlu_pro",
        prompt_mmlu_pro,
        [armenian_mmlu_pro_metric],
        generation=True,
    ),
    ArmenianEvalTask(
        "exam_history",
        "exam_history",
        prompt_exam_history,
        [armenian_exam_metric],
        generation=True,
    ),
    ArmenianEvalTask(
        "exam_literature",
        "exam_literature",
        prompt_exam_literature,
        [armenian_exam_metric],
        generation=True,
    ),
    ArmenianEvalTask(
        "exam_math",
        "exam_math",
        prompt_exam_math,
        [armenian_exam_metric],
        generation=True,
    ),
]
