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

from typing import Literal

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


PUNCT = "-.!?،؟‽,。，？؟،। "
NLI_2_TEMPLATE = "{premise}{comma}{sentence_space}{question_word}{question_mark}{sentence_space}{label}{comma}"
NLI_2_CONT_TEMPLATE = "{sentence_space}{hypothesis}"


def decapitalize(word: str):
    if len(word) == 0:
        return word
    return word[0].lower() + word[1:]


def capitalize(word: str):
    if len(word) == 0:
        return word
    return word[0].upper() + word[1:]


def get_xnli_prompt():
    prompter = _get_nli_prompt(["entailment", "contradiction"])
    return lambda line, task_name: prompter(
        task_name,
        line["premise"],
        line["hypothesis"],
        int(line["label"]),
    )


def _get_nli_prompt(pos_labels: list[Literal["entailment", "contradiction"]], template_version: Literal[1, 2] = 1):
    """
    Create an NLI prompt function based on the specified language, labels, and template version.

    Args:
        lang: The language to use for the prompt.
        pos_labels: List of possible labels, order matters for gold_index.
        template_version: Which template version to use ("v1" or "v2").

    Returns:
        A function that generates an NLI prompt given task name, premise, hypothesis, and label.
    """
    # 0 is entailment contradiction is 2
    label_remap = {
        0: 0,
        2: 1,
    }

    def _get_pos_label(label: Literal["entailment", "contradiction"]) -> str | None:
        if label == "entailment":
            return "是的"
        elif label == "contradiction":
            return "不是"

    labels = [_get_pos_label(label) for label in pos_labels if _get_pos_label(label) is not None]

    def nli_prompt(task_name: str, premise: str, hypothesis: str, label: int):
        if label not in [0, 2]:
            return None

        label = label_remap[label]

        premise = capitalize(premise.rstrip(PUNCT))
        hypothesis = hypothesis.rstrip(PUNCT)
        if template_version == 1:
            hypothesis = capitalize(hypothesis)
        else:
            hypothesis = decapitalize(hypothesis)

        query_template = NLI_2_TEMPLATE
        cont_template = NLI_2_CONT_TEMPLATE

        queries = [
            query_template.format(
                premise=premise,
                hypothesis=hypothesis,
                full_stop="。",
                question_word="是不是",
                question_mark="？",
                comma="，",
                sentence_space="",
                label=label,
            )
            for label in labels
        ]

        choices = [cont_template.format(hypothesis=hypothesis, sentence_space="") for _ in labels]

        # We want as many choices as queries
        return Doc(
            task_name=task_name, query="", choices=choices, gold_index=label, specific={"local_contexts": queries}
        )

    return nli_prompt


TASKS_TABLE = [
    LightevalTaskConfig(
        name="xnli-2.0-bool",
        suite=["custom"],
        prompt_function=get_xnli_prompt(),
        hf_repo="Harsit/xnli2.0_chinese",
        hf_subset="default",
        evaluation_splits=["test"],
        few_shots_split=None,
        metric=(Metrics.loglikelihood_acc_multicontext,),
    )
]
