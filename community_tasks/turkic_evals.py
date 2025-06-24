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
Task to evaluate LLMs on TUMLU-mini benchmark: https://huggingface.co/datasets/jafarisbarov/TUMLU-mini

For more details, see the associated paper:

@misc{isbarov2025tumluunifiednativelanguage,
    title={{TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages}},
    author={Jafar Isbarov and Arofat Akhundjanova and Mammad Hajili and Kavsar Huseynova and Dmitry Gaynullin and Anar Rzayev and Osman Tursun and Ilshat Saetov and Rinat Kharisov and Saule Belginova and Ariana Kenbayeva and Amina Alisheva and Aizirek Turdubaeva and Abdullatif Köksal and Samir Rustamov and Duygu Ataman},
    year={2025},
    eprint={2502.11020},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2502.11020},
}
"""

import random
import re
from functools import partial
from typing import Any, Dict, List, Optional, Union

from lighteval.metrics.llm_as_judge import JudgeLM
from lighteval.metrics.metrics import Metric, MetricCategory, Metrics
from lighteval.metrics.utils.metric_utils import MetricUseCase
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# TUMLU
# fmt: off
TUMLU_SUBSETS = [
    "azerbaijani",
    "crimean-tatar",
    "karakalpak",
    "kazakh",
    "tatar",
    "turkish",
    "uyghur",
    "uzbek",
    "kyrgyz"
]
# fmt: on

INSTRUCTION_BY_LANGUAGE = {
    "azerbaijani": "Aşağıdakı sual çoxvariantlı sualdır. Düzgün cavabı seçin:\n\n",
    "crimean-tatar": "Aşağıdaki sual çoqtan-çoq cevaplı sualdir. Doğru cevapnı seçip alıñız:\n\n",
    "karakalpak": "Tómendegi soraw kóp tańlawlı soraw Tuwrı juwaptı saylań:\n\n",
    "kazakh": "Төмендегі сұрақ көп таңдау мүмкіндігі бар сұрақ. Дұрыс жауапты таңдаңыз:\n\n",
    "tatar": "Түбәндәге сорау - күп сорау. Дөрес җавапны сайлагыз:\n\n",
    "turkish": "Aşağıdaki soru çoktan seçmeli bir sorudur. Doğru cevabı seçin:\n\n",
    "uyghur": "تۆۋەندىكى سوئال كۆپ تاللاش سوئالى. توغرا جاۋابنى تاللاڭ:\n\n",
    "uzbek": "Quyidagi savol tanlovli savoldir. To‘g‘ri javobni tanlang:\n\n",
    "kyrgyz": "Төмөнкү суроо бир нече варианттуу суроо. Туура жоопту тандаңыз:\n\n",
}

ANSWER_BY_LANGUAGE = {
    "uzbek": "Javob:",
    "uzbek-cyrillic": "Жавоб",
    "crimean-tatar": "Cevap:",
    "crimean-tatar-cyrillic": "Джевап",
    "tatar": "Җавап:",
    "kazakh": "Жауап:",
    "kazakh-latin": "Jawap",
    "karakalpak": "Juwap:",
    "kyrgyz": "Жооп:",
    "turkish": "Cevap:",
    "uyghur": "جاۋاب:",
    "uyghur-latin": "Jawab:",
    "azerbaijani": "Cavab:",
}


def tumlu_pfn(line, task_name: str = None, language: str = None):
    instruction = INSTRUCTION_BY_LANGUAGE[language]

    # Create a list of valid choices with corresponding keys
    choices = line.get("choices")
    valid_keys = ["A", "B", "C", "D", "E"][: len(choices)]

    answer_index = valid_keys.index(line.get("answer"))

    # Construct the query
    query = f"{instruction}{line['question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(valid_keys, choices)])
    query += ANSWER_BY_LANGUAGE[language]

    return Doc(
        task_name=task_name,
        query=query,
        choices=valid_keys,  # Return only valid choices
        gold_index=answer_index,  # Correct index
        instruction=instruction,
    )


class CustomTUMLUTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=partial(tumlu_pfn, language=hf_subset),
            hf_repo="jafarisbarov/TUMLU-mini",
            metric=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["test", "dev"],
            evaluation_splits=["test"],
            few_shots_split=["dev"],
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=False,
            version=0,
        )


TUMLU_TASKS = [CustomTUMLUTask(name=f"tumlu:{subset}", hf_subset=subset) for subset in TUMLU_SUBSETS]

TASKS_TABLE = TUMLU_TASKS
