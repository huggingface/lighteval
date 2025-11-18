"""
name:
Turkic Evals

dataset:
jafarisbarov/TUMLU-mini

abstract:
TUMLU-mini is a benchmark for Turkic language understanding, comprising 1,000
prompts organized into 10 subsets.

languages:
turkic

tags:
knowledge, multiple-choice

paper:
https://arxiv.org/abs/2502.11020
"""

from functools import partial

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
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
            metrics=[Metrics.loglikelihood_acc(sample_params={"logprob_normalization": LogProbCharNorm()})],
            hf_avail_splits=["test", "dev"],
            evaluation_splits=["test"],
            few_shots_split=["dev"],
            few_shots_select="sequential",
            generation_size=-1,
            stop_sequence=None,
            version=0,
        )


TUMLU_TASKS = [CustomTUMLUTask(name=f"tumlu:{subset}", hf_subset=subset) for subset in TUMLU_SUBSETS]

TASKS_TABLE = TUMLU_TASKS
