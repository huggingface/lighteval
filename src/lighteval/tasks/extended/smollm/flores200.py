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


from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


TASKS_TABLE = []


class Flores:
    def __init__(self, lang1, lang2):
        self.lang1 = lang1
        self.lang2 = lang2

    def prompt(self, line, task_name: str = None):
        instruction = f"Translate the following query from {self.lang1} to {self.lang2}. Think step by step before answering.\n\n"
        source = (line[f"sentence_{self.lang1}"],)
        target = (line[f"sentence_{self.lang2}"],)
        query = f"{instruction}###\nQuery:\n{source}"

        return Doc(
            task_name=task_name,
            query=query,
            choices=[target],
            gold_index=0,
            instruction=instruction,
        )


flores200_tasks = [
    LightevalTaskConfig(
        name=f"flores200:{lang1}-{lang2}",
        prompt_function=Flores(lang1, lang2).prompt,
        suite=("lighteval",),
        hf_repo="facebook/flores",
        hf_subset=f"{lang1}-{lang2}",
        hf_avail_splits=["dev", "devtest"],
        evaluation_splits=["devtest"],
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=30000,
        metric=[Metrics.chrf_plus, Metrics.bleu, Metrics.bleu_1, Metrics.bleu_4],
        stop_sequence=[],
        trust_dataset=True,
        version=0,
    )
    for (lang1, lang2) in [
        ("fra_Latn", "eng_Latn"),
        ("eng_Latn", "fra_Latn"),
        ("spa_Latn", "eng_Latn"),
        ("eng_Latn", "spa_Latn"),
        ("deu_Latn", "eng_Latn"),
        ("eng_Latn", "deu_Latn"),
        ("ita_Latn", "eng_Latn"),
        ("eng_Latn", "ita_Latn"),
        ("por_Latn", "eng_Latn"),
        ("eng_Latn", "por_Latn"),
        ("zho_Hans", "eng_Latn"),
        ("eng_Latn", "zho_Hans"),
        ("rus_Cyrl", "eng_Latn"),
        ("eng_Latn", "rus_Cyrl"),
        ("arb_Arab", "eng_Latn"),
        ("eng_Latn", "arb_Arab"),
    ]
]

TASKS_TABLE.extend(
    [
        *flores200_tasks,
    ]
)
