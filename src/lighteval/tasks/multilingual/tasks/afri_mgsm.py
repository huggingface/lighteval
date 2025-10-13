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


from lighteval.metrics.dynamic_metrics import (
    MultilingualQuasiExactMatchMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


# African MGSM: MGSM for African Languages
# From https://arxiv.org/abs/2406.03368. Human translated MGSM.

TASKS_TABLE = []


afri_mgsm_tasks = [
    LightevalTaskConfig(
        name=f"afri_mgsm_{language.value}",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                # The cot is available but we have no use:
                # line["answer"]
                "choices": [str(line["answer_number"])],
            },
        ),
        suite=("lighteval",),
        hf_repo="masakhane/afrimgsm",
        hf_subset=language.value,
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=25,
        metrics=[
            MultilingualQuasiExactMatchMetric(language, "full"),
        ],
        stop_sequence=("\n",),
    )
    for language in [
        Language.AMHARIC,
        # Language.EWE,
        Language.FRENCH,
        # Language.HAUSA,
        # Language.IGBO,
        # Language.KINYARWANDA,
        # Language.LINGALA,
        # Language.LUGANDA,
        # Language.OROMO,
        # Language.SHONA,
        # Language.SOTHO,
        Language.SWAHILI,
        # Language.TWI,
        # Language.WOLOF,
        # Language.XHOSA,
        Language.YORUBA,
        # Language.ZULU,
    ]
]
