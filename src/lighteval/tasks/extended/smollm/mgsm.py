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


from functools import partial

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


TASKS_TABLE = []


def mgsm_prompt(line, task_name: str = None, language: Language = Language.ENGLISH):
    # Taken from: https://github.com/openai/simple-evals/blob/3ec4e9b5ae3931a1858580e2fd3ce80c7fcbe1d9/mgsm_eval.py#L32C1-L66C2
    LANG_TO_INSTRUCTIONS = {
        "eng": """
Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

{input}
""".strip(),
        "ben": """
এই গণিতের সমস্যাটি সমাধান করুন। চূড়ান্ত উত্তর দেওয়ার আগে যুক্তিসম্পন্ন পদক্ষেপ প্রদান করুন। চূড়ান্ত উত্তরটি একক সংখ্যা হিসাবে "উত্তর:" এর পরে শেষ লাইনে দিন। "উত্তর:" এর পরে অন্য কিছু যুক্ত করবেন না।.

{input}
""".strip(),
        "deu": """
Löse dieses Mathematikproblem. Gib die Schritte zur Begründung an, bevor du die endgültige Antwort in der letzten Zeile alleine im Format "Antwort:" gibst. Füge nichts anderes als die ganzzahlige Antwort nach "Antwort:" hinzu.

{input}
""".strip(),
        "spa": """
Resuelve este problema matemático. Proporciona los pasos de razonamiento antes de dar la respuesta final en la última línea por sí misma en el formato de "Respuesta:". No añadas nada más que la respuesta entera después de "Respuesta:".

{input}
""".strip(),
        "fra": """
Résolvez ce problème de mathématiques. Donnez les étapes de raisonnement avant de fournir la réponse finale sur la dernière ligne elle-même dans le format de "Réponse:". N'ajoutez rien d'autre que la réponse entière après "Réponse:".

{input}
""".strip(),
        "jpn": """
の数学の問題を解いてください。最終的な答えを出す前に、解答の推論過程を記述してください。そして最後の行には "答え:" の形式で答えを記述し、その後には整数の答え以外何も追加しないでください。

{input}
""".strip(),
        "rus": """
Решите эту математическую задачу. Объясните шаги рассуждения перед тем, как дать окончательный ответ в последней строке сам по себе в формате "Ответ:". Не добавляйте ничего, кроме целочисленного ответа после "Ответ:".

{input}
""".strip(),
        "swa": """
Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".

{input}
""".strip(),
        "tel": """
ఈ గణిత సమస్యను పరిష్కరించండి. చివరి సమాధానాన్ని ఇవ్వదానికి ముందు తర్కాత్మక అదుగులను ఇవ్వండి. చివరి పంక్తిలో మాత్రమే 'సమాధానం:' అనే ఆకారంలో చివరి సమాధానాద్ని ఇవ్వండి సమాధానం: తర్వాత పూర్ణాంక సమాధానానికి తప్పించి ఎదేనా చేర్చవద్దు.

{input}
""".strip(),
        "tha": """
แก้ปัญหาคณิตศาสตร์นี้ ให้ให้ขั้นตอนการใช้เหตุผลก่อนที่จะให้คำตอบสุดท้ายในบรรทัดสุดท้ายโดยอยู่ในรูปแบบ "คำตอบ:" ไม่ควรเพิ่มอะไรนอกจากคำตอบที่เป็นจำนวนเต็มหลังจาก "คำตอบ:"

{input}
""".strip(),
        "zho": """
解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。

{input}
""".strip(),
    }

    query = LANG_TO_INSTRUCTIONS[language].format(
        input=line["question"],
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=[str(line["answer_number"])],
        gold_index=0,
    )


MGSM_TASKS = [
    LightevalTaskConfig(
        name=f"mgsm_instruct_{language.value}",
        prompt_function=partial(mgsm_prompt, language=language.value),
        suite=("lighteval",),
        hf_repo="juletxara/mgsm",
        hf_subset=lang,
        evaluation_splits=("test",),
        few_shots_split="train",
        metric=[
            Metrics.math_pass_at_1_1n,
        ],
        generation_size=32768,  # needed for reasoning models like R1
        stop_sequence=[],  # no stop sequence, will use eos token
    )
    for lang, language in [
        ("bn", Language.BENGALI),
        ("de", Language.GERMAN),
        ("en", Language.ENGLISH),
        ("es", Language.SPANISH),
        ("fr", Language.FRENCH),
        ("ja", Language.JAPANESE),
        ("ru", Language.RUSSIAN),
        ("sw", Language.SWAHILI),
        ("te", Language.TELUGU),
        ("th", Language.THAI),
        ("zh", Language.CHINESE),
    ]
]
TASKS_TABLE.extend(MGSM_TASKS)
