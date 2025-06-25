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
    ExprExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


chinese_answer_type_dict = {"Numerical": "数值", "Expression": "表达式", "Equation": "方程", "Interval": "区间"}
english_answer_type_dict = {
    "Numerical": "a numerical value",
    "Expression": "an expression",
    "Equation": "an equation",
    "Interval": "an interval",
}


def get_single_answer_type_text(answer_type, is_chinese):
    if "-" in answer_type:  # No need now
        answer_type = answer_type[: answer_type.find("-")]
    for t in ["Numerical", "Expression", "Equation", "Interval"]:
        if t in answer_type:
            if is_chinese:
                return chinese_answer_type_dict[t]
            else:
                return english_answer_type_dict[t]
    exit(f"Error parsing answer type {answer_type}!")


def get_answer_type_text(answer_type, is_chinese, multiple_answer):
    if (
        ("Need_human_evaluate" in answer_type) or ("Tuple" in answer_type)
    ):  # 'Tuple' has various meanings in different context, such as position or values of a series of variable, so it may lead to confusion to directly use 'tuple' in the prompt.
        full_answer_text = ""
    else:
        if not multiple_answer:
            answer_text = get_single_answer_type_text(answer_type, is_chinese)
            if is_chinese:
                full_answer_text = f"，答案类型为{answer_text}"
            else:
                full_answer_text = f"The answer of The problem should be {answer_text}. "
        else:
            if "," not in answer_type:  # Same answer type for all answers
                answer_text = get_single_answer_type_text(answer_type, is_chinese)
                if is_chinese:
                    full_answer_text = f"，题目有多个答案，答案类型均为{answer_text}"
                else:
                    full_answer_text = f"The problem has multiple answers, each of them should be {answer_text}. "
            else:
                answer_types = answer_type.split(",")
                answer_types = [get_single_answer_type_text(t, is_chinese) for t in answer_types]
                if len(set(answer_types)) == 1:
                    answer_text = answer_types[0]
                    if is_chinese:
                        full_answer_text = f"，题目有多个答案，答案类型均为{answer_text}"
                    else:
                        full_answer_text = f"The problem has multiple answers, each of them should be {answer_text}. "
                else:
                    if is_chinese:
                        answer_text = "、".join(answer_types)
                        full_answer_text = f"，题目有多个答案，答案类型分别为{answer_text}"
                    else:
                        answer_text = ", ".join(answer_types)
                        full_answer_text = (
                            f"The problem has multiple answers, with the answers in order being {answer_text}. "
                        )
    return full_answer_text


# Very specific task where there are no precise outputs but instead we test if the format obeys rules
def olympiad_bench_prompt(line, task_name: str = None):
    is_math = "Math" in line["subject"]
    subject = "Math" if is_math else "Physics"
    is_chinese = line["language"] == "Chinese"
    is_theorem_proving = "TP" in task_name
    unit = line["unit"]
    is_multiple_answer = line["is_multiple_answer"]

    if is_chinese:
        subject_content = "数学" if is_math else "物理"
        if is_theorem_proving:
            instruction = f"以下是中国{subject_content}竞赛中的证明题。请根据题目的要求，运用逻辑推理及常用定理证明题目中的命题。证明过程中使用的变量和公式请使用LaTeX格式表示。"
        else:
            answer_type_text = get_answer_type_text(
                line["answer_type"], is_chinese=True, multiple_answer=is_multiple_answer
            )
            if is_multiple_answer:
                multiple_answer_text = "\\boxed{用英文逗号连接的多个答案}"
            else:
                multiple_answer_text = "\\boxed{答案}"
            unit_text = ""
            if unit:
                multiple_answer_text += "(单位)"
                unit_text = "，注意答案的单位不要放在\\boxed{}中"
            instruction = f"以下是中国{subject_content}竞赛中的解答题{answer_type_text}。请根据题目的要求和所提供的信息计算得出答案。解答过程和结果中使用的变量和公式请使用LaTeXæ ¼式表示。请在最后以“所以最终答案是{multiple_answer_text}。”显式给出结果{unit_text}。"
    else:
        if is_theorem_proving:
            instruction = f"The following is a theorem proving problem from an International {subject} competition. Please use logical reasoning and common theorems to prove the proposition in the problem according to the given requirements. Please use LaTeX format to represent the variables and formulas used in the proof."
        else:
            if is_multiple_answer:
                multiple_answer_text = "\\boxed{multiple answers connected with commas}"
            else:
                multiple_answer_text = "\\boxed{answer}"
            unit_text = ""
            if unit:
                multiple_answer_text += "(unit)"
                unit_text = ", note that the unit of the answer should not be included in \\boxed{}"

            answer_type_text = get_answer_type_text(
                line["answer_type"], is_chinese=False, multiple_answer=is_multiple_answer
            )

            instruction = f'The following is an open-ended problem from an International {subject} competition. {answer_type_text}Please calculate the answer according to the given requirements and the information provided. Please use LaTeX format to represent the variables and formulas used in the solution process and results. Please end your solution with "So the final answer is {multiple_answer_text}." and give the result explicitly{unit_text}.'

    choice = line["final_answer"]

    return Doc(
        task_name=task_name,
        query=instruction + "\n" + line["question"],
        choices=[choice],
        gold_index=0,
        instruction=instruction,
        specific={},
    )


# * OE: Open-ended questions
# * TP: Theorem proof problems
# * MM: Multimodal
# * TO: Text-only
# * physics: Physics problems
# * maths: Math problems
# * en: English
# * zh: Chinese
# * COMP: Competition problems
# * CEE: Chinese College Entrance Exam problems

question_type = ["OE"]  # TP  # TP cannot be replicated
multimodality = ["TO"]  # MM  # We do not allow multimodal tasks
subject = ["physics", "maths"]
language = ["en", "zh"]
source = ["COMP", "CEE"]

olympiad_bench_subsets = []

for qt in question_type:
    for mm in multimodality:
        for sub in subject:
            for lang in language:
                for src in source:
                    olympiad_bench_subsets.append(f"{qt}_{mm}_{sub}_{lang}_{src}")

available_subsets = [
    "OE_MM_maths_en_COMP",
    "OE_MM_maths_zh_CEE",
    "OE_MM_maths_zh_COMP",
    "OE_MM_physics_en_COMP",
    "OE_MM_physics_zh_CEE",
    "OE_TO_maths_en_COMP",
    "OE_TO_maths_zh_CEE",
    "OE_TO_maths_zh_COMP",
    "OE_TO_physics_en_COMP",
    "OE_TO_physics_zh_CEE",
    "TP_MM_maths_en_COMP",
    "TP_MM_maths_zh_CEE",
    "TP_MM_maths_zh_COMP",
    "TP_MM_physics_en_COMP",
    "TP_TO_maths_en_COMP",
    "TP_TO_maths_zh_CEE",
    "TP_TO_maths_zh_COMP",
    "TP_TO_physics_en_COMP",
]

olympiad_bench_subsets = set(olympiad_bench_subsets).intersection(available_subsets)

extraction_targets = [ExprExtractionConfig(), LatexExtractionConfig()]

metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=extraction_targets,
    pred_extraction_target=extraction_targets,
    precision=6,
)

task_configs = []

for subset in olympiad_bench_subsets:
    # We create the task config
    task_configs.append(
        LightevalTaskConfig(
            name="olympiad_bench:" + subset,
            prompt_function=olympiad_bench_prompt,
            suite=["extended"],
            hf_repo="Hothan/OlympiadBench",
            hf_subset=subset,
            metrics=[metric],
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split="train",
            few_shots_select="random_sampling",
            generation_size=2048,
            stop_sequence=[],  # no stop sequence, will use eot token
            version=1,
        )
    )


TASKS_TABLE = task_configs
