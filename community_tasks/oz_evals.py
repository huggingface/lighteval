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
Custom evaluation tasks for lighteval.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
This module implements the OZ Eval task for General Knowledge for Serbian language.
See: https://huggingface.co/datasets/DjMel/oz-eval

In order to have comparable results to ours, please do not forget to run with --use_chat_template
"""

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def prompt_fn_oz_eval_task(line, task_name: str = None):
    query_template = """Pitanje: {question}\n
    Ponuđeni odgovori:
    A. {choice_a}
    B. {choice_b}
    C. {choice_c}
    D. {choice_d}
    E. {choice_e}

    Krajnji odgovor:"""

    import ast

    options = line["options"].replace("\n", "")
    options = ast.literal_eval(options)

    query = query_template.format(
        question=line["questions"],
        choice_a=options[0],
        choice_b=options[1],
        choice_c=options[2],
        choice_d=options[3],
        choice_e=options[4],
    )

    choices = ["A", "B", "C", "D", "E"]
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=choices.index(line["answer"]),
    )


oz_eval_task = LightevalTaskConfig(
    name="serbian_rag_eval:oi_task",
    prompt_function=prompt_fn_oz_eval_task,
    suite=["community"],
    hf_repo="DjMel/oz-eval",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="test",
    few_shots_select="sequential",
    metric=["loglikelihood_acc"],
    version=0,
    trust_dataset=True,
)


# STORE YOUR EVALS
TASKS_TABLE = [oz_eval_task]


if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
