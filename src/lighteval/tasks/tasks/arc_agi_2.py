"""
name:
ArcAgi 2

dataset:
arc-agi-community/arc-agi-2

abstract:
ARC-AGI tasks are a series of three to five input and output tasks followed by a
final task with only the input listed. Each task tests the utilization of a
specific learned skill based on a minimal number of cognitive priors.
In their native form, tasks are a JSON lists of integers. These JSON can also be
represented visually as a grid of colors using an ARC-AGI task viewer. You can
view an example of a task here.
A successful submission is a pixel-perfect description (color and position) of
the final task's output.
100% of tasks in the ARC-AGI-2 dataset were solved by a minimim of 2 people in
less than or equal to 2 attempts (many were solved more). ARC-AGI-2 is more
difficult for AI.

languages:
english

tags:
multiple-choice

paper:
https://arcprize.org/guide
"""

import json
from textwrap import dedent

from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# query from: https://github.com/arcprize/model_baseline/blob/main/src/prompts/system_prompt.txt
PROMPT_TEMPLATE = dedent("""
You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.

Respond in the format of the training output examples

--Training Examples--
{training_examples}
--End of Training Examples--
--Test Input--
{test_input}
--End of Test Input--

Your response:""")


def __convert_2d_list_to_string(list_of_lists: list[list[int]]) -> str:
    """Convert a list of lists to a string"""
    string_list = ""
    for row in list_of_lists:
        string_list += json.dumps(row) + "\n"
    return string_list


def arc_agi_2_prompt(line, task_name: str = None):
    training_pairs = line["fewshots"]
    training_examples = ""
    for i, pair in enumerate(training_pairs):
        training_examples += f"--Example {i}-- \n\n INPUT: \n\n"
        training_examples += __convert_2d_list_to_string(pair["input"]) + "\n\n"
        training_examples += "OUTPUT: \n\n"
        training_examples += __convert_2d_list_to_string(pair["output"]) + "\n\n"

    test_input = __convert_2d_list_to_string(line["question"][0]["input"])

    gold = str(line["question"][0]["output"])
    query = PROMPT_TEMPLATE.format(training_examples=training_examples, test_input=test_input)

    return Doc(
        task_name=task_name,
        query=query,
        choices=[gold],
        gold_index=0,
    )


def record_to_sample(record):
    training_pairs = record["fewshots"]
    training_examples = ""

    for i, pair in enumerate(training_pairs):
        training_examples += f"--Example {i}-- \n\n INPUT: \n\n"
        training_examples += __convert_2d_list_to_string(pair["input"]) + "\n\n"
        training_examples += "OUTPUT: \n\n"
        training_examples += __convert_2d_list_to_string(pair["output"]) + "\n\n"

    test_input = __convert_2d_list_to_string(record["question"][0]["input"])
    query = PROMPT_TEMPLATE.format(training_examples=training_examples, test_input=test_input)

    target = str(record["question"][0]["output"])

    return Sample(
        input=query,
        target=target,
    )


arc_agi_2 = LightevalTaskConfig(
    name="arc_agi_2",
    prompt_function=arc_agi_2_prompt,
    hf_repo="arc-agi-community/arc-agi-2",
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[Metrics.exact_match],
    stop_sequence=None,
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=exact(),
    version=0,
)

TASKS_TABLE = [arc_agi_2]
