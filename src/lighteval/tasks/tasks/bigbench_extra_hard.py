"""
name:
BIG-Bench Extra Hard

dataset:
jgyasu/bbeh

abstract:
BIG-Bench Extra Hard (BBEH) is a successor to BIG-Bench Hard (BBH), created to evaluate large
language models on substantially more difficult general-reasoning tasks. Each BBH task is replaced
with a new task targeting the same underlying reasoning skill but at a significantly higher difficulty.

languages:
english

tags:
reasoning

paper:
https://arxiv.org/abs/2502.19187

starred:
true
"""

from inspect_ai.dataset import Sample
from inspect_ai.scorer import answer
from inspect_ai.solver import generate, system_message

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def bbeh_prompt(line, task_name: str = None):
    line = {k: v for k, v in line.items() if v is not None}

    query = "Question: \n"
    query += line["input"]
    query += "\nAnswer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=line["target"],
        gold_index=0,
        instruction="",
    )


def record_to_sample(record):
    query = f"{record['input']}"
    target = record["target"]

    return Sample(input=query, target=target)


SYSTEM_MESSAGE = """Submit your answer in the following format:
ANSWER: {your answer}
"""


COMMON_TASK_ARGS = {
    "prompt_function": bbeh_prompt,
    "hf_repo": "jgyasu/bbeh",
    "hf_avail_splits": ["train"],
    "evaluation_splits": ["train"],
    "few_shots_split": None,
    "few_shots_select": None,
    "generation_size": -1,
    "metrics": [Metrics.loglikelihood_acc],
    "stop_sequence": ["</s>", "Q=", "\n\n"],
    "version": 0,
    "sample_fields": record_to_sample,
    "solver": [system_message(SYSTEM_MESSAGE), generate(cache=True)],
    "scorer": answer(pattern="line"),
}

boardgame_qa = LightevalTaskConfig(
    name="bigbench_extra_hard:boardgame_qa",
    hf_subset="boardgame_qa",
    **COMMON_TASK_ARGS,
)

boolean_expressions = LightevalTaskConfig(
    name="bigbench_extra_hard:boolean_expressions",
    hf_subset="boolean_expressions",
    **COMMON_TASK_ARGS,
)

buggy_tables = LightevalTaskConfig(
    name="bigbench_extra_hard:buggy_tables",
    hf_subset="buggy_tables",
    **COMMON_TASK_ARGS,
)

causal_understanding = LightevalTaskConfig(
    name="bigbench_extra_hard:causal_understanding",
    hf_subset="causal_understanding",
    **COMMON_TASK_ARGS,
)

disambiguation_qa = LightevalTaskConfig(
    name="bigbench_extra_hard:disambiguation_qa",
    hf_subset="disambiguation_qa",
    **COMMON_TASK_ARGS,
)

dyck_languages = LightevalTaskConfig(
    name="bigbench_extra_hard:dyck_languages",
    hf_subset="dyck_languages",
    **COMMON_TASK_ARGS,
)

geometric_shapes = LightevalTaskConfig(
    name="bigbench_extra_hard:geometric_shapes",
    hf_subset="geometric_shapes",
    **COMMON_TASK_ARGS,
)

hyperbaton = LightevalTaskConfig(
    name="bigbench_extra_hard:hyperbaton",
    hf_subset="hyperbaton",
    **COMMON_TASK_ARGS,
)

linguini = LightevalTaskConfig(
    name="bigbench_extra_hard:linguini",
    hf_subset="linguini",
    **COMMON_TASK_ARGS,
)

movie_recommendation = LightevalTaskConfig(
    name="bigbench_extra_hard:movie_recommendation",
    hf_subset="movie_recommendation",
    **COMMON_TASK_ARGS,
)

multistep_arithmetic = LightevalTaskConfig(
    name="bigbench_extra_hard:multistep_arithmetic",
    hf_subset="multistep_arithmetic",
    **COMMON_TASK_ARGS,
)

nycc = LightevalTaskConfig(
    name="bigbench_extra_hard:nycc",
    hf_subset="nycc",
    **COMMON_TASK_ARGS,
)

object_counting = LightevalTaskConfig(
    name="bigbench_extra_hard:object_counting",
    hf_subset="object_counting",
    **COMMON_TASK_ARGS,
)

object_properties = LightevalTaskConfig(
    name="bigbench_extra_hard:object_properties",
    hf_subset="object_properties",
    **COMMON_TASK_ARGS,
)

sarc_triples = LightevalTaskConfig(
    name="bigbench_extra_hard:sarc_triples",
    hf_subset="sarc_triples",
    **COMMON_TASK_ARGS,
)

shuffled_objects = LightevalTaskConfig(
    name="bigbench_extra_hard:shuffled_objects",
    hf_subset="shuffled_objects",
    **COMMON_TASK_ARGS,
)

spatial_reasoning = LightevalTaskConfig(
    name="bigbench_extra_hard:spatial_reasoning",
    hf_subset="spatial_reasoning",
    **COMMON_TASK_ARGS,
)

sportqa = LightevalTaskConfig(
    name="bigbench_extra_hard:sportqa",
    hf_subset="sportqa",
    **COMMON_TASK_ARGS,
)

temporal_sequence = LightevalTaskConfig(
    name="bigbench_extra_hard:temporal_sequence",
    hf_subset="temporal_sequence",
    **COMMON_TASK_ARGS,
)

time_arithmetic = LightevalTaskConfig(
    name="bigbench_extra_hard:time_arithmetic",
    hf_subset="time_arithmetic",
    **COMMON_TASK_ARGS,
)

web_of_lies = LightevalTaskConfig(
    name="bigbench_extra_hard:web_of_lies",
    hf_subset="web_of_lies",
    **COMMON_TASK_ARGS,
)

word_sorting = LightevalTaskConfig(
    name="bigbench_extra_hard:word_sorting",
    hf_subset="word_sorting",
    **COMMON_TASK_ARGS,
)

zebra_puzzles = LightevalTaskConfig(
    name="bigbench_extra_hard:zebra_puzzles",
    hf_subset="zebra_puzzles",
    **COMMON_TASK_ARGS,
)

TASKS_TABLE = [
    boardgame_qa,
    boolean_expressions,
    buggy_tables,
    causal_understanding,
    disambiguation_qa,
    dyck_languages,
    geometric_shapes,
    hyperbaton,
    linguini,
    movie_recommendation,
    multistep_arithmetic,
    nycc,
    object_counting,
    object_properties,
    sarc_triples,
    shuffled_objects,
    spatial_reasoning,
    sportqa,
    temporal_sequence,
    time_arithmetic,
    web_of_lies,
    word_sorting,
    zebra_puzzles,
]
