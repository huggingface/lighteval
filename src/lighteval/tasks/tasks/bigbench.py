"""
name:
Bigbench

dataset:
tasksource/bigbench

abstract:
Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models
166 tasks from bigbench benchmark.

languages:
english

tags:
reasoning

paper:
https://arxiv.org/abs/2206.04615
"""

from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def bigbench_linefeed_before_and_after_query_prompt(line, task_name: str = None):
    if len(line["multiple_choice_scores"]) == 0:
        choices = line["targets"]
        gold_index = [i for i, _ in enumerate(line["targets"])]
    else:
        choices = line["multiple_choice_targets"]
        gold_index = [i for i, a in enumerate(line["multiple_choice_scores"]) if a == 1]

    return Doc(
        task_name=task_name,
        query=f"\n{line['inputs']}\n",
        choices=choices,
        gold_index=gold_index,
    )


def bigbench_linefeed_before_whitespace_after_query_prompt(line, task_name: str = None):
    if len(line["multiple_choice_scores"]) == 0:
        choices = line["targets"]
        gold_index = [i for i, _ in enumerate(line["targets"])]
    else:
        choices = line["multiple_choice_targets"]
        gold_index = [i for i, a in enumerate(line["multiple_choice_scores"]) if a == 1]

    return Doc(
        task_name=task_name,
        query=f"\n{line['inputs']} ",
        choices=choices,
        gold_index=gold_index,
    )


def bigbench_whitespace_after_query_prompt(line, task_name: str = None):
    if len(line["multiple_choice_scores"]) == 0:
        choices = line["targets"]
        gold_index = [i for i, _ in enumerate(line["targets"])]
    else:
        choices = line["multiple_choice_targets"]
        gold_index = [i for i, a in enumerate(line["multiple_choice_scores"]) if a == 1]

    return Doc(
        task_name=task_name,
        query=f"{line['inputs']} ",
        choices=choices,
        gold_index=gold_index,
    )


def bigbench_prompt(line, task_name: str = None):
    if len(line["multiple_choice_scores"]) == 0:
        choices = line["targets"]
        gold_index = [i for i, _ in enumerate(line["targets"])]
    else:
        choices = line["multiple_choice_targets"]
        gold_index = [i for i, a in enumerate(line["multiple_choice_scores"]) if a == 1]

    return Doc(task_name=task_name, query=line["inputs"], choices=choices, gold_index=gold_index)


def record_to_sample(record):
    query = record["inputs"]
    choices = record["multiple_choice_targets"]
    target = ascii_uppercase[record["multiple_choice_scores"].index(1)]
    return Sample(input=query, target=target, choices=choices)


abstract_narrative_understanding = LightevalTaskConfig(
    name="bigbench:abstract_narrative_understanding",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="abstract_narrative_understanding",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

anachronisms = LightevalTaskConfig(
    name="bigbench:anachronisms",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="anachronisms",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

analogical_similarity = LightevalTaskConfig(
    name="bigbench:analogical_similarity",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="analogical_similarity",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

analytic_entailment = LightevalTaskConfig(
    name="bigbench:analytic_entailment",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="analytic_entailment",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

arithmetic_bb = LightevalTaskConfig(
    name="bigbench:arithmetic_bb",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="arithmetic",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc, Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

ascii_word_recognition = LightevalTaskConfig(
    name="bigbench:ascii_word_recognition",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="ascii_word_recognition",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

authorship_verification = LightevalTaskConfig(
    name="bigbench:authorship_verification",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="authorship_verification",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

auto_categorization = LightevalTaskConfig(
    name="bigbench:auto_categorization",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="auto_categorization",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.bleu],
    stop_sequence=["\n"],
    version=0,
)

auto_debugging = LightevalTaskConfig(
    name="bigbench:auto_debugging",
    prompt_function=bigbench_linefeed_before_and_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="auto_debugging",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=None,
    version=0,
)

bbq_lite_json = LightevalTaskConfig(
    name="bigbench:bbq_lite_json",
    prompt_function=bigbench_linefeed_before_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="bbq_lite_json",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

bridging_anaphora_resolution_barqa = LightevalTaskConfig(
    name="bigbench:bridging_anaphora_resolution_barqa",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="bridging_anaphora_resolution_barqa",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

causal_judgment = LightevalTaskConfig(
    name="bigbench:causal_judgment",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="causal_judgment",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

cause_and_effect = LightevalTaskConfig(
    name="bigbench:cause_and_effect",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="cause_and_effect",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

checkmate_in_one = LightevalTaskConfig(
    name="bigbench:checkmate_in_one",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="checkmate_in_one",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc, Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

chess_state_tracking = LightevalTaskConfig(
    name="bigbench:chess_state_tracking",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="chess_state_tracking",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

chinese_remainder_theorem = LightevalTaskConfig(
    name="bigbench:chinese_remainder_theorem",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="chinese_remainder_theorem",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

cifar10_classification = LightevalTaskConfig(
    name="bigbench:cifar10_classification",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="cifar10_classification",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

code_line_description = LightevalTaskConfig(
    name="bigbench:code_line_description",
    prompt_function=bigbench_linefeed_before_and_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="code_line_description",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

codenames = LightevalTaskConfig(
    name="bigbench:codenames",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="codenames",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.rouge_t5, Metrics.bleu],
    stop_sequence=["\n"],
    version=0,
)

color = LightevalTaskConfig(
    name="bigbench:color",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="color",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.rouge_t5,
        Metrics.bleu,
        Metrics.loglikelihood_acc,
        Metrics.exact_match(sample_params={"strip_strings": False}),
    ],
    stop_sequence=["\n"],
    version=0,
)

common_morpheme = LightevalTaskConfig(
    name="bigbench:common_morpheme",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="common_morpheme",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

conceptual_combinations = LightevalTaskConfig(
    name="bigbench:conceptual_combinations",
    prompt_function=bigbench_linefeed_before_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="conceptual_combinations",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

conlang_translation = LightevalTaskConfig(
    name="bigbench:conlang_translation",
    prompt_function=bigbench_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="conlang_translation",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.rouge_t5, Metrics.bleu, Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=[".", ";", "!", "?"],
    version=0,
)

contextual_parametric_knowledge_conflicts = LightevalTaskConfig(
    name="bigbench:contextual_parametric_knowledge_conflicts",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="contextual_parametric_knowledge_conflicts",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.rouge_t5, Metrics.loglikelihood_acc, Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

crash_blossom = LightevalTaskConfig(
    name="bigbench:crash_blossom",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="crash_blossom",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

crass_ai = LightevalTaskConfig(
    name="bigbench:crass_ai",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="crass_ai",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

cryobiology_spanish = LightevalTaskConfig(
    name="bigbench:cryobiology_spanish",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="cryobiology_spanish",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

cryptonite = LightevalTaskConfig(
    name="bigbench:cryptonite",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="cryptonite",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

cs_algorithms = LightevalTaskConfig(
    name="bigbench:cs_algorithms",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="cs_algorithms",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

dark_humor_detection = LightevalTaskConfig(
    name="bigbench:dark_humor_detection",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="dark_humor_detection",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

date_understanding = LightevalTaskConfig(
    name="bigbench:date_understanding",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="date_understanding",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

disambiguation_qa = LightevalTaskConfig(
    name="bigbench:disambiguation_qa",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="disambiguation_qa",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

discourse_marker_prediction = LightevalTaskConfig(
    name="bigbench:discourse_marker_prediction",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="discourse_marker_prediction",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

disfl_qa = LightevalTaskConfig(
    name="bigbench:disfl_qa",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="disfl_qa",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

dyck_languages = LightevalTaskConfig(
    name="bigbench:dyck_languages",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="dyck_languages",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

elementary_math_qa = LightevalTaskConfig(
    name="bigbench:elementary_math_qa",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="elementary_math_qa",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

emoji_movie = LightevalTaskConfig(
    name="bigbench:emoji_movie",
    prompt_function=bigbench_linefeed_before_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="emoji_movie",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.rouge_t5,
        Metrics.bleu,
        Metrics.loglikelihood_acc,
        Metrics.exact_match(sample_params={"strip_strings": False}),
    ],
    stop_sequence=["\n"],
    version=0,
)

emojis_emotion_prediction = LightevalTaskConfig(
    name="bigbench:emojis_emotion_prediction",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="emojis_emotion_prediction",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

empirical_judgments = LightevalTaskConfig(
    name="bigbench:empirical_judgments",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="empirical_judgments",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

english_proverbs = LightevalTaskConfig(
    name="bigbench:english_proverbs",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="english_proverbs",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

english_russian_proverbs = LightevalTaskConfig(
    name="bigbench:english_russian_proverbs",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="english_russian_proverbs",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

entailed_polarity = LightevalTaskConfig(
    name="bigbench:entailed_polarity",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="entailed_polarity",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

entailed_polarity_hindi = LightevalTaskConfig(
    name="bigbench:entailed_polarity_hindi",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="entailed_polarity_hindi",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

epistemic_reasoning = LightevalTaskConfig(
    name="bigbench:epistemic_reasoning",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="epistemic_reasoning",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

evaluating_information_essentiality = LightevalTaskConfig(
    name="bigbench:evaluating_information_essentiality",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="evaluating_information_essentiality",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

fact_checker = LightevalTaskConfig(
    name="bigbench:fact_checker",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="fact_checker",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

fantasy_reasoning = LightevalTaskConfig(
    name="bigbench:fantasy_reasoning",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="fantasy_reasoning",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

few_shot_nlg = LightevalTaskConfig(
    name="bigbench:few_shot_nlg",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="few_shot_nlg",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.bleu, Metrics.bleurt],
    stop_sequence=["\n"],
    version=0,
)

figure_of_speech_detection = LightevalTaskConfig(
    name="bigbench:figure_of_speech_detection",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="figure_of_speech_detection",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

formal_fallacies_syllogisms_negation = LightevalTaskConfig(
    name="bigbench:formal_fallacies_syllogisms_negation",
    prompt_function=bigbench_linefeed_before_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="formal_fallacies_syllogisms_negation",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

gem = LightevalTaskConfig(
    name="bigbench:gem",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="gem",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.bleu, Metrics.rouge_t5],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

gender_inclusive_sentences_german = LightevalTaskConfig(
    name="bigbench:gender_inclusive_sentences_german",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="gender_inclusive_sentences_german",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

general_knowledge = LightevalTaskConfig(
    name="bigbench:general_knowledge",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="general_knowledge",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

geometric_shapes = LightevalTaskConfig(
    name="bigbench:geometric_shapes",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="geometric_shapes",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.rouge_t5,
        Metrics.bleu,
        Metrics.loglikelihood_acc,
        Metrics.exact_match(sample_params={"strip_strings": False}),
    ],
    stop_sequence=["\n"],
    version=0,
)

goal_step_wikihow = LightevalTaskConfig(
    name="bigbench:goal_step_wikihow",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="goal_step_wikihow",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

gre_reading_comprehension = LightevalTaskConfig(
    name="bigbench:gre_reading_comprehension",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="gre_reading_comprehension",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

hhh_alignment = LightevalTaskConfig(
    name="bigbench:hhh_alignment",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="hhh_alignment",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

hindi_question_answering = LightevalTaskConfig(
    name="bigbench:hindi_question_answering",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="hindi_question_answering",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.bleu, Metrics.rouge_t5, Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

hindu_knowledge = LightevalTaskConfig(
    name="bigbench:hindu_knowledge",
    prompt_function=bigbench_linefeed_before_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="hindu_knowledge",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

hinglish_toxicity = LightevalTaskConfig(
    name="bigbench:hinglish_toxicity",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="hinglish_toxicity",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

human_organs_senses = LightevalTaskConfig(
    name="bigbench:human_organs_senses",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="human_organs_senses",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

hyperbaton = LightevalTaskConfig(
    name="bigbench:hyperbaton",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="hyperbaton",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

identify_math_theorems = LightevalTaskConfig(
    name="bigbench:identify_math_theorems",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="identify_math_theorems",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

identify_odd_metaphor = LightevalTaskConfig(
    name="bigbench:identify_odd_metaphor",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="identify_odd_metaphor",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

implicatures = LightevalTaskConfig(
    name="bigbench:implicatures",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="implicatures",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

implicit_relations = LightevalTaskConfig(
    name="bigbench:implicit_relations",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="implicit_relations",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

intent_recognition = LightevalTaskConfig(
    name="bigbench:intent_recognition",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="intent_recognition",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

international_phonetic_alphabet_nli = LightevalTaskConfig(
    name="bigbench:international_phonetic_alphabet_nli",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="international_phonetic_alphabet_nli",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

international_phonetic_alphabet_transliterate = LightevalTaskConfig(
    name="bigbench:international_phonetic_alphabet_transliterate",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="international_phonetic_alphabet_transliterate",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.bleu, Metrics.rouge_t5, Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

intersect_geometry = LightevalTaskConfig(
    name="bigbench:intersect_geometry",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="intersect_geometry",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

irony_identification = LightevalTaskConfig(
    name="bigbench:irony_identification",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="irony_identification",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

kanji_ascii = LightevalTaskConfig(
    name="bigbench:kanji_ascii",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="kanji_ascii",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

kannada = LightevalTaskConfig(
    name="bigbench:kannada",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="kannada",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

key_value_maps = LightevalTaskConfig(
    name="bigbench:key_value_maps",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="key_value_maps",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

known_unknowns = LightevalTaskConfig(
    name="bigbench:known_unknowns",
    prompt_function=bigbench_linefeed_before_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="known_unknowns",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

language_games = LightevalTaskConfig(
    name="bigbench:language_games",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="language_games",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.bleu, Metrics.rouge_t5, Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

language_identification = LightevalTaskConfig(
    name="bigbench:language_identification",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="language_identification",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

linguistic_mappings = LightevalTaskConfig(
    name="bigbench:linguistic_mappings",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="linguistic_mappings",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

linguistics_puzzles = LightevalTaskConfig(
    name="bigbench:linguistics_puzzles",
    prompt_function=bigbench_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="linguistics_puzzles",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.bleu, Metrics.rouge_t5, Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=None,
    version=0,
)

logic_grid_puzzle = LightevalTaskConfig(
    name="bigbench:logic_grid_puzzle",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="logic_grid_puzzle",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

logical_args = LightevalTaskConfig(
    name="bigbench:logical_args",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="logical_args",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

logical_deduction = LightevalTaskConfig(
    name="bigbench:logical_deduction",
    prompt_function=bigbench_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="logical_deduction",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

logical_fallacy_detection = LightevalTaskConfig(
    name="bigbench:logical_fallacy_detection",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="logical_fallacy_detection",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

logical_sequence = LightevalTaskConfig(
    name="bigbench:logical_sequence",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="logical_sequence",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

mathematical_induction = LightevalTaskConfig(
    name="bigbench:mathematical_induction",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="mathematical_induction",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

matrixshapes = LightevalTaskConfig(
    name="bigbench:matrixshapes",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="matrixshapes",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

metaphor_boolean = LightevalTaskConfig(
    name="bigbench:metaphor_boolean",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="metaphor_boolean",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

metaphor_understanding = LightevalTaskConfig(
    name="bigbench:metaphor_understanding",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="metaphor_understanding",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

minute_mysteries_qa = LightevalTaskConfig(
    name="bigbench:minute_mysteries_qa",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="minute_mysteries_qa",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc, Metrics.rouge_t5],
    stop_sequence=["\n"],
    version=0,
)

misconceptions = LightevalTaskConfig(
    name="bigbench:misconceptions",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="misconceptions",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

misconceptions_russian = LightevalTaskConfig(
    name="bigbench:misconceptions_russian",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="misconceptions_russian",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

mnist_ascii = LightevalTaskConfig(
    name="bigbench:mnist_ascii",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="mnist_ascii",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

modified_arithmetic = LightevalTaskConfig(
    name="bigbench:modified_arithmetic",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="modified_arithmetic",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

moral_permissibility = LightevalTaskConfig(
    name="bigbench:moral_permissibility",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="moral_permissibility",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

movie_dialog_same_or_different = LightevalTaskConfig(
    name="bigbench:movie_dialog_same_or_different",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="movie_dialog_same_or_different",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

movie_recommendation = LightevalTaskConfig(
    name="bigbench:movie_recommendation",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="movie_recommendation",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

mult_data_wrangling = LightevalTaskConfig(
    name="bigbench:mult_data_wrangling",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="mult_data_wrangling",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

navigate = LightevalTaskConfig(
    name="bigbench:navigate",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="navigate",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

nonsense_words_grammar = LightevalTaskConfig(
    name="bigbench:nonsense_words_grammar",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="nonsense_words_grammar",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

novel_concepts = LightevalTaskConfig(
    name="bigbench:novel_concepts",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="novel_concepts",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

object_counting = LightevalTaskConfig(
    name="bigbench:object_counting",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="object_counting",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

odd_one_out = LightevalTaskConfig(
    name="bigbench:odd_one_out",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="odd_one_out",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

operators = LightevalTaskConfig(
    name="bigbench:operators",
    prompt_function=bigbench_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="operators",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

paragraph_segmentation = LightevalTaskConfig(
    name="bigbench:paragraph_segmentation",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="paragraph_segmentation",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

parsinlu_qa = LightevalTaskConfig(
    name="bigbench:parsinlu_qa",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="parsinlu_qa",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

parsinlu_reading_comprehension = LightevalTaskConfig(
    name="bigbench:parsinlu_reading_comprehension",
    prompt_function=bigbench_linefeed_before_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="parsinlu_reading_comprehension",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=None,
    version=0,
)

penguins_in_a_table = LightevalTaskConfig(
    name="bigbench:penguins_in_a_table",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="penguins_in_a_table",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc, Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

periodic_elements = LightevalTaskConfig(
    name="bigbench:periodic_elements",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="periodic_elements",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

persian_idioms = LightevalTaskConfig(
    name="bigbench:persian_idioms",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="persian_idioms",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

phrase_relatedness = LightevalTaskConfig(
    name="bigbench:phrase_relatedness",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="phrase_relatedness",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

physical_intuition = LightevalTaskConfig(
    name="bigbench:physical_intuition",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="physical_intuition",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

physics = LightevalTaskConfig(
    name="bigbench:physics",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="physics",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

physics_questions = LightevalTaskConfig(
    name="bigbench:physics_questions",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="physics_questions",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.bleu, Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

play_dialog_same_or_different = LightevalTaskConfig(
    name="bigbench:play_dialog_same_or_different",
    prompt_function=bigbench_linefeed_before_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="play_dialog_same_or_different",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

polish_sequence_labeling = LightevalTaskConfig(
    name="bigbench:polish_sequence_labeling",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="polish_sequence_labeling",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.f1_score],
    stop_sequence=["\n"],
    version=0,
)

presuppositions_as_nli = LightevalTaskConfig(
    name="bigbench:presuppositions_as_nli",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="presuppositions_as_nli",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

qa_wikidata = LightevalTaskConfig(
    name="bigbench:qa_wikidata",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="qa_wikidata",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.bleurt,
        Metrics.bleu,
        Metrics.rouge_t5,
        Metrics.exact_match(sample_params={"strip_strings": False}),
    ],
    stop_sequence=["\n"],
    version=0,
)

question_selection = LightevalTaskConfig(
    name="bigbench:question_selection",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="question_selection",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

real_or_fake_text = LightevalTaskConfig(
    name="bigbench:real_or_fake_text",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="real_or_fake_text",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

reasoning_about_colored_objects = LightevalTaskConfig(
    name="bigbench:reasoning_about_colored_objects",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="reasoning_about_colored_objects",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

repeat_copy_logic = LightevalTaskConfig(
    name="bigbench:repeat_copy_logic",
    prompt_function=bigbench_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="repeat_copy_logic",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

rephrase = LightevalTaskConfig(
    name="bigbench:rephrase",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="rephrase",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.rouge_t5,
        Metrics.bleu,
        Metrics.loglikelihood_acc,
        Metrics.exact_match(sample_params={"strip_strings": False}),
    ],
    stop_sequence=["\n"],
    version=0,
)

rhyming = LightevalTaskConfig(
    name="bigbench:rhyming",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="rhyming",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

riddle_sense = LightevalTaskConfig(
    name="bigbench:riddle_sense",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="riddle_sense",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

ruin_names = LightevalTaskConfig(
    name="bigbench:ruin_names",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="ruin_names",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

salient_translation_error_detection = LightevalTaskConfig(
    name="bigbench:salient_translation_error_detection",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="salient_translation_error_detection",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

scientific_press_release = LightevalTaskConfig(
    name="bigbench:scientific_press_release",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="scientific_press_release",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.bleu, Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

semantic_parsing_in_context_sparc = LightevalTaskConfig(
    name="bigbench:semantic_parsing_in_context_sparc",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="semantic_parsing_in_context_sparc",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.bleu, Metrics.rouge_t5, Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

semantic_parsing_spider = LightevalTaskConfig(
    name="bigbench:semantic_parsing_spider",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="semantic_parsing_spider",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.bleu, Metrics.rouge_t5, Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

sentence_ambiguity = LightevalTaskConfig(
    name="bigbench:sentence_ambiguity",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="sentence_ambiguity",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

similarities_abstraction = LightevalTaskConfig(
    name="bigbench:similarities_abstraction",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="similarities_abstraction",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.bleu, Metrics.rouge_t5, Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

simp_turing_concept = LightevalTaskConfig(
    name="bigbench:simp_turing_concept",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="simp_turing_concept",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

simple_arithmetic_json = LightevalTaskConfig(
    name="bigbench:simple_arithmetic_json",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="simple_arithmetic_json",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

simple_arithmetic_json_multiple_choice = LightevalTaskConfig(
    name="bigbench:simple_arithmetic_json_multiple_choice",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="simple_arithmetic_json_multiple_choice",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

simple_arithmetic_json_subtasks = LightevalTaskConfig(
    name="bigbench:simple_arithmetic_json_subtasks",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="simple_arithmetic_json_subtasks",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

simple_arithmetic_multiple_targets_json = LightevalTaskConfig(
    name="bigbench:simple_arithmetic_multiple_targets_json",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="simple_arithmetic_multiple_targets_json",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.bleu, Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

simple_ethical_questions = LightevalTaskConfig(
    name="bigbench:simple_ethical_questions",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="simple_ethical_questions",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

simple_text_editing = LightevalTaskConfig(
    name="bigbench:simple_text_editing",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="simple_text_editing",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

snarks = LightevalTaskConfig(
    name="bigbench:snarks",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="snarks",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

social_iqa = LightevalTaskConfig(
    name="bigbench:social_iqa",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="social_iqa",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

social_support = LightevalTaskConfig(
    name="bigbench:social_support",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="social_support",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.f1_score_macro],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

sports_understanding = LightevalTaskConfig(
    name="bigbench:sports_understanding",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="sports_understanding",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

strange_stories = LightevalTaskConfig(
    name="bigbench:strange_stories",
    prompt_function=bigbench_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="strange_stories",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

strategyqa = LightevalTaskConfig(
    name="bigbench:strategyqa",
    prompt_function=bigbench_linefeed_before_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="strategyqa",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.bleu, Metrics.rouge_t5, Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

sufficient_information = LightevalTaskConfig(
    name="bigbench:sufficient_information",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="sufficient_information",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

suicide_risk = LightevalTaskConfig(
    name="bigbench:suicide_risk",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="suicide_risk",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

swahili_english_proverbs = LightevalTaskConfig(
    name="bigbench:swahili_english_proverbs",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="swahili_english_proverbs",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

swedish_to_german_proverbs = LightevalTaskConfig(
    name="bigbench:swedish_to_german_proverbs",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="swedish_to_german_proverbs",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

symbol_interpretation = LightevalTaskConfig(
    name="bigbench:symbol_interpretation",
    prompt_function=bigbench_linefeed_before_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="symbol_interpretation",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

tellmewhy = LightevalTaskConfig(
    name="bigbench:tellmewhy",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="tellmewhy",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.bleu, Metrics.rouge_t5],
    stop_sequence=["\n"],
    version=0,
)

temporal_sequences = LightevalTaskConfig(
    name="bigbench:temporal_sequences",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="temporal_sequences",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

tense = LightevalTaskConfig(
    name="bigbench:tense",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="tense",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

timedial = LightevalTaskConfig(
    name="bigbench:timedial",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="timedial",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

topical_chat = LightevalTaskConfig(
    name="bigbench:topical_chat",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="topical_chat",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.bleu, Metrics.rouge_t5, Metrics.loglikelihood_acc, Metrics.bleurt],
    stop_sequence=["\n"],
    version=0,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
)

tracking_shuffled_objects = LightevalTaskConfig(
    name="bigbench:tracking_shuffled_objects",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="tracking_shuffled_objects",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

understanding_fables = LightevalTaskConfig(
    name="bigbench:understanding_fables",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="understanding_fables",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

undo_permutation = LightevalTaskConfig(
    name="bigbench:undo_permutation",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="undo_permutation",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

unit_conversion = LightevalTaskConfig(
    name="bigbench:unit_conversion",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="unit_conversion",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

unit_interpretation = LightevalTaskConfig(
    name="bigbench:unit_interpretation",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="unit_interpretation",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

unnatural_in_context_learning = LightevalTaskConfig(
    name="bigbench:unnatural_in_context_learning",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="unnatural_in_context_learning",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

vitaminc_fact_verification = LightevalTaskConfig(
    name="bigbench:vitaminc_fact_verification",
    prompt_function=bigbench_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="vitaminc_fact_verification",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

what_is_the_tao = LightevalTaskConfig(
    name="bigbench:what_is_the_tao",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="what_is_the_tao",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

which_wiki_edit = LightevalTaskConfig(
    name="bigbench:which_wiki_edit",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="which_wiki_edit",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

winowhy = LightevalTaskConfig(
    name="bigbench:winowhy",
    prompt_function=bigbench_whitespace_after_query_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="winowhy",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=0,
)

word_sorting = LightevalTaskConfig(
    name="bigbench:word_sorting",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="word_sorting",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

word_unscrambling = LightevalTaskConfig(
    name="bigbench:word_unscrambling",
    prompt_function=bigbench_prompt,
    hf_repo="tasksource/bigbench",
    hf_subset="word_unscrambling",
    hf_avail_splits=["default", "train", "validation"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match(sample_params={"strip_strings": False})],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    abstract_narrative_understanding,
    anachronisms,
    analogical_similarity,
    moral_permissibility,
    movie_dialog_same_or_different,
    movie_recommendation,
    mult_data_wrangling,
    simple_ethical_questions,
    simple_text_editing,
    snarks,
    social_iqa,
    social_support,
    sports_understanding,
    strange_stories,
    strategyqa,
    sufficient_information,
    suicide_risk,
    swahili_english_proverbs,
    swedish_to_german_proverbs,
    symbol_interpretation,
    tellmewhy,
    temporal_sequences,
    tense,
    timedial,
    topical_chat,
    tracking_shuffled_objects,
    understanding_fables,
    undo_permutation,
    unit_conversion,
    unit_interpretation,
    unnatural_in_context_learning,
    vitaminc_fact_verification,
    what_is_the_tao,
    which_wiki_edit,
    winowhy,
    word_sorting,
    word_unscrambling,
]
