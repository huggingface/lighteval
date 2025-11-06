"""
name:
Bigbench Hard

dataset:
lighteval/bbh

abstract:

languages:

tags:
reasoning

paper:
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


causal_judgment = LightevalTaskConfig(
    name="bigbench_hard:causal_judgment",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="causal_judgement",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

date_understanding = LightevalTaskConfig(
    name="bigbench_hard:date_understanding",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="date_understanding",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

disambiguation_qa = LightevalTaskConfig(
    name="bigbench_hard:disambiguation_qa",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="disambiguation_qa",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

geometric_shapes = LightevalTaskConfig(
    name="bigbench_hard:geometric_shapes",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="geometric_shapes",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

logical_deduction_five_objects = LightevalTaskConfig(
    name="bigbench_hard:logical_deduction_five_objects",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="logical_deduction_five_objects",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

logical_deduction_seven_objects = LightevalTaskConfig(
    name="bigbench_hard:logical_deduction_seven_objects",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="logical_deduction_seven_objects",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

logical_deduction_three_objects = LightevalTaskConfig(
    name="bigbench_hard:logical_deduction_three_objects",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="logical_deduction_three_objects",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

movie_recommendation = LightevalTaskConfig(
    name="bigbench_hard:movie_recommendation",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="movie_recommendation",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

navigate = LightevalTaskConfig(
    name="bigbench_hard:navigate",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="navigate",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

reasoning_about_colored_objects = LightevalTaskConfig(
    name="bigbench_hard:reasoning_about_colored_objects",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="reasoning_about_colored_objects",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

ruin_names = LightevalTaskConfig(
    name="bigbench_hard:ruin_names",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="ruin_names",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

salient_translation_error_detection = LightevalTaskConfig(
    name="bigbench_hard:salient_translation_error_detection",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="salient_translation_error_detection",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

snarks = LightevalTaskConfig(
    name="bigbench_hard:snarks",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="snarks",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

sports_understanding = LightevalTaskConfig(
    name="bigbench_hard:sports_understanding",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="sports_understanding",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

temporal_sequences = LightevalTaskConfig(
    name="bigbench_hard:temporal_sequences",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="temporal_sequences",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

tracking_shuffled_objects_five_objects = LightevalTaskConfig(
    name="bigbench_hard:tracking_shuffled_objects_five_objects",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="tracking_shuffled_objects_five_objects",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

tracking_shuffled_objects_seven_objects = LightevalTaskConfig(
    name="bigbench_hard:tracking_shuffled_objects_seven_objects",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="tracking_shuffled_objects_seven_objects",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

tracking_shuffled_objects_three_objects = LightevalTaskConfig(
    name="bigbench_hard:tracking_shuffled_objects_three_objects",
    prompt_function=prompt.bbh_lighteval,
    hf_repo="lighteval/bbh",
    hf_subset="tracking_shuffled_objects_three_objects",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["</s>", "Q=", "\n\n"],
    version=0,
)

TASKS_TABLE = [
    causal_judgment,
    date_understanding,
    disambiguation_qa,
    geometric_shapes,
    logical_deduction_five_objects,
    logical_deduction_seven_objects,
    logical_deduction_three_objects,
    movie_recommendation,
    navigate,
    reasoning_about_colored_objects,
    ruin_names,
    salient_translation_error_detection,
    snarks,
    sports_understanding,
    temporal_sequences,
    tracking_shuffled_objects_five_objects,
    tracking_shuffled_objects_seven_objects,
    tracking_shuffled_objects_three_objects,
]
