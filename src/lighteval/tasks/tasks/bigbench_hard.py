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

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


"""
hardest subset of bigbench benchmark.
"""


causal_judgment = LightevalTaskConfig(
    name="bigbench_hard:causal_judgment",
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
    suite=["lighteval"],
    prompt_function=prompt.bbh,
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
