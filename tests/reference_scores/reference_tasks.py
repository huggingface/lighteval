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

# todo: add original once we are sure of the results
MMLU_SUBSET = [
    "leaderboard|mmlu:abstract_algebra|5|0",
    "helm|mmlu:abstract_algebra|5|0",
    # "original|mmlu:abstract_algebra|5",
    "leaderboard|mmlu:college_chemistry|5|0",
    "helm|mmlu:college_chemistry|5|0",
    # "original|mmlu:college_chemistry|5",
    "leaderboard|mmlu:computer_security|5|0",
    "helm|mmlu:computer_security|5|0",
    # "original|mmlu:computer_security|5",
    "leaderboard|mmlu:us_foreign_policy|5|0",
    "helm|mmlu:us_foreign_policy|5|0",
    # "original|mmlu:us_foreign_policy|5",
]

LEADERBOARD_SUBSET = [
    "leaderboard|arc:challenge|25|0",
    "leaderboard|truthfulqa:mc|0|0",
    "leaderboard|hellaswag|10|0",
    "leaderboard|mmlu:abstract_algebra|5|0",
    "leaderboard|mmlu:college_chemistry|5|0",
    "leaderboard|mmlu:computer_security|5|0",
    "leaderboard|mmlu:us_foreign_policy|5|0",
    "leaderboard|gsm8k|5|0",
]

STABLE_SUBSET = [
    "helm|mmlu:abstract_algebra|5|0",
    "helm|mmlu:college_chemistry|5|0",
    "helm|mmlu:computer_security|5|0",
    "helm|mmlu:us_foreign_policy|5|0",
    "lighteval|anli:r1|0|0",
    "lighteval|blimp:adjunct_island|0|0",
    "lighteval|blimp:ellipsis_n_bar_1|0|0",
]

HELM_SUBSET = [
    "helm|boolq|5|0",
]

AGIEVAL_SUBSET = [
    "lighteval|agieval:aqua-rat|0|0",
    "lighteval|agieval:logiqa-en|0|0",
    "lighteval|agieval:lsat-ar|0|0",
    "lighteval|agieval:lsat-lr|0|0",
    "lighteval|agieval:lsat-rc|0|0",
    "lighteval|agieval:sat-en-without-passage|0|0",
    "lighteval|agieval:sat-en|0|0",
    "lighteval|agieval:sat-math|0|0",
]

BBH_SUBSET = [
    "lighteval|bigbench:causal_judgment|3|0",
    "harness|bigbench:causal_judgment|3|0",
    "lighteval|bigbench:date_understanding|3|0",
    "harness|bigbench:date_understanding|3|0",
    "lighteval|bigbench:disambiguation_qa|3|0",
    "harness|bigbench:disambiguation_qa|3|0",
    "lighteval|bigbench:geometric_shapes|3|0",
    "harness|bigbench:geometric_shapes|3|0",
    "lighteval|bigbench:logical_deduction_five_objects|3|0",
    "harness|bigbench:logical_deduction_five_objects|3|0",
    "lighteval|bigbench:logical_deduction_seven_objects|3|0",
    "harness|bigbench:logical_deduction_seven_objects|3|0",
    "lighteval|bigbench:logical_deduction_three_objects|3|0",
    "harness|bigbench:logical_deduction_three_objects|3|0",
    "lighteval|bigbench:movie_recommendation|3|0",
    "harness|bigbench:movie_recommendation|3|0",
    "lighteval|bigbench:navigate|3|0",
    "harness|bigbench:navigate|3|0",
    "lighteval|bigbench:reasoning_about_colored_objects|3|0",
    "harness|bigbench:reasoning_about_colored_objects|3|0",
    "lighteval|bigbench:ruin_names|3|0",
    "harness|bigbench:ruin_names|3|0",
    "lighteval|bigbench:salient_translation_error_detection|3|0",
    "harness|bigbench:salient_translation_error_detection|3|0",
    "lighteval|bigbench:snarks|3|0",
    "harness|bigbench:snarks|3|0",
    "lighteval|bigbench:sports_understanding|3|0",
    "harness|bigbench:sports_understanding|3|0",
    "lighteval|bigbench:temporal_sequences|3|0",
    "harness|bigbench:temporal_sequences|3|0",
    "lighteval|bigbench:tracking_shuffled_objects_five_objects|3|0",
    "harness|bigbench:tracking_shuffled_objects_five_objects|3|0",
    "lighteval|bigbench:tracking_shuffled_objects_seven_objects|3|0",
    "harness|bigbench:tracking_shuffled_objects_seven_objects|3|0",
    "lighteval|bigbench:tracking_shuffled_objects_three_objects|3|0",
    "harness|bigbench:tracking_shuffled_objects_three_objects|3|0",
]

ALL_SUBSETS = LEADERBOARD_SUBSET + STABLE_SUBSET + HELM_SUBSET + AGIEVAL_SUBSET + BBH_SUBSET
