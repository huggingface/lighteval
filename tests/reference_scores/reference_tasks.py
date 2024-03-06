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
    "lighteval|mmlu:abstract_algebra|5|0",
    "helm|mmlu:abstract_algebra|5|0",
    # "original|mmlu:abstract_algebra|5",
    "lighteval|mmlu:college_chemistry|5|0",
    "helm|mmlu:college_chemistry|5|0",
    # "original|mmlu:college_chemistry|5",
    "lighteval|mmlu:computer_security|5|0",
    "helm|mmlu:computer_security|5|0",
    # "original|mmlu:computer_security|5",
    "lighteval|mmlu:us_foreign_policy|5|0",
    "helm|mmlu:us_foreign_policy|5|0",
    # "original|mmlu:us_foreign_policy|5",
]

LEADERBOARD_SUBSET = [
    "lighteval|arc:challenge|25|0",
    "lighteval|truthfulqa:mc|0|0",
    "lighteval|hellaswag|10|0",
    "lighteval|mmlu:abstract_algebra|5|0",
    "lighteval|mmlu:college_chemistry|5|0",
    "lighteval|mmlu:computer_security|5|0",
    "lighteval|mmlu:us_foreign_policy|5|0",
    "lighteval|gsm8k|5|0",
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
    "helm|hellaswag|5|0",
]
