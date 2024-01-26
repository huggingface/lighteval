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
