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

RESULTS_NANOTRON_LITE = {
    "LLama-119M": {
        "helm:boolq:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.0,
            "qem_stderr": 0.0,
            "pem": 0.0,
            "pem_stderr": 0.0,
            "pqem": 0.0,
            "pqem_stderr": 0.0,
        },
        "helm:hellaswag:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.0,
            "qem_stderr": 0.0,
            "pem": 0.0,
            "pem_stderr": 0.0,
            "pqem": 0.0,
            "pqem_stderr": 0.0,
        },
        "helm:mmlu:abstract_algebra:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.25,
            "qem_stderr": 0.25,
            "pem": 0.0,
            "pem_stderr": 0.0,
            "pqem": 0.25,
            "pqem_stderr": 0.25,
        },
        "helm:mmlu:college_chemistry:5": {
            "em": 0.25,
            "em_stderr": 0.25,
            "qem": 0.25,
            "qem_stderr": 0.25,
            "pem": 0.25,
            "pem_stderr": 0.25,
            "pqem": 0.25,
            "pqem_stderr": 0.25,
        },
        "helm:mmlu:computer_security:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.0,
            "qem_stderr": 0.0,
            "pem": 0.0,
            "pem_stderr": 0.0,
            "pqem": 0.0,
            "pqem_stderr": 0.0,
        },
        "helm:mmlu:us_foreign_policy:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.0,
            "qem_stderr": 0.0,
            "pem": 0.0,
            "pem_stderr": 0.0,
            "pqem": 0.25,
            "pqem_stderr": 0.25,
        },
        "leaderboard:gsm8k:5": {"qem": 0.0, "qem_stderr": 0.0},
        "leaderboard:arc:challenge:25": {
            "acc": 0.5,
            "acc_stderr": 0.28867513459481287,
            "acc_norm": 0.5,
            "acc_norm_stderr": 0.28867513459481287,
        },
        "leaderboard:hellaswag:10": {"acc": 0.0, "acc_stderr": 0.0, "acc_norm": 0.0, "acc_norm_stderr": 0.0},
        "leaderboard:mmlu:abstract_algebra:5": {"acc": 0.25, "acc_stderr": 0.25},
        "leaderboard:mmlu:college_chemistry:5": {"acc": 0.0, "acc_stderr": 0.0},
        "leaderboard:mmlu:computer_security:5": {"acc": 0.25, "acc_stderr": 0.25},
        "leaderboard:mmlu:us_foreign_policy:5": {"acc": 0.25, "acc_stderr": 0.25},
        "leaderboard:truthfulqa:mc:0": {
            "truthfulqa_mc1": 0.5,
            "truthfulqa_mc1_stderr": 0.28867513459481287,
            "truthfulqa_mc2": 0.4317633664159167,
            "truthfulqa_mc2_stderr": 0.25500097927438214,
        },
        "lighteval:blimp:adjunct_island:0": {"acc": 0.5, "acc_stderr": 0.28867513459481287},
        "lighteval:blimp:ellipsis_n_bar_1:0": {"acc": 0.25, "acc_stderr": 0.25},
        "lighteval:anli:r1:0": {"acc": 0.25, "acc_stderr": 0.25},
        "helm:mmlu:_average:5": {
            "em": 0.0625,
            "em_stderr": 0.0625,
            "qem": 0.125,
            "qem_stderr": 0.125,
            "pem": 0.0625,
            "pem_stderr": 0.0625,
            "pqem": 0.1875,
            "pqem_stderr": 0.1875,
        },
        "leaderboard:mmlu:_average:5": {"acc": 0.1875, "acc_stderr": 0.1875},
        "lighteval:blimp:_average:0": {"acc": 0.375, "acc_stderr": 0.26933756729740643},
    }
}
RESULTS_NANOTRON_FULL = {
    "LLama-119M": {
        "helm:boolq:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.0006116207951070336,
            "qem_stderr": 0.0004324150578206582,
            "pem": 0.0003058103975535168,
            "pem_stderr": 0.00030581039755354006,
            "pqem": 0.0024464831804281344,
            "pqem_stderr": 0.0008640358432108371,
        },
        "helm:hellaswag:5": {
            "em": 0.0016928898625771759,
            "em_stderr": 0.00041025884285982294,
            "qem": 0.0016928898625771759,
            "qem_stderr": 0.00041025884285982294,
            "pem": 0.0016928898625771759,
            "pem_stderr": 0.00041025884285982294,
            "pqem": 0.0016928898625771759,
            "pqem_stderr": 0.00041025884285982294,
        },
        "helm:mmlu:abstract_algebra:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.0,
            "qem_stderr": 0.0,
            "pem": 0.12,
            "pem_stderr": 0.03265986323710906,
            "pqem": 0.36,
            "pqem_stderr": 0.04824181513244218,
        },
        "helm:mmlu:college_chemistry:5": {
            "em": 0.02,
            "em_stderr": 0.014070529413628952,
            "qem": 0.02,
            "qem_stderr": 0.014070529413628952,
            "pem": 0.02,
            "pem_stderr": 0.014070529413628952,
            "pqem": 0.22,
            "pqem_stderr": 0.04163331998932269,
        },
        "helm:mmlu:computer_security:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.01,
            "qem_stderr": 0.009999999999999998,
            "pem": 0.07,
            "pem_stderr": 0.025643239997624283,
            "pqem": 0.35,
            "pqem_stderr": 0.04793724854411019,
        },
        "helm:mmlu:us_foreign_policy:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.0,
            "qem_stderr": 0.0,
            "pem": 0.02,
            "pem_stderr": 0.014070529413628954,
            "pqem": 0.32,
            "pqem_stderr": 0.046882617226215034,
        },
        "leaderboard:gsm8k:5": {"qem": 0.0, "qem_stderr": 0.0},
        "leaderboard:arc:challenge:25": {
            "acc": 0.20733788395904437,
            "acc_stderr": 0.011846905782971364,
            "acc_norm": 0.24829351535836178,
            "acc_norm_stderr": 0.012624912868089772,
        },
        "leaderboard:hellaswag:10": {
            "acc": 0.2577175861382195,
            "acc_stderr": 0.004364838000335622,
            "acc_norm": 0.26030671181039633,
            "acc_norm_stderr": 0.00437905135702414,
        },
        "leaderboard:mmlu:abstract_algebra:5": {"acc": 0.29, "acc_stderr": 0.045604802157206845},
        "leaderboard:mmlu:college_chemistry:5": {"acc": 0.2, "acc_stderr": 0.04020151261036846},
        "leaderboard:mmlu:computer_security:5": {"acc": 0.32, "acc_stderr": 0.04688261722621503},
        "leaderboard:mmlu:us_foreign_policy:5": {"acc": 0.24, "acc_stderr": 0.042923469599092816},
        "leaderboard:truthfulqa:mc:0": {
            "truthfulqa_mc1": 0.23011015911872704,
            "truthfulqa_mc1_stderr": 0.01473455795980776,
            "truthfulqa_mc2": 0.4796459449168539,
            "truthfulqa_mc2_stderr": 0.016677952132527703,
        },
        "lighteval:blimp:adjunct_island:0": {"acc": 0.506, "acc_stderr": 0.015818160898606715},
        "lighteval:blimp:ellipsis_n_bar_1:0": {"acc": 0.513, "acc_stderr": 0.015813952101896622},
        "lighteval:anli:r1:0": {"acc": 0.315, "acc_stderr": 0.014696631960792496},
        "helm:mmlu:_average:5": {
            "em": 0.005,
            "em_stderr": 0.003517632353407238,
            "qem": 0.0075,
            "qem_stderr": 0.006017632353407238,
            "pem": 0.057499999999999996,
            "pem_stderr": 0.021611040515497813,
            "pqem": 0.3125,
            "pqem_stderr": 0.046173750223022524,
        },
        "leaderboard:mmlu:_average:5": {"acc": 0.2625, "acc_stderr": 0.04390310039822079},
        "lighteval:blimp:_average:0": {"acc": 0.5095000000000001, "acc_stderr": 0.01581605650025167},
    }
}
