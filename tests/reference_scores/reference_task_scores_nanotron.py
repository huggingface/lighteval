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
            "qem": 0.0,
            "qem_stderr": 0.0,
            "pem": 0.0,
            "pem_stderr": 0.0,
            "pqem": 0.25,
            "pqem_stderr": 0.25,
        },
        "helm:mmlu:college_chemistry:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.0,
            "qem_stderr": 0.0,
            "pem": 0.0,
            "pem_stderr": 0.0,
            "pqem": 0.0,
            "pqem_stderr": 0.0,
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
        "leaderboard:hellaswag:10": {"acc": 0.0, "acc_stderr": 0.0, "acc_norm": 0.25, "acc_norm_stderr": 0.25},
        "leaderboard:mmlu:abstract_algebra:5": {"acc": 0.5, "acc_stderr": 0.28867513459481287},
        "leaderboard:mmlu:college_chemistry:5": {"acc": 0.0, "acc_stderr": 0.0},
        "leaderboard:mmlu:computer_security:5": {"acc": 1.0, "acc_stderr": 0.0},
        "leaderboard:mmlu:us_foreign_policy:5": {"acc": 0.25, "acc_stderr": 0.25},
        "leaderboard:truthfulqa:mc:0": {
            "truthfulqa_mc1": 0.0,
            "truthfulqa_mc1_stderr": 0.0,
            "truthfulqa_mc2": 0.2509311177276107,
            "truthfulqa_mc2_stderr": 0.1476758333226878,
        },
        "lighteval:blimp:adjunct_island:0": {"acc": 0.5, "acc_stderr": 0.28867513459481287},
        "lighteval:blimp:ellipsis_n_bar_1:0": {"acc": 0.5, "acc_stderr": 0.28867513459481287},
        "lighteval:anli:r1:0": {"acc": 0.25, "acc_stderr": 0.25},
        "helm:mmlu:_average:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.0,
            "qem_stderr": 0.0,
            "pem": 0.0,
            "pem_stderr": 0.0,
            "pqem": 0.125,
            "pqem_stderr": 0.125,
        },
        "leaderboard:mmlu:_average:5": {"acc": 0.4375, "acc_stderr": 0.13466878364870322},
        "lighteval:blimp:_average:0": {"acc": 0.5, "acc_stderr": 0.28867513459481287},
    }
}
RESULTS_NANOTRON_FULL = {
    "LLama-119M": {
        "helm:boolq:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.0,
            "qem_stderr": 0.0,
            "pem": 0.0,
            "pem_stderr": 0.0,
            "pqem": 0.0009174311926605505,
            "pqem_stderr": 0.0005295170903140158,
        },
        "helm:hellaswag:5": {
            "em": 0.0008962358095996813,
            "em_stderr": 0.00029862623598600317,
            "qem": 0.0008962358095996813,
            "qem_stderr": 0.00029862623598600317,
            "pem": 0.0008962358095996813,
            "pem_stderr": 0.00029862623598600317,
            "pqem": 0.0008962358095996813,
            "pqem_stderr": 0.00029862623598600317,
        },
        "helm:mmlu:abstract_algebra:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.0,
            "qem_stderr": 0.0,
            "pem": 0.0,
            "pem_stderr": 0.0,
            "pqem": 0.24,
            "pqem_stderr": 0.042923469599092816,
        },
        "helm:mmlu:college_chemistry:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.01,
            "qem_stderr": 0.009999999999999998,
            "pem": 0.0,
            "pem_stderr": 0.0,
            "pqem": 0.22,
            "pqem_stderr": 0.04163331998932269,
        },
        "helm:mmlu:computer_security:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.01,
            "qem_stderr": 0.009999999999999998,
            "pem": 0.0,
            "pem_stderr": 0.0,
            "pqem": 0.28,
            "pqem_stderr": 0.045126085985421276,
        },
        "helm:mmlu:us_foreign_policy:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.0,
            "qem_stderr": 0.0,
            "pem": 0.0,
            "pem_stderr": 0.0,
            "pqem": 0.28,
            "pqem_stderr": 0.04512608598542128,
        },
        "leaderboard:gsm8k:5": {"qem": 0.0, "qem_stderr": 0.0},
        "leaderboard:arc:challenge:25": {
            "acc": 0.21331058020477817,
            "acc_stderr": 0.011970971742326334,
            "acc_norm": 0.2593856655290102,
            "acc_norm_stderr": 0.012808273573927092,
        },
        "leaderboard:hellaswag:10": {
            "acc": 0.25712009559848636,
            "acc_stderr": 0.004361529679492746,
            "acc_norm": 0.25941047600079664,
            "acc_norm_stderr": 0.004374153847826758,
        },
        "leaderboard:mmlu:abstract_algebra:5": {"acc": 0.21, "acc_stderr": 0.040936018074033256},
        "leaderboard:mmlu:college_chemistry:5": {"acc": 0.18, "acc_stderr": 0.03861229196653694},
        "leaderboard:mmlu:computer_security:5": {"acc": 0.31, "acc_stderr": 0.04648231987117316},
        "leaderboard:mmlu:us_foreign_policy:5": {"acc": 0.23, "acc_stderr": 0.04229525846816505},
        "leaderboard:truthfulqa:mc:0": {
            "truthfulqa_mc1": 0.23745410036719705,
            "truthfulqa_mc1_stderr": 0.014896277441041843,
            "truthfulqa_mc2": 0.47183673282563937,
            "truthfulqa_mc2_stderr": 0.01683985739593103,
        },
        "lighteval:blimp:adjunct_island:0": {"acc": 0.531, "acc_stderr": 0.015788865959539006},
        "lighteval:blimp:ellipsis_n_bar_1:0": {"acc": 0.489, "acc_stderr": 0.01581547119529269},
        "lighteval:anli:r1:0": {"acc": 0.366, "acc_stderr": 0.015240612726405756},
        "helm:mmlu:_average:5": {
            "em": 0.0,
            "em_stderr": 0.0,
            "qem": 0.005,
            "qem_stderr": 0.004999999999999999,
            "pem": 0.0,
            "pem_stderr": 0.0,
            "pqem": 0.255,
            "pqem_stderr": 0.04370224038981452,
        },
        "leaderboard:mmlu:_average:5": {"acc": 0.23249999999999998, "acc_stderr": 0.042081472094977104},
        "lighteval:blimp:_average:0": {"acc": 0.51, "acc_stderr": 0.01580216857741585},
    }
}
