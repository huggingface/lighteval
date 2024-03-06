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

"""Results on the full suite."""
RESULTS_FULL = {
    "gpt2-xl": {
        "lighteval|anli:r1|0|0": {"acc": 0.337, "acc_stderr": 0.014955087918653605},
        "lighteval|blimp:adjunct_island|0|0": {"acc": 0.893, "acc_stderr": 0.009779910359847165},
        "lighteval|blimp:ellipsis_n_bar_1|0|0": {"acc": 0.909, "acc_stderr": 0.009099549538400246},
        "lighteval|arc:challenge|25|0": {
            "acc": 0.257679180887372,
            "acc_stderr": 0.0127807705627684,
            "acc_norm": 0.302901023890785,
            "acc_norm_stderr": 0.013428241573185347,
        },
        "lighteval|hellaswag|10|0": {
            "acc": 0.3981278629755029,
            "acc_stderr": 0.004885116465550274,
            "acc_norm": 0.5139414459271061,
            "acc_norm_stderr": 0.004987841367402517,
        },
        "lighteval|mmlu:abstract_algebra|5|0": {
            "acc": 0.26,
            "acc_stderr": 0.04408440022768081,
        },
        "lighteval|mmlu:college_chemistry|5|0": {
            "acc": 0.24,
            "acc_stderr": 0.04292346959909284,
        },
        "lighteval|mmlu:computer_security|5|0": {
            "acc": 0.29,
            "acc_stderr": 0.04560480215720684,
        },
        "lighteval|mmlu:us_foreign_policy|5|0": {
            "acc": 0.22,
            "acc_stderr": 0.041633319989322695,
        },
        "lighteval|truthfulqa:mc|0|0": {
            "truthfulqa_mc1": 0.22031823745410037,
            "truthfulqa_mc1_stderr": 0.0145090451714873,
            "truthfulqa_mc2": 0.3853407807086726,
            "truthfulqa_mc2_stderr": 0.014058180381569934,
        },
        "helm|mmlu:abstract_algebra|5|0": {
            "em": 0.26,
            "em_stderr": 0.04408440022768081,
            "pqem": 0.48,
            "pqem_stderr": 0.05021167315686779,
        },
        "helm|mmlu:college_chemistry|5|0": {
            "em": 0.22,
            "em_stderr": 0.041633319989322695,
            "pqem": 0.33,
            "pqem_stderr": 0.04725815626252604,
        },
        "helm|mmlu:computer_security|5|0": {
            "em": 0.22,
            "em_stderr": 0.04163331998932269,
            "pqem": 0.41,
            "pqem_stderr": 0.04943110704237101,
        },
        "helm|mmlu:us_foreign_policy|5|0": {
            "em": 0.2,
            "em_stderr": 0.04020151261036846,
            "pqem": 0.48,
            "pqem_stderr": 0.050211673156867795,
        },
        "helm|boolq|5|0": {
            "em": 0.5963302752293578,
            "em_stderr": 0.008581220435616823,
            "qem": 0.5966360856269113,
            "qem_stderr": 0.008580168554889729,
            "pem": 0.6048929663608563,
            "pem_stderr": 0.008550454248280891,
            "pqem": 0.6051987767584098,
            "pqem_stderr": 0.008549304887647408,
        },
        "helm|hellaswag|5|0": {
            "em": 0.2524397530372436,
            "em_stderr": 0.00433524343448683,
            "qem": 0.2524397530372436,
            "qem_stderr": 0.00433524343448683,
            "pem": 0.2524397530372436,
            "pem_stderr": 0.00433524343448683,
            "pqem": 0.2526389165504879,
            "pqem_stderr": 0.004336375492801806,
        },
        "lighteval|gsm8k|5|0": {"qem": 0.009097801364670205, "qem_stderr": 0.002615326510775673},
        # "gsm8k": {"acc": 0.009097801364670205, "acc_stderr": 0.002615326510775673}, Actual harness results
    },
    "gpt2": {
        "lighteval|anli:r1|0|0": {"acc": 0.341, "acc_stderr": 0.014998131348402704},
        "lighteval|blimp:adjunct_island|0|0": {"acc": 0.913, "acc_stderr": 0.00891686663074591},
        "lighteval|blimp:ellipsis_n_bar_1|0|0": {"acc": 0.842, "acc_stderr": 0.011539894677559568},
        "lighteval|arc:challenge|25|0": {
            "acc": 0.20051194539249148,
            "acc_stderr": 0.011700318050499373,
            "acc_norm": 0.21928327645051193,
            "acc_norm_stderr": 0.012091245787615723,
        },
        "lighteval|hellaswag|10|0": {
            "acc": 0.29267078271260705,
            "acc_stderr": 0.004540586983229992,
            "acc_norm": 0.3157737502489544,
            "acc_norm_stderr": 0.0046387332023738815,
        },
        "lighteval|mmlu:abstract_algebra|5|0": {
            "acc": 0.21,
            "acc_stderr": 0.040936018074033256,
        },
        "lighteval|mmlu:college_chemistry|5|0": {
            "acc": 0.2,
            "acc_stderr": 0.04020151261036846,
        },
        "lighteval|mmlu:computer_security|5|0": {
            "acc": 0.16,
            "acc_stderr": 0.03684529491774709,
        },
        "lighteval|mmlu:us_foreign_policy|5|0": {
            "acc": 0.27,
            "acc_stderr": 0.04461960433384739,
        },
        "lighteval|truthfulqa:mc|0|0": {
            "truthfulqa_mc1": 0.22766217870257038,
            "truthfulqa_mc1_stderr": 0.01467925503211107,
            "truthfulqa_mc2": 0.40693581786045147,
            "truthfulqa_mc2_stderr": 0.014921948720110469,
        },
        "helm|mmlu:abstract_algebra|5|0": {
            "em": 0.21,
            "em_stderr": 0.040936018074033256,
            "pqem": 0.37,
            "pqem_stderr": 0.048523658709391,
        },
        "helm|mmlu:college_chemistry|5|0": {
            "em": 0.25,
            "em_stderr": 0.04351941398892446,
            "pqem": 0.3,
            "pqem_stderr": 0.046056618647183814,
        },
        "helm|mmlu:computer_security|5|0": {
            "em": 0.14,
            "em_stderr": 0.03487350880197769,
            "pqem": 0.41,
            "pqem_stderr": 0.04943110704237102,
        },
        "helm|mmlu:us_foreign_policy|5|0": {
            "em": 0.27,
            "em_stderr": 0.044619604333847394,
            "pqem": 0.51,
            "pqem_stderr": 0.05024183937956911,
        },
        "helm|boolq|5|0": {
            "em": 0.5406727828746177,
            "em_stderr": 0.00871607349717106,
            "qem": 0.5406727828746177,
            "qem_stderr": 0.00871607349717106,
            "pem": 0.5406727828746177,
            "pem_stderr": 0.00871607349717106,
            "pqem": 0.5406727828746177,
            "pqem_stderr": 0.00871607349717106,
        },
        "helm|hellaswag|5|0": {
            "em": 0.24417446723760208,
            "em_stderr": 0.004287186932515093,
            "qem": 0.24417446723760208,
            "qem_stderr": 0.004287186932515093,
            "pem": 0.24417446723760208,
            "pem_stderr": 0.004287186932515093,
            "pqem": 0.24427404899422425,
            "pqem_stderr": 0.00428777857558616,
        },
        "lighteval|gsm8k|5|0": {"qem": 0.006065200909780136, "qem_stderr": 0.0021386703014604626},
        # "harness|gsm8k|5|0": {"acc": 0.004548900682335102, "acc_stderr": 0.0018535550440036204}, Actual harness results
    },
}

"""Results on 10 samples, using no parallelism"""
RESULTS_LITE = {
    "gpt2-xl": {
        "lighteval|anli:r1|0|0": {"acc": 0.4, "acc_stderr": 0.16329931618554522},
        "lighteval|blimp:adjunct_island|0|0": {"acc": 0.9, "acc_stderr": 0.09999999999999999},
        "lighteval|blimp:ellipsis_n_bar_1|0|0": {"acc": 0.8, "acc_stderr": 0.13333333333333333},
        "lighteval|arc:challenge|25|0": {
            "acc": 0.2,
            "acc_stderr": 0.13333333333333333,
            "acc_norm": 0.1,
            "acc_norm_stderr": 0.09999999999999999,
        },
        "lighteval|hellaswag|10|0": {
            "acc": 0.4,
            "acc_stderr": 0.16329931618554522,
            "acc_norm": 0.8,
            "acc_norm_stderr": 0.13333333333333333,
        },
        "lighteval|mmlu:abstract_algebra|5|0": {
            "acc": 0.3,
            "acc_stderr": 0.15275252316519466,
        },
        "lighteval|mmlu:college_chemistry|5|0": {
            "acc": 0.2,
            "acc_stderr": 0.13333333333333333,
        },
        "lighteval|mmlu:computer_security|5|0": {
            "acc": 0.4,
            "acc_stderr": 0.1632993161855452,
        },
        "lighteval|mmlu:us_foreign_policy|5|0": {
            "acc": 0.3,
            "acc_stderr": 0.15275252316519466,
        },
        "lighteval|truthfulqa:mc|0|0": {
            "truthfulqa_mc1": 0.3,
            "truthfulqa_mc1_stderr": 0.15275252316519466,
            "truthfulqa_mc2": 0.4528717362471066,
            "truthfulqa_mc2_stderr": 0.14740763841220644,
        },
        "helm|mmlu:abstract_algebra|5|0": {
            "em": 0.3,
            "em_stderr": 0.15275252316519466,
            "pqem": 0.4,
            "pqem_stderr": 0.16329931618554522,
        },
        "helm|mmlu:college_chemistry|5|0": {
            "em": 0.1,
            "em_stderr": 0.09999999999999999,
            "pqem": 0.2,
            "pqem_stderr": 0.13333333333333333,
        },
        "helm|mmlu:computer_security|5|0": {
            "em": 0.1,
            "em_stderr": 0.09999999999999999,
            "pqem": 0.3,
            "pqem_stderr": 0.15275252316519464,
        },
        "helm|mmlu:us_foreign_policy|5|0": {
            "em": 0.2,
            "em_stderr": 0.13333333333333333,
            "pqem": 0.5,
            "pqem_stderr": 0.16666666666666666,
        },
        "helm|boolq|5|0": {
            "em": 0.6,
            "em_stderr": 0.16329931618554522,
            "qem": 0.6,
            "qem_stderr": 0.16329931618554522,
            "pem": 0.6,
            "pem_stderr": 0.16329931618554522,
            "pqem": 0.6,
            "pqem_stderr": 0.16329931618554522,
        },
        "helm|hellaswag|5|0": {
            "em": 0.2,
            "em_stderr": 0.13333333333333333,
            "qem": 0.2,
            "qem_stderr": 0.13333333333333333,
            "pem": 0.2,
            "pem_stderr": 0.13333333333333333,
            "pqem": 0.2,
            "pqem_stderr": 0.13333333333333333,
        },
        "lighteval|gsm8k|5|0": {"qem": 0.1, "qem_stderr": 0.09999999999999999},
    },
    "gpt2": {
        "lighteval|anli:r1|0|0": {"acc": 0.5, "acc_stderr": 0.16666666666666666},
        "lighteval|blimp:adjunct_island|0|0": {"acc": 0.8, "acc_stderr": 0.13333333333333333},
        "lighteval|blimp:ellipsis_n_bar_1|0|0": {"acc": 0.7, "acc_stderr": 0.15275252316519466},
        "lighteval|arc:challenge|25|0": {
            "acc": 0.3,
            "acc_stderr": 0.15275252316519466,
            "acc_norm": 0.0,
            "acc_norm_stderr": 0.0,
        },
        "lighteval|hellaswag|10|0": {
            "acc": 0.4,
            "acc_stderr": 0.16329931618554522,
            "acc_norm": 0.6,
            "acc_norm_stderr": 0.16329931618554522,
        },
        "lighteval|mmlu:abstract_algebra|5|0": {
            "acc": 0.4,
            "acc_stderr": 0.16329931618554522,
        },
        "lighteval|mmlu:college_chemistry|5|0": {
            "acc": 0.1,
            "acc_stderr": 0.09999999999999999,
        },
        "lighteval|mmlu:computer_security|5|0": {
            "acc": 0.1,
            "acc_stderr": 0.09999999999999999,
        },
        "lighteval|mmlu:us_foreign_policy|5|0": {
            "acc": 0.3,
            "acc_stderr": 0.15275252316519466,
        },
        "lighteval|truthfulqa:mc|0|0": {
            "truthfulqa_mc1": 0.3,
            "truthfulqa_mc1_stderr": 0.15275252316519466,
            "truthfulqa_mc2": 0.4175889390166028,
            "truthfulqa_mc2_stderr": 0.14105533101540416,
        },
        "helm|mmlu:abstract_algebra|5|0": {
            "em": 0.4,
            "em_stderr": 0.16329931618554522,
            "pqem": 0.4,
            "pqem_stderr": 0.16329931618554522,
        },
        "helm|mmlu:college_chemistry|5|0": {
            "em": 0.3,
            "em_stderr": 0.15275252316519466,
            "pqem": 0.4,
            "pqem_stderr": 0.16329931618554522,
        },
        "helm|mmlu:computer_security|5|0": {
            "em": 0.1,
            "em_stderr": 0.09999999999999999,
            "pqem": 0.3,
            "pqem_stderr": 0.15275252316519464,
        },
        "helm|mmlu:us_foreign_policy|5|0": {
            "em": 0.3,
            "em_stderr": 0.15275252316519466,
            "pqem": 0.6,
            "pqem_stderr": 0.16329931618554522,
        },
        "helm|boolq|5|0": {
            "em": 0.4,
            "em_stderr": 0.16329931618554522,
            "qem": 0.4,
            "qem_stderr": 0.16329931618554522,
            "pem": 0.4,
            "pem_stderr": 0.16329931618554522,
            "pqem": 0.4,
            "pqem_stderr": 0.16329931618554522,
        },
        "helm|hellaswag|5|0": {
            "em": 0.1,
            "em_stderr": 0.09999999999999999,
            "qem": 0.1,
            "qem_stderr": 0.09999999999999999,
            "pem": 0.1,
            "pem_stderr": 0.09999999999999999,
            "pqem": 0.1,
            "pqem_stderr": 0.09999999999999999,
        },
        "lighteval|gsm8k|5|0": {"qem": 0.0, "qem_stderr": 0.0},
    },
}
