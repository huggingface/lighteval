# MIT License

# Copyright (c) 2024 The HuggingFace Team & Felipe Maia Polo

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

# ruff: noqa: F405, F403, F401
"""
See https://github.com/felipemaiapolo/tinyBenchmarks/ for the original code.

Test with `python run_evals_accelerate.py --model_args "pretrained=EleutherAI/pythia-70m" --tasks "extended|tiny:winogrande|0|0,extended|tiny:gsm8k|0|0,extended|tiny:hellaswag|0|0,extended|tiny:arc|0|0,extended|tiny:truthfulqa|0|0" --extended_tasks extended_tasks --output_dir "./evals"`
"""

import os
import pathlib
import pickle

import numpy as np
import requests
from aenum import extend_enum
from scipy.optimize import minimize

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import CorpusLevelMetricGrouping, Metrics
from lighteval.metrics.metrics_sample import ExactMatches, LoglikelihoodAcc
from lighteval.metrics.normalizations import gsm8k_normalizer
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig


# Utility functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def item_curve(theta, a, b):
    z = np.clip(a * theta - b, -30, 30).sum(axis=1)
    return sigmoid(z)


def fit_theta(responses_test, seen_items, A, B, theta_init=None, eps=1e-10, optimizer="BFGS"):
    D = A.shape[1]

    # Define the negative log likelihood function
    def neg_log_like(x):
        P = item_curve(x.reshape(1, D, 1), A[:, :, seen_items], B[:, :, seen_items]).squeeze()
        log_likelihood = np.sum(
            responses_test[seen_items] * np.log(P + eps) + (1 - responses_test[seen_items]) * np.log(1 - P + eps)
        )
        return -log_likelihood

    # Use the minimize function to find the ability parameters that minimize the negative log likelihood
    optimal_theta = minimize(neg_log_like, np.zeros(D), method=optimizer).x[None, :, None]
    return optimal_theta


# Evaluation function
class TinyCorpusAggregator:
    LEADEBRBOARD_SCENARIOS = ["truthfulqa", "gsm8k", "winogrande", "arc", "hellaswag"]
    BENCHS = ["lb", "mmlu"]
    METRICS = ["irt", "pirt", "gpirt"]
    # Not included yet:
    # - helm_lite (not avail on datasets)
    # - alpaca (needs to be added to lighteval first)

    def __init__(self, task: str):
        self.number_of_examples = 100
        if task not in self.LEADEBRBOARD_SCENARIOS + self.BENCHS:
            raise ValueError(f"Bench name must be one of {','.join(self.LEADEBRBOARD_SCENARIOS + self.BENCHS)}.")
        self.task = task
        self.scenario = "lb" if task in self.LEADEBRBOARD_SCENARIOS else task
        self.download()
        self.estimates = None
        self.num_samples = 0

    def download(self):
        # Likely to crash in // processes if we don't include the pkl
        path_dld = os.path.join(pathlib.Path(__file__).parent.resolve(), "tinyBenchmarks.pkl")
        # Downloading files
        if not os.path.isfile(path_dld):
            url = "https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/main/tinyBenchmarks/tinyBenchmarks.pkl"
            response = requests.get(url)
            if response.status_code == 200:
                # Write the content to a file
                with open(path_dld, "wb") as file:
                    file.write(response.content)

    def compute(self, **args):
        if self.task == "gsm8k":
            res = ExactMatches(
                strip_strings=True, normalize_pred=gsm8k_normalizer, normalize_gold=gsm8k_normalizer
            ).compute(**args)
            return dict.fromkeys(self.METRICS, res)
        else:
            res = LoglikelihoodAcc().compute(**args)
            return dict.fromkeys(self.METRICS, res)

    def aggregate(self, y_input):
        if len(y_input) == self.num_samples and self.estimates is not None:
            return self.estimates[self.task]

        # We load the weights for the relevant examples
        with open("extended_tasks/tiny_benchmarks/tinyBenchmarks.pkl", "rb") as handle:
            tinyBenchmarks = pickle.load(handle)

        seen_examples = tinyBenchmarks[self.scenario]["seen_examples"]
        examples_weights = tinyBenchmarks[self.scenario]["examples_weights"]
        irt_parameters = tinyBenchmarks[self.scenario]["irt_parameters"]
        A, B = irt_parameters["A"], irt_parameters["B"]
        optimal_lambdas = tinyBenchmarks[self.scenario]["optimal_lambdas"]
        scenarios_position = tinyBenchmarks[self.scenario]["scenarios_position"]
        subscenarios_position = tinyBenchmarks[self.scenario]["subscenarios_position"]

        N = np.max([np.max(x) for x in scenarios_position.values()]) + 1
        balance_weights = np.ones(N)
        for scenario in scenarios_position.keys():
            N_sce = len(scenarios_position[scenario])
            n_sub = len(subscenarios_position[scenario])
            for sub in subscenarios_position[scenario].keys():
                n_i = len(subscenarios_position[scenario][sub])
                balance_weights[subscenarios_position[scenario][sub]] = N_sce / (n_sub * n_i)

        # In case we use the big IRT model to estimate the performance of individual scenarios
        if self.task not in self.BENCHS:
            scenarios = [self.task]
            ind_scenario = (
                self.number_of_examples * ([i for i, s in enumerate(scenarios_position.keys()) if s == self.task][0])
            )
            seen_examples = seen_examples[ind_scenario : ind_scenario + self.number_of_examples]
        else:
            scenarios = list(scenarios_position.keys())

        # Creating vector y and estimating theta
        y = np.zeros(N)
        for i, j in enumerate(seen_examples):
            y[j] = y_input[i]

        # Getting estimates
        theta = fit_theta(y, seen_examples, A, B)
        estimates = {}
        unseen_examples = [i for i in range(N) if i not in seen_examples]

        for scenario in scenarios:
            N_sce = len(scenarios_position[scenario])
            seen_examples_sce = [s for s in seen_examples if s in scenarios_position[scenario]]
            unseen_examples_sce = [s for s in unseen_examples if s in scenarios_position[scenario]]

            data_part_IRTp = ((balance_weights * y)[seen_examples_sce]).mean()
            irt_part = (balance_weights * item_curve(theta.reshape(1, A.shape[1], 1), A, B))[
                0, [unseen_examples_sce]
            ].mean()
            IRTp_lambd = self.number_of_examples / N_sce
            IRT = (examples_weights[scenario] * y[seen_examples_sce]).sum()
            IRTp = IRTp_lambd * data_part_IRTp + (1 - IRTp_lambd) * irt_part
            IRTpp = optimal_lambdas[scenario] * IRT + (1 - optimal_lambdas[scenario]) * IRTp

            estimates[scenario] = {}
            estimates[scenario]["irt"] = IRT
            estimates[scenario]["pirt"] = IRTp
            estimates[scenario]["gpirt"] = IRTpp

        self.num_samples = len(y_input)
        self.estimates = estimates

        return estimates[self.task]


# TASK CREATION
task_params = [
    {
        "name": "winogrande",
        "dataset": "tinyBenchmarks/tinyWinogrande",
        "subset": "winogrande_xl",
        "prompt": prompt.winogrande,
        "splits": ["train", "validation", "test"],
        "evaluation_split": ["validation"],
    },
    {
        "name": "arc",
        "dataset": "tinyBenchmarks/tinyAI2_arc",
        "subset": "ARC-Challenge",
        "prompt": prompt.arc,
        "splits": ["train", "validation", "test"],
        "evaluation_split": ["validation"],
    },
    {
        "name": "hellaswag",
        "dataset": "tinyBenchmarks/tinyHellaswag",
        "subset": "default",
        "prompt": prompt.hellaswag_harness,
        "splits": ["train", "validation", "test"],
        "evaluation_split": ["validation"],
    },
    {
        "name": "mmlu",
        "dataset": "tinyBenchmarks/tinyMMLU",
        "subset": "all",
        "prompt": prompt.mmlu_harness,
        "splits": ["validation", "dev", "test"],
        "evaluation_split": ["test"],
    },
    {
        "name": "truthfulqa",
        "dataset": "tinyBenchmarks/tinyTruthfulQA",
        "subset": "multiple_choice",
        "prompt": prompt.truthful_qa_multiple_choice,
        "splits": ["validation"],
        "evaluation_split": ["validation"],
    },
    {
        "name": "gsm8k",
        "dataset": "tinyBenchmarks/tinyGSM8k",
        "subset": "main",
        "prompt": prompt.gsm8k,
        "splits": ["train", "test"],
        "evaluation_split": ["test"],
    },
    #    {
    #        "name": "alpacaeval",
    #        "dataset": "tinyBenchmarks/tinyAlpacaEval",
    #        "subset": "default"
    #    },
]

TASKS_TABLE = []
for task in task_params:
    name = task["name"]
    generation_size = None
    stop_sequence = None
    if name == "gsm8k":
        generation_size = 256
        stop_sequence = ["Question:", "Question"]
    task = LightevalTaskConfig(
        name=f"tiny:{name}",
        prompt_function=task["prompt"],
        suite=["extended"],
        hf_repo=task["dataset"],
        hf_subset=task["subset"],
        hf_avail_splits=task["splits"],
        evaluation_splits=task["evaluation_split"],
        few_shots_split=None,
        few_shots_select="random_sampling",
        metric=[f"tinybench_metric_{name}"],
        generation_size=generation_size,
        stop_sequence=stop_sequence,
    )
    TASKS_TABLE.append(task)

# CUSTOM METRIC
for task_param in task_params:
    name = task_param["name"]
    if name == "gsm8k":
        category = MetricCategory.GENERATIVE
        use_case = MetricUseCase.MATH
    else:
        category = MetricCategory.MULTICHOICE
        use_case = MetricUseCase.ACCURACY

    extend_enum(
        Metrics,
        f"tinybench_metric_{name}",
        CorpusLevelMetricGrouping(
            metric_name=TinyCorpusAggregator.METRICS,
            higher_is_better=dict.fromkeys(TinyCorpusAggregator.METRICS, True),
            sample_level_fn=TinyCorpusAggregator(name).compute,
            category=category,
            use_case=use_case,
            corpus_level_fn=TinyCorpusAggregator(name).aggregate,
        ),
    )
