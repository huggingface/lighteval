# MIT License

# Copyright (c) 2024 The HuggingFace Team & Felipe Maia Polo

# See https://github.com/felipemaiapolo/tinyBenchmarks/ for the original code

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
Custom evaluation tasks for lighteval. Copy this file and complete it with the info for your task.

This file generally create just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Author:
"""
import os
import pickle

import numpy as np
import requests
from aenum import extend_enum
from scipy.optimize import minimize

from lighteval.metrics import Metrics
from lighteval.metrics.metrics import CorpusLevelMetricGrouping
from lighteval.metrics.metrics_sample import LoglikelihoodAcc
from lighteval.metrics.utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


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
    # Not included yet:
    # - helm_lite (not avail on datasets)
    # - alpaca (needs to be added to lighteval first)

    def __init__(self, task: str, metric: str):
        self.number_of_examples = 100
        if task not in self.LEADEBRBOARD_SCENARIOS + self.BENCHS:
            raise ValueError(f"Bench name must be one of {','.join(self.LEADEBRBOARD_SCENARIOS + self.BENCHS)}.")
        self.task = task
        self.scenario = "lb" if task in self.LEADEBRBOARD_SCENARIOS else task
        self.download()
        self.estimates = None
        self.num_samples = 0
        self.metric = metric

    def download(self):
        # Downloading files
        if not os.path.isfile("extended_tasks/tiny_benchmarks/tinyBenchmarks.pkl"):
            url = "https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/main/tinyBenchmarks/tinyBenchmarks.pkl"
            response = requests.get(url)
            if response.status_code == 200:
                # Write the content to a file
                with open("extended_tasks/tiny_benchmarks/tinyBenchmarks.pkl", "wb") as file:
                    file.write(response.content)

    def aggregate(self, y_input):
        if len(y_input) == self.num_samples and self.estimates is not None:
            return self.estimates[self.task][self.metric]

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

        return estimates[self.task][self.metric]


task_params = [
    {
        "name": "winogrande",
        "dataset": "tinyBenchmarks/tinyWinogrande",
        "subset": "winogrande_xl",
        "prompt": "winogrande",
    },
    {"name": "arc", "dataset": "tinyBenchmarks/tinyAI2_arc", "subset": "ARC-Challenge", "prompt": "arc"},
    {
        "name": "hellaswag",
        "dataset": "tinyBenchmarks/tinyHellaswag",
        "subset": "default",
        "prompt": "hellaswag_harness",
    },
    {
        "name": "gsm8k",
        "dataset": "tinyBenchmarks/tinyGSM8k",
        "subset": "main",
        "prompt": "gsm8k",
    },
    {"name": "mmlu", "dataset": "tinyBenchmarks/tinyMMLU", "subset": "all", "prompt": "mmlu_harness"},
    {
        "name": "truthfulqa",
        "dataset": "tinyBenchmarks/tinyTruthfulQA",
        "subset": "multiple_choice",
        "prompt": "truthful_qa_multiple_choice",
    },
    #    {
    #        "name": "alpacaeval",
    #        "dataset": "tinyBenchmarks/tinyAlpacaEval",
    #        "subset": "default"
    #    },
]

_TASKS = []
for task_param in task_params:
    name, dataset, subset, prompt = (
        task_param["name"],
        task_param["dataset"],
        task_param["subset"],
        task_params["prompt"],
    )
    task = LightevalTaskConfig(
        name="tiny:{name}",
        prompt_function=prompt,
        suite=["extended"],
        hf_repo=dataset,
        hf_subset=subset,
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["validation"],
        few_shots_split=None,
        few_shots_select="random_sampling",
        metric=[f"custom_metric_{name}"],
    )
    _TASKS.append(task)

# CUSTOM METRIC
metrics = ["irt", "pirt", "gpirt"]


def sample_level(**args):
    result = LoglikelihoodAcc().compute(**args)
    return {m: result for m in metrics}


def custom_metric(name):
    return CorpusLevelMetricGrouping(
        metric=metrics,
        higher_is_better={m: True for m in metrics},
        sample_level_fn=sample_level,
        category=MetricCategory.MULTICHOICE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn={m: TinyCorpusAggregator(name, m).aggregate for m in metrics},
    )


for task_param in task_params:
    extend_enum(Metrics, "custom_metric_{name}", custom_metric(name))

# MODULE LOGIC
# You should not need to touch this
# Convert to dict for lighteval
TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
