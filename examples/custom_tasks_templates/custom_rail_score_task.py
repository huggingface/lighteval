# MIT License
#
# Copyright (c) 2024 Responsible AI Labs
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

"""RAIL Score evaluation for LightEval.

Evaluates model outputs across 8 responsible AI dimensions using the
RAIL Score API (https://responsibleailabs.ai). Each dimension appears
as a separate metric in LightEval results.

Dimensions: fairness, safety, reliability, transparency, privacy,
accountability, inclusivity, user_impact. Each scored 0-10 by the API,
normalized to 0-1 for LightEval.

Setup:
    pip install rail-score-sdk
    export RAIL_API_KEY="rail_..."

Usage:
    lighteval accelerate \
        "model_name=HuggingFaceH4/zephyr-7b-beta" \
        "rail_score:default|0" \
        --custom-tasks custom_rail_score_task.py
"""

import logging
import os

import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import MetricGrouping
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


logger = logging.getLogger(__name__)

DIMENSIONS = [
    "fairness",
    "safety",
    "reliability",
    "transparency",
    "privacy",
    "accountability",
    "inclusivity",
    "user_impact",
]

METRIC_NAMES = [f"rail_{dim}" for dim in DIMENSIONS] + ["rail_overall"]


# ---------------------------------------------------------------------------
# Metric: RAIL Score computation
# ---------------------------------------------------------------------------


class RAILScoreComputation(SampleLevelComputation):
    """Call RAIL Score API and return per-dimension scores (0-1).

    The client is lazily initialized on first use so that importing this
    module does not require RAIL_API_KEY to be set at import time.
    """

    def __init__(self, mode: str = "basic", domain: str = "general"):
        self.mode = mode
        self.domain = domain
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from rail_score_sdk import RailScoreClient

            api_key = os.environ.get("RAIL_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "RAIL_API_KEY environment variable is not set. Get a free key at https://responsibleailabs.ai"
                )
            self._client = RailScoreClient(api_key=api_key)
        return self._client

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> dict[str, float]:
        response_text = model_response.text[0] if model_response.text else ""

        if not response_text or len(response_text.strip()) < 10:
            return dict.fromkeys(METRIC_NAMES, 0.0)

        # Use domain from doc metadata if available, fall back to default
        domain = (doc.specific or {}).get("domain", self.domain)

        try:
            result = self.client.eval(
                content=response_text,
                mode=self.mode,
                domain=domain,
                context=doc.query if doc.query else None,
                include_explanations=(self.mode == "deep"),
            )
        except Exception:
            logger.exception("RAIL Score API call failed")
            return dict.fromkeys(METRIC_NAMES, 0.0)

        scores = {f"rail_{dim}": ds.score / 10.0 for dim, ds in result.dimension_scores.items()}
        scores["rail_overall"] = result.rail_score.score / 10.0
        return scores


# Build the MetricGrouping -- all 8 dimensions + overall as separate metrics
rail_score_metric = MetricGrouping(
    metric_name=METRIC_NAMES,
    higher_is_better=dict.fromkeys(METRIC_NAMES, True),
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=RAILScoreComputation(mode="basic"),
    corpus_level_fn=dict.fromkeys(METRIC_NAMES, np.mean),
)

# Register with LightEval's metric registry
extend_enum(Metrics, "RAIL_SCORE", rail_score_metric)


# ---------------------------------------------------------------------------
# Prompt function
# ---------------------------------------------------------------------------


def rail_score_prompt(line: dict, task_name: str) -> Doc:
    """Convert a dataset row to a LightEval Doc for generative evaluation."""
    # Handle common column name variations
    prompt = line.get("prompt") or line.get("question") or line.get("input", "")
    domain = line.get("domain", "general")

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[""],
        gold_index=0,
        specific={"domain": domain},
    )


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

# Uses the RAIL-HH-10K dataset which contains prompt/response pairs
# across multiple domains for responsible AI evaluation.
task = LightevalTaskConfig(
    name="rail_score:default",
    prompt_function=rail_score_prompt,
    hf_repo="responsible-ai-labs/RAIL-HH-10K",
    hf_subset="default",
    evaluation_splits=["test"],
    metrics=[Metrics.RAIL_SCORE],
    generation_size=512,
    stop_sequence=["\n\n"],
)

TASKS_TABLE = [task]

if __name__ == "__main__":
    print("RAIL Score custom task loaded successfully.")
    print(f"Registered metric: {Metrics.RAIL_SCORE}")
    print(f"Metric names: {METRIC_NAMES}")
    print(f"Task: {task.name}")
