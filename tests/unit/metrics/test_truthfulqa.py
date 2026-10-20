# MIT License

# Copyright (c) 2024 The HuggingFace Team

import numpy as np
import pytest

from lighteval.metrics.harness_compatibility.truthful_qa import TruthfulqaMCMetrics
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.tasks.truthfulqa import truthful_qa_multiple_choice_prompt


@pytest.mark.parametrize(
    "mc2_labels,mc2_probabilities,expected_mc2",
    [
        ([1, 1, 0], [0.1, 0.25, 0.15], 0.7),
        ([1, 0, 1], [0.1, 0.15, 0.25], 0.7),
        ([0, 1, 1], [0.15, 0.1, 0.25], 0.7),
        ([0, 1, 0], [0.1, 0.15, 0.25], 0.3),
    ],
    ids=["contiguous", "gap", "false-first", "single-nonleading-gold"],
)
@pytest.mark.parametrize("mc1_probabilities,expected_mc1", [([0.8, 0.2], True), ([0.2, 0.8], False)])
def test_truthfulqa_mc2_follows_labels(mc2_labels, mc2_probabilities, expected_mc2, mc1_probabilities, expected_mc1):
    doc = truthful_qa_multiple_choice_prompt(
        {
            "question": "Which answer is true?",
            "mc1_targets": {"choices": ["True", "False"], "labels": [1, 0]},
            "mc2_targets": {
                "choices": [f"Answer {probability}" for probability in mc2_probabilities],
                "labels": mc2_labels,
            },
        }
    )
    model_response = ModelResponse(logprobs=np.log(mc1_probabilities + mc2_probabilities).tolist())

    result = TruthfulqaMCMetrics().compute(doc, model_response)

    assert result["truthfulqa_mc1"] == expected_mc1
    assert result["truthfulqa_mc2"] == pytest.approx(expected_mc2)
