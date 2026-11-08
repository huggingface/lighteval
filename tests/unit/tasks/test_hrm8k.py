# MIT License

# Copyright (c) 2026 The HuggingFace Team

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

import pytest

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.registry import Registry
from lighteval.tasks.tasks.hrm8k import (
    HRM8K_SUBSETS,
    TASKS_TABLE,
    _gold,
    hrm8k_metric,
    hrm8k_mmmlu_prompt,
    hrm8k_prompt,
)


REVISION = "c360cabf8d733a82455565358b3dc965aab9ba8d"


def _score(gold, prediction: str) -> float:
    # Synthetic rows only; no dataset text is redistributed.
    doc = hrm8k_prompt({"question": "문제", "answer": gold}, "hrm8k:ksm")
    return hrm8k_metric.sample_level_fn.compute(doc=doc, model_response=ModelResponse(text=[prediction]))


# ---------- prompt ----------


def test_prompt_exact_snapshot():
    doc = hrm8k_prompt({"question": "  1 더하기 1은?  ", "answer": 2.0}, "hrm8k:gsm8k")
    assert doc.query == (
        "주어진 문제를 풀어보세요.\n"
        "문제를 푼 후, 최종 답변을 다음과 같은 형식으로 작성하세요: $\\boxed{N}$.\n\n"
        "문제: 1 더하기 1은?\n답변:"
    )
    assert doc.choices == ["$2$"]
    assert doc.gold_index == 0
    assert doc.task_name == "hrm8k:gsm8k"


def test_mmmlu_prompt_asks_for_option_number():
    doc = hrm8k_mmmlu_prompt({"question": "보기 중 무엇?\n1. 가\n2. 나\n3. 다\n4. 라", "answer": 3}, "hrm8k:mmmlu")
    assert doc.query == (
        "주어진 문제를 풀어보세요.\n"
        "문제를 푼 후, 주어진 선택지 (1, 2, 3, 4) 중 최종 선택지를 다음 형식으로 작성하세요: $\\boxed{N}$.\n\n"
        "문제: 보기 중 무엇?\n1. 가\n2. 나\n3. 다\n4. 라\n답변:"
    )
    assert doc.choices == ["$3$"]


@pytest.mark.parametrize(
    "answer,gold",
    [
        (18.0, "$18$"),
        (-10.0, "$-10$"),
        (3, "$3$"),
        (0.5, "$0.5$"),
        ("-8\\pi", "$-8\\pi$"),
        ("\\frac{1}{6}", "$\\frac{1}{6}$"),
    ],
)
def test_gold_wraps_answer_and_drops_integer_float_suffix(answer, gold):
    assert _gold(answer) == gold


# ---------- metric ----------


@pytest.mark.parametrize(
    "gold,prediction",
    [
        (18.0, "따라서 최종 답변은 $\\boxed{18}$ 입니다."),
        (-10.0, "$\\boxed{-10}$"),
        (2880000.0, "$\\boxed{2,880,000}$"),
        (0.5, "$\\boxed{\\frac{1}{2}}$"),
        ("\\frac{1}{2}", "$\\boxed{0.5}$"),
        ("-8\\pi", "$\\boxed{-8\\pi}$"),
        ("-8\\pi", "$\\boxed{-8 \\pi}$"),
        (66071772829247409, "$\\boxed{66071772829247409}$"),
        (5.0, "먼저 $\\boxed{3}$을 구하고, 최종 답은 $\\boxed{5}$입니다."),
    ],
)
def test_metric_accepts_equivalent_answers(gold, prediction):
    assert _score(gold, prediction) == 1.0


@pytest.mark.parametrize(
    "gold,prediction",
    [
        (66071772829247409, "$\\boxed{66071772829247408}$"),
        ("-8\\pi", "$\\boxed{-25.13}$"),
        (18.0, "$\\boxed{19}$"),
    ],
)
def test_metric_rejects_wrong_answers(gold, prediction):
    assert _score(gold, prediction) == 0.0


def test_metric_falls_back_to_rightmost_number_without_boxed():
    assert _score(18.0, "따라서 답은 18입니다.") == 1.0
    # The fallback takes the rightmost number, even one copied from the question.
    assert _score(12.0, "최종 답변: 12\n(참고로 문제에서 사과는 5개였습니다.)") == 0.0


def test_metric_rejects_korean_units_inside_boxed():
    # Known limitation: a Korean unit inside \boxed{} is not stripped.
    assert _score(18.0, "$\\boxed{18달러}$") == 0.0


# ---------- task configs ----------


def test_task_configs():
    assert [config.name for config in TASKS_TABLE] == [f"hrm8k:{subset.lower()}" for subset in HRM8K_SUBSETS]
    for config in TASKS_TABLE:
        assert config.hf_repo == "HAERAE-HUB/HRM8K"
        assert config.hf_revision == REVISION
        assert list(config.evaluation_splits) == ["test"]
        assert config.few_shots_split is None
        assert config.generation_size == 2048
        assert list(config.stop_sequence) == []
        expected = hrm8k_mmmlu_prompt if config.hf_subset == "MMMLU" else hrm8k_prompt
        assert config.prompt_function is expected


def test_superset_expands_to_all_subsets():
    registry = Registry(tasks="hrm8k|0")
    assert set(registry.task_to_configs) == {f"hrm8k:{subset.lower()}" for subset in HRM8K_SUBSETS}
