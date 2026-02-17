"""
name:
JFinQA - Japanese Financial Numerical Reasoning QA

dataset:
ajtgjmdjp/jfinqa

abstract:
JFinQA is a benchmark for numerical reasoning over Japanese corporate financial
disclosures. It contains 1,000 questions across three subtasks—numerical
reasoning (550), consistency checking (200), and temporal reasoning (250)—drawn
from 68 companies' EDINET filings covering J-GAAP, IFRS, and US-GAAP.

languages:
japanese

tags:
finance, qa, numerical_reasoning

paper:
https://github.com/ajtgjmdjp/jfinqa
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


# ---------------------------------------------------------------------------
# Normalisation & number parsing (mirrors jfinqa._metrics)
# ---------------------------------------------------------------------------

NUMERICAL_TOLERANCE: float = 0.01

_UNIT_SUFFIXES = (
    "百万円",
    "千円",
    "億円",
    "兆円",
    "円",
    "ドル",
    "ポイント",
    "pt",
    "bps",
)

_KANJI_MULTIPLIERS: dict[str, int] = {
    "千": 1_000,
    "百万": 1_000_000,
    "億": 100_000_000,
    "兆": 1_000_000_000_000,
}


def _normalize(text: str) -> str:
    """Normalize an answer string for comparison."""
    s = text.strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"^[△▲]", "-", s)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    if s.endswith("しました"):
        s = s.removesuffix("しました")
    elif s.endswith("した"):
        s = s.removesuffix("した")
    return s.lower().strip()


def _try_parse_number(text: str) -> float | None:
    """Try to extract a numeric value from *text*."""
    s = _normalize(text)
    for suffix in _UNIT_SUFFIXES:
        s = s.removesuffix(suffix)

    for kanji, multiplier in _KANJI_MULTIPLIERS.items():
        if kanji in s:
            num_part = s.replace(kanji, "").strip()
            num_part = re.sub(r"[^\d.\-+]", "", num_part)
            try:
                return float(num_part) * multiplier
            except ValueError:
                return None

    is_percent = s.endswith("%")
    if is_percent:
        s = s.removesuffix("%")

    s = re.sub(r"[^\d.\-+]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def _numerical_match(predicted: str, gold: str, tolerance: float = NUMERICAL_TOLERANCE) -> bool:
    """Check numerical equivalence within *tolerance* (relative)."""
    pred_num = _try_parse_number(predicted)
    gold_num = _try_parse_number(gold)

    if pred_num is None or gold_num is None:
        return _normalize(predicted) == _normalize(gold)

    if gold_num == 0:
        return pred_num == 0

    return abs(pred_num - gold_num) / abs(gold_num) <= tolerance


def _extract_answer(text: str) -> str:
    """Extract the answer portion from model output."""
    match = re.search(r"(?:Answer|answer|A|回答)\s*[:\uff1a]\s*(.+)", text)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# Custom metric: numerical match with 1 % relative tolerance
# ---------------------------------------------------------------------------


class NumericalMatch(SampleLevelComputation):
    """Numerical matching with relative tolerance for Japanese financial answers.

    Handles fullwidth digits, triangle negatives (△/▲), kanji multipliers
    (千/百万/億/兆), and unit suffixes (円/ドル/bps/pt).
    """

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        golds = doc.get_golds()
        best = 0.0
        for gold in golds:
            for pred in model_response.final_text:
                pred_answer = _extract_answer(pred)
                if _numerical_match(pred_answer, gold):
                    best = 1.0
                    break
            if best == 1.0:
                break
        return best


numerical_match_metric = SampleLevelMetric(
    metric_name="numerical_match",
    sample_level_fn=NumericalMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


# ---------------------------------------------------------------------------
# Custom exact match with Japanese financial normalisation
# ---------------------------------------------------------------------------


class JFinQAExactMatch(SampleLevelComputation):
    """Exact match with Japanese financial answer normalisation and extraction."""

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        golds = doc.get_golds()
        best = 0.0
        for gold in golds:
            for pred in model_response.final_text:
                pred_answer = _extract_answer(pred)
                if _normalize(pred_answer) == _normalize(gold):
                    best = 1.0
                    break
            if best == 1.0:
                break
        return best


jfinqa_exact_match = SampleLevelMetric(
    metric_name="em",
    sample_level_fn=JFinQAExactMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


# ---------------------------------------------------------------------------
# Prompt function
# ---------------------------------------------------------------------------


def jfinqa_prompt(line: dict, task_name: str = None) -> Doc:
    """Convert a dataset row into a Doc for evaluation."""
    parts: list[str] = []

    # Pre-text paragraphs
    pre_text = line.get("pre_text", [])
    if pre_text:
        parts.append("\n".join(pre_text))

    # Table as markdown
    headers = line.get("table_headers", [])
    rows = line.get("table_rows", [])
    if headers:
        header_line = "| " + " | ".join(str(h) for h in headers) + " |"
        sep_line = "| " + " | ".join("---" for _ in headers) + " |"
        row_lines = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
        parts.append("\n".join([header_line, sep_line, *row_lines]))

    # Post-text paragraphs
    post_text = line.get("post_text", [])
    if post_text:
        parts.append("\n".join(post_text))

    # Question
    question = line.get("question", "")
    parts.append(f"Question: {question}\nAnswer:")

    query = "\n\n".join(parts)

    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["answer"]],
        gold_index=0,
    )


# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

_COMMON = {
    "prompt_function": jfinqa_prompt,
    "hf_repo": "ajtgjmdjp/jfinqa",
    "hf_avail_splits": ["test"],
    "evaluation_splits": ["test"],
    "few_shots_split": None,
    "few_shots_select": None,
    "generation_size": 256,
    "stop_sequence": ["\n\n", "Question:"],
    "metrics": [jfinqa_exact_match, numerical_match_metric],
    "version": 0,
}

jfinqa_numerical = LightevalTaskConfig(
    name="jfinqa:numerical",
    hf_subset="numerical_reasoning",
    **_COMMON,
)

jfinqa_consistency = LightevalTaskConfig(
    name="jfinqa:consistency",
    hf_subset="consistency_checking",
    **_COMMON,
)

jfinqa_temporal = LightevalTaskConfig(
    name="jfinqa:temporal",
    hf_subset="temporal_reasoning",
    **_COMMON,
)

TASKS_TABLE = [
    jfinqa_numerical,
    jfinqa_consistency,
    jfinqa_temporal,
]
