"""Tests for ID-based few-shot example selection (issue #634)."""

import logging
import random
from unittest.mock import MagicMock

import pytest

from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.prompt_manager import FewShotSampler
from lighteval.tasks.requests import Doc


def _make_doc(doc_id: str, query: str, fewshot_id: str | None = None) -> Doc:
    specific = {"__fewshot_id": fewshot_id} if fewshot_id is not None else None
    return Doc(query=query, choices=["A", "B"], gold_index=0, id=doc_id, specific=specific)


def _make_task_with_fewshot_docs(fewshot_docs, fewshot_id_column=None, fewshot_id_list=None):
    task = MagicMock(spec=LightevalTask)
    task.name = "test_task"
    task.fewshot_docs.return_value = fewshot_docs
    task.fewshot_selection = "balanced"
    task.fewshot_split = "train"
    task.fewshot_id_column = fewshot_id_column
    task.fewshot_id_list = fewshot_id_list
    return task


class TestFewShotIdSelection:
    def test_id_based_selection_returns_correct_docs(self):
        docs = [
            _make_doc("0", "What is 1+1?", fewshot_id="q_001"),
            _make_doc("1", "What is 2+2?", fewshot_id="q_002"),
            _make_doc("2", "What is 3+3?", fewshot_id="q_003"),
            _make_doc("3", "What is 4+4?", fewshot_id="q_004"),
        ]
        task = _make_task_with_fewshot_docs(docs, fewshot_id_column="id", fewshot_id_list=["q_002", "q_004"])
        sampler = FewShotSampler(task)
        sampler._init_fewshot_pool(num_fewshot=2, variance_seed=0)
        pool = sampler._fewshot_cache[0]
        pool_ids = [d.specific["__fewshot_id"] for d in pool]
        assert pool_ids == ["q_002", "q_004"]

    def test_id_based_selection_preserves_order(self):
        docs = [
            _make_doc("0", "Q1", fewshot_id="a"),
            _make_doc("1", "Q2", fewshot_id="b"),
            _make_doc("2", "Q3", fewshot_id="c"),
        ]
        task = _make_task_with_fewshot_docs(docs, fewshot_id_column="id", fewshot_id_list=["c", "a"])
        sampler = FewShotSampler(task)
        sampler._init_fewshot_pool(num_fewshot=2, variance_seed=0)
        pool = sampler._fewshot_cache[0]
        pool_ids = [d.specific["__fewshot_id"] for d in pool]
        assert pool_ids == ["c", "a"]

    def test_id_based_selection_warns_on_missing_ids(self, caplog):
        docs = [_make_doc("0", "Q1", fewshot_id="exists")]
        task = _make_task_with_fewshot_docs(docs, fewshot_id_column="id", fewshot_id_list=["exists", "does_not_exist"])
        sampler = FewShotSampler(task)
        with caplog.at_level(logging.WARNING):
            sampler._init_fewshot_pool(num_fewshot=2, variance_seed=0)
        assert "does_not_exist" in caplog.text

    def test_id_based_selection_raises_on_all_missing(self):
        docs = [_make_doc("0", "Q1", fewshot_id="x")]
        task = _make_task_with_fewshot_docs(docs, fewshot_id_column="id", fewshot_id_list=["nonexistent_1", "nonexistent_2"])
        sampler = FewShotSampler(task)
        with pytest.raises(ValueError, match="no few-shot examples matched"):
            sampler._init_fewshot_pool(num_fewshot=2, variance_seed=0)

    def test_no_id_list_falls_back_to_default(self):
        docs = [_make_doc("0", "Q1", fewshot_id="a"), _make_doc("1", "Q2", fewshot_id="b")]
        task = _make_task_with_fewshot_docs(docs, fewshot_id_column=None, fewshot_id_list=None)
        sampler = FewShotSampler(task)
        sampler._init_fewshot_pool(num_fewshot=2, variance_seed=0)
        pool = sampler._fewshot_cache[0]
        assert len(pool) == 2

    def test_sample_fewshot_examples_with_id_selection(self):
        docs = [
            _make_doc("0", "Q1", fewshot_id="id_A"),
            _make_doc("1", "Q2", fewshot_id="id_B"),
            _make_doc("2", "Q3", fewshot_id="id_C"),
            _make_doc("3", "Q4", fewshot_id="id_D"),
        ]
        eval_doc = _make_doc("99", "Eval question", fewshot_id="id_EVAL")
        task = _make_task_with_fewshot_docs(docs, fewshot_id_column="id", fewshot_id_list=["id_B", "id_D"])
        sampler = FewShotSampler(task)
        rnd = random.Random(42)
        result = sampler.sample_fewshot_examples(num_fewshot=2, variance_seed=0, formatted_doc=eval_doc, sampler=rnd)
        result_ids = [d.specific["__fewshot_id"] for d in result]
        assert result_ids == ["id_B", "id_D"]
        assert len(result) == 2
