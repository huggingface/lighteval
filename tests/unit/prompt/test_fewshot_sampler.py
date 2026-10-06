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

import random
from unittest.mock import Mock

from lighteval.tasks.prompt_manager import FewShotSampler
from lighteval.tasks.requests import Doc


def _make_sampler(selection: str, pool: list[Doc]) -> FewShotSampler:
    """Build a FewShotSampler whose task returns the same pool list by reference.

    task.fewshot_docs() returning one shared list mirrors the real task, whose
    _fewshot_docs is memoized and handed back by reference.
    """
    task = Mock()
    task.fewshot_selection = selection
    task.fewshot_split = "train"
    task.fewshot_docs = Mock(return_value=pool)
    return FewShotSampler(task)


def test_sequential_does_not_mutate_shared_pool() -> None:
    """Sequential rotation must not mutate the shared memoized pool (issue #1307)."""
    pool = [Doc(query=f"q{i}", choices=["a", "b"], gold_index=0) for i in range(5)]
    sampler = _make_sampler("sequential", pool)

    sampler._init_fewshot_pool(num_fewshot=2, variance_seed=1)

    assert [d.query for d in pool] == [f"q{i}" for i in range(5)]
    assert sampler._fewshot_cache[1] is not pool


def test_sequential_variance_seeds_are_independent() -> None:
    """Each seed rotates from the original order, not from the previous seed's result (issue #1307).

    seed s rotates by num_fewshot * s, so with num_fewshot=2 seed 1 rotates by 2 and seed 2 by 4. If the
    shared pool were mutated in place, seed 2 would rotate the already-rotated list and land on the wrong offset.
    """
    pool = [Doc(query=f"q{i}", choices=["a", "b"], gold_index=0) for i in range(5)]
    sampler = _make_sampler("sequential", pool)

    sampler._init_fewshot_pool(num_fewshot=2, variance_seed=1)
    sampler._init_fewshot_pool(num_fewshot=2, variance_seed=2)

    assert [d.query for d in sampler._fewshot_cache[1]] == ["q2", "q3", "q4", "q0", "q1"]
    assert [d.query for d in sampler._fewshot_cache[2]] == ["q4", "q0", "q1", "q2", "q3"]


def test_balanced_falsy_label_does_not_truncate() -> None:
    """A present but falsy label (here an empty-string gold) must not cut the balanced selection short (issue #1309).

    The empty-string label has the highest count so it sorts first in the cycle. With a plain `if not next_label`
    guard the loop breaks on it immediately and returns zero examples.
    """
    pool = [Doc(query=f"z{i}", choices=[""], gold_index=0) for i in range(3)]  # label "" (falsy), count 3
    pool.append(Doc(query="zx", choices=["x"], gold_index=0))  # label "x", count 1
    sampler = _make_sampler("balanced", pool)

    sampler._init_fewshot_pool(num_fewshot=2, variance_seed=3)

    # num_instances_to_sample = min(len(pool), num_fewshot + 1) = min(4, 3) = 3
    assert len(sampler._fewshot_cache[3]) == 3


def test_balanced_uses_local_rng_and_leaves_global_state_untouched() -> None:
    """Balanced selection must use a seeded local RNG, not the global random module (issue #1309)."""
    pool = [Doc(query=f"z{i}", choices=[str(i % 2)], gold_index=0) for i in range(6)]
    sampler = _make_sampler("balanced", pool)

    random.seed(0)
    before = random.getstate()
    sampler._init_fewshot_pool(num_fewshot=3, variance_seed=3)
    after = random.getstate()

    assert before == after

    # Same seed reproduces the same selection.
    other = _make_sampler("balanced", list(pool))
    other._init_fewshot_pool(num_fewshot=3, variance_seed=3)
    assert [d.query for d in sampler._fewshot_cache[3]] == [d.query for d in other._fewshot_cache[3]]
