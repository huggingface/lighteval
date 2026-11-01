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

import math

import pytest

from lighteval.logging.info_loggers import MetricsLogger


def test_average_and_all_rows_combine_subtask_stderrs_in_quadrature():
    # The `_average` and `all` rows are unweighted means of G subtask scores computed on disjoint samples,
    # so their standard error is sqrt(sum_i stderr_i**2) / G, not the mean of the subtask stderrs.
    logger = MetricsLogger()
    subtasks = {
        "mmlu:anatomy|5": {"acc": 0.5, "acc_stderr": 0.05},
        "mmlu:astronomy|5": {"acc": 0.7, "acc_stderr": 0.03},
        "mmlu:virology|5": {"acc": 0.6, "acc_stderr": 0.04},
    }
    for task, metrics in subtasks.items():
        logger.metric_aggregated[task] = dict(metrics)

    logger.aggregate(task_dict={}, bootstrap_iters=1000)

    expected_stderr = math.sqrt(0.05**2 + 0.03**2 + 0.04**2) / 3
    for row in ("mmlu:_average|5", "all"):
        assert logger.metric_aggregated[row]["acc"] == pytest.approx(0.6)
        assert logger.metric_aggregated[row]["acc_stderr"] == pytest.approx(expected_stderr)
