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

import unittest
from unittest.mock import MagicMock

from lighteval.logging.info_loggers import DetailsLogger


class TestDetailsLoggerAggregate(unittest.TestCase):
    def _make_detail(self):
        # `DetailsLogger.aggregate` only reads the length of the per-task detail
        # list, so a plain mock stands in for a real `Detail` instance here.
        return MagicMock()

    def test_num_samples_matches_number_of_logged_details(self):
        logger = DetailsLogger()
        logger.details["task1"] = [self._make_detail() for _ in range(3)]
        logger.details["task2"] = [self._make_detail() for _ in range(7)]

        logger.aggregate()

        self.assertEqual(logger.compiled_details["task1"].num_samples, 3)
        self.assertEqual(logger.compiled_details["task2"].num_samples, 7)

    def test_num_samples_over_all_tasks_is_the_sum_of_all_tasks(self):
        logger = DetailsLogger()
        logger.details["task1"] = [self._make_detail() for _ in range(3)]
        logger.details["task2"] = [self._make_detail() for _ in range(7)]

        logger.aggregate()

        self.assertEqual(logger.compiled_details_over_all_tasks.num_samples, 10)

    def test_task_with_no_logged_samples_reports_zero(self):
        logger = DetailsLogger()
        logger.details["empty_task"] = []

        logger.aggregate()

        self.assertEqual(logger.compiled_details["empty_task"].num_samples, 0)


if __name__ == "__main__":
    unittest.main()
