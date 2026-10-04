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
from types import SimpleNamespace

from lighteval.utils.utils import make_results_table, remove_reasoning_tags


class TestMakeResultsTable(unittest.TestCase):
    def _base_result_dict(self):
        return {
            "results": {
                "mmlu:abstract_algebra": {"acc": 0.4, "acc_stderr": 0.05},
                "hellaswag": {"acc_norm": 0.75},
            },
            "versions": {
                "mmlu:abstract_algebra": 1,
                "hellaswag": 0,
            },
        }

    def test_num_samples_column_present(self):
        result_dict = self._base_result_dict()
        result_dict["config_tasks"] = {
            "mmlu:abstract_algebra": SimpleNamespace(effective_num_docs=100),
            "hellaswag": SimpleNamespace(effective_num_docs=10042),
        }
        table = make_results_table(result_dict)
        self.assertIn("Num. samples", table)
        self.assertIn("100", table)
        self.assertIn("10042", table)

    def test_num_samples_empty_without_config_tasks(self):
        # config_tasks absent — column header still present, counts are blank
        table = make_results_table(self._base_result_dict())
        self.assertIn("Num. samples", table)

    def test_num_samples_only_on_first_metric_row(self):
        # Tasks with multiple metrics should only show the count on the first row
        result_dict = {
            "results": {"task_a": {"acc": 0.5, "acc_stderr": 0.01, "f1": 0.6, "f1_stderr": 0.02}},
            "versions": {"task_a": 0},
            "config_tasks": {"task_a": SimpleNamespace(effective_num_docs=50)},
        }
        table = make_results_table(result_dict)
        self.assertEqual(table.count("50"), 1)

    def test_stderr_row_format(self):
        table = make_results_table(self._base_result_dict())
        self.assertIn("±", table)
        self.assertIn("0.40", table)
        self.assertIn("0.05", table)


class TestRemoveReasoningTags(unittest.TestCase):
    def test_remove_reasoning_tags(self):
        text = "<think> Reasoning section </think> Answer section"
        tag_pairs = [("<think>", "</think>")]
        result = remove_reasoning_tags(text, tag_pairs)
        self.assertEqual(result, " Answer section")

    def test_remove_multiple_tags(self):
        text = "<think> Reasoning </think> Interlude <think> More reasoning </think> Answer"
        tag_pairs = [("<think>", "</think>")]
        result = remove_reasoning_tags(text, tag_pairs)
        self.assertEqual(result, " Interlude  Answer")

    def test_no_tags(self):
        text = "No reasoning tags here."
        tag_pairs = [("<think>", "</think>")]
        result = remove_reasoning_tags(text, tag_pairs)
        self.assertEqual(result, "No reasoning tags here.")

    def test_empty_text(self):
        text = ""
        tag_pairs = [("<think>", "</think>")]
        result = remove_reasoning_tags(text, tag_pairs)
        self.assertEqual(result, "")

    def test_no_opening_tag(self):
        text = "No opening tag <think> Reasoning section. </think> Answer section"
        tag_pairs = [("<think>", "</think>")]
        result = remove_reasoning_tags(text, tag_pairs)
        self.assertEqual(result, "No opening tag  Answer section")

    def test_no_closing_tag(self):
        text = "<think> Reasoning section. Answer section"
        tag_pairs = [("<think>", "</think>")]
        result = remove_reasoning_tags(text, tag_pairs)
        self.assertEqual(result, "<think> Reasoning section. Answer section")
