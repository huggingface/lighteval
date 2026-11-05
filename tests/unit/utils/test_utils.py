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
from dataclasses import dataclass

from lighteval.utils.utils import make_results_table, remove_reasoning_tags


@dataclass
class MockTaskConfig:
    effective_num_docs: int = -1


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


class TestMakeResultsTable(unittest.TestCase):
    def test_results_table_with_count(self):
        result_dict = {
            "results": {
                "task_a": {"accuracy": 0.85, "accuracy_stderr": 0.02},
                "task_b": {"f1": 0.92},
            },
            "versions": {"task_a": "1.0", "task_b": "2.0"},
            "config_tasks": {
                "task_a": MockTaskConfig(effective_num_docs=100),
                "task_b": MockTaskConfig(effective_num_docs=50),
            },
        }
        
        table = make_results_table(result_dict)
        
        self.assertIn("Count", table)
        self.assertIn("100", table)
        self.assertIn("50", table)
        self.assertIn("task_a", table)
        self.assertIn("task_b", table)
        self.assertIn("0.85", table)
        self.assertIn("0.92", table)

    def test_results_table_without_config_tasks(self):
        result_dict = {
            "results": {
                "task_a": {"accuracy": 0.85},
            },
            "versions": {"task_a": "1.0"},
        }
        
        table = make_results_table(result_dict)
        
        self.assertIn("Count", table)
        self.assertIn("task_a", table)
        self.assertIn("0.85", table)

    def test_results_table_with_missing_count(self):
        result_dict = {
            "results": {
                "task_a": {"accuracy": 0.85},
            },
            "versions": {"task_a": "1.0"},
            "config_tasks": {
                "task_a": MockTaskConfig(effective_num_docs=-1),
            },
        }
        
        table = make_results_table(result_dict)
        
        self.assertIn("Count", table)
        self.assertIn("task_a", table)
        lines = table.split("\n")
        count_column_present = any("Count" in line for line in lines)
        self.assertTrue(count_column_present)

    def test_results_table_with_zero_count(self):
        result_dict = {
            "results": {
                "task_a": {"accuracy": 0.0},
            },
            "versions": {"task_a": "1.0"},
            "config_tasks": {
                "task_a": MockTaskConfig(effective_num_docs=0),
            },
        }
        
        table = make_results_table(result_dict)
        
        self.assertIn("Count", table)
        self.assertIn("task_a", table)
        lines = table.split("\n")
        count_column_present = any("Count" in line for line in lines)
        self.assertTrue(count_column_present)

    def test_results_table_multiple_metrics_same_task(self):
        result_dict = {
            "results": {
                "task_a": {"accuracy": 0.85, "accuracy_stderr": 0.02, "f1": 0.90, "f1_stderr": 0.01},
            },
            "versions": {"task_a": "1.0"},
            "config_tasks": {
                "task_a": MockTaskConfig(effective_num_docs=100),
            },
        }
        
        table = make_results_table(result_dict)
        
        self.assertIn("Count", table)
        self.assertIn("100", table)
        self.assertIn("accuracy", table)
        self.assertIn("f1", table)
        lines = table.split("\n")
        count_in_table = sum(1 for line in lines if "100" in line)
        self.assertGreaterEqual(count_in_table, 1)
