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

from lighteval.logging.info_loggers import DetailsLogger
from lighteval.utils.utils import make_results_table, remove_reasoning_tags


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
    def test_includes_sample_count_per_task(self):
        result_dict = {
            "results": {
                "task1": {"accuracy": 0.8, "accuracy_stderr": 0.05},
            },
            "versions": {"task1": "0"},
            "summary_tasks": {
                "task1": DetailsLogger.CompiledDetail(num_samples=10),
            },
        }
        table = make_results_table(result_dict)
        rows = [row for row in table.splitlines() if "task1" in row]
        self.assertEqual(len(rows), 1)
        self.assertIn("10", rows[0])

    def test_distinguishes_zero_score_from_zero_samples(self):
        # A task that truly evaluated zero samples should be visibly different
        # from a task that evaluated many samples and scored zero on all of them.
        result_dict = {
            "results": {
                "misconfigured_task": {"accuracy": 0.0},
                "genuinely_failing_task": {"accuracy": 0.0},
            },
            "versions": {},
            "summary_tasks": {
                "misconfigured_task": DetailsLogger.CompiledDetail(num_samples=0),
                "genuinely_failing_task": DetailsLogger.CompiledDetail(num_samples=25),
            },
        }
        table = make_results_table(result_dict)
        misconfigured_row = next(row for row in table.splitlines() if "misconfigured_task" in row)
        failing_row = next(row for row in table.splitlines() if "genuinely_failing_task" in row)
        self.assertIn("|0|", misconfigured_row.replace(" ", ""))
        self.assertIn("|25|", failing_row.replace(" ", ""))

    def test_missing_summary_tasks_defaults_to_blank(self):
        # Older callers (or the doctest example) may not provide `summary_tasks`
        # at all; the table should still render without raising.
        result_dict = {
            "results": {"task1": {"accuracy": 0.5}},
            "versions": {"task1": "0"},
        }
        table = make_results_table(result_dict)
        self.assertIn("task1", table)
