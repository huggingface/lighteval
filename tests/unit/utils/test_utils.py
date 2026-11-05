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
    def test_count_column_shows_num_samples_on_first_metric_row(self):
        results = {
            "results": {
                "squad": {"accuracy": 0.85, "accuracy_stderr": 0.02, "f1": 0.90, "f1_stderr": 0.01},
            },
            "versions": {"squad": "2.0"},
            "num_samples": {"squad": 100},
        }
        table = make_results_table(results)
        self.assertIn("Count", table)
        squad_rows = [line for line in table.splitlines() if "squad" in line]
        self.assertEqual(len(squad_rows), 1)
        self.assertIn("100", squad_rows[0])
        # The count is a per-task value: it must not repeat on the second metric row.
        f1_rows = [line for line in table.splitlines() if "f1" in line]
        self.assertEqual(len(f1_rows), 1)
        self.assertNotIn("100", f1_rows[0])

    def test_missing_num_samples_renders_blank_count(self):
        results = {
            "results": {"squad": {"accuracy": 0.85, "accuracy_stderr": 0.02}},
            "versions": {"squad": "2.0"},
        }
        table = make_results_table(results)
        self.assertIn("Count", table)
        self.assertIn("squad", table)

    def test_partial_num_samples(self):
        results = {
            "results": {
                "glue": {"f1": 0.92, "f1_stderr": 0.01},
                "squad": {"accuracy": 0.85, "accuracy_stderr": 0.02},
            },
            "versions": {"squad": "2.0", "glue": "1.0"},
            "num_samples": {"squad": 100},
        }
        table = make_results_table(results)
        squad_rows = [line for line in table.splitlines() if "squad" in line]
        self.assertIn("100", squad_rows[0])
