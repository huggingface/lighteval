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
    def test_shows_scored_sample_count(self):
        result_dict = {
            "results": {
                "all": {"ether0_accuracy": 0.0, "ether0_accuracy_stderr": 0.0},
                "community:ether0:loose:0": {"ether0_accuracy": 0.0, "ether0_accuracy_stderr": 0.0},
            },
            "versions": {"community:ether0:loose:0": 0},
            "n_samples": {"all": 10, "community:ether0:loose:0": 10},
        }

        table = make_results_table(result_dict)

        self.assertIn("Count", table.splitlines()[0])
        rows = {line.split("|")[1].strip(): line for line in table.splitlines() if line.startswith("|")}
        self.assertIn("10", rows["all"])
        self.assertIn("10", rows["community:ether0:loose:0"])

    def test_leaves_count_blank_when_samples_were_not_recorded(self):
        result_dict = {
            "results": {"task_a": {"accuracy": 0.5, "accuracy_stderr": 0.1}},
            "versions": {"task_a": "1"},
        }

        table = make_results_table(result_dict)
        task_row = next(line for line in table.splitlines() if line.startswith("|task_a"))
        count_cell = task_row.split("|")[5]

        self.assertEqual(count_cell.strip(), "")

    def test_count_is_printed_on_the_first_metric_only(self):
        result_dict = {
            "results": {"task_a": {"accuracy": 0.5, "accuracy_stderr": 0.1, "f1": 0.25, "f1_stderr": 0.05}},
            "versions": {"task_a": "1"},
            "n_samples": {"task_a": 4},
        }

        table = make_results_table(result_dict)
        data_rows = [line for line in table.splitlines() if "accuracy" in line or "|f1" in line]

        self.assertEqual(data_rows[0].split("|")[5].strip(), "4")
        self.assertEqual(data_rows[1].split("|")[5].strip(), "")
