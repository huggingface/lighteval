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

from lighteval.metrics.imports.data_stats_utils import Fragments


class TestFragmentsStrings(unittest.TestCase):
    def test_source_offset_and_original_case(self):
        fragments = Fragments("cat sat", "The Cat sat on the mat")
        self.assertEqual(fragments.overlaps(), [Fragments.Match(0, 1, 2)])
        self.assertEqual(fragments.strings(summary_base=False), [["Cat", "sat"]])

    def test_summary_offset_is_unchanged(self):
        fragments = Fragments("cat sat", "The Cat sat on the mat")
        self.assertEqual(fragments.strings(), [["cat", "sat"]])

    def test_multiple_source_spans(self):
        fragments = Fragments(["cat", "sat", "new", "dog"], ["prefix", "Dog", "then", "Cat", "sat"])
        self.assertEqual(fragments.strings(summary_base=False), [["Cat", "sat"], ["Dog"]])
        self.assertEqual(fragments.strings(), [["cat", "sat"], ["dog"]])

    def test_threshold_retains_existing_strict_semantics(self):
        fragments = Fragments("cat sat new dog", "prefix Dog then Cat sat")
        self.assertEqual(fragments.strings(min_length=1, summary_base=False), [["Cat", "sat"]])
        self.assertEqual(fragments.strings(min_length=2, summary_base=False), [])

    def test_no_overlap(self):
        self.assertEqual(Fragments("cat", "dog").strings(summary_base=False), [])
        self.assertEqual(Fragments("", "dog").strings(summary_base=False), [])

    def test_metrics_and_matches_do_not_mutate(self):
        fragments = Fragments("cat sat", "The Cat sat on the mat")
        before = (fragments.overlaps()[:], fragments.coverage(), fragments.density(), fragments.compression())
        fragments.strings(summary_base=False)
        after = (fragments.overlaps(), fragments.coverage(), fragments.density(), fragments.compression())
        self.assertEqual(before, after)
