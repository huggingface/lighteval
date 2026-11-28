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

from lighteval.utils.utils import clean_s3_links, remove_reasoning_tags


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


class TestCleanS3Links(unittest.TestCase):
    def test_bucket_only_uri_links_to_bucket_without_prefix(self):
        result = clean_s3_links("s3://my-bucket")
        self.assertEqual(
            result,
            '<a href="https://s3.console.aws.amazon.com/s3/buckets/my-bucket" target="_blank"> s3://my-bucket </a>',
        )

    def test_uri_with_path_keeps_prefix_link(self):
        result = clean_s3_links("s3://my-bucket/models/gpt2")
        self.assertEqual(
            result,
            '<a href="https://s3.console.aws.amazon.com/s3/buckets/my-bucket?prefix=models/gpt2/" target="_blank"> s3://my-bucket/models/gpt2 </a>',
        )

    def test_uri_with_trailing_slash_keeps_single_prefix_separator(self):
        result = clean_s3_links("s3://my-bucket/models/gpt2/")
        self.assertEqual(
            result,
            '<a href="https://s3.console.aws.amazon.com/s3/buckets/my-bucket?prefix=models/gpt2/" target="_blank"> s3://my-bucket/models/gpt2/ </a>',
        )
