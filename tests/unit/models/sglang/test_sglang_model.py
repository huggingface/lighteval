# MIT License

# Copyright (c) 2025 The HuggingFace Team

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
from unittest.mock import Mock

from lighteval.models.sglang.sglang_model import SGLangModel
from lighteval.tasks.requests import Doc


class FakeTokenizer:
    """Maps each context to a token list whose length is unrelated to its character length."""

    def __init__(self, token_lengths_by_prefix: dict):
        self.token_lengths_by_prefix = token_lengths_by_prefix

    def __call__(self, contexts, add_special_tokens=True):
        input_ids = [[1] * self.token_lengths_by_prefix[context[0]] for context in contexts]
        return {"input_ids": input_ids}


class TestSGLangGreedyUntilTruncation(unittest.TestCase):
    def test_truncation_uses_longest_tokenized_input_in_batch(self):
        """The batch is sorted by CHARACTER length, so inputs[0] is not necessarily the
        longest in TOKENS. Truncation must be sized from the longest tokenized input,
        otherwise a shorter-in-characters but longer-in-tokens prompt is sent to the
        engine above max_length. Same bug class as the vllm one reported in #1204."""
        max_length = 50
        max_new_tokens = 10

        model = SGLangModel.__new__(SGLangModel)
        model._max_length = max_length
        model._add_special_tokens = False
        model.use_chat_template = False
        model.prompt_manager = Mock(prepare_prompt=lambda doc: doc.query)
        # "a..." prompts: many characters, few tokens. "b..." prompts: fewer characters,
        # many tokens (e.g. CJK text). The "a" doc sorts first by character length.
        model._tokenizer = FakeTokenizer({"a": 10, "b": 60})

        generate_calls = []

        def fake_generate(inputs, max_new_tokens=None, stop_tokens=None, num_samples=1):
            generate_calls.append(inputs)
            return [{"text": "out", "meta_info": {"output_token_logprobs": [(-0.5, 7)]}} for _ in inputs]

        model._generate = fake_generate

        docs = [
            Doc(query="a" * 100, choices=[""], gold_index=0, generation_size=max_new_tokens),
            Doc(query="b" * 90, choices=[""], gold_index=0, generation_size=max_new_tokens),
        ]

        responses = model._greedy_until(docs)

        self.assertEqual(len(responses), 2)
        self.assertTrue(generate_calls)
        for inputs in generate_calls:
            for input_ids in inputs:
                self.assertLessEqual(
                    len(input_ids) + max_new_tokens,
                    max_length,
                    f"an input of {len(input_ids)} tokens plus {max_new_tokens} new tokens "
                    f"exceeds max_length={max_length}: truncation was sized from the wrong input",
                )


if __name__ == "__main__":
    unittest.main()
