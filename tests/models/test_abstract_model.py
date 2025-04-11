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

from transformers import AutoTokenizer

from lighteval.models.dummy.dummy_model import DummyModel, DummyModelConfig


def test_tok_encode_pair():
    model = DummyModel(config=DummyModelConfig(seed=42))
    model._tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M")
    context = "答案："
    continuation = "1"
    non_pairwise_tokens = model.tok_encode_pair(context, continuation, pairwise=False)
    pairwise_tokens = model.tok_encode_pair(context, continuation, pairwise=True)
    # Non-pairwise merged "：1" to one token
    assert non_pairwise_tokens == ([6, 47873], [34871])
    # Pairwise separated "：" and "1"
    assert pairwise_tokens == ([6, 47873, 13], [82])
