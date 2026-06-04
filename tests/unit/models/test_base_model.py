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

import math

import pytest
import torch

from lighteval.models.model_loader import load_model
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.tasks.requests import Doc


def test_empty_requests():
    model_config = TransformersModelConfig(
        model_name="hf-internal-testing/tiny-random-LlamaForCausalLM", model_parallel=False, revision="main"
    )
    model: TransformersModel = load_model(config=model_config)

    assert model.loglikelihood([]) == []
    assert model.loglikelihood_rolling([]) == []
    assert model.greedy_until([]) == []


def test_loglikelihood_rolling_matches_manual_scoring():
    """Rolling perplexity on a `choices=None` doc (the_pile/wikitext style) should not
    crash and should match a plain autoregressive logits[:-1] vs ids[1:] scoring."""
    model_config = TransformersModelConfig(
        model_name="hf-internal-testing/tiny-random-LlamaForCausalLM", model_parallel=False, revision="main"
    )
    model: TransformersModel = load_model(config=model_config)

    text = "The quick brown fox jumps over the lazy dog."
    doc = Doc(task_name="ppl", query=text, choices=None, gold_index=None)

    (response,) = model.loglikelihood_rolling([doc])
    total_logprob = float(sum(response.logprobs))
    assert math.isfinite(total_logprob)

    # Reference: score the exact tokens the model sees (context + continuation) with the
    # standard one-position shift.
    ctx_ids = model.tok_encode("", add_special_tokens=model.add_special_tokens)
    cont_ids = model.tok_encode(text, add_special_tokens=False)
    input_ids = torch.tensor([ctx_ids + cont_ids], device=model.device)
    with torch.no_grad():
        logits = model.model(input_ids).logits.float()
    logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
    reference = logprobs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum().item()

    assert total_logprob == pytest.approx(reference, rel=1e-3, abs=1e-2)
