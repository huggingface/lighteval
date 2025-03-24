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

from lighteval.models.model_loader import load_model
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.utils.utils import EnvConfig


def test_empty_requests():
    model_config = TransformersModelConfig(
        "hf-internal-testing/tiny-random-LlamaForCausalLM", model_parallel=False, revision="main"
    )
    model: TransformersModel = load_model(config=model_config, env_config=EnvConfig(cache_dir="."))

    assert model.loglikelihood([]) == []
    assert model.loglikelihood_single_token([]) == []
    assert model.loglikelihood_rolling([]) == []
    assert model.greedy_until([]) == []
    assert model.greedy_until_multi_turn([]) == []
