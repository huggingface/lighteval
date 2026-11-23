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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from lighteval.models.transformers.adapter_model import AdapterModel, AdapterModelConfig
from lighteval.utils.imports import is_package_available


REPO_ROOT = Path(__file__).parents[3]


def load_example_config() -> AdapterModelConfig:
    with (REPO_ROOT / "examples/model_configs/peft_model.yaml").open() as config_file:
        config = yaml.safe_load(config_file)["model_parameters"]

    assert config.pop("adapter_weights") is True
    return AdapterModelConfig(**config)


def test_peft_example_uses_base_model_for_config_and_tokenizer():
    config = load_example_config()

    assert config.model_name == "ybelkada/opt-350m-lora"
    assert config.base_model == "facebook/opt-350m"
    assert config.tokenizer == config.base_model
    assert config.generation_parameters.max_new_tokens == 256

    expected_config = MagicMock()
    with patch(
        "lighteval.models.transformers.transformers_model.AutoConfig.from_pretrained",
        return_value=expected_config,
    ) as from_pretrained:
        assert config.get_transformers_config() is expected_config

    from_pretrained.assert_called_once_with(
        config.base_model,
        revision="main",
        trust_remote_code=False,
    )


@pytest.mark.skipif(not is_package_available("peft"), reason="requires the adapters extra")
def test_adapter_model_loads_weights_from_model_name():
    config = load_example_config()
    adapter_model = AdapterModel.__new__(AdapterModel)
    adapter_model.config = config
    adapter_model.accelerator = MagicMock(is_local_main_process=True)
    adapter_model._tokenizer = [0, 1]
    adapter_model.init_model_parallel = MagicMock(return_value=(False, None, None))

    base_model = MagicMock()
    base_model.config.vocab_size = len(adapter_model._tokenizer)
    peft_model = MagicMock()
    peft_model.merge_and_unload.return_value = MagicMock()
    loaded_model = MagicMock()

    with (
        patch(
            "lighteval.models.transformers.adapter_model.AutoModelForCausalLM.from_pretrained",
            side_effect=[base_model, loaded_model],
        ),
        patch(
            "lighteval.models.transformers.adapter_model.PeftModel.from_pretrained",
            return_value=peft_model,
        ) as from_pretrained,
    ):
        assert adapter_model._create_auto_model() is loaded_model

    from_pretrained.assert_called_once_with(base_model, config.model_name)
