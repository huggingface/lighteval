from typing import Iterator
import pytest

from lighteval.models.model_config import BaseModelConfig
from lighteval.models.abstract_model import EnvConfig
from lighteval.models.base_model import BaseModel


@pytest.fixture(scope="module")
def base_model() -> Iterator[BaseModel]:
    config = BaseModelConfig("hf-internal-testing/tiny-random-LlamaForCausalLM")
    return BaseModel(config, EnvConfig())