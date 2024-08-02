from typing import Tuple, cast

from lighteval.models.model_loader import load_model, ModelInfo
from lighteval.models.model_config import BaseModelConfig, EnvConfig
from lighteval.models.base_model import BaseModel

def test_empty_requests():
    model_config = BaseModelConfig("trl-internal-testing/tiny-random-LlamaForCausalLM")
    model, _ = cast(Tuple[BaseModel, ModelInfo], load_model(config=model_config, env_config=EnvConfig()))

    assert model.loglikelihood([]) == []
    assert model.loglikelihood_single_token([]) == []
    assert model.loglikelihood_rolling([]) == []
    assert model.greedy_until([]) == []
    assert model.greedy_until_multi_turn([]) == []