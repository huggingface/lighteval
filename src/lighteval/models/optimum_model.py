from lighteval.models.base_model import BaseModel
from lighteval.models.model_config import EnvConfig, OptimumModelConfig
from lighteval.models.utils import _get_dtype
from lighteval.utils import is_optimum_available


if is_optimum_available():
    # from optimum import OptimumConfig
    from optimum.intel.openvino import OVModelForCausalLM


class OptimumModel(BaseModel):
    def _create_auto_model(self, config: OptimumModelConfig, env_config: EnvConfig):
        # TODO : Get loading class from optimum config (add support for ORTModelForCausalLM / INCModelForCausalLM / IPEXModelForCausalLM)
        # optimum_config = OptimumConfig.from_pretrained(config.pretrained)

        config.model_parallel, max_memory, device_map = self.init_model_parallel(config.model_parallel)
        torch_dtype = _get_dtype(config.dtype, self._config)

        model = OVModelForCausalLM.from_pretrained(
            config.pretrained,
            revision=config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
            torch_dtype=torch_dtype,
            trust_remote_code=config.trust_remote_code,
            cache_dir=env_config.cache_dir,
            use_auth_token=env_config.token,
            quantization_config=config.quantization_config,
        )

        return model
