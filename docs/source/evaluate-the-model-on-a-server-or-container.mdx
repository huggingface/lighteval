An alternative to launching the evaluation locally is to serve the model on a
TGI-compatible server/container and then run the evaluation by sending requests
to the server. The command is the same as before, except you specify a path to
a yaml config file (detailed below):

```
lighteval accelerate \
    --model_config_path="/path/to/config/file"\
    --tasks <task parameters> \
    --output_dir output_dir
```

There are two types of configuration files that can be provided for running on
the server:

### Hugging Face Inference Endpoints

To launch a model using HuggingFace's Inference Endpoints, you need to provide
the following file: `endpoint_model.yaml`. Lighteval will automatically deploy
the endpoint, run the evaluation, and finally delete the endpoint (unless you
specify an endpoint that was already launched, in which case the endpoint won't
be deleted afterwards).

__configuration file example:__

```yaml
model:
  type: "endpoint"
  base_params:
    endpoint_name: "llama-2-7B-lighteval" # needs to be lower case without special characters
    model: "meta-llama/Llama-2-7b-hf"
    revision: "main"
    dtype: "float16" # can be any of "awq", "eetq", "gptq", "4bit' or "8bit" (will use bitsandbytes), "bfloat16" or "float16"
    reuse_existing: false # if true, ignore all params in instance, and don't delete the endpoint after evaluation
  instance:
    accelerator: "gpu"
    region: "eu-west-1"
    vendor: "aws"
    instance_size: "medium"
    instance_type: "g5.2xlarge"
    framework: "pytorch"
    endpoint_type: "protected"
    namespace: null # The namespace under which to launch the endopint. Defaults to the current user's namespace
    image_url: null # Optionally specify the docker image to use when launching the endpoint model. E.g., launching models with later releases of the TGI container with support for newer models.
    env_vars:
      null # Optional environment variables to include when launching the endpoint. e.g., `MAX_INPUT_LENGTH: 2048`
  generation:
    add_special_tokens: true
```

### Text Generation Inference (TGI)

To use a model already deployed on a TGI server, for example on HuggingFace's
serverless inference.

__configuration file example:__

```yaml
model:
  type: "tgi"
  instance:
    inference_server_address: ""
    inference_server_auth: null
    model_id: null # Optional, only required if the TGI container was launched with model_id pointing to a local directory
```
