# Use VLLM as backend

Lighteval allows you to use `vllm` as backend allowing great speedups.
To use, simply change the `model_args` to reflect the arguments you want to pass to vllm.

```bash
lighteval accelerate \
    --model_args="vllm,pretrained=HuggingFaceH4/zephyr-7b-beta,dtype=float16" \
    --tasks "leaderboard|truthfulqa:mc|0|0" \
    --output_dir="./evals/"
```

`vllm` is able to distribute the model across multiple GPUs using data
parallelism, pipeline parallelism or tensor parallelism.
You can choose the parallelism method by setting in the the `model_args`.

For example if you have 4 GPUs you can split it across using `tensor_parallelism`:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn && lighteval accelerate \
    --model_args="vllm,pretrained=HuggingFaceH4/zephyr-7b-beta,dtype=float16,tensor_parallel_size=4" \
    --tasks "leaderboard|truthfulqa:mc|0|0" \
    --output_dir="./evals/"
```

Or, if your model fits on a single GPU, you can use `data_parallelism` to speed up the evaluation:

```bash
lighteval accelerate \
    --model_args="vllm,pretrained=HuggingFaceH4/zephyr-7b-beta,dtype=float16,data_parallel_size=4" \
    --tasks "leaderboard|truthfulqa:mc|0|0" \
    --output_dir="./evals/"
```

Available arguments for `vllm` can be found in the `VLLMModelConfig`:

[[autodoc]] lighteval.models.model_config.VLLMModelConfig
