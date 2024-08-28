# Quicktour

We provide two main entry points to evaluate models:

- `lighteval accelerate` : evaluate models on CPU or one or more GPUs using 🤗 Accelerate.
- `lighteval nanotron`: evaluate models in distributed settings using ⚡️ Nanotron.

## Accelerate

### Evaluate a model on one or more GPUs

To evaluate a model on one or more GPUs, first create a multi-gpu config by running.

```bash
accelerate config
```

You can then evaluate a model using data parallelism as follows:

```bash
accelerate launch --multi_gpu --num_processes=<num_gpus> -m \
    lighteval accelerate \
    --model_args="pretrained=<path to model on the hub>" \
    --tasks <task parameters> \
    --output_dir output_dir
```

Here, --tasks refers to either a comma-separated list of supported tasks from
the tasks_list in the format: Tasks details can also be found in the file
implementing them.

```bash
suite|task|num_few_shot|{0 or 1 to automatically reduce `num_few_shot` if prompt is too long}
```

or a file path like ``examples/tasks/recommended_set.txt`` which specifies
multiple task configurations. For example, to evaluate GPT-2 on the Truthful QA
benchmark run:

```bash
accelerate launch --multi_gpu --num_processes=8 -m \
    lighteval accelerate \
    --model_args "pretrained=gpt2" \
    --tasks "leaderboard|truthfulqa:mc|0|0" \
    --override_batch_size 1 \
    --output_dir="./evals/"
```

Here, --override_batch_size defines the batch size per device, so the effective
batch size will be override_batch_size x num_gpus. To evaluate on multiple
benchmarks, separate each task configuration with a comma, e.g.

```bash
accelerate launch --multi_gpu --num_processes=8 -m \
    lighteval accelerate \
    --model_args "pretrained=gpt2" \
    --tasks "leaderboard|truthfulqa:mc|0|0,leaderboard|gsm8k|0|0" \
    --override_batch_size 1 \
    --output_dir="./evals/"
```

## Nanotron

...
