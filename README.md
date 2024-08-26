<div align="center">

[![Tests](https://github.com/huggingface/lighteval/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/huggingface/lighteval/actions/workflows/tests.yaml?query=branch%3Amain)
[![Quality](https://github.com/huggingface/lighteval/actions/workflows/quality.yaml/badge.svg?branch=main)](https://github.com/huggingface/lighteval/actions/workflows/quality.yaml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lighteval)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/huggingface/lighteval/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lighteval)](https://pypi.org/project/lighteval/)
[![Version](https://img.shields.io/pypi/v/lighteval)](https://pypi.org/project/lighteval/)

</div>

# LightEval 🌤️

A lightweight framework for LLM evaluation

## Context
LightEval is a lightweight LLM evaluation suite that Hugging Face has been using internally with the recently released LLM data processing library [datatrove](https://github.com/huggingface/datatrove) and LLM training library [nanotron](https://github.com/huggingface/nanotron).

We're releasing it with the community in the spirit of building in the open.

Note that it is still very much early so don't expect 100% stability ^^'
In case of problems or questions, feel free to open an issue!

## Installation

Clone the repo:

```bash
git clone https://github.com/huggingface/lighteval.git
cd lighteval
```

Create a virtual environment using virtualenv or conda depending on your preferences. We require Python 3.10 or above:

```bash
conda create -n lighteval python=3.10 && conda activate lighteval
```

Install the dependencies. For the default installation, you just need:

```bash
pip install .
```

If you want to evaluate models with frameworks like `accelerate` or `peft`, you will need to specify the optional dependencies group that fits your use case (`accelerate`,`tgi`,`optimum`,`quantization`,`adapters`,`nanotron`,`tensorboardX`):

```bash
pip install '.[optional1,optional2]'
```

The setup tested most is:

```bash
pip install '.[accelerate,quantization,adapters]'
```

If you want to push your results to the Hugging Face Hub, don't forget to add your access token to the environment variable `HF_TOKEN`. You can do this by running:

```shell
huggingface-cli login
```

and pasting your access token.

### Optional steps

- to load and push big models/datasets, your machine likely needs Git LFS. You can install it with `sudo apt-get install git-lfs`
- If you want to run bigbench evaluations, install bigbench `pip install "bigbench@https://storage.googleapis.com/public_research_data/bigbench/bigbench-0.0.1.tar.gz"`

Lastly, if you intend to push to the code base, you'll need to install the precommit hook for styling tests:

```bash
pip install .[dev]
pre-commit install
```

## Usage

We provide two main entry points to evaluate models:

* `lighteval accelerate`: evaluate models on CPU or one or more GPUs using [🤗 Accelerate](https://github.com/huggingface/accelerate).
* `lighteval nanotron`: evaluate models in distributed settings using [⚡️ Nanotron](https://github.com/huggingface/nanotron).

For most users, we recommend using the 🤗 Accelerate backend - see below for specific commands.

### Evaluate a model on one or more GPUs (recommended)

To evaluate a model on one or more GPUs, first create a `multi-gpu` config by running:

```shell
accelerate config
```

You can then evaluate a model using data parallelism as follows:

```shell
accelerate launch --multi_gpu --num_processes=<num_gpus> -m \
    lighteval accelerate \
    --model_args="pretrained=<path to model on the hub>" \
    --tasks <task parameters> \
    --output_dir output_dir
```

Here, `--tasks` refers to either a _comma-separated_ list of supported tasks from the [tasks_list](examples/tasks/all_tasks.txt) in the format:
Tasks details can also be found in the [file implementing them](src/lighteval/tasks/default_tasks.py).

```
suite|task|num_few_shot|{0 or 1 to automatically reduce `num_few_shot` if prompt is too long}
```

or a file path like [`examples/tasks/recommended_set.txt`](./examples/tasks/recommended_set.txt) which specifies multiple task configurations. For example, to evaluate GPT-2 on the Truthful QA benchmark run:

```shell
accelerate launch --multi_gpu --num_processes=8 -m \
    lighteval accelerate \
    --model_args "pretrained=gpt2" \
    --tasks "leaderboard|truthfulqa:mc|0|0" \
    --override_batch_size 1 \
    --output_dir="./evals/"
```

Here, `--override_batch_size` defines the _batch size per device_, so the effective batch size will be `override_batch_size x num_gpus`. To evaluate on multiple benchmarks, separate each task configuration with a comma, e.g.

```shell
accelerate launch --multi_gpu --num_processes=8 -m \
    lighteval accelerate \
    --model_args "pretrained=gpt2" \
    --tasks "leaderboard|truthfulqa:mc|0|0,leaderboard|gsm8k|0|0" \
    --override_batch_size 1 \
    --output_dir="./evals/"
```

See the [`examples/tasks/recommended_set.txt`](./examples/tasks/recommended_set.txt) file for a list of recommended task configurations.

### Evaluating a model with a complex configuration

If you want to evaluate a model by spinning up inference endpoints, use adapter/delta weights, or more complex configuration options, you can load models using a configuration file. This is done as follows:

```shell
accelerate launch --multi_gpu --num_processes=<num_gpus> -m \
    lighteval accelerate \
    --model_config_path="<path to your model configuration>" \
    --tasks <task parameters> \
    --output_dir output_dir
```

You can find the template of the expected model configuration in [examples/model_configs/base_model.yaml_](./examples/model_configs/base_model.yaml).

### Evaluating a quantized model

If you want to evaluate a model by quantizing it, then the model can be loaded in `4bit` or `8bit`. Implicitly, this makes use of `BitsAndBytesConfig` and can drastically reduce memory requirements for consumer-grade hardware.

An example configuration can be found in [examples/model_configs/quantized_model.yaml](./examples/model_configs/quantized_model.yaml).

### Evaluating a PEFT model

If you want to evaluate a model trained with `peft`, check out [examples/model_configs/peft_model.yaml](./examples/model_configs/peft_model.yaml).

Currently, `lighteval` supports `adapter` and `delta` weights to be applied to the base model.

### Evaluating a large model with pipeline parallelism

To evaluate models larger that ~40B parameters in 16-bit precision, you will need to shard the model across multiple GPUs to fit it in VRAM. You can do this by passing `model_parallel=True` and adapting `--num_processes` to be the number of processes to use for data parallel. For example, on a single node of 8 GPUs, you can run:

```shell
# PP=2, DP=4 - good for models < 70B params
accelerate launch --multi_gpu --num_processes=4 -m \
    lighteval accelerate \
    --model_args="pretrained=<path to model on the hub>,model_parallel=True" \
    --tasks <task parameters> \
    --output_dir output_dir

# PP=4, DP=2 - good for huge models >= 70B params
accelerate launch --multi_gpu --num_processes=2 -m \
    lighteval accelerate \
    --model_args="pretrained=<path to model on the hub>,model_parallel=True" \
    --tasks <task parameters> \
    --output_dir output_dir
```

### Evaluate a model on the Open LLM Leaderboard benchmarks

To evaluate a model on all the benchmarks of the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) using a single node of 8 GPUs, run:

```shell
accelerate launch --multi_gpu --num_processes=8 -m \
    lighteval accelerate \
    --model_args "pretrained=<model name>" \
    --tasks examples/tasks/open_llm_leaderboard_tasks.txt \
    --override_batch_size 1 \
    --output_dir="./evals/"
```

### Evaluate a model on CPU

You can also use `lighteval` to evaluate models on CPU, although note this will typically be very slow for large models. To do so, run:

```shell
lighteval accelerate \
    --model_args="pretrained=<path to model on the hub>"\
    --tasks <task parameters> \
    --output_dir output_dir
```

### Evaluate the model on a server/container.

An alternative to launching the evaluation locally is to serve the model on a TGI-compatible server/container and then run the evaluation by sending requests to the server. The command is the same as before, except you specify a path to a yaml config file (detailed below):

```shell
python run_evals_accelerate.py \
    --model_config_path="/path/to/config/file"\
    --tasks <task parameters> \
    --output_dir output_dir
```

There are two types of configuration files that can be provided for running on the server:

1. [endpoint_model.yaml](./examples/model_configs/endpoint_model.yaml): This configuration allows you to launch the model using [HuggingFace's Inference Endpoints](https://huggingface.co/inference-endpoints/dedicated). You can specify in the configuration file all the relevant parameters, and then `lighteval` will automatically deploy the endpoint, run the evaluation, and finally delete the endpoint (unless you specify an endpoint that was already launched, in which case the endpoint won't be deleted afterwards).

2. [tgi_model.yaml](./examples/model_configs/tgi_model.yaml): This configuration lets you specify the URL of a model running in a TGI container, such as one deployed on HuggingFace's serverless inference.

Templates for these configurations can be found in [examples/model_configs](./examples/model_configs/).

### Evaluate a model on extended, community, or custom tasks.

Independently of the default tasks provided in `lighteval` that you will find in the `tasks_table.jsonl` file, you can use `lighteval` to evaluate models on tasks that require special processing (or have been added by the community). These tasks have their own evaluation suites and are defined as follows:

* `extended`: tasks that have complex pre- or post-processing and are added by the `lighteval` maintainers. See the [`extended`](./src/lighteval/tasks/extended) folder for examples.
* `community`: tasks that have been added by the community. See the [`community_tasks`](./community_tasks) folder for examples.
* `custom`: tasks that are defined locally and not present in the core library. Use this suite if you want to experiment with designing a special metric or task.

For example, to run an extended task like `ifeval`, you can run:
```shell
lighteval accelerate \
    --model_args "pretrained=HuggingFaceH4/zephyr-7b-beta" \
    --use_chat_template \ # optional, if you want to run the evaluation with the chat template
    --tasks "extended|ifeval|0|0" \
    --output_dir "./evals"
```

To run a community or custom task, you can use (note the custom_tasks flag):

```shell
lighteval accelerate \
    --model_args="pretrained=<path to model on the hub>"\
    --tasks <task parameters> \
    --custom_tasks <path to your custom or community task> \
    --output_dir output_dir
```

For example, to launch `lighteval` on `arabic_mmlu:abstract_algebra` for `HuggingFaceH4/zephyr-7b-beta`, run:

```shell
lighteval accelerate \
    --model_args "pretrained=HuggingFaceH4/zephyr-7b-beta" \
    --use_chat_template \ # optional, if you want to run the evaluation with the chat template
    --tasks "community|arabic_mmlu:abstract_algebra|5|1" \
    --custom_tasks "community_tasks/arabic_evals" \
    --output_dir "./evals"
```

### Using the dummy model
To debug or obtain random baseline scores for a given set of tasks, you can use the `dummy` model:
```shell
python run_evals_accelerate.py \
    --model_args "dummy"\
    --tasks <task parameters> \
    --output_dir output_dir
```
This "model" randomly generates logprobs (for selection/accuracy tasks) and the string "random baseline" for generation tasks.
You can also select a specific seed for the random logprob values generated by the dummy model: `--model_args "dummy,seed=123"`.

## Deep thanks
`lighteval` was originally built on top of the great [Eleuther AI Harness](https://github.com/EleutherAI/lm-evaluation-harness) (we use the latter to power the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)). We also took a lot of inspiration from the amazing [HELM](https://crfm.stanford.edu/helm/latest/), notably for metrics.

Through adding more and more logging functionalities, and making it compatible with increasingly different workflows and model codebases (including 3D parallelism) as well as allowing custom evaluation experiments, metrics and benchmarks, we ended up needing to change the code more and more deeply until `lighteval` became the small standalone library that it is now.

However, we are very grateful to the Harness and HELM teams for their continued work on better evaluations.

## How to navigate this project
`lighteval` is supposed to be used as a standalone evaluation library.
- To run the evaluations, you can use `run_evals_accelerate.py` or `run_evals_nanotron.py`.
- [src/lighteval](https://github.com/huggingface/lighteval/tree/main/src/lighteval) contains the core of the lib itself
    - [lighteval](https://github.com/huggingface/lighteval/tree/main/src/lighteval) contains the core of the library, divided in the following section
        - [main_accelerate.py](https://github.com/huggingface/lighteval/blob/main/src/lighteval/main_accelerate.py) and [main_nanotron.py](https://github.com/huggingface/lighteval/blob/main/src/lighteval/main_nanotron.py) are our entry points to run evaluation
        - [logging](https://github.com/huggingface/lighteval/tree/main/src/lighteval/logging): Our loggers, to display experiment information and push it to the hub after a run
        - [metrics](https://github.com/huggingface/lighteval/tree/main/src/lighteval/metrics): All the available metrics you can use. They are described in metrics, and divided between sample metrics (applied at the sample level, such as prediction accuracy) and corpus metrics (applied over the whole corpus). You'll also find available normalisation functions.
        - [models](https://github.com/huggingface/lighteval/tree/main/src/lighteval/models): Possible models to use. We cover transformers (base_model), with adapter or delta weights, as well as TGI models locally deployed (it's likely the code here is out of date though), and brrr/nanotron models.
        - [tasks](https://github.com/huggingface/lighteval/tree/main/src/lighteval/tasks): Available tasks. The complete list is in `default_tasks.py`, and you'll find all the prompts in `tasks_prompt_formatting.py`. Popular tasks requiring custom logic are exceptionally added in the [extended tasks](https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/extended).
- [examples/tasks](https://github.com/huggingface/lighteval/tree/main/examples/tasks) contains a list of available tasks you can launch. We advise using tasks in the `recommended_set`, as it's possible that some of the other tasks need double checking.
- [tests](https://github.com/huggingface/lighteval/tree/main/tests) contains our test suite, which we run at each PR to prevent regressions in metrics/prompts/tasks, for a subset of important tasks.

## Customization
If your new task or metric has requirements, add a specific `requirements.txt` file with your evaluation.

### Adding a new task
To add a new task, first either open an issue, to determine whether it will be integrated in the core evaluations of lighteval, in the extended tasks, or the community tasks, and **add its dataset** on the hub.

- Core evaluations are evaluations that only require standard logic in their metrics and processing, and that we will add to our test suite to ensure non regression through time. They already see high usage in the community.
- Extended evaluations are evaluations that require custom logic in their metrics (complex normalisation, an LLM as a judge, ...), that we added to facilitate the life of users. They already see high usage in the community.
- Community evaluations are submissions by the community of new tasks.

A popular community evaluation can move to become an extended or core evaluation over time.

#### Core evaluations
Prompt function: **find a suitable prompt function** in `src.lighteval.tasks.task_prompt_formatting.py`, or code your own. This function must output a `Doc` object, which should contain the `query`, your prompt, and either `gold`, the gold output, or `choices` and `gold_index`, the list of choices and index or indices of correct answers. If your query contains an instruction that should not be repeated in a few shot setup, add it to an `instruction` field.

Summary: create a `LightevalTaskConfig` summary of your evaluation, in `src/lighteval/tasks/default_tasks.py`. This summary should contain the following fields:
- `name` (str), your evaluation name
- `suite` (list), the suite(s) to which your evaluation should belong. This field allows us to compare different task implementations and is used as a task selection to differentiate the versions to launch. At the moment, you'll find the keywords ["helm", "bigbench", "original", "lighteval", "community", "custom"]; for core evals, please choose `lighteval`.
- `prompt_function` (Callable), the prompt function you defined in the step above
- `hf_repo` (str), the path to your evaluation dataset on the hub
- `hf_subset` (str), the specific subset you want to use for your evaluation (note: when the dataset has no subset, fill this field with `"default"`, not with `None` or `""`)
- `hf_avail_splits` (list), all the splits available for your dataset (train, valid or validation, test, other...)
- `evaluation_splits` (list), the splits you want to use for evaluation
- `few_shots_split` (str, can be `null`), the specific split from which you want to select samples for your few-shot examples. It should be different from the sets included in `evaluation_splits`
- `few_shots_select` (str, can be `null`), the method that you will use to select items for your few-shot examples. Can be `null`, or one of:
    - `balanced` select examples from the `few_shots_split` with balanced labels, to avoid skewing the few shot examples (hence the model generations) toward one specific label
    - `random` selects examples at random from the `few_shots_split`
    - `random_sampling` selects new examples at random from the `few_shots_split` for every new item, but if a sampled item is equal to the current one, it is removed from the available samples
    - `random_sampling_from_train` selects new examples at random from the `few_shots_split` for every new item, but if a sampled item is equal to the current one, it is kept! Only use this if you know what you are doing.
`sequential` selects the first `n` examples of the `few_shots_split`
- `generation_size` (int), the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
- `stop_sequence` (list), a list of strings acting as end of sentence tokens for your generation
- `metric` (list), the metrics you want to use for your evaluation (see next section for a detailed explanation)
- `output_regex` (str), A regex string that will be used to filter your generation. (Generative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching `\n` and a generation `\nModel generation output\nSome other text` the metric will only be fed with `Model generation output`)
- `frozen` (bool), for now, is set to False, but we will steadily pass all stable tasks to True.
- `trust_dataset` (bool), set to True if you trust the dataset.

Make sure you can launch your model with your new task using `--tasks lighteval|yournewtask|2|0`.

#### Community evaluations
Copy the `community_tasks/_template.py` to `community_tasks/yourevalname.py` and edit it to add your custom tasks (the parameters you can use are explained above). It contains an interesting mechanism if the dataset you are adding contains a lot of subsets.

Make sure you can launch your model with your new task using `--tasks community|yournewtask|2|0 --custom_tasks community_tasks/yourevalname.py`.

### Adding a new metric
First, check if you can use one of the parametrized functions in `src.lighteval.metrics.metrics_corpus` or `src.lighteval.metrics.metrics_sample`.

If not, you can use the custom_task system to register your new metric:
- create a new Python file which should contain the full logic of your metric.
- the file also needs to start with these imports
```python
from aenum import extend_enum
from lighteval.metrics import Metrics

# And any other class you might need to redefine your specific metric, depending on whether it's a sample or corpus metric.
```

- and to end with the following, so that it adds your metric to our metrics list when loaded as a module.

```python
# Adds the metric to the metric list!
extend_enum(Metrics, "metric_name", metric_function)
if __name__ == "__main__":
    print("Imported metric")
```

You can then give your custom metric to lighteval by using `--custom-tasks path_to_your_file` when launching it.

To see an example of a custom metric added along with a custom task, look at `examples/tasks/custom_tasks_with_custom_metrics/ifeval/ifeval.py`.

## Available metrics
### Metrics for multiple choice tasks
These metrics use log-likelihood of the different possible targets.
- `loglikelihood_acc` (Harness): Fraction of instances where the choice with the best logprob was correct - also exists in a faster version for tasks where the possible choices include only one token (`loglikelihood_acc_single_token`)
- `loglikelihood_acc_norm` (Harness): Fraction of instances where the choice with the best logprob, normalized by sequence length, was correct - also exists in a faster version for tasks where the possible choices include only one token (`loglikelihood_acc_norm_single_token`)
- `loglikelihood_acc_norm_nospace` (Harness): Fraction of instances where the choice with the best logprob, normalized by sequence length, was correct, with the first space ignored
- `loglikelihood_f1` (Harness): Corpus level F1 score of the multichoice selection - also exists in a faster version for tasks where the possible choices include only one token (`loglikelihood_f1_single_token`)
- `mcc` (Harness): Matthew's correlation coefficient (a measure of agreement between statistical distributions),
- `recall_at_1` (Harness): Fraction of instances where the choice with the best logprob was correct - also exists in a faster version for tasks where the possible choices include only one token per choice (`recall_at_1_single_token`)
- `recall_at_2` (Harness): Fraction of instances where the choice with the 2nd best logprob or better was correct  - also exists in a faster version for tasks where the possible choices include only one token per choice (`recall_at_2_single_token`)
- `mrr` (Harness): Mean reciprocal rank, a measure of the quality of a ranking of choices ordered by correctness/relevance  - also exists in a faster version for tasks where the possible choices include only one token (`mrr_single_token`)
- `target_perplexity` (Harness): Perplexity of the different choices available.
- `acc_golds_likelihood`: (Harness): A bit different, it actually checks if the average logprob of a single target is above or below 0.5
- `multi_f1_numeric`: Loglikelihood F1 score for multiple gold targets

All these metrics also exist in a "single token" version (`loglikelihood_acc_single_token`, `loglikelihood_acc_norm_single_token`, `loglikelihood_f1_single_token`, `mcc_single_token`, `recall@2_single_token` and `mrr_single_token`). When the multichoice option compares only one token (ex: "A" vs "B" vs "C" vs "D", or "yes" vs "no"), using these metrics in the single token version will divide the time spent by the number of choices. Single token evals also include:
- `multi_f1_numeric` (Harness, for CB): computes the f1 score of all possible choices and averages it.

### Metrics for perplexity and language modeling
These metrics use log-likelihood of prompt.
- `word_perplexity` (Harness): Perplexity (log probability of the input) weighted by the number of words of the sequence.
- `byte_perplexity` (Harness): Perplexity (log probability of the input) weighted by the number of bytes of the sequence.
- `bits_per_byte` (HELM): Average number of bits per byte according to model probabilities.
- `log_prob` (HELM): Predicted output's average log probability (input's log prob for language modeling).

### Metrics for generative tasks
These metrics need the model to generate an output. They are therefore slower.
- Base:
    - `perfect_exact_match` (Harness): Fraction of instances where the prediction matches the gold exactly.
    - `exact_match` (HELM): Fraction of instances where the prediction matches the gold with the exception of the border whitespaces (= after a `strip` has been applied to both).
    - `quasi_exact_match` (HELM): Fraction of instances where the normalized prediction matches the normalized gold (normalization done on whitespace, articles, capitalization, ...). Other variations exist, with other normalizers, such as `quasi_exact_match_triviaqa`, which only normalizes the predictions after applying a strip to all sentences.
    - `prefix_exact_match` (HELM): Fraction of instances where the beginning of the prediction matches the gold at the exception of the border whitespaces (= after a `strip` has been applied to both).
    - `prefix_quasi_exact_match` (HELM): Fraction of instances where the normalized beginning of the prediction matches the normalized gold (normalization done on whitespace, articles, capitalization, ...)
    - `exact_match_indicator`: Exact match with some preceding context (before an indicator) removed
    - `f1_score_quasi` (HELM): Average F1 score in terms of word overlap between the model output and gold, with both being normalized first
    - `f1_score`:  Average F1 score in terms of word overlap between the model output and gold without normalisation
    - `f1_score_macro`: Corpus level macro F1 score
    - `f1_score_macro`: Corpus level micro F1 score
    - `maj_at_5` and `maj_at_8`: Model majority vote. Takes n (5 or 8) generations from the model and assumes the most frequent is the actual prediction.
- Summarization:
    - `rouge` (Harness): Average ROUGE score [(Lin, 2004)](https://aclanthology.org/W04-1013/)
    - `rouge1` (HELM): Average ROUGE score [(Lin, 2004)](https://aclanthology.org/W04-1013/) based on 1-gram overlap.
    - `rouge2` (HELM): Average ROUGE score [(Lin, 2004)](https://aclanthology.org/W04-1013/) based on 2-gram overlap.
    - `rougeL` (HELM): Average ROUGE score [(Lin, 2004)](https://aclanthology.org/W04-1013/) based on longest common subsequence overlap.
    - `rougeLsum` (HELM): Average ROUGE score [(Lin, 2004)](https://aclanthology.org/W04-1013/) based on longest common subsequence overlap.
    - `rouge_t5` (BigBench): Corpus level ROUGE score for all available ROUGE metrics
    - `faithfulness` (HELM): Faithfulness scores based on the SummaC method of [Laban et al. (2022)](https://aclanthology.org/2022.tacl-1.10/).
    - `extractiveness` (HELM): Reports, based on [(Grusky et al., 2018)](https://aclanthology.org/N18-1065/)
        - `summarization_coverage`: Extent to which the model-generated summaries are extractive fragments from the source document,
        - `summarization_density`: Extent to which the model-generated summaries are extractive summaries based on the source document,
        - `summarization_compression`: Extent to which the model-generated summaries are compressed relative to the source document.
    - `bert_score` (HELM): Reports the average BERTScore precision, recall, and f1 score [(Zhang et al., 2020)](https://openreview.net/pdf?id=SkeHuCVFDr) between model generation and gold summary.
- Translation
    - `bleu`: Corpus level BLEU score [(Papineni et al., 2002)](https://aclanthology.org/P02-1040/) - uses the sacrebleu implementation.
    - `bleu_1` (HELM): Average sample BLEU score [(Papineni et al., 2002)](https://aclanthology.org/P02-1040/) based on 1-gram overlap - uses the nltk implementation.
    - `bleu_4` (HELM): Average sample BLEU score [(Papineni et al., 2002)](https://aclanthology.org/P02-1040/) based on 4-gram overlap - uses the nltk implementation.
    - `chrf` (Harness): Character n-gram matches f-score.
    - `ter` (Harness): Translation edit/error rate.
- Copyright
    - `copyright` (HELM): Reports:
        - `longest_common_prefix_length`: average length of longest common prefix between model generation and reference,
        - `edit_distance`: average Levenshtein edit distance between model generation and reference,
        - `edit_similarity`: average Levenshtein edit similarity (normalized by length of longer sequence) between model generation and reference.
- Math:
    - `quasi_exact_match_math` (HELM): Fraction of instances where the normalized prediction matches the normalized gold (normalization done for math, where latex symbols, units, etc are removed)
    - `maj_at_4_math` (Lighteval): Majority choice evaluation, using the math normalisation for the predictions and gold
    - `quasi_exact_match_gsm8k` (Harness): Fraction of instances where the normalized prediction matches the normalized gold (normalization done for gsm8k, where latex symbols, units, etc are removed)
    - `maj_at_8_gsm8k` (Lighteval): Majority choice evaluation, using the gsm8k normalisation for the predictions and gold
- LLM-as-Judge:
    - `llm_judge_gpt3p5`: Can be used for any generative task, the model will be scored by a GPT3.5 model using the openai API
    - `llm_judge_llama_3_405b`: Can be used for any generative task, the model will be scored by a Llama 3.405B model using the openai API
    - `llm_judge_multi_turn_gpt3p5`: Can be used for any generative task, the model will be scored by a GPT3.5 model using the openai API. It is used for multiturn tasks like mt-bench.
    - `llm_judge_multi_turn_llama_3_405b`: Can be used for any generative task, the model will be scored by a Llama 3.405B model using the openai API. It is used for multiturn tasks like mt-bench.

### Metrics for specific tasks
To keep compatibility with the Harness for some specific tasks, we ported their evaluations more or less as such. They include `drop` (for the DROP dataset) and `truthfulqa_mc_metrics` (for TruthfulQA). In general, except for tasks where the dataset has very different formatting than usual (another language, programming language, math, ...), we want to use standard implementations of the above metrics. It makes little sense to have 10 different versions of an exact match depending on the task. However, most of the above metrics are parametrizable so that you can change the normalization applied easily for experimental purposes.

### Not working yet
These metrics need both the generation and its logprob. They are not working at the moment, as this fn is not in the AI Harness.
- `prediction_perplexity` (HELM): Measure of the logprob of a given input.

## Examples of scripts to launch lighteval on the cluster
### Evaluate a whole suite on one node, 8 GPUs
1) Create a config file for accelerate

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

2) Create a slurm file

```bash
#!/bin/bash
#SBATCH --job-name=kirby-one-node
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:8
#SBATCH --mem-per-cpu=11G # This is essentially 1.1T / 96
#SBATCH --partition=production-cluster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=clementine@huggingface.co

set -x -e
export TMPDIR=/scratch

echo "START TIME: $(date)"

# Activate your relevant virtualenv
source <path_to_your_venv>/activate #or conda activate yourenv

cd <path_to_your_lighteval>/lighteval

export CUDA_LAUNCH_BLOCKING=1
srun accelerate launch --multi_gpu --num_processes=8 -m lighteval accelerate --model_args "pretrained=your model name" --tasks examples/tasks/open_llm_leaderboard_tasks.txt --override_batch_size 1 --save_details --output_dir=your output dir
```

## Authentication

For authentication of HuggingFace models (i.e `base` models), a HuggingFace token is used. The `HF_TOKEN` used is picked up directly from the environment.

For `tgi` models, authentication is provided in the config file. An example can be found at [tgi_model.yaml](./examples/model_configs/tgi_model.yaml).

## Releases

### Building the package
```bash
pip install build
python3 -m build .
```

## Cite as

```bibtex
@misc{lighteval,
  author = {Fourrier, Clémentine and Habib, Nathan and Wolf, Thomas and Tunstall, Lewis},
  title = {LightEval: A lightweight framework for LLM evaluation},
  year = {2023},
  version = {0.3.0},
  url = {https://github.com/huggingface/lighteval}
}
```