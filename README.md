# LightEval 🌤️
A lightweight LLM evaluation

## Context
LightEval is a lightweight LLM evaluation suite that Hugging Face has been using internally with the recently released LLM data processing library [datatrove](https://github.com/huggingface/datatrove) and LLM training library [nanotron](https://github.com/huggingface/nanotron).

We're releasing it with the community in the spirit of building in the open.

Note that it is still very much early so don't expect 100% stability ^^'
In case of problems or question, feel free to open an issue!

## News
- **Feb 08, 2024**: Release of `lighteval``

## Deep thanks
`lighteval` was originally built on top of the great [Eleuther AI Harness](https://github.com/EleutherAI/lm-evaluation-harness) (which is powering the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)). We also took a lot of inspiration from the amazing [HELM](https://crfm.stanford.edu/helm/latest/), notably for metrics.

Through adding more and more logging functionalities, and making it compatible with increasingly different workflows and model codebases (including 3D parallelism) as well as allowing custom evaluation experiments, metrics and benchmarks, we ended up needing to change the code more and more deeply until `lighteval` became the small standalone library that it is now.

However, we are very grateful to the Harness and HELM teams for their continued work on better evaluations.

## How to navigate this project
`lighteval` is supposed to be used as a standalone evaluation library.
- To run the evaluations, you can use `run_evals_accelerate.py` or `run_evals_nanotron.py`.
- [src/lighteval](https://github.com/huggingface/lighteval/tree/main/src/lighteval) contains the core of the lib itself
    - [lighteval](https://github.com/huggingface/lighteval/tree/main/src/lighteval) contains the core of the library, divided in the following section
        - [main_accelerate.py](https://github.com/huggingface/lighteval/blob/main/src/main_accelerate.py) and [main_accelerate.py](https://github.com/huggingface/lighteval/blob/main/src/main_nanotron.py) are our entry points to run evaluation
        - [logging](https://github.com/huggingface/lighteval/tree/main/src/lighteval/logging): Our loggers, to display experiment information and push it to the hub after a run
        - [metrics](https://github.com/huggingface/lighteval/tree/main/src/lighteval/metrics): All the available metrics you can use. They are described in metrics, and divided between sample metrics (applied at the sample level, such as a prediction accuracy) and corpus metrics (applied over the whole corpus). You'll also find available normalisation functions.
        - [models](https://github.com/huggingface/lighteval/tree/main/src/lighteval/models): Possible models to use. We cover transformers (base_model), with adapter or delta weights, as well as TGI models locally deployed (it's likely the code here is out of date though), and brrr/nanotron models.
        - [tasks](https://github.com/huggingface/lighteval/tree/main/src/lighteval/tasks): Available tasks. The complete list is in `tasks_table.jsonl`, and you'll find all the prompts in `ŧasks_prompt_formatting.py`.
- [tasks_examples](https://github.com/huggingface/lighteval/tree/main/tasks_examples) contains a list of available tasks you can launch. We advise using tasks in the `recommended_set`, as it's possible that some of the other tasks need double checking.
- [tests](https://github.com/huggingface/lighteval/tree/main/tests) contains our test suite, that we run at each PR to prevent regressions in metrics/prompts/tasks, for a subset of important tasks.

## How to install and use

Note:
- Use the Eleuther AI Harness (`lm_eval`) to share comparable numbers with everyone (e.g. on the Open LLM Leaderboard).
- Use `lighteval` during training with the nanotron/datatrove LLM training stack and/or for quick eval/benchmark experimentations.

### Installation
Create your virtual environment using virtualenv or conda depending on your preferences. We require Python3.10 or above.
```bash
conda create -n lighteval python==3.10
```

Clone the package
```bash
git clone
cd lighteval-harness
```

Install the dependencies. For the default installation, you just need:
```bash
pip install -e .
```

If you want to run your models using accelerate, tgi or optimum, do quantization, or use adapter weights, you will need to specify the optional dependencies group fitting your use case (`accelerate`,`tgi`,`optimum`,`quantization`,`adapters`,`nanotron`) at install time
```bash
pip install -e .[optional1,optional2]
```

The setup we tested most is:
```bash
pip install -e .[accelerate,quantization,adapters]
```

If you want to push your results to the hub, don't forget to add your user token to the environment variable `HUGGING_FACE_HUB_TOKEN`.

Lastly, if you intend to push to the code base, you'll need to install the precommit hook for styling tests.
```bash
pip install pre-commit
pre-commit install
```

Optional steps.
- to load and push big models/datasets, your machine likely needs Git LFS. You can install it with `sudo apt-get install git-lfs`
- If you want to run bigbench evaluations, install bigbench `pip install "bigbench@https://storage.googleapis.com/public_research_data/bigbench/bigbench-0.0.1.tar.gz"`


### Testing that everything was installed correctly
If you want to test your install, you can run your first evaluation on GPUs (8GPU, single node), using
```bash
mkdir tmp
python -m accelerate launch --multi_gpu --num_processes=8 run_evals_accelerate.py --model_args "pretrained=gpt2" --tasks tasks_examples/open_llm_leaderboard_tasks.txt --override_batch_size 1 --save_details --output_dir="tmp/"
```

### Usage
- Launching on CPU
    - `python run_evals_accelerate.py --model_args="pretrained=<path to your model on the hub>" <task parameters> --output_dir output_dir`
- Using data parallelism on several GPUs (recommended)
    - If you want to use data parallelism, first configure accelerate (`accelerate config`).
    - `accelerate launch <accelerate parameters> run_evals_accelerate.py --model_args="pretrained=<path to your model on the hub>" <task parameters>  --output_dir=<your output dir>`
    for instance: `python -m accelerate launch --multi_gpu --num_processes=8 run_evals_accelerate.py --model_args "pretrained=gpt2" --tasks tasks_examples/open_llm_leaderboard_tasks.txt --override_batch_size 1 --save_details --output_dir=tmp/`
    - Note: if you use model_parallel, accelerate will use 2 processes for model parallel, num_processes for data parallel

The task parameters indicate which tasks you want to launch. You can select:
- one or several tasks, with `--tasks task_names`, with task_names in the [metadata table](metadata_table.json), separated by commas. You must specify which version of the task you want (= in which suite it is), by prepending the suite name, as well as the number of training few_shots prompts for the given task, and whether you want to automatically reduce the number of few_shots if they make the prompt too long (`suite|task|few_shot|1 or 0 to automatically reduce the number of few_shots or not`).
- a file path, which contains tasks following the above format.

Example
If you want to compare hellaswag from helm and the harness on Gpt-6j, you can do
`python run_evals_accelerate.py --model hf_causal --model_args="pretrained=EleutherAI/gpt-j-6b" --tasks helm|hellaswag|0|0,lighteval|hellaswag|0|0 --output_dir output_dir`

## Customisation
### Adding a new metric
If you want to add a new metric, first check if you can use one of the parametrized functions in `src.lighteval.metrics.metrics_corpus` or `src.lighteval.metrics.metrics_sample`. If not, add it to either of these files depending on the level at which it is applied.
Then, follow the example in `src.lighteval.metrics.metrics` to register your metric.

### Adding a new task
To add a new task, first **add its dataset** on the hub.

Then, **find a suitable prompt function** or **create a new prompt function** in `src.lighteval.tasks.task_prompt_formatting.py`. This function must output a `Doc` object, which should contain `query`, your prompt, and either `gold`, the gold output, or `choices` and `gold_index`, the list of choices and index or indices of correct answers. If your query contains an instruction which should not be repeated in a few shot setup, add it to an `instruction` field.

Lastly, create a **line summary** of your evaluation, in `src/lighteval/tasks/tasks_table.jsonl`. This summary should contain the following fields:
- `name` (str), your evaluation name
- `suite` (list), the suite(s) to which your evaluation should belong. This field allows us to compare different tasks implementation, and is used a task selection to differentiate the versions to launch. At the moment, you'll find the keywords ["helm", "bigbench", "original", "lighteval"]; you can add also add new ones (for test, we recommend using "custom").
- `prompt_function` (str), the name of the prompt function you defined in the step above
- `hf_repo` (str), the path to your evaluation dataset on the hub
- `hf_subset` (str), the specific subset you want to use for your evaluation (note: when the dataset has no subset, fill this field with `"default"`, not with `None` or `""`)
- `hf_avail_splits` (list), all the splits available for your dataset (train, valid or validation, test, other...)
- `evaluation_splits` (list), the splits you want to use for evaluation
- `few_shots_split` (str, can be `null`), the specific split from which you want to select samples for your few-shot examples. It should be different from the sets included in `evaluation_splits`
- `few_shots_select` (str, can be `null`), the method that you will use to select items for your few-shot examples. Can be `null`, or one of:
    - `balanced` selects examples from the `few_shots_split` with balanced labels, to avoid skewing the few shot examples (hence the model generations) towards one specific label
    - `random` selects examples at random from the `few_shots_split`
    - `random_sampling` selects new examples at random from the `few_shots_split` for every new item, but if a sampled item is equal to the current one, it is removed from the available samples
    - `random_sampling_from_train` selects new examples at random from the `few_shots_split` for every new item, but if a sampled item is equal to the current one, it is kept! Only use this if you know what you are doing.
    - `sequential` selects the first `n` examples of the `few_shots_split`
- `generation_size` (int), the maximum number of tokens allowed for a generative evaluation. If your evaluation is a log likelihood evaluation (multi-choice), this value should be -1
- `stop_sequence` (list), a list of strings acting as end of sentence tokens for your generation
- `metric` (list), the metrics you want to use for your evaluation (see next section for a detailed explanation)
- `output_regex` (str), A regex string that will be used to filter your generation. (Genrative metrics will only select tokens that are between the first and the second sequence matched by the regex. For example, for a regex matching `\n` and a generation `\nModel generation output\nSome other text` the metric will only be fed with `Model generation output`)
- `frozen` (bool), for now is set to False, but we will steadily pass all stable tasks to True.

## Available metrics
### Metrics for multiple choice tasks
These metrics use log-likelihood of the different possible targets.
- `loglikelihood_acc` (Harness): Fraction of instances where the choice with the best logprob was correct - also exists in a faster version for tasks where the possible choices include only one token (`loglikelihood_acc_single_token`)
- `loglikelihood_acc_norm` (Harness): Fraction of instances where the choice with the best logprob, normalized by sequence length, was correct - also exists in a faster version for tasks where the possible choices include only one token (`loglikelihood_acc_norm_single_token`)
- `loglikelihood_acc_norm_nospace` (Harness): Fraction of instances where the choice with the best logprob, normalized by sequence length, was correct, with the first space ignored
- `loglikelihood_f1` (Harness): Corpus level F1 score of the multichoice selection - also exists in a faster version for tasks where the possible choices include only one token (`loglikelihood_f1_single_token`)
- `mcc` (Harness): Matthew's correlation coefficient (measure of agreement between statistical distributions),
- `recall_at_1` (Harness): Fraction of instances where the choice with the best logprob was correct - also exists in a faster version for tasks where the possible choices include only one token per choice (`recall_at_1_single_token`)
- `recall_at_2` (Harness): Fraction of instances where the choice with the 2nd best logprob or better was correct  - also exists in a faster version for tasks where the possible choices include only one token per choice (`recall_at_2_single_token`)
- `mrr` (Harness): Mean reciprocal rank, measure of the quality of a ranking of choices ordered by correctness/relevance  - also exists in a faster version for tasks where the possible choices include only one token (`mrr_single_token`)
- `target_perplexity` (Harness): Perplexity of the different choices available.
- `acc_golds_likelihood`: (Harness): A bit different, it actually checks if the average logprob of a single target is above or below 0.5
- `multi_f1_numeric`: Loglikelihood F1 score for multiple gold targets

All these metrics also exist in a "single token" version (`loglikelihood_acc_single_token`, `loglikelihood_acc_norm_single_token`, `loglikelihood_f1_single_token`, `mcc_single_token`, `recall@2_single_token` and `mrr_single_token`). When the multichoice option compare only one token (ex: "A" vs "B" vs "C" vs "D", or "yes" vs "no"), using these metrics in the single token version will divide the time spent by the number of choices. Single token evals also include:
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
    - `exact_match` (HELM): Fraction of instances where the prediction matches the gold at the exception of the border whitespaces (= after a `strip` has been applied to both).
    - `quasi_exact_match` (HELM): Fraction of instances where the normalized prediction matches the normalized gold (normalization done on whitespace, articles, capitalization, ...). Other variations exist, with other normalizers, such as `quasi_exact_match_triviaqa`, which only normalizes the predictions after applying a strip to all sentences.
    - `prefix_exact_match` (HELM): Fraction of instances where the beginning of the prediction matches the gold at the exception of the border whitespaces (= after a `strip` has been applied to both).
    - `prefix_quasi_exact_match` (HELM): Fraction of instances where the normalized beginning of the prediction matches the normalized gold (normalization done on whitespace, articles, capitalization, ...)
    - `exact_match_indicator`: Exact match with some preceding context (before an indicator) removed
    - `f1_score_quasi` (HELM): Average F1 score in terms of word overlap between the model output and gold, with both being normalized first
    - `f1_score`:  Average F1 score in terms of word overlap between the model output and gold without normalisation
    - `f1_score_macro`: Corpus level macro F1 score
    - `f1_score_macro`: Corpus level micro F1 score
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
- Bias, toxicity, copyright
    - `bias` (HELM): Reports uneven association of test groups (race, gender, demographic) and target adjectives or professions, based on cooccurence statistics between the test terms (word list from [Bolukbasi et al., 2016](https://papers.nips.cc/paper/2016/hash/a486cd07e4ac3d270571622f4f316ec5-Abstract.html)) and the target adjectives (word list from [Bolukbasi et al., 2016](https://papers.nips.cc/paper/2016/hash/a486cd07e4ac3d270571622f4f316ec5-Abstract.html)).
    - `copyright` (HELM): Reports:
        - `longest_common_prefix_length`: average length of longest common prefix between model generation and reference,
        - `edit_distance`: average Levenshtein edit distance between model generation and reference,
        - `edit_similarity`: average Levenshtein edit similarity (normalized by length of longer sequence) between model generation and reference.
- Math:
    - `quasi_exact_match_math` (HELM): Fraction of instances where the normalized prediction matches the normalized gold (normalization done for math, where latex symbols, units, etc are removed)
    - `quasi_exact_match_gsm8k` (Harness): Fraction of instances where the normalized prediction matches the normalized gold (normalization done for gsm8k, where latex symbols, units, etc are removed)

### Metrics for specific tasks
To keep compatibility with the Harness for some specific tasks, we ported their evaluations more or less as such. They include `drop` (for the DROP dataset) and `truthfulqa_mc_metrics` (for TruthfulQA). In general, except for tasks where the dataset has a very different formatting than usual (an other language, programming language, math, ...), we want to use standard implementations of the above metrics. It makes little sense to have 10 different versions of an exact match depending on the task. However, most of the above metrics are parametrizable so that you can change the normalization applied easily for experimental purposes.

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
srun accelerate launch --multi_gpu --num_processes=8 run_evals_accelerate.py --model_args "pretrained=your model name" --tasks tasks_examples/open_llm_leaderboard_tasks.txt --override_batch_size 1 --save_details --output_dir=your output dir
```

## Releases

### Building the package
```bash
pip install build
python3 -m build .
```
