<p align="center">
  <br/>
    <img alt="lighteval library logo" src="./assets/lighteval-doc.svg" width="376" height="59" style="max-width: 100%;">
  <br/>
</p>


<p align="center">
    <i>Your go-to toolkit for lightning-fast, flexible LLM evaluation, from Hugging Face's Leaderboard and Evals Team.</i>
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lighteval/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/huggingface/lighteval/actions/workflows/tests.yaml?query=branch%3Amain)
[![Quality](https://github.com/huggingface/lighteval/actions/workflows/quality.yaml/badge.svg?branch=main)](https://github.com/huggingface/lighteval/actions/workflows/quality.yaml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lighteval)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/huggingface/lighteval/blob/main/LICENSE)
[![Version](https://img.shields.io/pypi/v/lighteval)](https://pypi.org/project/lighteval/)

</div>

---

**Documentation**: <a href="https://github.com/huggingface/lighteval/wiki" target="_blank">Lighteval's Wiki</a>

---

### Unlock the Power of LLM Evaluation with Lighteval 🚀

**Lighteval** is your all-in-one toolkit for evaluating LLMs across multiple
backends—whether it's
[transformers](https://github.com/huggingface/transformers),
[tgi](https://github.com/huggingface/text-generation-inference),
[vllm](https://github.com/vllm-project/vllm), or
[nanotron](https://github.com/huggingface/nanotron)—with
ease. Dive deep into your model’s performance by saving and exploring detailed,
sample-by-sample results to debug and see how your models stack-up.

Customization at your fingertips: letting you effortlessly create [new
tasks](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task) and
[metrics](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
tailored to your needs.

Seamlessly experiment, benchmark, and store your results on the Hugging Face
Hub, S3, or locally. Empower your AI research with precision and flexibility.



## 🔑 Key Features

- **Speed**: [Use vllm as backend for fast evals](https://github.com/huggingface/lighteval/wiki/Use-VLLM-as-backend).
- **Seamless Storage**: [Save results in S3 or Hugging Face Datasets](https://github.com/huggingface/lighteval/wiki/Saving-results).
- **Python API**: [Simple integration with the Python API](https://github.com/huggingface/lighteval/wiki/Use-the-Python-API).
- **Custom Tasks**: [Easily add custom tasks](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task).
- **Versatility**: Tons of [metrics](https://github.com/huggingface/lighteval/wiki/Metric-List) and [tasks](https://github.com/huggingface/lighteval/wiki/Available-Tasks) ready to go.


## ⚡️ Installation

```bash
git clone https://github.com/huggingface/lighteval.git
cd lighteval
pip install -e .[{extras}]
```

| extra name   | description                                                               |
|--------------|---------------------------------------------------------------------------|
| accelerate   | To use accelerate for model and data parallelism with transformers models |
| tgi          | To use Text Generation Inference API to evaluate your model               |
| nanotron     | To evaluate nanotron models                                               |
| quantization | To evaluate quantized models                                              |
| adapters     | To evaluate adapters models (delta and peft)                              |
| tensorboardX | To upload your results to tensorboard                                     |
| vllm         | To use vllm as backend for inference                                      |
| s3           | To push your results to s3                                                |
| dev          | Will download all the tools necessary for contributing to the lib         |


If you want to push results to the Hugging Face Hub, add your access token as
an environment variable:

```shell
huggingface-cli login
```

## 🚀 Quickstart

Lighteval offers two main entry points for model evaluation:


* `lighteval accelerate`: evaluate models on CPU or one or more GPUs using [🤗
  Accelerate](https://github.com/huggingface/accelerate).
* `lighteval nanotron`: evaluate models in distributed settings using [⚡️
  Nanotron](https://github.com/huggingface/nanotron).

Here’s a quick command to evaluate using the Accelerate backend:

```shell
lighteval accelerate \
    --model_args "pretrained=gpt2" \
    --tasks "leaderboard|truthfulqa:mc|0|0" \
    --override_batch_size 1 \
    --output_dir="./evals/"
```

## 🙏 Acknowledgements

Lighteval started as an extension of the fantastic [Eleuther AI
Harness](https://github.com/EleutherAI/lm-evaluation-harness) (which powers the
[Open LLM
Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard))
and draws inspiration from the amazing
[HELM](https://crfm.stanford.edu/helm/latest/) framework.

While evolving Lighteval into its own standalone tool, we are grateful to the
Harness and HELM teams for their pioneering work on LLM evaluations.

## 🌟 Contributions Welcome 💙💚💛💜🧡

Got ideas? Found a bug? Want to add a
[task](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task) or
[metric](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)?
Contributions are warmly
welcomed!

## 📜 Citation

```bibtex
@misc{lighteval,
  author = {Fourrier, Clémentine and Habib, Nathan and Wolf, Thomas and Tunstall, Lewis},
  title = {LightEval: A lightweight framework for LLM evaluation},
  year = {2023},
  version = {0.4.0},
  url = {https://github.com/huggingface/lighteval}
}
```
