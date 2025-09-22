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

<p align="center">
  <a href="https://huggingface.co/docs/lighteval/main/en/index" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/Documentation-4F4F4F?style=for-the-badge&logo=readthedocs&logoColor=white" />
  </a>
</p>

---

**Lighteval** is your *all-in-one toolkit* for evaluating LLMs across multiple
backends—whether your model is being **served somewhere** or **already loaded in memory**.
Dive deep into your model's performance by saving and exploring *detailed,
sample-by-sample results* to debug and see how your models stack-up.

*Customization at your fingertips*: letting you either browse all our existing tasks and [metrics](https://huggingface.co/docs/lighteval/metric-list) or effortlessly create your own [custom task](https://huggingface.co/docs/lighteval/adding-a-custom-task) and [custom metric](https://huggingface.co/docs/lighteval/adding-a-new-metric), tailored to your needs.


## Available Tasks

Lighteval supports **7,000+ evaluation tasks** across multiple domains and languages. Here's an overview of some *popular benchmarks*:


### 📚 **Knowledge**
- **General Knowledge**: MMLU, MMLU-Pro, MMMU, BIG-Bench
- **Question Answering**: TriviaQA, Natural Questions, SimpleQA, Humanity's Last Exam (HLE)
- **Specialized**: GPQA, AGIEval

### 🧮 **Math and Code**
- **Math Problems**: GSM8K, GSM-Plus, MATH, MATH500
- **Competition Math**: AIME24, AIME25
- **Multilingual Math**: MGSM (Grade School Math in 10+ languages)
- **Coding Benchmarks**: LCB (LiveCodeBench)

### 🎯 **Chat Model Evaluation**
- **Instruction Following**: IFEval, IFEval-fr
- **Reasoning**: MUSR, DROP (discrete reasoning)
- **Long Context**: RULER
- **Dialogue**: MT-Bench
- **Holistic Evaluation**: HELM, BIG-Bench

### 🌍 **Multilingual Evaluation**
- **Cross-lingual**: XTREME, Flores200 (200 languages), XCOPA, XQuAD
- **Language-specific**: 
  - **Arabic**: ArabicMMLU
  - **Filipino**: FilBench
  - **French**: IFEval-fr, GPQA-fr, BAC-fr
  - **German**: German RAG Eval
  - **Serbian**: Serbian LLM Benchmark, OZ Eval
  - **Turkic**: TUMLU (9 Turkic languages)
  - **Chinese**: CMMLU, CEval, AGIEval
  - **Russian**: RUMMLU, Russian SQuAD
  - **And many more...**

### 🧠 **Core Language Understanding**
- **NLU**: GLUE, SuperGLUE, TriviaQA, Natural Questions
- **Commonsense**: HellaSwag, WinoGrande, ProtoQA
- **Natural Language Inference**: XNLI
- **Reading Comprehension**: SQuAD, XQuAD, MLQA, Belebele


## ⚡️ Installation

> **Note**: lighteval is currently *completely untested on Windows*, and we don't support it yet. (*Should be fully functional on Mac/Linux*)

```bash
pip install lighteval
```

Lighteval allows for *many extras* when installing, see [here](https://huggingface.co/docs/lighteval/installation) for a **complete list**.

If you want to push results to the **Hugging Face Hub**, add your access token as
an environment variable:

```shell
huggingface-cli login
```

## 🚀 Quickstart

Lighteval offers the following entry points for model evaluation:

- `lighteval accelerate`: Evaluate models on CPU or one or more GPUs using [🤗
  Accelerate](https://github.com/huggingface/accelerate)
- `lighteval nanotron`: Evaluate models in distributed settings using [⚡️
  Nanotron](https://github.com/huggingface/nanotron)
- `lighteval vllm`: Evaluate models on one or more GPUs using [🚀
  VLLM](https://github.com/vllm-project/vllm)
- `lighteval sglang`: Evaluate models using [SGLang](https://github.com/sgl-project/sglang) as backend
- `lighteval endpoint`: Evaluate models using various endpoints as backend
  - `lighteval endpoint inference-endpoint`: Evaluate models using Hugging Face's [Inference Endpoints API](https://huggingface.co/inference-endpoints/dedicated)
  - `lighteval endpoint tgi`: Evaluate models using [🔗 Text Generation Inference](https://huggingface.co/docs/text-generation-inference/en/index) running locally
  - `lighteval endpoint litellm`: Evaluate models on any compatible API using [LiteLLM](https://www.litellm.ai/)
  - `lighteval endpoint inference-providers`: Evaluate models using [HuggingFace's inference providers](https://huggingface.co/docs/inference-providers/en/index) as backend

Did not find what you need ? You can always make your custom model API by following [this guide](https://huggingface.co/docs/lighteval/main/en/evaluating-a-custom-model)
- `lighteval custom`: Evaluate custom models (can be anything)

Here's a **quick command** to evaluate using the *Accelerate backend*:

```shell
lighteval accelerate \
    "model_name=gpt2" \
    "leaderboard|truthfulqa:mc|0"
```

Or use the **Python API** to run a model *already loaded in memory*!

```python
from transformers import AutoModelForCausalLM

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
BENCHMARKS = "lighteval|gsm8k|0"

evaluation_tracker = EvaluationTracker(output_dir="./results")
pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.NONE,
    max_samples=2
)

model = AutoModelForCausalLM.from_pretrained(
  MODEL_NAME, device_map="auto"
)
config = TransformersModelConfig(model_name=MODEL_NAME, batch_size=1)
model = TransformersModel.from_model(model, config)

pipeline = Pipeline(
    model=model,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    tasks=BENCHMARKS,
)

results = pipeline.evaluate()
pipeline.show_results()
results = pipeline.get_results()
```

## 🙏 Acknowledgements

Lighteval took inspiration from the following *amazing* frameworks: Eleuther's [AI Harness](https://github.com/EleutherAI/lm-evaluation-harness) and Stanford's
[HELM](https://crfm.stanford.edu/helm/latest/). We are grateful to their teams for their **pioneering work** on LLM evaluations.

We'd also like to offer our thanks to all the community members who have contributed to the library, adding new features and reporting or fixing bugs.

## 🌟 Contributions Welcome 💙💚💛💜🧡

**Got ideas?** Found a bug? Want to add a
[task](https://huggingface.co/docs/lighteval/adding-a-custom-task) or
[metric](https://huggingface.co/docs/lighteval/adding-a-new-metric)?
Contributions are *warmly welcomed*!

If you're adding a **new feature**, please *open an issue first*.

If you open a PR, don't forget to **run the styling**!

```bash
pip install -e .[dev]
pre-commit install
pre-commit run --all-files
```
## 📜 Citation

```bibtex
@misc{lighteval,
  author = {Habib, Nathan and Fourrier, Clémentine and Kydlíček, Hynek and Wolf, Thomas and Tunstall, Lewis},
  title = {LightEval: A lightweight framework for LLM evaluation},
  year = {2023},
  version = {0.11.0},
  url = {https://github.com/huggingface/lighteval}
}
```
