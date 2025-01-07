# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
The Japanese benchmark JGLUE has been implemented, covering four of its five
benchmark tasks (MARC-ja is unavailable as the dataset has been removed at
Amazon's request). This effort is part of the plan to reimplement llm-jp-eval
in the lighteval framework.

Tasks:
- JSTS
- JNLI
- JSQUAD
- JCommonsenseQA

The datasets were developed by Yahoo Japan, with prompts inspired by
Stability-AI's fork of lm-evaluation-harness from last year, though the fork is
no longer directly runnable.

Dataset: https://github.com/yahoojapan/JGLUE
Prompts: https://github.com/Stability-AI/lm-evaluation-harness
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr

from lighteval.metrics.metrics import CorpusLevelMetric, Metrics
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


_CITATION = """
@inproceedings{kurihara-etal-2022-jglue,
    title = "{JGLUE}: {J}apanese General Language Understanding Evaluation",
    author = "Kurihara, Kentaro  and
      Kawahara, Daisuke  and
      Shibata, Tomohide",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.317",
    pages = "2957--2966",
    abstract = "To develop high-performance natural language understanding (NLU) models, it is necessary to have a benchmark to evaluate and analyze NLU ability from various perspectives. While the English NLU benchmark, GLUE, has been the forerunner, benchmarks are now being released for languages other than English, such as CLUE for Chinese and FLUE for French; but there is no such benchmark for Japanese. We build a Japanese NLU benchmark, JGLUE, from scratch without translation to measure the general NLU ability in Japanese. We hope that JGLUE will facilitate NLU research in Japanese.",
}
"""


# Metrics
def correlation_metric(golds: list[int], predictions: list[str], **kwargs):
    def convert_to_float(score):
        try:
            return float(score)
        except ValueError:
            return None

    predicted_score = convert_to_float(predictions[0])
    gold_score = convert_to_float(golds[0])

    return {
        "predicted_score": predicted_score,
        "gold_score": gold_score,
    }


def spearman_corpus_metric(items):
    predicted_scores, gold_scores = zip(
        *[
            (item["predicted_score"], item["gold_score"])
            for item in items
            if (item["gold_score"] is not None and item["predicted_score"] is not None)
        ]
    )
    r, _ = spearmanr(predicted_scores, gold_scores)
    if np.isnan(r):
        return 0.0
    frac = len(predicted_scores) / len(items)

    return r * frac


def pearson_corpus_metric(items):
    predicted_scores, gold_scores = zip(
        *[
            (item["predicted_score"], item["gold_score"])
            for item in items
            if (item["gold_score"] is not None and item["predicted_score"] is not None)
        ]
    )
    r, _ = pearsonr(predicted_scores, gold_scores)
    if np.isnan(r):
        return 0.0
    frac = len(predicted_scores) / len(items)
    return r * frac


spearman_metric = CorpusLevelMetric(
    metric_name="spearman_correlation",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.NONE,
    sample_level_fn=correlation_metric,
    corpus_level_fn=spearman_corpus_metric,
)

pearson_metric = CorpusLevelMetric(
    metric_name="pearson_correlation",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.NONE,
    sample_level_fn=correlation_metric,
    corpus_level_fn=pearson_corpus_metric,
)

# JSQUAD

JSQUAD_INSTRUCTION = "[題名]と[問題]から[質問]に対する[答え]を抜き出しなさい\n\n"

JSQUAD_PROMPT_TEMPLAT = """\
[題名]:
{title}
[問題]:
{context}
[質問]:
{question}
[答え]: """


def jsquad_prompt_fn(line, task_name: str = None):
    prompt = JSQUAD_PROMPT_TEMPLAT.format(title=line["title"], context=line["context"], question=line["question"])
    query = JSQUAD_INSTRUCTION + "\n\n" + prompt
    answer = line["answers"][0]["text"]

    doc = Doc(
        task_name=task_name,
        query=query,
        choices=[answer],
        gold_index=0,
        instruction=JSQUAD_INSTRUCTION,
    )
    return doc


jsquad_task = LightevalTaskConfig(
    name="jglue:jsquad",
    prompt_function=jsquad_prompt_fn,
    suite=["community"],
    hf_repo="zenless-lab/jsquad",
    hf_subset="default",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.f1_score_macro,
        Metrics.f1_score_micro,
    ],
)

# JCommonsenceQA

JCOMMONSENSE_QA_INSTRUCTION = "[問題]に対する[答え]を[選択肢]の中から選んでください。\n\n"

JCOMMONSENSE_QA_PROMPT_TEMPLAT = """\
[問題]: {question}
[選択肢]: {choices}
[答え]: """


def jcommonsenseqa_prompt_fn(line, task_name: str = None):
    choices = [line[f"choice{i}"] for i in range(5)]
    prompt = JCOMMONSENSE_QA_PROMPT_TEMPLAT.format(question=line["question"], choices=str(choices))
    query = JCOMMONSENSE_QA_INSTRUCTION + "\n\n" + prompt
    label = line["label"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=label,
        instruction=JCOMMONSENSE_QA_INSTRUCTION,
    )


jcommonsenseqa_task = LightevalTaskConfig(
    name="jglue:jcommonsenseqa",
    prompt_function=jcommonsenseqa_prompt_fn,
    suite=["community"],
    hf_repo="zenless-lab/jcommonsenseqa",
    hf_subset="default",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm,
        Metrics.loglikelihood_acc_norm_nospace,
    ],
)

# JSTS

JSTS_INSTRUCTION = (
    "日本語の文ペアの意味がどのくらい近いかを判定し、類似度を0.0〜5.0までの間の値で付与してください。"
    "0.0に近いほど文ペアの意味が異なり、5.0に近いほど文ペアの意味が似ていることを表しています。"
    "整数値のみを返し、それ以外には何も含めないことを厳守してください。"
)
JSTS_PROMPT_TEMPLAT = """\
[文1]: {sentence1}
[文2]: {sentence2}
[類似度]: """


def jsts_prompt_fn(line, task_name: str = None):
    prompt = JSTS_PROMPT_TEMPLAT.format(sentence1=line["sentence1"], sentence2=line["sentence2"])
    query = JSTS_INSTRUCTION + "\n\n" + prompt
    answer = line["label"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[answer],
        gold_index=0,
        instruction=JSTS_INSTRUCTION,
    )


jsts_task = LightevalTaskConfig(
    name="jglue:jsts",
    prompt_function=jsts_prompt_fn,
    suite=["community"],
    hf_repo="zenless-lab/jsts",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[spearman_metric, pearson_metric],
)

# JNLI

JNLI_INSTRUCTION = """
前提と仮説の関係を「含意」、「矛盾」、「中立」の中から回答してください。
制約:
- 前提から仮説が、論理的知識や常識的知識を用いて導出可能である場合は 含意 と出力
- 前提と仮説が両立しえない場合は 矛盾 と出力
- そのいずれでもない場合は 中立 と出力"""

JNLI_PROMPT_TEMPLAT = """\
[前提]: {premise}
[仮説]: {hypothesis}
[関係]: """

JNLI_LABELS = ["含意", "中立", "矛盾"]


def jnli_prompt_fn(line, task_name: str = None):
    prompt = JNLI_PROMPT_TEMPLAT.format(premise=line["premise"], hypothesis=line["hypothesis"])
    query = JNLI_INSTRUCTION + "\n\n" + prompt
    label = line["label"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=JNLI_LABELS,
        gold_index=label,
        instruction=JNLI_INSTRUCTION,
    )


jnli_task = LightevalTaskConfig(
    name="jglue:jnli",
    prompt_function=jnli_prompt_fn,
    suite=["community"],
    hf_repo="zenless-lab/jnli",
    hf_subset="default",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm,
        Metrics.loglikelihood_acc_norm_nospace,
    ],
)

TASKS_TABLE = [jsquad_task, jcommonsenseqa_task, jsts_task, jnli_task]
