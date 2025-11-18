"""
name:
Legal Summarization

dataset:
lighteval/legal_summarization

abstract:
LegalSummarization is a dataset for legal summarization.

languages:
english

tags:
legal, summarization

paper:
https://arxiv.org/abs/2210.13448
https://arxiv.org/abs/2210.13448
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def legal_summarization_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"###\nArticle:{line['text']}\n\nSummarize the above article in 3 sentences.\n",
        choices=[str(line["summary"])],
        gold_index=0,
        specific={"text": line["text"]},
    )


def multilexsum_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"###\nArticle: {line['article']}\n\nSummarize the above article in 2 sentences.\n",
        gold_index=0,
        choices=[line["summary"]],
        specific={"text": line["article"]},
    )


legal_summarization_billsum = LightevalTaskConfig(
    name="legal_summarization:billsum",
    prompt_function=legal_summarization_prompt,
    hf_repo="lighteval/legal_summarization",
    hf_subset="BillSum",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1024,
    metrics=[
        Metrics.rouge1,
        Metrics.rouge2,
        Metrics.rougeL,
        Metrics.faithfulness,
        Metrics.extractiveness,
        Metrics.bert_score,
    ],
    stop_sequence=["\n"],
    version=0,
)


legal_summarization_eurlexsum = LightevalTaskConfig(
    name="legal_summarization:eurlexsum",
    prompt_function=legal_summarization_prompt,
    hf_repo="lighteval/legal_summarization",
    hf_subset="EurLexSum",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metrics=[
        Metrics.rouge1,
        Metrics.rouge2,
        Metrics.rougeL,
        Metrics.faithfulness,
        Metrics.extractiveness,
        Metrics.bert_score,
    ],
    stop_sequence=["\n"],
    version=0,
)


legal_summarization_multilexsum = LightevalTaskConfig(
    name="legal_summarization:multilexsum",
    prompt_function=multilexsum_prompt,
    hf_repo="lighteval/legal_summarization",
    hf_subset="MultiLexSum",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=256,
    metrics=[
        Metrics.rouge1,
        Metrics.rouge2,
        Metrics.rougeL,
        Metrics.faithfulness,
        Metrics.extractiveness,
        Metrics.bert_score,
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    legal_summarization_billsum,
    legal_summarization_eurlexsum,
    legal_summarization_multilexsum,
]
