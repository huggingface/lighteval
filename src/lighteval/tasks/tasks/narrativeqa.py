"""
name:
Narrativeqa

dataset:
lighteval/narrative_qa_helm

abstract:
NarrativeQA is a reading comprehension benchmark that tests deep understanding
of full narratives—books and movie scripts—rather than shallow text matching. To
answer its questions, models must integrate information across entire stories.

languages:
english

tags:
qa, reading-comprehension

paper:
https://aclanthology.org/Q18-1023/
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


narrativeqa = LightevalTaskConfig(
    name="narrativeqa",
    suite=["lighteval"],
    prompt_function=prompt.narrativeqa,
    hf_repo="lighteval/narrative_qa_helm",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)
