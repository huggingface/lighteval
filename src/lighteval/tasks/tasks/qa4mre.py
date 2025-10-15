"""
name:
Qa4Mre

dataset:
qa4mre

abstract:
QA4MRE is a machine reading comprehension benchmark from the CLEF 2011-2013
challenges. It evaluates systems' ability to answer questions requiring deep
understanding of short texts, supported by external background knowledge.
Covering tasks like modality, negation, biomedical reading, and entrance exams,
QA4MRE tests reasoning beyond surface-level text matching.

languages:
english

tags:
biomedical, health, qa

paper:
https://link.springer.com/chapter/10.1007/978-3-642-40802-1_29
"""

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


qa4mre_2011 = LightevalTaskConfig(
    name="qa4mre:2011",
    suite=["lighteval"],
    prompt_function=prompt.qa4mre,
    hf_repo="qa4mre",
    hf_subset="2011.main.EN",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
)


qa4mre_2012 = LightevalTaskConfig(
    name="qa4mre:2012",
    suite=["lighteval"],
    prompt_function=prompt.qa4mre,
    hf_repo="qa4mre",
    hf_subset="2012.main.EN",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
)


qa4mre_2013 = LightevalTaskConfig(
    name="qa4mre:2013",
    suite=["lighteval"],
    prompt_function=prompt.qa4mre,
    hf_repo="qa4mre",
    hf_subset="2013.main.EN",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[
        Metrics.loglikelihood_acc,
    ],
    stop_sequence=["\n"],
    version=0,
)
