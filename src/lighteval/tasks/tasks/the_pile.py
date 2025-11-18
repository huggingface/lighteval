"""
name:
The Pile

dataset:
lighteval/pile_helm

abstract:
The Pile corpus for measuring lanugage model performance across various domains.

languages:
english

tags:
language-modeling

paper:
https://arxiv.org/abs/2101.00027
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def the_pile_prompt(line, task_name: str = None):
    return Doc(task_name=task_name, query=line["text"], gold_index=None, choices=None)


the_pile_arxiv_helm = LightevalTaskConfig(
    name="the_pile:arxiv",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="arxiv",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_bibliotik_helm = LightevalTaskConfig(
    name="the_pile:bibliotik",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="bibliotik",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_commoncrawl_helm = LightevalTaskConfig(
    name="the_pile:commoncrawl",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="commoncrawl",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_dm_mathematics_helm = LightevalTaskConfig(
    name="the_pile:dm-mathematics",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="dm-mathematics",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_enron_helm = LightevalTaskConfig(
    name="the_pile:enron",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="enron",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_europarl_helm = LightevalTaskConfig(
    name="the_pile:europarl",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="europarl",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_freelaw_helm = LightevalTaskConfig(
    name="the_pile:freelaw",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="freelaw",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_github_helm = LightevalTaskConfig(
    name="the_pile:github",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="github",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_gutenberg_helm = LightevalTaskConfig(
    name="the_pile:gutenberg",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="gutenberg",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_hackernews_helm = LightevalTaskConfig(
    name="the_pile:hackernews",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="hackernews",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_nih_exporter_helm = LightevalTaskConfig(
    name="the_pile:nih-exporter",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="nih-exporter",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_opensubtitles_helm = LightevalTaskConfig(
    name="the_pile:opensubtitles",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="opensubtitles",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_openwebtext2_helm = LightevalTaskConfig(
    name="the_pile:openwebtext2",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="openwebtext2",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)


the_pile_pubmed_abstracts_helm = LightevalTaskConfig(
    name="the_pile:pubmed-abstracts",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="pubmed-abstracts",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_pubmed_central_helm = LightevalTaskConfig(
    name="the_pile:pubmed-central",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="pubmed-central",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_stackexchange_helm = LightevalTaskConfig(
    name="the_pile:stackexchange",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="stackexchange",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_upsto_helm = LightevalTaskConfig(
    name="the_pile:upsto",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="uspto",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_wikipedia_helm = LightevalTaskConfig(
    name="the_pile:wikipedia",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="wikipedia",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

the_pile_youtubesubtitles_helm = LightevalTaskConfig(
    name="the_pile:youtubesubtitles",
    prompt_function=the_pile_prompt,
    hf_repo="lighteval/pile_helm",
    hf_subset="youtubesubtitles",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    the_pile_arxiv_helm,
    the_pile_bibliotik_helm,
    the_pile_commoncrawl_helm,
    the_pile_dm_mathematics_helm,
    the_pile_enron_helm,
    the_pile_europarl_helm,
    the_pile_freelaw_helm,
    the_pile_github_helm,
    the_pile_gutenberg_helm,
    the_pile_hackernews_helm,
    the_pile_nih_exporter_helm,
    the_pile_opensubtitles_helm,
    the_pile_openwebtext2_helm,
    the_pile_pubmed_abstracts_helm,
    the_pile_pubmed_central_helm,
    the_pile_stackexchange_helm,
    the_pile_upsto_helm,
    the_pile_wikipedia_helm,
    the_pile_youtubesubtitles_helm,
]
