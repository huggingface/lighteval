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

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


"""
abstract:
The Pile corpus for measuring lanugage model performance across various domains.

languages:
en

tags:
language-modeling

paper:
https://arxiv.org/abs/2101.00027
"""

the_pile_arxiv_helm = LightevalTaskConfig(
    name="the_pile:arxiv",
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
    suite=["lighteval"],
    prompt_function=prompt.the_pile,
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
