"""
name:
Wikitext

dataset:
EleutherAI/wikitext_document_level

abstract:
The WikiText language modeling dataset is a collection of over 100 million
tokens extracted from the set of verified Good and Featured articles on
Wikipedia. The dataset is available under the Creative Commons
Attribution-ShareAlike License.

languages:
english

tags:
language-modeling

paper:
https://arxiv.org/abs/1609.07843
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def wikitext_prompt(line, task_name: str = None):  # perplexity metric
    def wikitext_detokenizer(cur_string):
        import re

        cur_string = cur_string.replace("s '", "s'")
        cur_string = re.sub(r"/' [0-9]/", r"/'[0-9]/", cur_string)
        cur_string = cur_string.replace(" @-@ ", "-")
        cur_string = cur_string.replace(" @,@ ", ",")
        cur_string = cur_string.replace(" @.@ ", ".")
        cur_string = cur_string.replace(" : ", ": ")
        cur_string = cur_string.replace(" ; ", "; ")
        cur_string = cur_string.replace(" . ", ". ")
        cur_string = cur_string.replace(" ! ", "! ")
        cur_string = cur_string.replace(" ? ", "? ")
        cur_string = cur_string.replace(" , ", ", ")
        cur_string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", cur_string)
        cur_string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", cur_string)
        cur_string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", cur_string)
        cur_string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', cur_string)
        cur_string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", cur_string)
        cur_string = cur_string.replace("= = = =", "====")
        cur_string = cur_string.replace("= = =", "===")
        cur_string = cur_string.replace("= =", "==")
        cur_string = cur_string.replace(" " + chr(176) + " ", chr(176))
        cur_string = cur_string.replace(" \n", "\n")
        cur_string = cur_string.replace("\n ", "\n")
        cur_string = cur_string.replace(" N ", " 1 ")
        cur_string = cur_string.replace(" 's", "'s")
        return cur_string

    return Doc(
        task_name=task_name,
        query=wikitext_detokenizer(line["page"]),
        original_query=line["page"],
        choices=None,
        gold_index=None,
    )


wikitext_103_document_level = LightevalTaskConfig(
    name="wikitext:103:document_level",
    prompt_function=wikitext_prompt,
    hf_repo="EleutherAI/wikitext_document_level",
    hf_subset="wikitext-103-raw-v1",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[Metrics.word_perplexity, Metrics.byte_perplexity, Metrics.bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    wikitext_103_document_level,
]
