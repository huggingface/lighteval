import re
import string
import sys
import unicodedata
from typing import Literal

import numpy as np

from ..utils.translation_literals import LANGS
from lighteval.metrics.metrics_sample import ExactMatches, F1_score
from lighteval.metrics.utils import MetricCategory, MetricUseCase, SampleLevelMetric


PUNCT = {chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")}.union(
    string.punctuation
)
WHITESPACE_LANGS = ["en", "es", "hi", "vi", "de", "ar"]
MIXED_SEGMENTATION_LANGS = ["zh"]
EVAL_TYPE = Literal["exact", "f1"]


# MLQA normalizer
# TODO: support rest
# supports: en, zh, ar, hi
# https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py
def get_answer_normalizer(lang: LANGS):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def whitespace_tokenize(text):
        return text.split()

    def mixed_segmentation(text):
        segs_out = []
        temp_str = ""
        for char in text:
            if re.search(r"[\u4e00-\u9fa5]", char) or char in PUNCT:
                if temp_str != "":
                    ss = whitespace_tokenize(temp_str)
                    segs_out.extend(ss)
                    temp_str = ""
                segs_out.append(char)
            else:
                temp_str += char

        if temp_str != "":
            ss = whitespace_tokenize(temp_str)
            segs_out.extend(ss)

        return segs_out

    def remove_articles(text):
        if lang == "en":
            return re.sub(r"\b(a|an|the)\b", " ", text)
        elif lang == "hi" or lang == "sw" or lang == "te" or lang == "zh" or lang == "ru":
            return text  # Hindi does not have formal articles
        elif lang == "ar":
            return re.sub(r"\sال^|ال", " ", text)
        elif lang == "fr":
            return re.sub(r"\b(l|le|la)\b", " ", text)
        elif lang == "tr":
            return re.sub(r"\b(bir)\b", " ", text)
        elif lang == "th":
            return re.sub(r"(คน|ตัว|ลูก)", " ", text)
        else:
            # TODO: raise Exception("Unknown Language {}".format(lang))
            return text

    def white_space_fix(text):
        if lang in WHITESPACE_LANGS:
            tokens = whitespace_tokenize(text)
        elif lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
            
        elif lang == "th":
            from pythainlp.tokenize import word_tokenize
            tokens = word_tokenize(text, keep_whitespace=False, engine="newmm")
        else:
            # TODO: raise Exception("Unknown Language {}".format(lang))
            # assume whitespace
            tokens = whitespace_tokenize(text)
        return " ".join([t for t in tokens if t.strip() != ""])

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in PUNCT)

    def lower(text):
        return text.lower()

    return lambda s: white_space_fix(remove_articles(remove_punc(lower(s))))


def get_qa_scorer(lang: LANGS, evalType: EVAL_TYPE):
    if evalType == "exact":
        return ExactMatches(
            normalize_gold=get_answer_normalizer(lang),
            normalize_pred=get_answer_normalizer(lang),
            type_exact_match="prefix"
        ).compute
    elif evalType == "f1":
        return F1_score(
            normalize_gold=get_answer_normalizer(lang),
            normalize_pred=get_answer_normalizer(lang),
        ).compute

    raise ValueError(f"Unknown eval type {evalType}")


def get_qa_metric(lang: LANGS, evalType: EVAL_TYPE):
    return SampleLevelMetric(
        metric=f"qa_{lang}_{evalType}",
        sample_level_fn=get_qa_scorer(lang, evalType),
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )