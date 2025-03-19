# MIT License

# Copyright (c) 2021 Alex Fabbri, Wojciech Kryściński, Bryan
# McCann, Caiming Xiong, Richard Socher, and Dragomir Radev and The HuggingFace
# Team

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

# pylint: disable=C0103,W0221,W0106
# Replace summ_eval.data_stats_metric
import logging
from collections import Counter
from multiprocessing import Pool

from lighteval.metrics.imports.data_stats_utils import Fragments
from lighteval.utils.imports import NO_SPACY_ERROR_MSG, is_spacy_available


logger = logging.getLogger(__name__)


_en = None


class Metric:
    def evaluate_example(self, summary, reference):
        raise NotImplementedError

    def evaluate_batch(self, summaries, references, aggregate=True):
        raise NotImplementedError


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


class DataStatsMetric(Metric):
    def __init__(self, n_gram=3, n_workers=24, case=False, tokenize=True):
        """
        Data Statistics metric
        Makes use of Newsroom code: \
            https://github.com/lil-lab/newsroom/blob/master/newsroom/analyze/fragments.py
        Calculates extractive statistics such as coverage, density, compression as
            defined in Newsroom paper as well as the percentage of novel n-grams in the
            summary vs the input text and the percentage of n-grams in the summary which are
            repeated

        NOTE: these statistics are meant to be calculated with respect to the source text
            (e.g. news article) as opposed to the reference.

        Args:
                :param n_gram: compute statistics for n-grams up to and including this length
                :param n_workers: number of processes to use if using multiprocessing
                :param case: whether to lowercase input before calculating statistics
                :param tokenize: whether to tokenize the input; otherwise assumes that the input
                    is a string of space-separated tokens
        """
        if not is_spacy_available():
            raise ImportError(NO_SPACY_ERROR_MSG)
        import spacy

        self.n_gram = n_gram
        self.n_workers = n_workers
        self.case = case
        self.tokenize = tokenize

        global _en
        try:
            _en = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading the spacy en_core_web_sm model\n(don't worry, this will only happen once)")
            from spacy.cli import download

            download("en_core_web_sm")
            _en = spacy.load("en_core_web_sm")

    def evaluate_example(self, summary, input_text):
        if self.tokenize:
            input_text = _en(input_text, disable=["tagger", "parser", "ner", "textcat"])
            input_text = [tok.text for tok in input_text]
            summary = _en(summary, disable=["tagger", "parser", "ner", "textcat"])
            summary = [tok.text for tok in summary]
        fragments = Fragments(summary, input_text, case=self.case)
        coverage = fragments.coverage()
        density = fragments.density()
        compression = fragments.compression()
        score_dict = {"coverage": coverage, "density": density, "compression": compression}
        tokenized_summary = fragments.summary
        tokenized_text = fragments.text
        score_dict["summary_length"] = len(tokenized_summary)
        for i in range(1, self.n_gram + 1):
            input_ngrams = list(find_ngrams(tokenized_text, i))
            summ_ngrams = list(find_ngrams(tokenized_summary, i))
            input_ngrams_set = set(input_ngrams)
            summ_ngrams_set = set(summ_ngrams)
            intersect = summ_ngrams_set.intersection(input_ngrams_set)
            try:
                score_dict[f"percentage_novel_{i}-gram"] = (len(summ_ngrams_set) - len(intersect)) / float(
                    len(summ_ngrams_set)
                )
                ngramCounter = Counter()
                ngramCounter.update(summ_ngrams)
                repeated = [key for key, val in ngramCounter.items() if val > 1]
                score_dict[f"percentage_repeated_{i}-gram_in_summ"] = len(repeated) / float(len(summ_ngrams_set))
            except ZeroDivisionError:
                continue
        return score_dict

    def evaluate_batch(self, summaries, input_texts, aggregate=True):
        corpus_score_dict = Counter()
        p = Pool(processes=self.n_workers)
        results = p.starmap(self.evaluate_example, zip(summaries, input_texts))
        p.close()
        if aggregate:
            [corpus_score_dict.update(x) for x in results]
            for key in corpus_score_dict.keys():
                corpus_score_dict[key] /= float(len(input_texts))
            return corpus_score_dict
        else:
            return results

    @property
    def supports_multi_ref(self):
        return False
