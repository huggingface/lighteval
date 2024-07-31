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

import re
import string
from typing import List, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def drop_metrics(predictions: list[str], formatted_doc, **kwargs):  # noqa: C901
    """F1 score from bag of words: comes from Harness Drop. DROP offers two metrics,
    a quasi exact match and a numeracy-focused F1 score. Quasi in the sense that it
    does some normalizations before matching and numeracy-focused in the sense that
    if there's number mismatch between the target and prediction F1 score is set to 0.
    F1 score is computed using the intersection of target and prediction's BoW
    representations with the additional spice that if the answer and/or prediction is
    comprised of multiple spans, a greedy matching is done between the two sets of spans
    (based on the very BoW overlap) and the average over F1 of pairs is returned.
    DROP also accepts multiple answers in which case, the maximum of F1/ Exact Match
    between prediction and the different answers is taken.

    For more information, please refer to the section 5 of the DROP paper (https://aclanthology.org/N19-1246/).

    Todo: this code is really hard to follow, simplify when possible
    """

    def _answer_to_bags(answer: List[str]) -> Tuple[List[str], List[Set[str]]]:
        if isinstance(answer, (list, tuple)):
            raw_spans = answer
        else:
            raw_spans = [answer]
        normalized_spans = []
        token_bags = []
        for raw_span in raw_spans:
            normalized_span = _normalize(raw_span)
            normalized_spans.append(normalized_span)
            token_bags.append(set(normalized_span.split()))
        return normalized_spans, token_bags

    def _get_metrics(predicted: List[str], gold: List[str]):
        """
        Takes a predicted answer and a gold answer (that are both either a string or a list of
        strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
        writing a script for evaluating objects in memory (say, the output of predictions during
        validation, or while training), this is the function you want to call, after using
        :func:`answer_json_to_strings` when reading the gold answer from the released data file.
        """
        pred_normalized_spans, pred_bags = _answer_to_bags(predicted)
        gold_normalized_spans, gold_bags = _answer_to_bags(gold)

        if set(pred_normalized_spans) == set(gold_normalized_spans) and len(gold_normalized_spans) == len(
            gold_normalized_spans
        ):
            exact_match = 1.0
        else:
            exact_match = 0.0

        f1_per_bag = _align_bags(pred_bags, gold_bags)
        f1 = np.mean(f1_per_bag)
        f1 = round(f1, 2)
        return exact_match, f1

    def _is_number(text):
        try:
            float(text)
            return True
        except ValueError:
            return False

    def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]):
        gold_numbers = set()
        predicted_numbers = set()
        for word in gold_bag:
            if _is_number(word):
                gold_numbers.add(word)
        for word in predicted_bag:
            if _is_number(word):
                predicted_numbers.add(word)
        if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
            return True
        return False

    def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> np.array:
        """
        Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
        between them and gets maximum metric values over all the answers.
        """
        scores = np.zeros([len(gold), len(predicted)])
        for gold_index, gold_item in enumerate(gold):
            for pred_index, pred_item in enumerate(predicted):
                if _match_numbers_if_present(gold_item, pred_item):
                    scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
        row_ind, col_ind = linear_sum_assignment(-scores)

        max_scores = np.zeros([max(len(gold), len(predicted))])
        for row, column in zip(row_ind, col_ind):
            max_scores[row] = max(max_scores[row], scores[row, column])
        return max_scores

    def _compute_f1(predicted_bag, gold_bag):
        intersection = len(gold_bag.intersection(predicted_bag))
        if not predicted_bag:
            precision = 1.0
        else:
            precision = intersection / float(len(predicted_bag))
        if not gold_bag:
            recall = 1.0
        else:
            recall = intersection / float(len(gold_bag))
        if precision == 0.0 and recall == 0.0:
            return 0
        return (2 * precision * recall) / (precision + recall)

    def _remove_articles(text):
        return re.compile(r"\b(a|an|the)\b", re.UNICODE).sub(" ", text)

    def _white_space_fix(text):
        return " ".join(text.split())

    def _remove_punc(text):
        exclude = set(string.punctuation)
        if not _is_number(text):
            return "".join(ch for ch in text if ch not in exclude)
        else:
            return text

    def _fix_number(text):
        return str(float(text)) if _is_number(text) else text

    def _tokenize(text):
        return re.split(" |-", text)

    def _normalize(answer: str):
        tokens = [
            _white_space_fix(_remove_articles(_fix_number(_remove_punc(token.lower())))) for token in _tokenize(answer)
        ]
        tokens = [token for token in tokens if token.strip()]
        normalized = " ".join(tokens).strip()
        return normalized

    max_em = 0
    max_f1 = 0
    for gold_answer in formatted_doc.specific["golds_no_preprocessing"]:
        exact_match, f1_score = _get_metrics(predictions, gold_answer)
        if isinstance(gold_answer, list):
            gold_answer = gold_answer[0]
        if gold_answer.strip():
            max_em = max(max_em, exact_match)
            max_f1 = max(max_f1, f1_score)
    return {"qem": max_em, "f1": max_f1}
