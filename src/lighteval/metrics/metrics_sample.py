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

"""This module manages all the metrics occurring at the sample level. The results of said metrics are then aggregated
using simple function (min, mean, max, ...) at the corpus level. Most metrics fall under this category.
"""
import os
from typing import Union

import nltk
import numpy as np
from nltk.metrics.distance import edit_distance
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer, scoring
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lighteval.logging.hierarchical_logger import hlog_warn
from lighteval.metrics.imports.bert_scorer import BERTScorer
from lighteval.metrics.imports.data_stats_metric import DataStatsMetric
from lighteval.metrics.imports.summac import SummaCZS
from lighteval.metrics.llm_as_judge import JudgeOpenAI
from lighteval.metrics.normalizations import remove_braces, remove_braces_and_strip
from lighteval.tasks.requests import Doc
from lighteval.utils import as_list


class ExactMatches:
    def __init__(
        self,
        aggregation_function: callable = None,
        normalize_gold: callable = None,
        normalize_pred: callable = None,
        strip_strings: bool = False,
        type_exact_match: str = "full",
    ):
        """An exact match class.

        Args:
            aggregation_function (callable, optional): How to aggregate the item results. Defaults to max.
                Used if there are several golds or predictions on which scores were computed.
            normalize_gold (callable, optional): Function to use to normalize the reference strings.
                Defaults to None if no normalization is applied.
            normalize_pred (callable, optional): Function to use to normalize the predicted strings.
                Defaults to None if no normalization is applied.
            strip_strings (bool, optional): Whether to strip both reference and predictions. Defaults to False.
            type_exact_match (str, optional): Defines what type of match to apply (post normalization if present).
                Can be any of `prefix`, `suffix` or `full`. Defaults to "full".
                `prefix` checks if the prediction starts with the gold,
                `suffix` if the prediction ends with the gold,
                `full` if the prediction and gold are equal
        """
        if aggregation_function is None:
            aggregation_function = max
        self.aggregation_function = aggregation_function
        self.normalize_gold = normalize_gold
        self.normalize_pred = normalize_pred
        self.strip_strings = strip_strings

        if type_exact_match not in ["prefix", "suffix", "full"]:
            # todo: we could add a set exact match
            raise ValueError(
                f"type_exact_match (used in parametrized_exact_match) must be one of prefix, suffix, or full. Was {type_exact_match} instead."
            )
        self.type_exact_match = type_exact_match

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        """Computes the metric over a list of golds and predictions for one single sample.

        Args:
            golds (list[str]): Reference targets
            predictions (list[str]): Predicted strings

        Returns:
            float: Aggregated score over the current sample's items.
        """
        results = []
        # We might need to flatten golds if they are a list of lists
        for gold in golds:
            for pred in predictions:
                results.append(self.compute_one_item(gold=gold, pred=pred))
        return self.aggregation_function(results)

    def compute_one_item(
        self,
        gold: str,
        pred: str,
    ) -> float:
        """Compares two strings only.

        Args:
            gold (str): One of the possible references
            pred (str): One of the possible predictions

        Returns:
            float: The exact match score. Will be 1 for a match, 0 otherwise.
        """
        if not pred:
            return 0

        if self.strip_strings:
            gold = gold.strip()
            pred = pred.strip()

        if self.normalize_gold:
            gold = self.normalize_gold(gold)
        if self.normalize_pred:
            pred = self.normalize_pred(pred)

        if self.type_exact_match == "prefix":
            return 1 if pred.startswith(gold) else 0
        if self.type_exact_match == "suffix":
            return 1 if pred.endswith(gold) else 0
        return 1 if gold == pred else 0


class F1_score:
    def __init__(
        self,
        aggregation_function: callable = None,
        normalize_gold: callable = None,
        normalize_pred: callable = None,
        strip_strings: bool = False,
    ):
        """An F1 score class. F1 is computed over the bag of words of the golds and predictions.

        Args:
            aggregation_function (callable, optional): How to aggregate the item results. Defaults to max.
                Used if there are several golds or predictions on which scores were computed.
            normalize_gold (callable, optional): Function to use to normalize the reference strings.
                Defaults to None if no normalization is applied.
            normalize_pred (callable, optional): Function to use to normalize the predicted strings.
                Defaults to None if no normalization is applied.
            strip_strings (bool, optional): Whether to strip both reference and predictions. Defaults to False.
        """
        if aggregation_function is None:
            aggregation_function = max
        self.aggregation_function = aggregation_function

        self.normalize_gold = normalize_gold
        self.normalize_pred = normalize_pred
        self.strip_strings = strip_strings

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        """Computes the metric over a list of golds and predictions for one single sample.

        Args:
            golds (list[str]): Reference targets
            predictions (list[str]): Predicted strings

        Returns:
            float: Aggregated score over the current sample's items.
        """
        results = []
        # We might need to flatten golds if they are a list of lists
        for gold in golds:
            for pred in predictions:
                results.append(self.compute_one_item(gold=gold, pred=pred))
        return self.aggregation_function(results)

    def compute_one_item(self, gold: str, pred: str) -> float:
        """Compares two strings only.

        Args:
            gold (str): One of the possible references
            pred (str): One of the possible predictions

        Returns:
            float: The f1 score over the bag of words, computed using nltk.
        """
        if self.normalize_gold:
            gold = self.normalize_gold(gold)

        if self.normalize_pred:
            pred = self.normalize_pred(pred)

        gold_bow = set(gold.split())
        pred_bow = set(pred.split())

        ret = nltk.scores.f_measure(gold_bow, pred_bow)

        if ret is None:
            return 0.0
        return ret


class LoglikelihoodAcc:
    def __init__(self, length_normalization: bool = False, ignore_first_space: bool = False) -> None:
        """Log likelihood accuracy class. It tests if the highest log-probability of the possible choices
        is actually in the gold ones.

        Args:
            length_normalization (bool, optional): Whether log-likelihood scores should be normalized for sentence length. Defaults to False.
                Should be True for most cases.
            ignore_first_space (bool, optional): Whether to ignore the first token's log prob (if it's a space only). Defaults to False.
                Only case when it should be True is when the possible choices (for example `A`,`B` ...) have an extra
                space added in front of them to manage tokenization issues (` A`, ` B`, ...) for some models.
        """
        self.length_normalization = length_normalization
        self.ignore_first_space = ignore_first_space

    def compute(self, gold_ixs: list[int], choices_logprob: list[float], formatted_doc: Doc, **kwargs) -> int:
        """Computs the log likelihood accuracy: is the choice with the highest logprob in `choices_logprob` present
        in the `gold_idxs`?

        Args:
            gold_ixs (list[int]): All the gold choices indices
            choices_logprob (list[float]): Summed log-probabilities of all the possible choices for the model, ordered as the choices.
            formatted_doc (Doc): Original document for the sample.
                Used to get the original choices's length for possible normalisation

        Returns:
            int: The eval score: 1 if the best log-prob choice is in gold, 0 otherwise.
        """
        if self.length_normalization:
            normalized_log_probs = []
            for ix, choice in enumerate(formatted_doc.choices):
                if self.ignore_first_space and choice[0] == " ":
                    normalized_log_probs.append(choices_logprob[ix] / (len(choice) - 1))
                else:
                    normalized_log_probs.append(choices_logprob[ix] / len(choice))

            best_choice = np.argmax(normalized_log_probs)
        else:
            best_choice = np.argmax(choices_logprob)
        return int(best_choice in gold_ixs)


class Recall:
    def __init__(self, at: int) -> None:
        """Recall metric class. It checks if the top `at` best choices include one of the golds or not.

        Args:
            at (int): Depth level of the recall.
                Recall at 1 is equivalent to a logprob accuracy without normalization.
        """
        self.recall_depth = at

    def compute(self, choices_logprob: list[float], gold_ixs: list[int], **kwargs) -> int:
        """Computes the recall at the requested depth level: looks at the `n` best predicted choices (with the
        highest log probabilies) and see if there is an actual gold among them.

        Args:
            gold_ixs (list[int]): All the gold choices indices
            choices_logprob (list[float]): Summed log-probabilities of all the possible choices for the model, ordered as the choices.

        Returns:
            int: Score: 1 if one of the top level predicted choices was correct, 0 otherwise.
        """
        if self.recall_depth == 1:
            return int(np.argmax(choices_logprob) in gold_ixs)
        return (int(any(ix in gold_ixs for ix in np.array(choices_logprob).argsort()[::-1][: self.recall_depth])),)


class MRR:
    def __init__(self, length_normalization: bool = False):
        """A mean reciprocal rank class.

        Args:
            length_normalization (bool, optional): Whether to use normalisation be choice length when computing the best log-probabilities. Defaults to False.
        """
        self.length_normalization = length_normalization

    def compute(self, choices_logprob: list[float], gold_ixs: list[float], formatted_doc: Doc, **kwargs) -> float:
        """Mean reciprocal rank. Measures the quality of a ranking of choices (ordered by correctness).

        Args:
            gold_ixs (list[int]): All the gold choices indices
            choices_logprob (list[float]): Summed log-probabilities of all the possible choices for the model, ordered as the choices.
            formatted_doc (Doc): Original document for the sample.
                Used to get the original choices's length for possible normalisation

        Returns:
            float: MRR score.
        """
        if self.length_normalization:
            choices_logprob = [choices_logprob[ix] / len(formatted_doc.choices[ix]) for ix in len(choices_logprob)]
        ranked_choices = [sorted(choices_logprob, reverse=True).index(choices_logprob[gold]) for gold in gold_ixs]
        return 1.0 / (min(ranked_choices) + 1)


def acc_golds_likelihood(target_acc: Union[list[int], int], **kwargs) -> int:
    """Tests if at least one of predicted gold targets' log-likelihood is above 0.5.

    Args:
        target_acc (list[int]): List of scores indicating whether the predictions log-probabilities are above 0.5 aggregated.

    Returns:
        int: 1 if at least one of the possible golds had a log-likelihood above 0.5.
    """
    return max([int(acc_ppl) for acc_ppl in as_list(target_acc)])


class ROUGE:
    ALLOWED_ROUGE_METHODS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    def __init__(
        self,
        methods: str | list[str],
        multiple_golds: bool = False,
        bootstrap: bool = False,
        normalize_gold: callable = None,
        normalize_pred: callable = None,
        aggregation_function: callable = None,
    ):
        """A ROUGE wrapper method. Relies on `rouge_scorer`.

        Args:
            methods (str | list[str]): What type of ROUGE scoring to use. Can be one or any of `rouge1`, `rouge2`, `rougeL` or `rougeLsum`.
            multiple_golds (bool, optional): Whether to compute ROUGE by allowing the comparision to several golds
                at once, or to compute ROUGE on individual gold/prediction pairs and aggregate afterwards. Defaults to False.
            bootstrap (bool, optional): Whether to use bootstrapping. Defaults to False.
            aggregation_function (callable, optional): How to aggregate the item results. Defaults to max.
                Used if there are several golds or predictions on which scores were computed.
            normalize_gold (callable, optional): Function to use to normalize the reference strings.
                Defaults to None if no normalization is applied.
            normalize_pred (callable, optional): Function to use to normalize the predicted strings.
                Defaults to None if no normalization is applied.
        """
        if aggregation_function and bootstrap:
            hlog_warn("Can't use both bootstrapping and an aggregation function in Rouge. Keeping bootstrap.")
        self.aggregation_function = aggregation_function
        if self.aggregation_function is None:
            self.aggregation_function = np.mean

        self.methods = as_list(methods)
        if any(method not in self.ALLOWED_ROUGE_METHODS for method in self.methods):
            raise ValueError(
                f"Rouge was initialised with method {methods}, which is not in {','.join(self.ALLOWED_ROUGE_METHODS)}"
            )
        self.scorer = rouge_scorer.RougeScorer([methods])
        self.multiple_golds = multiple_golds
        self.bootstrap = bootstrap
        self.normalize_gold = normalize_gold
        self.normalize_pred = normalize_pred

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float | dict:
        """Computes the metric(s) over a list of golds and predictions for one single sample.

        Args:
            golds (list[str]): Reference targets
            predictions (list[str]): Predicted strings

        Returns:
            float or dict: Aggregated score over the current sample's items.
                If several rouge functions have been selected, returns a dict which maps name and scores.
        """
        # Normalize
        if self.normalize_gold:
            golds = [self.normalize_gold(g) for g in golds]

        if self.normalize_pred:
            predictions = [self.normalize_pred(p) for p in predictions]

        if self.bootstrap:  # For t5 style rouge score
            scores = self._rouge_score_with_bootsrap(golds=golds, predictions=predictions)
        elif self.multiple_golds:
            scores = self._rouge_score_multi_golds(golds=golds, preds=predictions)
        else:
            scores = self._rouge_score(golds=golds, preds=predictions)

        if len(scores) == 1:
            return list(scores.values())[0]
        return scores

    def _rouge_score(self, golds: list[str], preds: list[str]):
        scores = {m: [] for m in self.methods}
        for pred in preds:
            for gold in golds:
                cur_scores = self.scorer.score(gold, pred)
                for method in self.methods:
                    scores[method].append(cur_scores[method].fmeasure)
        return {method: self.aggregation_function(scores[method]) for method in self.methods}

    def _rouge_score_multi_golds(self, golds: list[str], preds: list[str]):
        scores = {m: [] for m in self.methods}
        for pred in preds:
            cur_scores = self.scorer.score_multi(golds, pred)
            for method in self.methods:
                scores[method].append(cur_scores[method].fmeasure)
        return {method: self.aggregation_function(scores[method]) for method in self.methods}

    def _rouge_score_with_bootsrap(self, golds: list[str], preds: list[str]):
        aggregator = scoring.BootstrapAggregator()
        for g, p in zip(golds, preds):
            aggregator.add_scores(self.scorer.score(g, p))
        result = aggregator.aggregate()
        return {method: result[method].mid.fmeasure * 100 for method in self.methods}


class BertScore:
    def __init__(
        self,
        normalize_gold: callable = None,
        normalize_pred: callable = None,
    ):
        """A BERT scorer class. Relies on some called extracted from `bert-score`. By default, will use the
        `microsoft/deberta-large-mnli` as scorer

        Args:
            normalize_gold (callable, optional): Function to use to normalize the reference strings.
                Defaults to None if no normalization is applied.
            normalize_pred (callable, optional): Function to use to normalize the predicted strings.
                Defaults to None if no normalization is applied.
        """
        self.bert_scorer = BERTScorer(
            model_type="microsoft/deberta-large-mnli", lang="en", rescale_with_baseline=True, num_layers=9
        )

        self.normalize_gold = normalize_gold
        self.normalize_pred = normalize_pred

    def compute(self, golds: list[str], predictions: list[str]) -> dict:
        """Computes the prediction, recall and f1 score using the bert scorer.

        Args:
            golds (list[str]): Reference targets
            predictions (list[str]): Predicted strings

        Returns:
            dict: Scores over the current sample's items.
        """
        golds = as_list(golds)
        predictions = as_list(predictions)
        # Normalize
        if self.normalize_gold:
            golds = [self.normalize_gold(g) for g in golds]

        if self.normalize_pred:
            predictions = [self.normalize_pred(p) for p in predictions]

        p, r, f = self.bert_scorer.score(predictions, golds)
        return {"BERTScore-P": p[0].item(), "BERTScore-R": r[0].item(), "BERTScore-F": f[0].item()}


# todo: make into clean classes with call to normalizer
def extractiveness(formatted_doc: Doc, predictions: list[str], **kwargs):
    inp = remove_braces(formatted_doc.specific["text"])
    pred = remove_braces_and_strip(predictions[0])
    stats = DataStatsMetric().evaluate_example(pred, inp)
    return {
        "summarization_coverage": stats["coverage"],
        "summarization_density": stats["density"],
        "summarization_compression": stats["compression"],
    }


# todo: make into clean classes with call to normalizer
def faithfulness(formatted_doc: Doc, predictions: list[str], **kwargs):
    inp = remove_braces(formatted_doc.specific["text"])
    pred = remove_braces_and_strip(predictions[0])
    summac = SummaCZS(granularity="sentence", model_name="vitc", imager_load_cache=False)  # , device=device)
    return summac.score_one(inp, pred)["score"]


class BLEURT:
    def __init__(self):
        """Creates a BLEURT scorer using a light bleurt-tiny-512 model.
        For more complex use cases, could also be Elron/bleurt-base-128
        """
        self.tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-tiny-512")
        self.model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-tiny-512")
        self.model.eval()

    def compute(self, golds: list[str], predictions: list[str]) -> float:
        """Uses the stored BLEURT scorer to compute the score on the current sample.

        Args:
            golds (list[str]): Reference targets
            predictions (list[str]): Predicted strings

        Returns:
            float: Score over the current sample's items.
        """
        if len(predictions) == 1:
            predictions = predictions * len(golds)
        scores = self.model(**self.tokenizer(golds, predictions, return_tensors="pt"))[0].squeeze()

        return scores


class BLEU:
    def __init__(self, n_gram: int):
        """BLEU scorer class. Relies on `nltk`'s sentencebleu for scoring.
        TODO: Will have to move this to sacrebleu.

        Args:
            n_gram (int): Number of n_grams to use for scoring.
        """
        self.n_gram = n_gram

    def compute(self, golds: list[str], predictions: list[str], **kwargs):
        """Computes the sentence level BLEU between the golds and each prediction, then takes the average.

        Args:
            golds (list[str]): Reference targets
            predictions (list[str]): Predicted strings

        Returns:
            float: Score over the current sample's items.
        """
        return np.mean([self._bleu_score(golds, p) for p in predictions])

    def _bleu_score(self, gold: list[str], pred: str) -> float:
        """Computes the BLEU score between a list of golds and the current prediction.

        Args:
            golds (list[str]): Reference targets
            predictions (str): One of the predicted strings

        Returns:
            float: Score over the current prediction.
        """
        weights = [1 if ix == self.n_gram else 0 for ix in range(1, 5)]
        return sentence_bleu([word_tokenize(g) for g in gold], word_tokenize(pred), weights=weights)


class StringDistance:
    def __init__(
        self,
        metric_types: list[str] | str,
        strip_prediction: bool = True,
    ):
        """Contains a number of string distance and edition metrics. Relies on nltk to compute the edit distance.

        Args:
            metric_types (list[str] | str): Can be one or any of `longest_common_prefix_length`, `edit_distance` or `edit_similarity`.
            strip_prediction (bool, optional): Whether to strip the prediction. Defaults to True.
        """
        allowed_values = ["longest_common_prefix_length", "edit_distance", "edit_similarity"]
        metric_types = as_list(metric_types)
        if any(metric_type not in allowed_values for metric_type in metric_types):
            raise ValueError(
                f"{metric_types} is not a valid value for an EditDistance metric. Possible values are {','.join(allowed_values)}."
            )
        self.metric_types = metric_types
        self.strip_prediction = strip_prediction
        self.sample_aggregations = {"longest_common_prefix_length": max, "edit_distance": min, "edit_similarity": max}

    def compute(self, gold: list[str], predictions: list[str], **kwargs) -> dict:
        """Computes all the requested metrics on the golds and prediction.

        Args:
            gold (list[str]): A list of possible golds. If it contains more than one item, only the first one is kept.
            predictions (list[str]): Predicted strings.

        Returns:
           dict: The different scores computed
        """
        if len(gold) > 0:
            hlog_warn("Provided more than one gold to compute a string distance metric. Just using the first one.")
        reference = gold[0]

        result = {m: [] for m in self.metric_types}
        for sequence in predictions:
            if self.strip_prediction:
                completion = sequence.strip()

            # `reference` is the entire remaining book for each instance.
            # Truncate it here to be of the same length as the completion to ensure edit-distance is meaningful.
            truncated_reference = reference[: len(completion)]

            completion_tokens = np.array(TreebankWordTokenizer().tokenize(completion))
            truncated_reference_tokens = np.array(TreebankWordTokenizer().tokenize(truncated_reference))

            if "edit_distance" in self.metric_types:
                result["edit_distance"].append(edit_distance(s1=completion_tokens, s2=truncated_reference_tokens))
            if "edit_similarity" in self.metric_types:
                result["edit_similarity"].append(
                    self.edit_similarity(s1=completion_tokens, s2=truncated_reference_tokens)
                )
            if "longest_common_prefix_length" in self.metric_types:
                result["longest_common_prefix_length"].append(
                    self.longest_common_prefix_length(s1=completion_tokens, s2=truncated_reference_tokens)
                )

        final_result = {}
        # We cast to float as final results can be numpy types, not JSON serializable
        for m, v in result.items():
            final_result[m] = float(self.sample_aggregations[m](v))

        return final_result

    def longest_common_prefix_length(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """Compute the length of the longest common prefix."""
        min_len = min(len(s1), len(s2))
        s1, s2 = s1[:min_len], s2[:min_len]
        (nonzeros,) = np.cumprod(s1 == s2).nonzero()
        return int(np.max(nonzeros)) + 1 if len(nonzeros) > 0 else 0

    def edit_similarity(self, s1, s2):
        """Compute the edit similarity between two lists of strings.

        Edit similarity is also used in the paper
            Lee, Katherine, et al.
            "Deduplicating training data makes language models better."
            arXiv preprint arXiv:2107.06499 (2021).
        """
        edist = edit_distance(s1, s2)
        return 1.0 - edist / max(len(s1), len(s2)) if len(s1) > 0 and len(s2) > 0 else 0


class JudgeLLM:
    available_models = ["gpt-3.5-turbo"]

    def __init__(self, judge_model_name: str, template_path: str, multi_turn: bool = False):
        if judge_model_name not in self.available_models:
            raise ValueError(f"{judge_model_name} not in available models for llm as a judge metric")

        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.multi_turn = multi_turn

        try:
            self.judge = JudgeOpenAI(
                model=judge_model_name,
                seed=42,
                temperature=0.0,
                templates_path=template_path,
                openai_api_key=OPENAI_API_KEY,
                multi_turn=multi_turn,
            )
        except Exception as e:
            print(f"Could not initialize the JudgeOpenAI model:\n{e}")
            self.judge = None

    def compute(self, predictions: list[str], formatted_doc: Doc, **kwargs) -> dict[str, float]:
        """
        Compute the score of a generative taks using a llm as a judge.
        The generative task can be multiturn with 2 turns max, in that case, we
        return scores for turn 1 and 2. Also returns user_prompt and judgment
        which are ignored later by the aggregator.
        """

        # If we are evaluating a multiturn task, we need to have specific field in the formated doc
        if self.multi_turn:
            questions = formatted_doc.specific["multi_turn_queries"]
            ref_answers = formatted_doc.specific.get("reference", None) if formatted_doc.specific is not None else None
        else:
            questions = [formatted_doc.query]
            ref_answers = [formatted_doc.choices[formatted_doc.gold_index]]

        scores, messages, judgements = self.judge.evaluate_answer(questions, predictions, ref_answers)

        # Multi turn only has 2 turns
        if self.multi_turn:
            return {
                "single_turn": scores[0],
                "multi_turn": scores[1],
                "user_prompt": [messages[0], messages[1]],
                "judgement": [judgements[0], judgements[1]],
            }

        return {
            "judge_score": scores[0],
            "user_prompt": messages[0],
            "judgement": judgements[0],
        }
