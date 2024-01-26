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
from lighteval.metrics.normalizations import remove_braces, remove_braces_and_strip
from lighteval.tasks.requests import Doc
from lighteval.utils import as_list


# Parametrized metrics are defined as classes
class ExactMatches:
    def __init__(
        self,
        aggregation_function: callable = None,
        normalize_gold: callable = None,
        normalize_pred: callable = None,
        strip_strings: bool = False,
        type_exact_match: str = "full",
    ):
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
        type_f1: str = "",
    ):
        if aggregation_function is None:
            aggregation_function = max
        self.aggregation_function = aggregation_function

        self.normalize_gold = normalize_gold
        self.normalize_pred = normalize_pred
        self.strip_strings = strip_strings
        self.type_f1 = type_f1

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        results = []
        # We might need to flatten golds if they are a list of lists
        for gold in golds:
            for pred in predictions:
                results.append(self.compute_one_item(gold=gold, pred=pred))
        return self.aggregation_function(results)

    def compute_one_item(self, gold: str, pred: str) -> float:
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
        self.length_normalization = length_normalization
        self.ignore_first_space = ignore_first_space

    def compute(self, gold_ixs: list[int], choices_logprob: list[list[float]], formatted_doc: Doc, **kwargs):
        if self.length_normalization:
            normalized_log_probs = []
            for ix, choice in enumerate(formatted_doc.choices):
                if self.ignore_first_space:
                    normalized_log_probs.append(
                        choices_logprob[ix] / (len(choice) - 1 if choice[0] == " " else len(choice))
                    )
                else:
                    normalized_log_probs.append(choices_logprob[ix] / len(choice))

            best_choice = np.argmax(normalized_log_probs)
        else:
            best_choice = np.argmax(choices_logprob)
        return int(best_choice in gold_ixs)


class Recall:
    def __init__(self, at: int) -> None:
        self.recall_depth = at

    def compute(self, choices_logprob, gold_ixs, **kwargs):
        if self.at == 1:
            return int(np.argmax(choices_logprob) in gold_ixs)
        return (int(any(ix in gold_ixs for ix in np.array(choices_logprob).argsort()[::-1][: self.recall_depth])),)


def mrr(choices_logprob: list[float], gold_ixs: list[float], **kwargs):
    ranked_choices = [sorted(choices_logprob, reverse=True).index(choices_logprob[gold]) for gold in gold_ixs]
    return 1.0 / (min(ranked_choices) + 1)


def acc_golds_likelihood(results: list[int], formatted_doc: Doc, **kwargs):
    results = results[: len(formatted_doc.get_golds())]  # todo: check, might not be needed
    return max([int(acc_ppl) for _, acc_ppl in results])


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
        if aggregation_function and bootstrap:
            hlog_warn("Can't use both bootstrapping and an aggreagation function in Rouge. Keeping bootstrap.")
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

    def compute(self, golds: list[str], predictions: list[str], **kwargs):
        # Normalize
        if self.normalize_gold:
            golds = [self.normalize_gold(g) for g in golds]

        if self.normalize_pred:
            predictions = [self.normalize_pred(p) for p in predictions]

        if self.bootstrap:  # For t5 style rouge score
            scores = self.rouge_score_with_bootsrap(golds=golds, predictions=predictions)
        elif self.multiple_golds:
            scores = self.rouge_score_multi_golds(golds=golds, preds=predictions)
        else:
            scores = self.rouge_score(golds=golds, preds=predictions)

        if len(scores) == 1:
            return list(scores.values())[0]
        return scores

    def rouge_score(self, golds: list[str], preds: list[str]):
        scores = {m: [] for m in self.methods}
        for pred in preds:
            for gold in golds:
                cur_scores = self.scorer.score(gold, pred)
                for method in self.methods:
                    scores[method].append(cur_scores[method].fmeasure)
        return {method: self.aggregation_function(scores[method]) for method in self.methods}

    def rouge_score_multi_golds(self, golds: list[str], preds: list[str]):
        scores = {m: [] for m in self.methods}
        for pred in preds:
            cur_scores = self.scorer.score_multi(golds, pred)
            for method in self.methods:
                scores[method].append(cur_scores[method].fmeasure)
        return {method: self.aggregation_function(scores[method]) for method in self.methods}

    def rouge_score_with_bootsrap(self, golds: list[str], preds: list[str]):
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
        self.bert_scorer = BERTScorer(
            model_type="microsoft/deberta-large-mnli", lang="en", rescale_with_baseline=True, num_layers=9
        )

        self.normalize_gold = normalize_gold
        self.normalize_pred = normalize_pred

    def compute(self, golds: list[str], predictions: list[str]):
        golds = as_list(golds)
        predictions = as_list(predictions)
        # Normalize
        if self.normalize_gold:
            golds = [self.normalize_gold(g) for g in golds]

        if self.normalize_pred:
            predictions = [self.normalize_pred(p) for p in predictions]

        p, r, f = self.bert_scorer.score(predictions, golds)
        return {"BERTScore-P": p[0].item(), "BERTScore-R": r[0].item(), "BERTScore-F": f[0].item()}


def extractiveness(formatted_doc: Doc, predictions: list[str], **kwargs):
    inp = remove_braces(formatted_doc.specific["text"])
    pred = remove_braces_and_strip(predictions[0])
    stats = DataStatsMetric().evaluate_example(pred, inp)
    return {
        "summarization_coverage": stats["coverage"],
        "summarization_density": stats["density"],
        "summarization_compression": stats["compression"],
    }


def faithfulness(formatted_doc: Doc, predictions: list[str], **kwargs):
    inp = remove_braces(formatted_doc.specific["text"])
    pred = remove_braces_and_strip(predictions[0])
    summac = SummaCZS(granularity="sentence", model_name="vitc", imager_load_cache=False)  # , device=device)
    return summac.score_one(inp, pred)["score"]


class BLEURT:
    # Model chosen could also be Elron/bleurt-base-128
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-tiny-512")
        self.model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-tiny-512")
        self.model.eval()

    def compute(self, golds: list[str], predictions: list[str]) -> float:
        if len(predictions) == 1:
            predictions = predictions * len(golds)
        scores = self.model(**self.tokenizer(golds, predictions, return_tensors="pt"))[0].squeeze()

        return scores


class BLEU:
    def __init__(self, n_gram: int):
        self.n_gram = n_gram

    def compute(self, golds: list[str], predictions: list[str], **kwargs):
        return np.mean([self.bleu_score(golds, p) for p in predictions])

    def bleu_score(self, gold: list[str], pred: str):
        weights = [1 if ix == self.n_gram else 0 for ix in range(1, 5)]
        return sentence_bleu([word_tokenize(g) for g in gold], word_tokenize(pred), weights=weights)


class StringDistance:
    def __init__(
        self,
        metric_types: list[str] | str,
        strip_prediction: bool = True,
    ):
        allowed_values = ["longest_common_prefix_length", "edit_distance", "edit_similarity"]
        metric_types = as_list(metric_types)
        if any(metric_type not in allowed_values for metric_type in metric_types):
            raise ValueError(
                f"{metric_types} is not a valid value for an EditDistance metric. Possible values are {','.join(allowed_values)}."
            )
        self.metric_types = metric_types
        self.strip_prediction = strip_prediction
        self.sample_aggregations = {"longest_common_prefix_length": max, "edit_distance": min, "edit_similarity": max}

    def compute(self, gold: list[str], predictions: list[str], **kwargs):
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
        (nonzeros,) = np.cumprod(s1 == s2).nonzero()  # Get indices (inclusive) up to which s1 and s2 are the same.
        return int(np.max(nonzeros)) + 1 if len(nonzeros) > 0 else 0

    def edit_similarity(self, s1, s2):
        """Compute the edit similarity between two lists of strings.

        Edit similarity is also used in the paper
            Lee, Katherine, et al.
            "Deduplicating training data makes language models better."
            arXiv preprint arXiv:2107.06499 (2021).
        """
        edist = edit_distance(s1, s2)

        # Some models can return an empty completion e.g., openai/text-davinci-002
        # returns '<|endoftext|>' token immediately for a certain request.
        return 1.0 - edist / max(len(s1), len(s2)) if len(s1) > 0 and len(s2) > 0 else 0
