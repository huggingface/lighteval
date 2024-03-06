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

import numpy as np
from aenum import Enum

from lighteval.metrics.harness_compatibility.drop import drop_metrics
from lighteval.metrics.harness_compatibility.truthful_qa import truthfulqa_mc_metrics
from lighteval.metrics.metrics_corpus import (
    CorpusLevelF1Score,
    CorpusLevelPerplexityMetric,
    CorpusLevelTranslationMetric,
    matthews_corrcoef,
)
from lighteval.metrics.metrics_sample import (
    BLEU,
    BLEURT,
    MRR,
    ROUGE,
    BertScore,
    ExactMatches,
    F1_score,
    LoglikelihoodAcc,
    Recall,
    StringDistance,
    acc_golds_likelihood,
    extractiveness,
    faithfulness,
)
from lighteval.metrics.normalizations import (
    bigbench_normalizer,
    gsm8k_normalizer,
    harness_triviaqa_normalizer,
    helm_normalizer,
    math_normalizer,
    math_normalizer_gold,
    remove_braces,
    remove_braces_and_strip,
)
from lighteval.metrics.sample_preparator import GenerativePreparator, LoglikelihoodPreparator, PerplexityPreparator
from lighteval.metrics.utils import (
    CorpusLevelMetric,
    CorpusLevelMetricGrouping,
    MetricCategory,
    MetricGrouping,
    MetricUseCase,
    SampleLevelMetric,
    SampleLevelMetricGrouping,
)
from lighteval.utils import as_list


class Metrics(Enum):
    acc_golds_likelihood = SampleLevelMetric(  # todo: we need a better name for this!
        metric="acc",
        sample_level_fn=acc_golds_likelihood,
        category=MetricCategory.TARGET_PERPLEXITY,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    bert_score = SampleLevelMetricGrouping(
        metric=["BERTScore-P", "BERTScore-R", "BERTScore-F"],
        sample_level_fn=BertScore(normalize_gold=remove_braces, normalize_pred=remove_braces_and_strip).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.SUMMARIZATION,
        corpus_level_fn={"BERTScore-P": np.mean, "BERTScore-R": np.mean, "BERTScore-F": np.mean},
        higher_is_better={"BERTScore-P": True, "BERTScore-R": True, "BERTScore-F": True},
    )
    bits_per_byte = CorpusLevelMetric(
        metric="bits_per_byte",
        sample_level_fn=PerplexityPreparator(units_type="bytes").prepare,
        category=MetricCategory.PERPLEXITY,
        use_case=MetricUseCase.PERPLEXITY,
        corpus_level_fn=CorpusLevelPerplexityMetric("bits_per_byte").compute,
        higher_is_better=False,
    )
    bleu = CorpusLevelMetric(
        metric="bleu",
        sample_level_fn=GenerativePreparator().prepare,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.TRANSLATION,
        corpus_level_fn=CorpusLevelTranslationMetric("bleu").compute,
        higher_is_better=True,
    )
    bleu_1 = SampleLevelMetric(
        metric="bleu_1",
        sample_level_fn=BLEU(n_gram=1).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.TRANSLATION,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    bleu_4 = SampleLevelMetric(
        metric="bleu_4",
        sample_level_fn=BLEU(n_gram=4).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.TRANSLATION,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    bleurt = SampleLevelMetric(
        metric="bleurt",
        sample_level_fn=BLEURT.compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.TRANSLATION,
        corpus_level_fn=lambda x: np.mean(x.flatten()),  # flatten, then average
        higher_is_better=True,
    )
    byte_perplexity = CorpusLevelMetric(
        metric="byte_perplexity",
        sample_level_fn=PerplexityPreparator(units_type="bytes").prepare,
        category=MetricCategory.PERPLEXITY,
        use_case=MetricUseCase.PERPLEXITY,
        corpus_level_fn=CorpusLevelPerplexityMetric("weighted_perplexity").compute,
        higher_is_better=False,
    )
    chrf = CorpusLevelMetric(
        metric="chrf",
        sample_level_fn=GenerativePreparator().prepare,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.TRANSLATION,
        corpus_level_fn=CorpusLevelTranslationMetric("chrf").compute,
        higher_is_better=True,
    )
    copyright = SampleLevelMetricGrouping(
        metric=["longest_common_prefix_length", "edit_distance", "edit_similarity"],
        sample_level_fn=StringDistance(
            metric_types=["longest_common_prefix_length", "edit_distance", "edit_similarity"], strip_prediction=True
        ).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.SOCIAL_IMPACTS,
        corpus_level_fn={"longest_common_prefix_length": max, "edit_distance": min, "edit_similarity": max},
        higher_is_better={"longest_common_prefix_length": True, "edit_distance": False, "edit_similarity": True},
    )
    drop = SampleLevelMetricGrouping(
        metric=["qem", "f1"],
        sample_level_fn=drop_metrics,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn={"qem": max, "f1": max},
        higher_is_better={"qem": True, "f1": True},
    )
    exact_match = SampleLevelMetric(
        metric="em",
        sample_level_fn=ExactMatches(strip_strings=True).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    extractiveness = SampleLevelMetricGrouping(
        metric=["summarization_coverage", "summarization_density", "summarization_compression"],
        sample_level_fn=extractiveness,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.SUMMARIZATION,
        corpus_level_fn={
            "summarization_coverage": np.mean,
            "summarization_density": np.mean,
            "summarization_compression": np.mean,
        },
        higher_is_better={
            "summarization_coverage": True,
            "summarization_density": True,
            "summarization_compression": True,
        },
    )
    f1_score_quasi = SampleLevelMetric(
        metric="f1_score_quasi",
        sample_level_fn=F1_score(normalize_gold=helm_normalizer, normalize_pred=helm_normalizer).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    f1_score = SampleLevelMetric(
        metric="f1",
        sample_level_fn=F1_score().compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    f1_score_macro = CorpusLevelMetric(
        metric="f1",
        sample_level_fn=GenerativePreparator().prepare,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=CorpusLevelF1Score(average="macro").compute,
        higher_is_better=True,
    )
    f1_score_micro = CorpusLevelMetric(
        metric="f1",
        sample_level_fn=GenerativePreparator().prepare,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=CorpusLevelF1Score(average="micro").compute,
        higher_is_better=True,
    )
    faithfulness = SampleLevelMetric(
        metric="summac",
        sample_level_fn=faithfulness,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.SUMMARIZATION,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    loglikelihood_acc = SampleLevelMetric(
        metric="acc",
        sample_level_fn=LoglikelihoodAcc().compute,
        category=MetricCategory.MULTICHOICE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    loglikelihood_acc_norm = SampleLevelMetric(
        metric="acc_norm",
        sample_level_fn=LoglikelihoodAcc(length_normalization=True).compute,
        category=MetricCategory.MULTICHOICE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    loglikelihood_acc_norm_nospace = SampleLevelMetric(
        metric="acc_norm",
        sample_level_fn=LoglikelihoodAcc(length_normalization=True, ignore_first_space=True).compute,
        category=MetricCategory.MULTICHOICE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    loglikelihood_acc_norm_single_token = SampleLevelMetric(
        metric="acc_norm",
        sample_level_fn=LoglikelihoodAcc(length_normalization=True).compute,
        category=MetricCategory.MULTICHOICE_ONE_TOKEN,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    loglikelihood_acc_single_token = SampleLevelMetric(
        metric="acc",
        sample_level_fn=LoglikelihoodAcc().compute,
        category=MetricCategory.MULTICHOICE_ONE_TOKEN,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    loglikelihood_f1 = CorpusLevelMetric(
        metric="loglikelihood_f1",
        sample_level_fn=LoglikelihoodPreparator().prepare,
        category=MetricCategory.MULTICHOICE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=CorpusLevelF1Score(None),
        higher_is_better=True,
    )
    loglikelihood_f1_single_token = CorpusLevelMetric(
        metric="loglikelihood_f1",
        sample_level_fn=LoglikelihoodPreparator(is_single_token=True).prepare,
        category=MetricCategory.MULTICHOICE_ONE_TOKEN,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=CorpusLevelF1Score(None),
        higher_is_better=True,
    )
    mcc = CorpusLevelMetric(
        metric="mcc",
        sample_level_fn=LoglikelihoodPreparator().prepare,
        category=MetricCategory.MULTICHOICE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=matthews_corrcoef,
        higher_is_better=True,
    )
    mcc_single_token = CorpusLevelMetric(
        metric="mcc",
        sample_level_fn=LoglikelihoodPreparator().prepare,
        category=MetricCategory.MULTICHOICE_ONE_TOKEN,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=matthews_corrcoef,
        higher_is_better=True,
    )
    mrr = SampleLevelMetric(
        metric="mrr",
        sample_level_fn=MRR().compute,
        category=MetricCategory.MULTICHOICE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    mrr_single_token = SampleLevelMetric(
        metric="mrr",
        sample_level_fn=mrr,
        category=MetricCategory.MULTICHOICE_ONE_TOKEN,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    multi_f1_numeric = CorpusLevelMetric(
        metric="mf1",
        sample_level_fn=LoglikelihoodPreparator(is_single_token=True).prepare,
        category=MetricCategory.MULTICHOICE_ONE_TOKEN,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=CorpusLevelF1Score(average=None, num_classes=3),
        higher_is_better=True,
    )
    perfect_exact_match = SampleLevelMetric(
        metric="perfect_em",
        sample_level_fn=ExactMatches().compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    prediction_perplexity = SampleLevelMetric(
        metric="ppl",
        sample_level_fn=None,  # todo!!!
        category=MetricCategory.IGNORED,
        use_case=MetricUseCase.PERPLEXITY,
        corpus_level_fn=CorpusLevelPerplexityMetric("perplexity").compute,
        higher_is_better=True,
    )
    prefix_exact_match = SampleLevelMetric(
        metric="pem",
        sample_level_fn=ExactMatches(strip_strings=True, type_exact_match="prefix").compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    prefix_quasi_exact_match = SampleLevelMetric(
        metric="pqem",
        sample_level_fn=ExactMatches(
            normalize_gold=helm_normalizer,
            normalize_pred=helm_normalizer,
            type_exact_match="prefix",
        ).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    quasi_exact_match = SampleLevelMetric(
        metric="qem",
        sample_level_fn=ExactMatches(
            normalize_gold=helm_normalizer,
            normalize_pred=helm_normalizer,
            strip_strings=True,
        ).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    quasi_exact_match_math = SampleLevelMetric(
        metric="qem",
        sample_level_fn=ExactMatches(
            strip_strings=True, normalize_pred=math_normalizer, normalize_gold=math_normalizer_gold
        ).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.MATH,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    quasi_exact_match_triviaqa = SampleLevelMetric(
        metric="qem",
        sample_level_fn=ExactMatches(strip_strings=True, normalize_pred=harness_triviaqa_normalizer).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    quasi_exact_match_gsm8k = SampleLevelMetric(
        metric="qem",
        sample_level_fn=ExactMatches(
            strip_strings=True, normalize_pred=gsm8k_normalizer, normalize_gold=gsm8k_normalizer
        ).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.MATH,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    recall_at_1_single_token = SampleLevelMetric(
        metric="acc",
        sample_level_fn=Recall(at=1).compute,
        category=MetricCategory.MULTICHOICE_ONE_TOKEN,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    recall_at_2_single_token = SampleLevelMetric(
        metric="recall@2",
        sample_level_fn=Recall(at=2).compute,
        category=MetricCategory.MULTICHOICE_ONE_TOKEN,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    recall_at_1 = SampleLevelMetric(
        metric="acc",
        sample_level_fn=Recall(at=1),
        category=MetricCategory.MULTICHOICE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    recall_at_2 = SampleLevelMetric(
        metric="recall@2",
        sample_level_fn=Recall(at=2),
        category=MetricCategory.MULTICHOICE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    rouge_t5 = CorpusLevelMetricGrouping(
        metric=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        sample_level_fn=ROUGE(
            ["rouge1", "rouge2", "rougeL", "rougeLsum"],
            bootstrap=True,
            normalize_gold=bigbench_normalizer,
            normalize_pred=bigbench_normalizer,
        ).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn={"rouge1": np.mean, "rouge2": np.mean, "rougeL": np.mean, "rougeLsum": np.mean},
        higher_is_better={"rouge1": True, "rouge2": True, "rougeL": True, "rougeLsum": True},
    )
    rouge1 = SampleLevelMetric(
        metric="rouge1",
        sample_level_fn=ROUGE("rouge1").compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.SUMMARIZATION,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    rouge2 = SampleLevelMetric(
        metric="rouge2",
        sample_level_fn=ROUGE("rouge2").compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.SUMMARIZATION,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    rougeL = SampleLevelMetric(
        metric="rougeL",
        sample_level_fn=ROUGE("rougeL").compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.SUMMARIZATION,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    rougeLsum = SampleLevelMetric(
        metric="rougeLsum",
        sample_level_fn=ROUGE("rougeLsum").compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.SUMMARIZATION,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    target_perplexity = SampleLevelMetric(
        metric="ppl",
        sample_level_fn=PerplexityPreparator(units_type="words").prepare,
        category=MetricCategory.TARGET_PERPLEXITY,
        use_case=MetricUseCase.PERPLEXITY,
        corpus_level_fn=CorpusLevelPerplexityMetric("perplexity").compute,
        higher_is_better=False,
    )
    ter = CorpusLevelMetric(
        metric="ter",
        sample_level_fn=GenerativePreparator().prepare,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.TRANSLATION,
        corpus_level_fn=CorpusLevelTranslationMetric("ter").compute,
        higher_is_better=False,
    )
    truthfulqa_mc_metrics = SampleLevelMetricGrouping(
        metric=["truthfulqa_mc1", "truthfulqa_mc2"],
        sample_level_fn=truthfulqa_mc_metrics,
        category=MetricCategory.MULTICHOICE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn={"truthfulqa_mc1": np.mean, "truthfulqa_mc2": np.mean},
        higher_is_better={"truthfulqa_mc1": True, "truthfulqa_mc2": True},
    )
    word_perplexity = CorpusLevelMetric(
        metric="word_perplexity",
        sample_level_fn=PerplexityPreparator(units_type="words").prepare,
        category=MetricCategory.PERPLEXITY,
        use_case=MetricUseCase.SUMMARIZATION,
        corpus_level_fn=CorpusLevelPerplexityMetric("weighted_perplexity").compute,
        higher_is_better=False,
    )

    def __str__(self):
        return self.name.replace("_at_", "@")

    @staticmethod
    def higher_is_better():
        res = {}
        for metric in Metrics:
            if metric.value.category == MetricCategory.IGNORED:
                continue
            if isinstance(metric.value, MetricGrouping):
                res.update(metric.value.higher_is_better)
            else:
                res[metric.value.metric] = metric.value.higher_is_better
        return res

    @staticmethod
    def corpus_level_fns() -> dict[str, callable]:
        res = {}
        for metric in Metrics:
            if metric.value.category == MetricCategory.IGNORED:
                continue
            if isinstance(metric.value, MetricGrouping):
                res.update(metric.value.corpus_level_fn)
            else:
                res[metric.value.metric] = metric.value.corpus_level_fn
        return res

    @staticmethod
    def all_metrics():
        res = []
        for metric in Metrics:
            if metric.value.category == MetricCategory.IGNORED:
                continue
            res.extend(as_list(metric.value.metric))
        return res
