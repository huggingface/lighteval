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


from copy import deepcopy

import numpy as np
from aenum import Enum

from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.harness_compatibility.drop import DropMetrics
from lighteval.metrics.harness_compatibility.truthful_qa import TruthfulqaMCMetrics
from lighteval.metrics.metrics_corpus import (
    CorpusLevelF1Score,
    CorpusLevelPerplexityMetric,
    CorpusLevelTranslationMetric,
    MatthewsCorrCoef,
)
from lighteval.metrics.metrics_sample import (
    BLEU,
    BLEURT,
    MRR,
    ROUGE,
    AccGoldLikelihood,
    AvgAtN,
    BertScore,
    ExactMatches,
    Extractiveness,
    F1_score,
    Faithfulness,
    GPassAtK,
    JudgeLLMSimpleQA,
    LoglikelihoodAcc,
    MajAtN,
    PassAtK,
    Recall,
    StringDistance,
)
from lighteval.metrics.normalizations import bigbench_normalizer, remove_braces, remove_braces_and_strip
from lighteval.metrics.sample_preparator import (
    GenerativePreparator,
    LoglikelihoodPreparator,
    PerplexityPreparator,
    TargetPerplexityPreparator,
)
from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
)
from lighteval.metrics.utils.metric_utils import (
    CorpusLevelMetric,
    CorpusLevelMetricGrouping,
    SampleLevelMetric,
    SampleLevelMetricGrouping,
    SamplingMethod,
)
from lighteval.utils.language import Language


class Metrics(Enum):
    acc_golds_likelihood = SampleLevelMetric(  # todo: we need a better name for this!
        metric_name="acc",
        sample_level_fn=AccGoldLikelihood(),
        category=SamplingMethod.LOGPROBS,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    avg_at_n = SampleLevelMetric(
        metric_name="avg@n",
        sample_level_fn=AvgAtN(strip_strings=True),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    avg_at_n_math = SampleLevelMetric(
        metric_name="avg@n",
        sample_level_fn=AvgAtN(
            sample_scoring_function=MultilingualExtractiveMatchMetric(
                language=Language.ENGLISH,
                gold_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig()],
                pred_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig()],
                precision=6,
            ),
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    bert_score = SampleLevelMetricGrouping(
        metric_name=["BERTScore-P", "BERTScore-R", "BERTScore-F"],
        sample_level_fn=BertScore(normalize_gold=remove_braces, normalize_pred=remove_braces_and_strip),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn={"BERTScore-P": np.mean, "BERTScore-R": np.mean, "BERTScore-F": np.mean},
        higher_is_better={"BERTScore-P": True, "BERTScore-R": True, "BERTScore-F": True},
    )
    bits_per_byte = CorpusLevelMetric(
        metric_name="bits_per_byte",
        sample_level_fn=PerplexityPreparator(units_type="bytes"),
        category=SamplingMethod.PERPLEXITY,
        corpus_level_fn=CorpusLevelPerplexityMetric("bits_per_byte"),
        higher_is_better=False,
    )
    bleu = CorpusLevelMetric(
        metric_name="bleu",
        sample_level_fn=GenerativePreparator(),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=CorpusLevelTranslationMetric("bleu"),
        higher_is_better=True,
    )
    bleu_1 = SampleLevelMetric(
        metric_name="bleu_1",
        sample_level_fn=BLEU(n_gram=1),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    bleu_4 = SampleLevelMetric(
        metric_name="bleu_4",
        sample_level_fn=BLEU(n_gram=4),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

    bleurt = SampleLevelMetric(
        metric_name="bleurt",
        sample_level_fn=BLEURT(),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    byte_perplexity = CorpusLevelMetric(
        metric_name="byte_perplexity",
        sample_level_fn=PerplexityPreparator(units_type="bytes"),
        category=SamplingMethod.PERPLEXITY,
        corpus_level_fn=CorpusLevelPerplexityMetric("weighted_perplexity"),
        higher_is_better=False,
    )
    chrf = CorpusLevelMetric(
        metric_name="chrf",
        sample_level_fn=GenerativePreparator(),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=CorpusLevelTranslationMetric("chrf"),
        higher_is_better=True,
    )
    chrf_plus = CorpusLevelMetric(
        metric_name="chrf++",
        sample_level_fn=GenerativePreparator(),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=CorpusLevelTranslationMetric("chrf++"),
        higher_is_better=True,
    )
    copyright = SampleLevelMetricGrouping(
        metric_name=["longest_common_prefix_length", "edit_distance", "edit_similarity"],
        sample_level_fn=StringDistance(
            metric_types=["longest_common_prefix_length", "edit_distance", "edit_similarity"], strip_prediction=True
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn={"longest_common_prefix_length": max, "edit_distance": min, "edit_similarity": max},
        higher_is_better={"longest_common_prefix_length": True, "edit_distance": False, "edit_similarity": True},
    )
    drop = SampleLevelMetricGrouping(
        metric_name=["em", "f1"],
        sample_level_fn=DropMetrics(),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn={"em": max, "f1": max},
        higher_is_better={"em": True, "f1": True},
    )
    exact_match = SampleLevelMetric(
        metric_name="em",
        sample_level_fn=ExactMatches(strip_strings=True),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    expr_gold_metric = SampleLevelMetric(
        metric_name="extractive_match",
        sample_level_fn=MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            fallback_mode="first_match",
            precision=5,
            gold_extraction_target=(ExprExtractionConfig(),),
            # Match boxed first before trying other regexes
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
            aggregation_function=max,
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    extractiveness = SampleLevelMetricGrouping(
        metric_name=["summarization_coverage", "summarization_density", "summarization_compression"],
        sample_level_fn=Extractiveness(
            normalize_input=remove_braces, normalize_pred=remove_braces_and_strip, input_column="text"
        ),
        category=SamplingMethod.GENERATIVE,
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
    extractiveness_de = SampleLevelMetricGrouping(
        metric_name=["summarization_coverage", "summarization_density", "summarization_compression"],
        sample_level_fn=Extractiveness(
            normalize_input=remove_braces, normalize_pred=remove_braces_and_strip, input_column="text", language="de"
        ),
        category=SamplingMethod.GENERATIVE,
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
    extractiveness_fr = SampleLevelMetricGrouping(
        metric_name=["summarization_coverage", "summarization_density", "summarization_compression"],
        sample_level_fn=Extractiveness(
            normalize_input=remove_braces, normalize_pred=remove_braces_and_strip, input_column="text", language="fr"
        ),
        category=SamplingMethod.GENERATIVE,
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
    extractiveness_it = SampleLevelMetricGrouping(
        metric_name=["summarization_coverage", "summarization_density", "summarization_compression"],
        sample_level_fn=Extractiveness(
            normalize_input=remove_braces, normalize_pred=remove_braces_and_strip, input_column="text", language="it"
        ),
        category=SamplingMethod.GENERATIVE,
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
    f1_score = SampleLevelMetric(
        metric_name="f1",
        sample_level_fn=F1_score(),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    f1_score_macro = CorpusLevelMetric(
        metric_name="f1",
        sample_level_fn=GenerativePreparator(),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=CorpusLevelF1Score(average="macro"),
        higher_is_better=True,
    )
    f1_score_micro = CorpusLevelMetric(
        metric_name="f1",
        sample_level_fn=GenerativePreparator(),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=CorpusLevelF1Score(average="micro"),
        higher_is_better=True,
    )
    faithfulness = SampleLevelMetric(
        metric_name="summac",
        sample_level_fn=Faithfulness(
            normalize_input=remove_braces, normalize_pred=remove_braces_and_strip, input_column="text"
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    g_pass_at_k = SampleLevelMetricGrouping(
        metric_name="g-pass@k",
        sample_level_fn=GPassAtK(strip_strings=True),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    g_pass_at_k_math = SampleLevelMetricGrouping(
        metric_name="math-g-pass@k",
        sample_level_fn=GPassAtK(
            name_prefix="math",
            strip_strings=True,
            sample_scoring_function=MultilingualExtractiveMatchMetric(
                language=Language.ENGLISH,
                fallback_mode="first_match",
                precision=5,
                gold_extraction_target=(ExprExtractionConfig(),),
                # Match boxed first before trying other regexes
                pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
                aggregation_function=max,
            ),
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    g_pass_at_k_latex = SampleLevelMetricGrouping(
        metric_name="latex-g-pass@k",
        sample_level_fn=GPassAtK(
            name_prefix="latex",
            strip_strings=True,
            sample_scoring_function=MultilingualExtractiveMatchMetric(
                language=Language.ENGLISH,
                fallback_mode="first_match",
                precision=5,
                gold_extraction_target=(LatexExtractionConfig(),),
                # Match boxed first before trying other regexes
                pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
                aggregation_function=max,
            ),
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    loglikelihood_acc = SampleLevelMetric(
        metric_name="acc",
        sample_level_fn=LoglikelihoodAcc(logprob_normalization=None),
        category=SamplingMethod.LOGPROBS,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    loglikelihood_f1 = CorpusLevelMetric(
        metric_name="loglikelihood_f1",
        sample_level_fn=LoglikelihoodPreparator(),
        category=SamplingMethod.LOGPROBS,
        corpus_level_fn=CorpusLevelF1Score(None),
        higher_is_better=True,
    )
    maj_at_n = SampleLevelMetric(
        metric_name="maj@n",
        sample_level_fn=MajAtN(),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    mcc = CorpusLevelMetric(
        metric_name="mcc",
        sample_level_fn=LoglikelihoodPreparator(),
        category=SamplingMethod.LOGPROBS,
        corpus_level_fn=MatthewsCorrCoef(),
        higher_is_better=True,
    )
    mrr = SampleLevelMetric(
        metric_name="mrr",
        sample_level_fn=MRR(),
        category=SamplingMethod.LOGPROBS,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    multi_f1_numeric = CorpusLevelMetric(
        metric_name="mf1",
        sample_level_fn=LoglikelihoodPreparator(is_single_token=True),
        category=SamplingMethod.LOGPROBS,
        corpus_level_fn=CorpusLevelF1Score(average="micro", num_classes=3),
        higher_is_better=True,
    )
    pass_at_k = SampleLevelMetric(
        metric_name="pass@k",
        sample_level_fn=PassAtK(strip_strings=True),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    pass_at_k_math = SampleLevelMetric(
        metric_name="pass@k",
        sample_level_fn=PassAtK(
            strip_strings=True,
            # Extracting mathematical expressions and latex expressions
            sample_scoring_function=MultilingualExtractiveMatchMetric(
                language=Language.ENGLISH,
                gold_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig()],
                pred_extraction_target=[ExprExtractionConfig(), LatexExtractionConfig()],
                precision=6,
            ),
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    pass_at_k_letters = SampleLevelMetric(
        metric_name="pass@k",
        sample_level_fn=PassAtK(
            sample_scoring_function=MultilingualExtractiveMatchMetric(
                language=Language.ENGLISH,
                gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
                pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
                precision=6,
            ),
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    prediction_perplexity = SampleLevelMetric(
        metric_name="ppl",
        sample_level_fn=PerplexityPreparator("words"),
        category=SamplingMethod.PERPLEXITY,
        corpus_level_fn=CorpusLevelPerplexityMetric("perplexity"),
        higher_is_better=True,
    )
    recall_at_k = SampleLevelMetric(
        metric_name="recall",
        sample_level_fn=Recall(k=1),
        category=SamplingMethod.LOGPROBS,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    rouge_t5 = CorpusLevelMetricGrouping(
        metric_name=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        sample_level_fn=ROUGE(
            ["rouge1", "rouge2", "rougeL", "rougeLsum"],
            bootstrap=True,
            normalize_gold=bigbench_normalizer,
            normalize_pred=bigbench_normalizer,
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn={"rouge1": np.mean, "rouge2": np.mean, "rougeL": np.mean, "rougeLsum": np.mean},
        higher_is_better={"rouge1": True, "rouge2": True, "rougeL": True, "rougeLsum": True},
    )
    rouge1 = SampleLevelMetric(
        metric_name="rouge1",
        sample_level_fn=ROUGE("rouge1"),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    rouge2 = SampleLevelMetric(
        metric_name="rouge2",
        sample_level_fn=ROUGE("rouge2"),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    rougeL = SampleLevelMetric(
        metric_name="rougeL",
        sample_level_fn=ROUGE("rougeL"),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    rougeLsum = SampleLevelMetric(
        metric_name="rougeLsum",
        sample_level_fn=ROUGE("rougeLsum"),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    simpleqa_judge = SampleLevelMetricGrouping(
        metric_name=["simpleqa_judge"],
        higher_is_better={"simpleqa_judge": True},
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=JudgeLLMSimpleQA(),
        corpus_level_fn={
            "simpleqa_judge": np.mean,
        },
        batched_compute=True,
    )
    target_perplexity = SampleLevelMetric(
        metric_name="ppl",
        sample_level_fn=TargetPerplexityPreparator(units_type="words"),
        category=SamplingMethod.LOGPROBS,
        corpus_level_fn=CorpusLevelPerplexityMetric("perplexity"),
        higher_is_better=False,
    )
    ter = CorpusLevelMetric(
        metric_name="ter",
        sample_level_fn=GenerativePreparator(),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=CorpusLevelTranslationMetric("ter"),
        higher_is_better=False,
    )
    truthfulqa_mc_metrics = SampleLevelMetricGrouping(
        metric_name=["truthfulqa_mc1", "truthfulqa_mc2"],
        sample_level_fn=TruthfulqaMCMetrics(),
        category=SamplingMethod.LOGPROBS,
        corpus_level_fn={"truthfulqa_mc1": np.mean, "truthfulqa_mc2": np.mean},
        higher_is_better={"truthfulqa_mc1": True, "truthfulqa_mc2": True},
    )
    word_perplexity = CorpusLevelMetric(
        metric_name="word_perplexity",
        sample_level_fn=PerplexityPreparator(units_type="words"),
        category=SamplingMethod.PERPLEXITY,
        corpus_level_fn=CorpusLevelPerplexityMetric("weighted_perplexity"),
        higher_is_better=False,
    )
    gpqa_instruct_metric = SampleLevelMetric(
        metric_name="extractive_match",
        sample_level_fn=MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            gold_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
            ],
            pred_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
            ],
            precision=6,
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )
    gpqa_instruct_pass_at_k = SampleLevelMetric(
        metric_name="gpqa_pass@k",
        sample_level_fn=PassAtK(
            sample_scoring_function=MultilingualExtractiveMatchMetric(
                language=Language.ENGLISH,
                gold_extraction_target=[
                    IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
                ],
                pred_extraction_target=[
                    IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
                ],
                precision=6,
            ),
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

    def __str__(self):
        return self.name.replace("_at_", "@")

    def __call__(self, sample_params):
        # When parametrizing, we don't look at the Metrics enum,
        # but at a specific single metric (a value)
        # Be very careful to not change the default value of the enum
        return deepcopy(self.value)(sample_params=sample_params)
