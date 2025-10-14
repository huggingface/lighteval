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

from dataclasses import dataclass
from typing import Callable

from lighteval.metrics.metrics_corpus import CorpusLevelComputation
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.sample_preparator import Preparator
from lighteval.tasks.requests import SamplingMethod


@dataclass
class Metric:
    metric_name: str
    higher_is_better: bool
    category: SamplingMethod
    sample_level_fn: SampleLevelComputation | Preparator
    corpus_level_fn: CorpusLevelComputation | Callable

    batched_compute: bool = False

    def get_doc(self):
        return self.sample_level_fn.__doc__

    def compute_sample(
        self, **kwargs
    ) -> dict:  # result: Union[list[ModelResponse], ModelResponse], formatted_doc: Doc) -> dict:
        if isinstance(self.sample_level_fn, SampleLevelComputation):
            sample_level_fn = self.sample_level_fn.compute
        elif isinstance(self.sample_level_fn, Preparator):
            sample_level_fn = self.sample_level_fn.prepare
        else:
            raise ValueError(
                f"Incorrect type for {self.sample_level_fn}, should be a SampleLevelComputation or Preparator"
            )

        if isinstance(self, MetricGrouping):
            return sample_level_fn(**kwargs)
        return {self.metric_name: sample_level_fn(**kwargs)}

    def get_corpus_aggregations(self) -> dict:
        if isinstance(self, MetricGrouping):
            if isinstance(self.corpus_level_fn, dict):
                corpus_level_fn = self.corpus_level_fn
            else:
                corpus_level_fn = dict.fromkeys(self.metric_name, self.corpus_level_fn)
        else:
            corpus_level_fn = {self.metric_name: self.corpus_level_fn}

        for name, item in corpus_level_fn.items():
            if isinstance(item, Callable):
                corpus_level_fn[name] = item
            else:
                corpus_level_fn[name] = item.compute_corpus

        return corpus_level_fn

    def __call__(self, sample_params: dict | None):
        """Allow creating new instances with modified parameters"""
        if sample_params is not None:
            for k, v in sample_params.items():
                setattr(self.sample_level_fn, k, v)

        # Once the parameters are updated, we need to adjust the
        # metric name to what will be returned
        # CAREFUL: do not change the following logic!
        # It must always provide the values of all parameters, so that people can evaluate using a range of metrics
        # For example, pass@k=1&n=16, pass@k=10&n=16, etc
        sample_params_name = "&".join(f"{k}={v}" for k, v in sample_params.items())
        if isinstance(self, MetricGrouping):
            if hasattr(self.sample_level_fn, "metric_names"):
                # this is mostly for the gpass@k metrics which redefine submetric names
                self.metric_name = self.sample_level_fn.metric_names
            else:
                self.metric_name = [f"{metric}:{sample_params_name}" for metric in self.metric_name]
        else:
            self.metric_name = f"{self.metric_name}:{sample_params_name}"
        return self

    @staticmethod
    def get_allowed_types_for_metrics():
        return (SampleLevelComputation, Preparator, CorpusLevelComputation, Callable)


@dataclass
class MetricGrouping(Metric):
    """Some metrics are more advantageous to compute together at once.
    For example, if a costly preprocessing is the same for all metrics, it makes more sense to compute it once.
    """

    metric_name: list[str]
    corpus_level_fn: dict[str, Callable]
    higher_is_better: dict[str, Callable]


@dataclass
class CorpusLevelMetric(Metric):
    """Metric computed over the whole corpora, with computations happening at the aggregation phase"""

    pass


@dataclass
class SampleLevelMetric(Metric):
    """Metric computed per sample, then aggregated over the corpus"""

    pass


@dataclass
class CorpusLevelMetricGrouping(MetricGrouping):
    """MetricGrouping computed over the whole corpora, with computations happening at the aggregation phase"""

    pass


@dataclass
class SampleLevelMetricGrouping(MetricGrouping):
    """MetricGrouping are computed per sample, then aggregated over the corpus"""

    pass
