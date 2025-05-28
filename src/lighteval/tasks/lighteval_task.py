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

import inspect
import logging
import random
from dataclasses import asdict, dataclass, field
from typing import Callable

from datasets import DatasetDict
from huggingface_hub import TextGenerationInputGrammarType
from multiprocess import Pool
from pytablewriter import MarkdownTableWriter

from lighteval.metrics import (
    apply_generative_metric,
)
from lighteval.metrics.metrics import Metric, Metrics
from lighteval.tasks.prompt_manager import FewShotSampler
from lighteval.tasks.requests import (
    Doc,
    SamplingMethod,
)
from lighteval.utils.utils import ListLike, as_list, download_dataset_worker


logger = logging.getLogger(__name__)


@dataclass
class LightevalTaskConfig:
    """Stored configuration of a given [`LightevalTask`].

    Arguments:
        name (str): Short name of the evaluation task.
        suite (list[str]): Evaluation suites to which the task belongs.
        prompt_function (Callable[[dict, str], Doc]): Function used to create the [`Doc`] samples from each line of the evaluation dataset.
        hf_repo (str): Path of the hub dataset repository containing the evaluation information.
        hf_subset (str): Subset used for the current task, will be default if none is selected.
        hf_avail_splits (list[str]): All the available splits in the evaluation dataset
        evaluation_splits (list[str]): List of the splits actually used for this evaluation
        few_shots_split (str): Name of the split from which to sample few-shot examples
        few_shots_select (str): Method with which to sample few-shot examples
        generation_size (int): Maximum allowed size of the generation
        generation_grammar (TextGenerationInputGrammarType): The grammar to generate completion according to. Currently only available for TGI and Inference Endpoint models.
        metric (list[str]): List of all the metrics for the current task.
        stop_sequence (list[str]): Stop sequence which interrupts the generation for generative metrics.
        original_num_docs (int): Number of documents in the task
        effective_num_docs (int): Number of documents used in a specific evaluation
        truncated_num_docs (bool): Whether less than the total number of documents were used
        trust_dataset (bool): Whether to trust the dataset at execution or not
        version (int): The version of the task. Defaults to 0. Can be increased if the underlying dataset or the prompt changes.
    """

    name: str
    prompt_function: Callable[
        [dict, str], Doc
    ]  # The prompt function should be used to map a line in the dataset to a Sample
    hf_repo: str
    hf_subset: str
    metrics: ListLike[Metric]  # List of metric , should be configurable

    # Additional hf dataset config
    hf_revision: str | None = None
    hf_filter: Callable[[dict], bool] | None = None
    hf_avail_splits: ListLike[str] = field(default_factory=lambda: ["train", "validation", "test"])

    # We default to false, to reduce security issues
    trust_dataset: bool = False

    # Splits
    evaluation_splits: ListLike[str] = field(default_factory=lambda: ["validation"])
    few_shots_split: str | None = None
    few_shots_select: str | None = None

    # Generation args
    generation_size: int | None = None
    generation_grammar: TextGenerationInputGrammarType | None = None
    stop_sequence: ListLike[str] | None = None
    num_samples: list[int] | None = None

    suite: ListLike[str] = field(default_factory=lambda: ["custom"])

    original_num_docs: int = -1
    effective_num_docs: int = -1

    must_remove_duplicate_docs: bool = False

    num_fewshots: int = 0
    truncate_fewshots: bool = False

    version: int = 0

    def __post_init__(self):
        # If we got a Metrics enums instead of a Metric, we convert
        self.metrics = [metric.value if isinstance(metric, Metrics) else metric for metric in self.metrics]

        # Convert list to tuple for hashing
        self.metrics = tuple(self.metrics)
        self.hf_avail_splits = tuple(self.hf_avail_splits)
        self.evaluation_splits = tuple(self.evaluation_splits)
        self.suite = tuple(self.suite)
        self.stop_sequence = tuple(self.stop_sequence) if self.stop_sequence is not None else ()
        self.full_name = f"{self.name}|{self.num_fewshots}"

    def print(self):
        md_writer = MarkdownTableWriter()
        md_writer.headers = ["Key", "Value"]

        values = []

        for k, v in asdict(self).items():
            if k == "metric":
                for ix, metrics in enumerate(v):
                    for metric_k, metric_v in metrics.items():
                        if inspect.ismethod(metric_v):
                            values.append([f"{k} {ix}: {metric_k}", metric_v.__qualname__])
                        else:
                            values.append([f"{k} {ix}: {metric_k}", repr(metric_v)])

            else:
                if isinstance(v, Callable):
                    values.append([k, v.__name__])
                else:
                    values.append([k, repr(v)])

        md_writer.value_matrix = values

        print(md_writer.dumps())


class LightevalTask:
    def __init__(
        self,
        config: LightevalTaskConfig,
    ):
        """
        Initialize a LightEval task.

        Args:
            config (dict): configuration dictionary containing
                task-specific settings (from the task_table.json file).
        """
        self.name = config.name
        self.full_name = f"{self.name}|{config.num_fewshots}"
        self.version = config.version
        self.suite = config.suite
        self.dataset_config = config

        # Dataset info
        self.dataset_path = config.hf_repo
        self.dataset_config_name = config.hf_subset
        self.dataset_revision = config.hf_revision
        self.dataset_filter = config.hf_filter
        self.trust_dataset = config.trust_dataset
        self.dataset: DatasetDict | None = None  # Delayed download
        self.evaluation_split = as_list(config.evaluation_splits)
        self._docs = None

        self._fewshot_docs = None
        self.fewshot_split: str | None = config.few_shots_split or self.get_first_possible_fewshot_splits(
            config.hf_avail_splits or []
        )
        self.fewshot_selection = config.few_shots_select

        self.formatter = config.prompt_function
        self.generation_size = config.generation_size
        self.generation_grammar = config.generation_grammar
        self.stop_sequence = config.stop_sequence
        self.must_remove_duplicate_docs = config.must_remove_duplicate_docs
        self.fewshot_sampler = FewShotSampler(self)

        # Metrics
        self.metrics = config.metrics

        self.sampling_methods = {metric.category for metric in self.metrics}
        self.has_metric_category = {category: (category in self.sampling_methods) for category in SamplingMethod}

        # We assume num_samples always contains 1 (for base generative evals)
        self.num_samples = [1]
        for metric in self.metrics:
            metric_names = as_list(metric.metric_name)

            for metric_name in metric_names:
                # Update the number of samples to generate using the information in the metric name
                self.num_samples.append(extract_num_samples(metric_name))

    def get_first_possible_fewshot_splits(self, available_splits: ListLike[str]) -> str | None:
        """
        Parses the possible fewshot split keys in order: train, then validation
        keys and matches them with the available keys.  Returns the first
        available.

        Returns:
            str: the first available fewshot splits or None if nothing is available
        """
        # Possible few shot splits are the available splits not used for evaluation
        possible_fewshot_splits = [k for k in available_splits if k not in self.evaluation_split]
        stored_splits = []

        # We look at these keys in order (first the training sets, then the validation sets)
        allowed_splits = ["train", "dev", "valid", "default"]
        for allowed_split in allowed_splits:
            # We do a partial match of the allowed splits
            available_splits = [k for k in possible_fewshot_splits if allowed_split in k]
            stored_splits.extend(available_splits)

        if len(stored_splits) > 0:
            return stored_splits[0]

        logger.warning(f"Careful, the task {self.name} is using evaluation data to build the few shot examples.")
        return None

    def _get_docs_from_split(self, splits: list[str], few_shots=False) -> list[Doc]:
        """
        Get the documents from the dataset for the given keys (splits).

        Args:
            splits (list[str]): List of splits, (e.g. ["train", "dev"])
            few_shots (bool, optional): Whether the documents are used for few
                shot examples. Defaults to False.

        Returns:
            list[Doc]: List of documents.
        """
        if self.dataset is None:
            self.dataset = download_dataset_worker(
                self.dataset_path,
                self.dataset_config_name,
                self.trust_dataset,
                self.dataset_filter,
                self.dataset_revision,
            )

        assert self.dataset is not None, f"Dataset {self.dataset_path} not found."

        docs = []
        for split in splits:
            for ix, item in enumerate(self.dataset[split]):
                # Some tasks formatting is applied differently when the document is used for fewshot examples
                # vs when it's used for the actual prompt. That's why we store whether we are currently using the
                # doc for a fewshot sample (few_shots=True) or not, which then leads to the creation of a different Doc.
                item["__few_shots"] = few_shots
                # Some tasks require to know which is the current item index in order to apply a different prompt template
                item["__index"] = ix
                doc = self.formatter(item, self.name)
                doc.id = str(ix)
                docs.append(doc)

        return docs

    def remove_duplicate_docs(self, docs: list[Doc]) -> list[Doc]:
        seen_examples, res = set(), []
        for doc in docs:
            if str(doc) not in seen_examples:
                res.append(doc)
                seen_examples.add(str(doc))
        return res

    def fewshot_docs(self) -> list[Doc]:
        """
        Returns the few shot documents. If the few shot documents are not
        available, it gets them from the few shot split or the evaluation split.

        Returns:
            list[Doc]: Documents that will be used for few shot examples. One
                document = one few shot example.
        """
        if self._fewshot_docs is None:
            self._fewshot_docs = []

            # If we have no available few shot split, the few shot data is the eval data!
            if self.fewshot_split is None:
                self._fewshot_docs = self._get_docs_from_split(self.evaluation_split, few_shots=True)
            else:  # Normal case
                self._fewshot_docs = self._get_docs_from_split([self.fewshot_split], few_shots=True)

        return self._fewshot_docs

    def eval_docs(self) -> list[Doc]:
        """
        Returns the evaluation documents.

        Returns:
            list[Doc]: Evaluation documents.
        """
        if self._docs is None:
            self._docs = self._get_docs_from_split(self.evaluation_split)
            if self.must_remove_duplicate_docs:
                self._docs = self.remove_duplicate_docs(self._docs)
        return self._docs

    def get_docs(self, max_samples: int | None = None) -> list[Doc]:
        eval_docs = self.eval_docs()

        if len(eval_docs) == 0:
            raise ValueError(f"Task {self.name} has no documents to evaluate skipping.")

        n_samples = min(max_samples, len(eval_docs)) if max_samples else len(eval_docs)
        random.shuffle(eval_docs)

        docs = []

        for doc in eval_docs[:n_samples]:
            num_fewshots = self.dataset_config.num_fewshots
            doc.task_name = f"{self.name}|{num_fewshots}"
            doc.fewshot_samples = self.fewshot_sampler.sample_fewshot_examples(num_fewshots, 1, formatted_doc=doc)
            doc.sampling_methods.update(self.sampling_methods)
            docs.append(doc)

        return docs

        #    def construct_requests_for_doc(
        #        self,
        #        doc: Doc,
        #    ) -> dict[RequestType, list[Request]]:
        #        """
        #        Constructs a list of requests from the task based on the given parameters.
        #
        #        Args:
        #            formatted_doc (Doc): Formatted document almost straight from the dataset.
        #            ctx (str): Context, which is the few shot examples + the query.
        #            document_id_seed (str): Index of the document in the task appended with the seed used for the few shot sampling.
        #            current_task_name (str): Name of the current task.
        #
        #        Returns:
        #            dict[RequestType, List[Request]]: List of requests.
        #        """
        #        requests: dict[RequestType, list[Request]] = collections.defaultdict(list)
        #
        #        if self.has_metric_category[MetricCategory.TARGET_PERPLEXITY]:
        #            golds = doc.get_golds()
        #            requests[RequestType.LOGLIKELIHOOD] += [
        #                LoglikelihoodRequest(
        #                    task_name=doc.task_name,
        #                    sample_index=doc.id,
        #                    request_index=i,
        #                    context=doc.query,
        #                    choice=gold,
        #                    metric_categories=[MetricCategory.TARGET_PERPLEXITY],
        #                    fewshots=doc.fewshots,
        #                )
        #                for i, gold in enumerate(golds)
        #            ]
        #        if self.has_metric_category[MetricCategory.PERPLEXITY]:
        #            requests[RequestType.LOGLIKELIHOOD_ROLLING] += [
        #                LoglikelihoodRollingRequest(
        #                    task_name=doc.task_name,
        #                    sample_index=doc.id,
        #                    request_index=0,
        #                    context=doc.query,
        #                    metric_categories=[MetricCategory.PERPLEXITY],
        #                    fewshots=doc.fewshots,
        #                )
        #            ]
        #        if self.has_metric_category[MetricCategory.GENERATIVE_SAMPLING]:
        #            # All the possible sampling tasks require the same generation process - we can do them in one step
        #            # so we select the maximum number of samples and the metrics will select only the
        #            # relevant number of tiems
        #            requests[RequestType.GREEDY_UNTIL] += [
        #                GreedyUntilRequest(
        #                    task_name=doc.task_name,
        #                    sample_index=doc.id,
        #                    request_index=0,
        #                    context=doc.query,
        #                    stop_sequence=self.stop_sequence,
        #                    generation_size=self.generation_size,
        #                    generation_grammar=self.generation_grammar,
        #                    num_samples=max(self.num_samples),
        #                    do_sample=True,
        #                    use_logits=False,
        #                    metric_categories=[MetricCategory.GENERATIVE_SAMPLING],
        #                    fewshots=doc.fewshots,
        #                )
        #            ]
        #        if (
        #            self.has_metric_category[MetricCategory.GENERATIVE]
        #            or self.has_metric_category[MetricCategory.GENERATIVE_LOGPROB]
        #        ):
        #            use_logits = self.has_metric_category[MetricCategory.GENERATIVE_LOGPROB]
        #            requests[RequestType.GREEDY_UNTIL] += [
        #                GreedyUntilRequest(
        #                    task_name=doc.task_name,
        #                    sample_index=doc.id,
        #                    request_index=0,
        #                    context=doc.query,
        #                    stop_sequence=self.stop_sequence,
        #                    generation_size=self.generation_size,
        #                    generation_grammar=self.generation_grammar,
        #                    num_samples=1,
        #                    use_logits=use_logits,
        #                    fewshots=doc.fewshots,
        #                    metric_categories=[
        #                        c
        #                        for c in [
        #                            MetricCategory.GENERATIVE,
        #                            MetricCategory.GENERATIVE_LOGPROB,
        #                        ]
        #                        if self.has_metric_category[c]
        #                    ],
        #                )
        #            ]
        #        if (
        #            self.has_metric_category[MetricCategory.MULTICHOICE]
        #            or self.has_metric_category[MetricCategory.MULTICHOICE_PMI]
        #        ):
        #            requests[RequestType.LOGLIKELIHOOD] += [
        #                LoglikelihoodRequest(
        #                    task_name=doc.task_name,
        #                    sample_index=doc.id,
        #                    request_index=i,
        #                    context=doc.query,
        #                    choice=choice,
        #                    fewshots=doc.fewshots,
        #                    metric_categories=[
        #                        c
        #                        for c in [MetricCategory.MULTICHOICE, MetricCategory.MULTICHOICE_PMI]
        #                        if self.has_metric_category[c]
        #                    ],
        #                )
        #                for i, choice in enumerate(doc.choices)
        #            ]
        #        if self.has_metric_category[MetricCategory.MULTICHOICE_PMI]:
        #            assert doc.unconditioned_query is not None, "Unconditioned query is required for PMI normalization"
        #            requests[RequestType.LOGLIKELIHOOD] += [
        #                LoglikelihoodRequest(
        #                    task_name=doc.task_name,
        #                    sample_index=doc.id,
        #                    # The normalization should come after the choices
        #                    request_index=i + len(doc.choices),
        #                    context=doc.unconditioned_query,
        #                    choice=choice,
        #                    metric_categories=[MetricCategory.MULTICHOICE_PMI],
        #                    fewshots=doc.fewshots,
        #                )
        #                for i, choice in enumerate(doc.choices)
        #            ]
        #        if self.has_metric_category[MetricCategory.LLM_AS_JUDGE]:
        #            requests[RequestType.GREEDY_UNTIL] += [
        #                GreedyUntilRequest(
        #                    task_name=doc.task_name,
        #                    sample_index=doc.id,
        #                    request_index=0,
        #                    context=doc.query,
        #                    stop_sequence=self.stop_sequence,
        #                    generation_size=self.generation_size,
        #                    generation_grammar=self.generation_grammar,
        #                    num_samples=1,
        #                    metric_categories=[MetricCategory.LLM_AS_JUDGE],
        #                    fewshots=doc.fewshots,
        #                )
        #            ]
        #
        #        return requests

    def get_metric_method_from_category(self, metric_category):
        if metric_category in [
            SamplingMethod.GENERATIVE,
        ]:
            return apply_generative_metric

    def aggregation(self):
        """
        Return a dict with metric name and its aggregation function for all
        metrics
        """
        return Metrics.corpus_level_fns(self.metrics)

    @staticmethod
    def load_datasets(tasks: dict[str, "LightevalTask"], dataset_loading_processes: int = 1) -> None:
        """
        Load datasets from the HuggingFace Hub for the given tasks.

        Args:
            tasks (list): A list of tasks.
            dataset_loading_processes (int, optional): number of processes to use for dataset loading. Defaults to 1.

        Returns:
            None
        """

        if dataset_loading_processes <= 1:
            datasets = [
                download_dataset_worker(
                    task.dataset_path,
                    task.dataset_config_name,
                    task.trust_dataset,
                    task.dataset_filter,
                    task.dataset_revision,
                )
                for task in tasks.values()
            ]
        else:
            with Pool(processes=dataset_loading_processes) as pool:
                datasets = pool.starmap(
                    download_dataset_worker,
                    [
                        (
                            task.dataset_path,
                            task.dataset_config_name,
                            task.trust_dataset,
                            task.dataset_filter,
                            task.dataset_revision,
                        )
                        for task in tasks.values()
                    ],
                )

        for task, dataset in zip(tasks, datasets):
            tasks[task].dataset = dataset


def extract_num_samples(metric_name: str) -> int:
    """Gets the number of samples to generate from the metric name.
    Assumes that any metric with @ in it's name depends on the number of samples.

    Args:
        metric_name (str): The metric name in the task.

    Returns:
        int: The number of samples to generate.
    """
    if "@" in metric_name:
        metric_name = metric_name.split("@")[-1]
        if "_" in metric_name:
            metric_name = metric_name.split("_")[0]
        if ":" in metric_name:
            return int(metric_name.split(":")[-1])
        else:
            return int(metric_name)
    return 1
