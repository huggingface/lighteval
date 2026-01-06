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

import functools
import logging
import random
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping, Sequence

from datasets import DatasetDict, load_dataset
from huggingface_hub import TextGenerationInputGrammarType
from inspect_ai.dataset import Sample
from multiprocess import Pool
from pytablewriter import MarkdownTableWriter

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import SamplingMetric
from lighteval.metrics.utils.metric_utils import Metric
from lighteval.tasks.prompt_manager import FewShotSampler
from lighteval.tasks.requests import (
    Doc,
)
from lighteval.utils.utils import ListLike, as_list


logger = logging.getLogger(__name__)


@dataclass
class LightevalTaskConfig:
    """Configuration dataclass for a LightevalTask.

    This class stores all the configuration parameters needed to define and run
    an evaluation task, including dataset information, prompt formatting,
    evaluation metrics, and generation parameters.

    Args:
        name (str): Short name of the evaluation task.
        prompt_function (Callable[[dict, str], Doc]): Function that converts dataset
            row to Doc objects for evaluation. Takes a dataset row dict and task
            name as input.
        hf_repo (str): HuggingFace Hub repository path containing the evaluation dataset.
        hf_data_files (str | Sequence[str] | Mapping[str, str | Sequence[str]] | None):
            Data files to load. Same as `data_files` argument of `datasets.load_dataset`.
        hf_subset (str): Dataset subset/configuration name to use for this task.
        metrics (ListLike[Metric | Metrics]): List of metrics or metric enums to compute for this task.

    Dataset Configuration:
        hf_revision (str | None, optional): Specific dataset revision to use.
            Defaults to None (latest).
        hf_filter (Callable[[dict], bool] | None, optional): Filter function to
            apply to dataset items. Defaults to None.
        hf_avail_splits (ListLike[str], optional): Available dataset splits.
            Defaults to ["train", "validation", "test"].

    Evaluation Splits:
        evaluation_splits (ListLike[str], optional): Dataset splits to use for
            evaluation. Defaults to ["validation"].
        few_shots_split (str | None, optional): Split to sample few-shot examples
            from. Defaults to None.
        few_shots_select (str | None, optional): Method for selecting few-shot
            examples. Defaults to None.

    Generation Parameters:
        generation_size (int | None, optional): Maximum token length for generated
            text. Defaults to None.
        generation_grammar (TextGenerationInputGrammarType | None, optional): Grammar
            for structured text generation. Only available for TGI and Inference
            Endpoint models. Defaults to None.
        stop_sequence (ListLike[str] | None, optional): Sequences that stop text
            generation. Defaults to None.
        num_samples (list[int] | None, optional): Number of samples to generate
            per input. Defaults to None.

    Task Configuration:
        version (int, optional): Task version number. Increment when dataset or
            prompt changes. Defaults to 0.
        num_fewshots (int, optional): Number of few-shot examples to include.
            Defaults to 0.
        truncate_fewshots (bool, optional): Whether to truncate few-shot examples.
            Defaults to False.
        must_remove_duplicate_docs (bool, optional): Whether to remove duplicate
            documents. Defaults to False.

    Document Tracking:
        original_num_docs (int, optional): Total number of documents in the task.
            Defaults to -1.
        effective_num_docs (int, optional): Number of documents actually used
            in evaluation. Defaults to -1.
    """

    name: str
    prompt_function: Callable[
        [dict, str], Doc
    ]  # The prompt function should be used to map a line in the dataset to a Sample
    hf_repo: str
    hf_subset: str
    metrics: ListLike[Metric | Metrics]  # Accept both Metric objects and Metrics enums
    hf_data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None

    # Inspect AI compatible parameters
    solver: None = None
    scorer: None = None
    sample_fields: Callable[[dict], Sample] | None = None
    sample_to_fewshot: Callable[[Sample], str] | None = None
    filter: Callable[[dict], bool] | None = None

    # Additional hf dataset config
    hf_revision: str | None = None
    hf_filter: Callable[[dict], bool] | None = None
    hf_avail_splits: ListLike[str] = field(default_factory=lambda: ["train", "validation", "test"])

    # Splits
    evaluation_splits: ListLike[str] = field(default_factory=lambda: ["validation"])
    few_shots_split: str | None = None
    few_shots_select: str | None = None

    # Generation args
    generation_size: int | None = None
    generation_grammar: TextGenerationInputGrammarType | None = None
    stop_sequence: ListLike[str] | None = None
    num_samples: list[int] | None = None

    original_num_docs: int = -1
    effective_num_docs: int = -1

    must_remove_duplicate_docs: bool = False

    num_fewshots: int = 0

    version: int = 0

    def __post_init__(self):
        # If we got a Metrics enums instead of a Metric, we convert
        self.metrics = [metric.value if isinstance(metric, Metrics) else metric for metric in self.metrics]
        # Convert list to tuple for hashing
        self.metrics = tuple(self.metrics)
        self.hf_avail_splits = tuple(self.hf_avail_splits)
        self.evaluation_splits = tuple(self.evaluation_splits)
        self.stop_sequence = self.stop_sequence if self.stop_sequence is not None else ()
        self.full_name = f"{self.name}|{self.num_fewshots}"  # todo clefourrier: this is likely incorrect

    def __str__(self, lite: bool = False):  # noqa: C901
        md_writer = MarkdownTableWriter()
        md_writer.headers = ["Key", "Value"]

        # These keys change through time
        to_ignore = ["original_num_docs", "effective_num_docs"]

        values = []

        for k, v in asdict(self).items():
            if lite and k in to_ignore:
                continue
            if k == "metrics":
                for ix, metrics in enumerate(v):
                    for metric_k, metric_v in metrics.items():
                        if isinstance(metric_v, functools.partial):
                            func_name = getattr(metric_v.func, "__name__", str(metric_v.func))
                            repr_v = f"partial({func_name}, ...)"
                        elif isinstance(metric_v, Callable):
                            repr_v = getattr(metric_v, "__name__", repr(metric_v))
                        elif isinstance(metric_v, Metric.get_allowed_types_for_metrics()):
                            repr_v = str(metric_v)
                        else:
                            repr_v = repr(metric_v)
                        values.append([f"{k} {ix}: {metric_k}", repr_v])

            else:
                if isinstance(v, functools.partial):
                    func_name = getattr(v.func, "__name__", str(v.func))
                    values.append([k, f"partial({func_name}, ...)"])
                elif isinstance(v, Callable):
                    values.append([k, getattr(v, "__name__", repr(v))])
                else:
                    values.append([k, repr(v)])

        md_writer.value_matrix = values

        return md_writer.dumps()

    def print(self, lite: bool = False):
        print(str(self, lite))


class LightevalTask:
    def __init__(
        self,
        config: LightevalTaskConfig,
    ):
        """Initialize a LightEval task.

        Args:
            config (dict): configuration dictionary containing
                task-specific settings (from the task_table.json file).
        """
        self.config = config
        self.name = config.name
        self.version = config.version
        self.dataset_config = config

        self.full_name = config.full_name

        # Dataset info
        self.dataset_path = config.hf_repo
        self.data_files = config.hf_data_files
        self.dataset_config_name = config.hf_subset
        self.dataset_revision = config.hf_revision
        self.dataset_filter = config.hf_filter
        self.dataset: DatasetDict | None = None  # Delayed download
        self.evaluation_split = as_list(config.evaluation_splits)
        self._docs = None

        self._fewshot_docs = None
        self.fewshot_split: str | None = config.few_shots_split or self.get_first_possible_fewshot_splits(
            config.hf_avail_splits or []
        )
        self.fewshot_selection = config.few_shots_select
        self.must_remove_duplicate_docs = config.must_remove_duplicate_docs

        self.formatter = config.prompt_function
        self.fewshot_sampler = FewShotSampler(self)

        # Metrics
        self.metrics = config.metrics
        self.sampling_methods = list({metric.category for metric in self.metrics})

        # generation parameters
        self.generation_size = config.generation_size
        self.generation_grammar = config.generation_grammar
        self.stop_sequence = config.stop_sequence

        # We assume num_samples always contains 1 (for base generative evals)
        self.num_samples = [1]
        for metric in self.metrics:
            if isinstance(metric.sample_level_fn, SamplingMetric):
                # Update the number of samples to generate using the information in the metric name
                self.num_samples.append(metric.sample_level_fn.num_samples())

    def get_first_possible_fewshot_splits(self, available_splits: ListLike[str]) -> str | None:
        """Parses the possible fewshot split keys in order: train, then validation
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
        """Get the documents from the dataset for the given keys (splits).

        Args:
            splits (list[str]): List of dataset splits to process (e.g. ["train", "dev"])
            few_shots (bool, optional): Whether the documents are used for few-shot
                examples. This affects how the formatter processes the items. Defaults to False.

        Returns:
            list[Doc]: List of documents.
        """
        if self.dataset is None:
            self.dataset = self.download_dataset_worker(self)

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
                # Skip if formatter returns None (e.g., to filter out certain samples)
                if doc is None or doc == []:
                    continue

                doc.id = str(ix)

                # Transfer task-level generation parameters to the document
                doc.generation_grammar = self.generation_grammar
                doc.generation_size = self.generation_size
                doc.stop_sequences = self.stop_sequence

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
        """Returns the few shot documents. If the few shot documents are not
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
        """Returns the evaluation documents.

        Returns:
            list[Doc]: Evaluation documents.
        """
        if self._docs is None:
            self._docs = self._get_docs_from_split(self.evaluation_split)
            if self.must_remove_duplicate_docs:
                self._docs = self.remove_duplicate_docs(self._docs)
        return self._docs

    def get_docs(self, max_samples: int | None = None) -> list[Doc]:
        """Get evaluation documents with few-shot examples and generation parameters configured.

        Retrieves evaluation documents, optionally limits the number of samples,
        shuffles them for reproducibility, and configures each document with
        few-shot examples and generation parameters for evaluation.

        Args:
            max_samples (int | None, optional): Maximum number of documents to return.
                If None, returns all available documents. Defaults to None.

        Returns:
            list[Doc]: List of documents ready for evaluation with few-shot examples
                and generation parameters configured.

        Raises:
            ValueError: If no documents are available for evaluation.
        """
        eval_docs = self.eval_docs()

        if len(eval_docs) == 0:
            raise ValueError(f"Task {self.name} has no documents to evaluate skipping.")

        n_samples = min(max_samples, len(eval_docs)) if max_samples else len(eval_docs)
        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(eval_docs)

        docs = []

        for doc in eval_docs[:n_samples]:
            num_fewshots = self.dataset_config.num_fewshots
            doc.task_name = self.full_name
            doc.fewshot_samples = self.fewshot_sampler.sample_fewshot_examples(
                num_fewshots, 0, formatted_doc=doc, sampler=rnd
            )
            doc.sampling_methods.extend(self.sampling_methods)
            doc.generation_size = self.generation_size
            doc.use_logits = doc.use_logits if doc.use_logits is not None else True
            doc.stop_sequences = self.stop_sequence
            doc.num_samples = max(self.num_samples)
            docs.append(doc)

        return docs

    def aggregation(self):
        """Return a dict with metric name and its aggregation function for all
        metrics
        """
        aggregations = {}
        for metric in self.metrics:
            aggregations.update(metric.get_corpus_aggregations())
        return aggregations

    @staticmethod
    def load_datasets(tasks: dict[str, "LightevalTask"], dataset_loading_processes: int = 1) -> None:
        """Load datasets from the HuggingFace Hub for the given tasks.

        Args:
            tasks (dict[str, LightevalTask]): Dictionary mapping task names to task objects.
            dataset_loading_processes (int, optional): Number of processes to use for
                parallel dataset loading. Defaults to 1 (sequential loading).
        """
        if dataset_loading_processes <= 1:
            # Useful for the test suite: we can mock loading tasks by overwriting the
            # individual download_dataset_worker functions
            datasets = [task.download_dataset_worker(task) for task in tasks.values()]
        else:
            with Pool(processes=dataset_loading_processes) as pool:
                datasets = pool.starmap(
                    LightevalTask.download_dataset_worker,
                    [tasks.values()],
                )

        for task, dataset in zip(tasks, datasets):
            tasks[task].dataset = dataset

    @staticmethod
    def download_dataset_worker(
        task: "LightevalTask",
    ) -> DatasetDict:
        """Worker function to download a dataset from the HuggingFace Hub.

        Downloads the dataset specified in the task configuration, optionally
        applies a filter if configured, and returns the dataset dictionary.
        This method is designed to be used for parallel dataset loading.

        Args:
            task (LightevalTask): The task object containing dataset configuration.

        Returns:
            DatasetDict: The loaded dataset dictionary containing all splits.
        """
        dataset = load_dataset(
            path=task.dataset_path,
            name=task.dataset_config_name,
            revision=task.dataset_revision,
            data_files=task.data_files,
        )

        if task.dataset_filter is not None:
            dataset = dataset.filter(task.dataset_filter)

        # It returns DatasetDict because we don't specify a split
        return dataset  # type: ignore
