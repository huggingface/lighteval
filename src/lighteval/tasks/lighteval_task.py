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

import collections
import inspect
import logging
import random
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from datasets import DatasetDict
from huggingface_hub import TextGenerationInputGrammarType
from multiprocess import Pool
from pytablewriter import MarkdownTableWriter

from lighteval.metrics import (
    apply_generative_metric,
    apply_llm_as_judge_metric,
    apply_multichoice_metric,
    apply_multichoice_metric_one_token,
    apply_perplexity_metric,
    apply_target_perplexity_metric,
)
from lighteval.metrics.metrics import Metric, MetricCategory, Metrics
from lighteval.models.transformers.transformers_model import TransformersModel
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import (
    Doc,
    GreedyUntilMultiTurnRequest,
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
    Request,
    RequestType,
    SampleUid,
)
from lighteval.utils.utils import ListLike, as_list, download_dataset_worker


if TYPE_CHECKING:
    from lighteval.logging.evaluation_tracker import EvaluationTracker

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
    prompt_function: Callable[[dict, str], Doc | None]
    hf_repo: str
    hf_subset: str
    metric: ListLike[Metric | Metrics]

    # Additional hf dataset config
    hf_revision: Optional[str] = None
    hf_filter: Optional[Callable[[dict], bool]] = None
    hf_avail_splits: Optional[ListLike[str]] = field(default_factory=lambda: ["train", "validation", "test"])
    # We default to false, to reduce security issues
    trust_dataset: bool = False

    # Splits
    evaluation_splits: ListLike[str] = field(default_factory=lambda: ["validation"])
    few_shots_split: Optional[str] = None
    few_shots_select: Optional[str] = None

    # Generation args
    generation_size: Optional[int] = None
    generation_grammar: Optional[TextGenerationInputGrammarType] = None
    stop_sequence: Optional[ListLike[str]] = None
    num_samples: Optional[list[int]] = None

    suite: ListLike[str] = field(default_factory=lambda: ["custom"])

    original_num_docs: int = -1
    effective_num_docs: int = -1

    must_remove_duplicate_docs: bool = False

    version: int = 0

    def __post_init__(self):
        # If we got a Metrics enums instead of a Metric, we convert
        self.metric = [metric.value if isinstance(metric, Metrics) else metric for metric in self.metric]

        # Convert list to tuple for hashing
        self.metric = tuple(self.metric)
        self.hf_avail_splits = tuple(self.hf_avail_splits) if self.hf_avail_splits is not None else None
        self.evaluation_splits = tuple(self.evaluation_splits)
        self.suite = tuple(self.suite)
        self.stop_sequence = tuple(self.stop_sequence) if self.stop_sequence is not None else ()

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
    def __init__(  # noqa: C901
        self, name: str, cfg: LightevalTaskConfig, cache_dir: Optional[str] = None
    ):
        """
        Initialize a LightEval task.

        Args:
            name (str): name of the task.
            cfg (dict): configuration dictionary containing
                task-specific settings (from the task_table.json file).
            cache_dir (Optional[str], optional): directory to cache the
                dataset. Defaults to None.
        """
        self.name = name
        self.version = cfg.version
        self.cache_dir = cache_dir
        self._cfg = cfg

        # Dataset info
        self.dataset_path = cfg.hf_repo
        self.dataset_config_name = cfg.hf_subset
        self.dataset_revision = cfg.hf_revision
        self.dataset_filter = cfg.hf_filter
        self.trust_dataset = cfg.trust_dataset
        self.dataset: Optional[DatasetDict] = None  # Delayed download
        logger.info(f"{self.dataset_path} {self.dataset_config_name}")
        self._fewshot_docs = None
        self._docs = None

        self.evaluation_split = as_list(cfg.evaluation_splits)

        self.fewshot_split: list[str] | None
        if cfg.few_shots_split is not None:
            self.fewshot_split = as_list(cfg.few_shots_split)
        else:
            self.fewshot_split = self.get_first_possible_fewshot_splits(cfg.hf_avail_splits or [])
        self.fewshot_selection = cfg.few_shots_select

        # Metrics
        self.metrics = as_list(cfg.metric)
        self.suite = as_list(cfg.suite)
        ignored = [metric for metric in self.metrics if metric.category == MetricCategory.IGNORED]

        if len(ignored) > 0:
            logger.warning(f"Not implemented yet: ignoring the metric {' ,'.join(ignored)} for task {self.name}.")

        current_categories = [metric.category for metric in self.metrics]
        self.has_metric_category = {category: (category in current_categories) for category in MetricCategory}

        # We assume num_samples always contains 1 (for base generative evals)
        self.num_samples = [1]
        for metric in self.metrics:
            metric_names = as_list(metric.metric_name)

            for metric_name in metric_names:
                # If we do maj_at_ metrics, we need to use the correct number of samples
                if "maj@" in metric_name:
                    self.num_samples.append(int(metric_name.replace("maj@", "").split("_")[0]))

        self.formatter = cfg.prompt_function

        self.generation_size = cfg.generation_size
        self.generation_grammar = cfg.generation_grammar
        self.stop_sequence = cfg.stop_sequence
        self.must_remove_duplicate_docs = cfg.must_remove_duplicate_docs

    @property
    def cfg(self):
        return self._cfg

    def get_first_possible_fewshot_splits(
        self, available_splits: ListLike[str], number_of_splits: int = 1
    ) -> list[str] | None:
        """
        Parses the possible fewshot split keys in order: train, then validation
        keys and matches them with the available keys.  Returns the first
        available.

        Args:
            number_of_splits (int, optional): Number of splits to return.
                Defaults to 1.

        Returns:
            list[str]: List of the first available fewshot splits.
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
            return stored_splits[:number_of_splits]

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
        splits = as_list(splits)

        docs = []
        for split in splits:
            for ix, item in enumerate(self.dataset[split]):
                # Some tasks formatting is applied differently when the document is used for fewshot examples
                # vs when it's used for the actual prompt. That's why we store whether we are currently using the
                # doc for a fewshot sample (few_shots=True) or not, which then leads to the creation of a different Doc.
                item["__few_shots"] = few_shots
                # Some tasks require to know which is the current item index in order to apply a different prompt template
                item["__index"] = ix
                cur_docs = self.formatter(item, self.name)
                if cur_docs is None:
                    continue
                docs.extend(as_list(cur_docs))
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
                self._fewshot_docs = self._get_docs_from_split(self.fewshot_split, few_shots=True)
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

    def construct_requests(
        self, formatted_doc: Doc, context: str, document_id_seed: str, current_task_name: str
    ) -> Dict[RequestType, List[Request]]:
        """
        Constructs a list of requests from the task based on the given parameters.

        Args:
            formatted_doc (Doc): Formatted document almost straight from the dataset.
            ctx (str): Context, which is the few shot examples + the query.
            document_id_seed (str): Index of the document in the task appended with the seed used for the few shot sampling.
            current_task_name (str): Name of the current task.

        Returns:
            dict[RequestType, List[Request]]: List of requests.
        """
        requests: dict[RequestType, list[Request]] = collections.defaultdict(list)

        if self.has_metric_category[MetricCategory.TARGET_PERPLEXITY]:
            golds = formatted_doc.get_golds()
            requests[RequestType.LOGLIKELIHOOD] += [
                LoglikelihoodRequest(
                    task_name=current_task_name,
                    sample_index=document_id_seed,
                    request_index=i,
                    context=context,
                    choice=gold,
                    metric_categories=[MetricCategory.TARGET_PERPLEXITY],
                )
                for i, gold in enumerate(golds)
            ]
        if self.has_metric_category[MetricCategory.PERPLEXITY]:
            requests[RequestType.LOGLIKELIHOOD_ROLLING] += [
                LoglikelihoodRollingRequest(
                    task_name=current_task_name,
                    sample_index=document_id_seed,
                    request_index=0,
                    context=context,
                    metric_categories=[MetricCategory.PERPLEXITY],
                )
            ]
        if self.has_metric_category[MetricCategory.GENERATIVE_SAMPLING]:
            # All the possible sampling tasks require the same generation process - we can do them in one step
            # so we select the maximum number of samples and the metrics will select only the
            # relevant number of tiems
            requests[RequestType.GREEDY_UNTIL] += [
                GreedyUntilRequest(
                    task_name=current_task_name,
                    sample_index=document_id_seed,
                    request_index=0,
                    context=context,
                    stop_sequence=self.stop_sequence,
                    generation_size=self.generation_size,
                    generation_grammar=self.generation_grammar,
                    num_samples=max(self.num_samples),
                    do_sample=True,
                    use_logits=False,
                    metric_categories=[MetricCategory.GENERATIVE_SAMPLING],
                )
            ]
        if (
            self.has_metric_category[MetricCategory.GENERATIVE]
            or self.has_metric_category[MetricCategory.GENERATIVE_LOGPROB]
        ):
            use_logits = self.has_metric_category[MetricCategory.GENERATIVE_LOGPROB]
            requests[RequestType.GREEDY_UNTIL] += [
                GreedyUntilRequest(
                    task_name=current_task_name,
                    sample_index=document_id_seed,
                    request_index=0,
                    context=context,
                    stop_sequence=self.stop_sequence,
                    generation_size=self.generation_size,
                    generation_grammar=self.generation_grammar,
                    num_samples=1,
                    use_logits=use_logits,
                    metric_categories=[
                        c
                        for c in [
                            MetricCategory.GENERATIVE,
                            MetricCategory.GENERATIVE_LOGPROB,
                        ]
                        if self.has_metric_category[c]
                    ],
                )
            ]
        if (
            self.has_metric_category[MetricCategory.MULTICHOICE]
            or self.has_metric_category[MetricCategory.MULTICHOICE_PMI]
        ):
            requests[RequestType.LOGLIKELIHOOD] += [
                LoglikelihoodRequest(
                    task_name=current_task_name,
                    sample_index=document_id_seed,
                    request_index=i,
                    context=context,
                    choice=choice,
                    metric_categories=[
                        c
                        for c in [MetricCategory.MULTICHOICE, MetricCategory.MULTICHOICE_PMI]
                        if self.has_metric_category[c]
                    ],
                )
                for i, choice in enumerate(formatted_doc.choices)
            ]
        if self.has_metric_category[MetricCategory.MULTICHOICE_PMI]:
            assert (
                formatted_doc.unconditioned_query is not None
            ), "Unconditioned query is required for PMI normalization"
            requests[RequestType.LOGLIKELIHOOD] += [
                LoglikelihoodRequest(
                    task_name=current_task_name,
                    sample_index=document_id_seed,
                    # The normalization should come after the choices
                    request_index=i + len(formatted_doc.choices),
                    context=formatted_doc.unconditioned_query,
                    choice=choice,
                    metric_categories=[MetricCategory.MULTICHOICE_PMI],
                )
                for i, choice in enumerate(formatted_doc.choices)
            ]
        if self.has_metric_category[MetricCategory.MULTICHOICE_ONE_TOKEN]:
            requests[RequestType.LOGLIKELIHOOD_SINGLE_TOKEN] += [
                LoglikelihoodSingleTokenRequest(
                    task_name=current_task_name,
                    sample_index=document_id_seed,
                    request_index=0,
                    context=context,
                    choices=formatted_doc.choices,
                    metric_categories=[MetricCategory.MULTICHOICE_ONE_TOKEN],
                )
            ]
        if self.has_metric_category[MetricCategory.LLM_AS_JUDGE_MULTI_TURN]:
            requests[RequestType.GREEDY_UNTIL_MULTI_TURN] += [
                GreedyUntilMultiTurnRequest(
                    task_name=current_task_name,
                    sample_index=document_id_seed,
                    request_index=0,
                    context=context,
                    stop_sequence=self.stop_sequence,
                    generation_size=self.generation_size,
                    metric_categories=[MetricCategory.LLM_AS_JUDGE_MULTI_TURN],
                )
            ]
        if self.has_metric_category[MetricCategory.LLM_AS_JUDGE]:
            requests[RequestType.GREEDY_UNTIL] += [
                GreedyUntilRequest(
                    task_name=current_task_name,
                    sample_index=document_id_seed,
                    request_index=0,
                    context=context,
                    stop_sequence=self.stop_sequence,
                    generation_size=self.generation_size,
                    generation_grammar=self.generation_grammar,
                    num_samples=1,
                    metric_categories=[MetricCategory.LLM_AS_JUDGE],
                )
            ]

        return requests

    def get_metric_method_from_category(self, metric_category):
        if not self.has_metric_category[metric_category]:
            raise ValueError(f"Requested a metric category {metric_category} absent from the task list.")

        return LightevalTask._get_metric_method_from_category(metric_category)

    @staticmethod
    def _get_metric_method_from_category(metric_category):
        if metric_category == MetricCategory.TARGET_PERPLEXITY:
            return apply_target_perplexity_metric
        if metric_category in [MetricCategory.MULTICHOICE, MetricCategory.MULTICHOICE_PMI]:
            return apply_multichoice_metric
        if metric_category == MetricCategory.MULTICHOICE_ONE_TOKEN:
            return apply_multichoice_metric_one_token
        if metric_category == MetricCategory.PERPLEXITY:
            return apply_perplexity_metric
        if metric_category in [
            MetricCategory.GENERATIVE,
            MetricCategory.GENERATIVE_SAMPLING,
            MetricCategory.GENERATIVE_LOGPROB,
        ]:
            return apply_generative_metric
        if metric_category in [MetricCategory.LLM_AS_JUDGE_MULTI_TURN, MetricCategory.LLM_AS_JUDGE]:
            return apply_llm_as_judge_metric

    def aggregation(self):
        """
        Return a dict with metric name and its aggregation function for all
        metrics
        """
        return Metrics.corpus_level_fns(self.metrics)

    @staticmethod
    def load_datasets(tasks: list["LightevalTask"], dataset_loading_processes: int = 1) -> None:
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
                for task in tasks
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
                        for task in tasks
                    ],
                )

        for task, dataset in zip(tasks, datasets):
            task.dataset = dataset


def create_requests_from_tasks(  # noqa: C901
    task_dict: dict[str, LightevalTask],
    fewshot_dict: dict[str, list[Tuple[int, bool]]],
    num_fewshot_seeds: int,
    lm: TransformersModel,
    max_samples: int | None,
    evaluation_tracker: "EvaluationTracker",
    use_chat_template: bool,
    system_prompt: str | None,
) -> Tuple[dict[RequestType, list[Request]], dict[SampleUid, Doc]]:
    """
    Takes a task dict and a fewshot dict and returns a dict of requests, a dict
    of docs, and a dict of requests origins. The construction of prompts and
    thus the managing of few shots is done here.

    Args:
        task_dict (dict[str, LightevalTask]): A dictionary of tasks.
        fewshot_dict (dict[str, list[Tuple[int, bool]]]): A dictionary of few
            shot examples.
        num_fewshot_seeds (int): number of few shot seeds.
        lm (TransformersModel): language model class that will be used to eventually
            truncate the few shot examples (we need the maximum input size of the
            model)
        max_samples (int): maximum number of samples.
        evaluation_tracker (EvaluationTracker): evaluation tracker.
        use_chat_template (bool): Whether to use the chat template.

    Raises:
        NotImplementedError: If the request type is not implemented for the
            task.

    Returns:
        Tuple[dict[RequestType, list[Request]], dict[SampleUid, Doc]]: A
            tuple containing the requests and the documents.
    """
    docs: dict[SampleUid, Doc] = {}
    requests: dict[RequestType, list[Request]] = collections.defaultdict(list)

    # Filter out tasks that don't have any docs
    task_dict_items = [(name, task) for name, task in task_dict.items() if len(task.eval_docs()) > 0]

    # Get lists of each type of request
    for task_name, task in task_dict_items:
        task_docs = list(task.eval_docs())
        n_samples = min(max_samples, len(task_docs)) if max_samples else len(task_docs)
        evaluation_tracker.task_config_logger.log_num_docs(task_name, len(task_docs), n_samples)

        # logs out the different versions of the tasks for every few shot
        for num_fewshot, _ in fewshot_dict[task_name]:
            cur_task_name = f"{task_name}|{num_fewshot}"
            evaluation_tracker.versions_logger.log(cur_task_name, task.version)

        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(task_docs)

        prompt_manager = PromptManager(lm=lm, task=task)
        seeds = prompt_manager.few_shot_sampler.get_fewshot_seeds(num_fewshot_seeds)

        # We can do several round of fewshots sampling to get some variance information
        for seed in seeds:
            for doc_id in range(n_samples):
                doc_id_seed = f"{doc_id}_{seed}"  # if we do several rounds of few shot sampling we have several seeds
                for num_fewshot, truncate_few_shots in fewshot_dict[task_name]:
                    doc = task_docs[doc_id]
                    doc = prompt_manager.add_context_to_doc(
                        doc,
                        num_fewshot=num_fewshot,
                        seed=seed,
                        sampler=rnd,
                        truncate_few_shots=truncate_few_shots,
                        use_chat_template=use_chat_template,
                        system_prompt=system_prompt,
                    )

                    # Constructing the requests
                    cur_task_name = f"{task_name}|{num_fewshot}"
                    docs[SampleUid(cur_task_name, doc_id_seed)] = doc
                    req_type_reqs_dict = task.construct_requests(doc, doc.ctx, doc_id_seed, cur_task_name)
                    for req_type, reqs in req_type_reqs_dict.items():
                        requests[req_type].extend(reqs)

    return requests, docs
