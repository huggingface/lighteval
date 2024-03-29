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
import random
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from datasets import load_dataset

from lighteval.few_shot_manager import FewShotSampler
from lighteval.logging.hierarchical_logger import hlog, hlog_warn
from lighteval.metrics import (
    apply_generative_logprob_metric,
    apply_generative_metric,
    apply_generative_multi_turn_metric,
    apply_multichoice_metric,
    apply_multichoice_metric_one_token,
    apply_perplexity_metric,
    apply_target_perplexity_metric,
)
from lighteval.metrics.metrics import MetricCategory, Metrics
from lighteval.models.base_model import BaseModel
from lighteval.models.model_output import ModelReturn
from lighteval.tasks.requests import (
    Doc,
    GreedyUntilMultiTurnRequest,
    GreedyUntilRequest,
    GreedyUntilWithLogitsRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
    Request,
    RequestType,
    TaskExampleId,
)
from lighteval.utils import as_list

from . import tasks_prompt_formatting


if TYPE_CHECKING:
    from lighteval.logging.evaluation_tracker import EvaluationTracker


@dataclass
class LightevalTaskConfig:
    """Stored configuration of a given [`LightevalTask`].

    Arguments:
        name (str): Short name of the evaluation task.
        suite (list[str]): Evaluation suites to which the task belongs.
        prompt_function (str): Name of the function used to create the [`Doc`] samples from each line of the evaluation dataset.
        hf_repo (str): Path of the hub dataset repository containing the evaluation information.
        hf_subset (str): Subset used for the current task, will be default if none is selected.
        hf_avail_splits (list[str]): All the available splits in the evaluation dataset
        evaluation_splits (list[str]): List of the splits actually used for this evaluation
        few_shots_split (str): Name of the split from which to sample few-shot examples
        few_shots_select (str): Method with which to sample few-shot examples
        generation_size (int): Maximum allowed size of the generation
        metric (list[str]): List of all the metrics for the current task.
        stop_sequence (list[str]): Stop sequence which interrupts the generation for generative metrics.
        original_num_docs (int): Number of documents in the task
        effective_num_docs (int): Number of documents used in a specific evaluation
        truncated_num_docs (bool): Whether less than the total number of documents were used
        output_regex (str)
        frozen (bool)
        trust_dataset (bool): Whether to trust the dataset at execution or not
    """

    name: str
    prompt_function: str
    hf_repo: str
    hf_subset: str
    metric: Tuple[Union[str, Metrics]]
    hf_avail_splits: Optional[Tuple[str]] = None
    evaluation_splits: Optional[Tuple[str]] = None
    few_shots_split: Optional[str] = None
    few_shots_select: Optional[str] = None
    generation_size: int = None
    stop_sequence: Optional[Tuple[str]] = None
    output_regex: Optional[str] = None

    frozen: bool = False
    suite: Optional[Tuple[str]] = None

    original_num_docs: int = -1
    effective_num_docs: int = -1

    trust_dataset: bool = None

    must_remove_duplicate_docs: bool = None

    def as_dict(self):
        return {
            "name": self.name,
            "prompt_function": self.prompt_function,
            "hf_repo": self.hf_repo,
            "hf_subset": self.hf_subset,
            "metric": tuple(str(m) for m in self.metric),
            "hf_avail_splits": self.hf_avail_splits,
            "evaluation_splits": self.evaluation_splits,
            "few_shots_split": self.few_shots_split,
            "few_shots_select": self.few_shots_select,
            "generation_size": self.generation_size,
            "stop_sequence": self.stop_sequence,
            "output_regex": self.output_regex,
            "frozen": self.frozen,
            "suite": self.suite,
        }

    def __post_init__(self):
        if self.suite is None:
            self.suite = ["custom"]
        if self.hf_avail_splits is None:
            self.hf_avail_splits = ["train", "validation", "test"]
        if self.evaluation_splits is None:
            self.evaluation_splits = ["validation"]

        # Convert list to tuple for hashing
        self.metric = tuple(self.metric)
        self.hf_avail_splits = tuple(self.hf_avail_splits) if self.hf_avail_splits is not None else None
        self.evaluation_splits = tuple(self.evaluation_splits) if self.evaluation_splits is not None else None
        self.suite = tuple(self.suite) if self.suite is not None else None
        self.stop_sequence = tuple(self.stop_sequence) if self.stop_sequence is not None else None


class LightevalTask:
    def __init__(  # noqa: C901
        self, name: str, cfg: LightevalTaskConfig, cache_dir: Optional[str] = None, custom_tasks_module: list = None
    ):
        """
        Initialize a LightEval task.

        Args:
            name (str): name of the task.
            cfg (dict): configuration dictionary containing
                task-specific settings (from the task_table.json file).
            cache_dir (Optional[str], optional): directory to cache the
                dataset. Defaults to None.
            custom_tasks_module ([type], optional): A custom module
                containing task-specific functions. Defaults to None.
        """
        self.name = name
        self.VERSION = 0
        self.is_main_process = False
        self.cache_dir = cache_dir
        self._cfg = cfg

        # Dataset info
        self.hf_repo = cfg.hf_repo
        self.hf_subset = cfg.hf_subset
        self.dataset_path = self.hf_repo
        self.dataset_config_name = self.hf_subset
        self.dataset = None  # Delayed download
        self.trust_dataset = cfg.trust_dataset
        hlog(f"{self.dataset_path} {self.dataset_config_name}")
        self._fewshot_docs = None
        self._docs = None

        # Managing splits and few shot
        self.all_available_splits = as_list(cfg.hf_avail_splits)
        if cfg.evaluation_splits is None:
            raise ValueError(f"The evaluation split for task {self.name} is None. Please select a valid split.")

        self.evaluation_split = as_list(cfg.evaluation_splits)
        if cfg.few_shots_split is not None:
            self.fewshot_split = as_list(cfg.few_shots_split)
        else:
            self.fewshot_split = as_list(self.get_first_possible_fewshot_splits())
        self.fewshot_sampler = FewShotSampler(
            few_shots_select=cfg.few_shots_select, few_shots_split=self.fewshot_split
        )

        # Metrics
        self.metrics = as_list(cfg.metric)
        self.suite = as_list(cfg.suite)
        ignored = [metric for metric in self.metrics if Metrics[metric].value.category == MetricCategory.IGNORED]
        if len(ignored) > 0:
            hlog_warn(f"[WARNING] Not implemented yet: ignoring the metric {' ,'.join(ignored)} for task {self.name}.")
        current_categories = [Metrics[metric].value.category for metric in self.metrics]
        self.has_metric_category = {category: (category in current_categories) for category in MetricCategory}

        # Data processing
        # to use once prompt formatting is managed as a module
        if custom_tasks_module is None:
            self.formatter = getattr(tasks_prompt_formatting, cfg.prompt_function)
        else:
            formatter = []
            for module in custom_tasks_module:
                if hasattr(module, cfg.prompt_function):
                    formatter.append(getattr(module, cfg.prompt_function))

            if len(formatter) == 0:  # Default version
                self.formatter = getattr(tasks_prompt_formatting, cfg.prompt_function)
            elif len(formatter) == 1:
                # If we have a prompt in both the module and our tasks_prompt_formatting
                # We take the prompt from the module
                if hasattr(tasks_prompt_formatting, cfg.prompt_function):
                    hlog_warn(
                        f"Be careful you are using custom prompt function {cfg.prompt_function} and not the default one."
                    )
                self.formatter = formatter[0]
            else:
                raise Exception(
                    f"You defined the prompt function {cfg.prompt_function} several times in the different custom modules you are loading."
                )
        self.generation_size = cfg.generation_size
        self.stop_sequence = cfg.stop_sequence
        self.output_regex = cfg.output_regex
        self.must_remove_duplicate_docs = cfg.must_remove_duplicate_docs
        if self.must_remove_duplicate_docs is None:
            self.must_remove_duplicate_docs = False

        # Save options
        self.save_queries: bool = False
        self.logfile_name: Optional[Path] = None
        self.is_main_process: bool = False

    @property
    def cfg(self):
        return self._cfg

    def doc_to_text_without_instructions(self, doc: Doc) -> str:
        """
        Returns the query of the document without the instructions. If the
        document has instructions, it removes them from the query:

        Args:
            doc (Doc): document class, containing the query and the
                instructions.

        Returns:
            str: Query of the document without the instructions.
        """
        if doc.instruction is not None:
            if not doc.query.startswith(doc.instruction):
                raise ValueError(f"Prompt query {doc.query} is not starting with instruction {doc.instruction}")
            return doc.query[len(doc.instruction) :]
        return doc.query

    def doc_to_text_and_instructions(self, doc: Doc) -> Tuple[str, str]:
        """
        Returns a tuple with the query of the document and the instructions.
        If the document has no instructions, the second element of the tuple is
        an empty string.

        Args:
            doc (Doc): document, containing the query and the instructions.

        Returns:
            Tuple[str, str]: A tuple with the query of the document and the
                instructions.
        """
        if doc.instruction is not None:
            if not doc.query.startswith(doc.instruction):
                raise ValueError(f"Prompt query {doc.query} is not starting with instruction {doc.instruction}")
            return (doc.query[len(doc.instruction) :], doc.instruction)
        return (doc.query, "")

    def get_first_possible_fewshot_splits(self, number_of_splits: int = 1) -> list[str]:
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
        possible_fewshot_splits = [k for k in self.all_available_splits if k not in self.evaluation_split]
        stored_splits = []

        # We look at these keys in order (first the training sets, then the validation sets)
        allowed_splits = ["train", "dev", "valid", "default"]
        for allowed_split in allowed_splits:
            # We do a partial match of the allowed splits
            available_splits = [k for k in possible_fewshot_splits if allowed_split in k]
            stored_splits.extend(available_splits)

        if len(stored_splits) > 0:
            return stored_splits[:number_of_splits]

        hlog_warn(f"Careful, the task {self.name} is using evaluation data to build the few shot examples.")
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
            self.dataset = download_dataset_worker((self.dataset_path, self.dataset_config_name, self.trust_dataset))
        splits = as_list(splits)

        docs = []
        for split in splits:
            for item in self.dataset[split]:
                # Some tasks formatting is applied differently when the document is used for fewshot examples
                # vs when it's used for the actual prompt. That's why we store whether we are currently using the
                # doc for a fewshot sample (few_shots=True) or not, which then leads to the creation of a different Doc.
                item["__few_shots"] = few_shots
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
            if self.fewshot_split in [None, [None]]:
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

    def doc_to_target(self, formatted_doc: Doc, few_shot: bool = False) -> str:
        """
        Returns the target of the given document.

        Args:
            formatted_doc (Doc): Formatted document.
            few_shot (bool, optional): Whether the document is used for few
                shot examples. Defaults to False.

        Returns:
            str: Target of the document, which is the correct answer for a document.
        """
        # likely we mostly need one example not all
        return as_list(formatted_doc.get_golds(few_shot=few_shot))[0]

    # Requests
    def get_request_type(self) -> list[RequestType]:
        """
        Returns the request types for the task.

        Returns:
            list[RequestType]: Request types for the task.

        Raises:
            NotImplementedError: If the request type is not implemented for the
                task.
        """
        request_types = []
        if self.has_metric_category[MetricCategory.TARGET_PERPLEXITY]:
            request_types.append(RequestType.LOGLIKELIHOOD)
        if self.has_metric_category[MetricCategory.PERPLEXITY]:
            request_types.append(RequestType.LOGLIKELIHOOD_ROLLING)
        if self.has_metric_category[MetricCategory.GENERATIVE]:
            request_types.append(RequestType.GREEDY_UNTIL)
        if self.has_metric_category[MetricCategory.GENERATIVE_MULTI_TURN]:
            request_types.append(RequestType.GREEDY_UNTIL_MULTI_TURN)
        if self.has_metric_category[MetricCategory.GENERATIVE_LOGPROB]:
            request_types.append(RequestType.GREEDY_UNTIL_WITH_LOGITS)
        if self.has_metric_category[MetricCategory.MULTICHOICE]:
            request_types.append(RequestType.LOGLIKELIHOOD)
        if self.has_metric_category[MetricCategory.MULTICHOICE_ONE_TOKEN]:
            request_types.append(RequestType.LOGLIKELIHOOD_SINGLE_TOKEN)

        if len(request_types) == 0:
            raise NotImplementedError(f"Request type not implemented for task {self.name}")

        return request_types

    def construct_requests(
        self, formatted_doc: Doc, context: str, document_id_seed: str, current_task_name: str
    ) -> List[Request]:
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
        requests = {type: [] for type in RequestType}

        if self.has_metric_category[MetricCategory.TARGET_PERPLEXITY]:
            golds = formatted_doc.get_golds()
            requests[RequestType.LOGLIKELIHOOD] += [
                LoglikelihoodRequest(
                    task_name=current_task_name,
                    example_index=document_id_seed,
                    request_index=i,
                    context=context,
                    choice=gold,
                )
                for i, gold in enumerate(golds)
            ]
        if self.has_metric_category[MetricCategory.PERPLEXITY]:
            requests[RequestType.LOGLIKELIHOOD_ROLLING] += [
                LoglikelihoodRollingRequest(
                    task_name=current_task_name, example_index=document_id_seed, request_index=0, context=context
                )
            ]
        if self.has_metric_category[MetricCategory.GENERATIVE]:
            requests[RequestType.GREEDY_UNTIL] += [
                GreedyUntilRequest(
                    task_name=current_task_name,
                    example_index=document_id_seed,
                    request_index=0,
                    context=context,
                    stop_sequence=self.stop_sequence,
                    generation_size=self.generation_size,
                )
            ]
        if self.has_metric_category[MetricCategory.GENERATIVE_LOGPROB]:
            requests[RequestType.GREEDY_UNTIL_WITH_LOGITS] += [
                GreedyUntilWithLogitsRequest(
                    task_name=current_task_name,
                    example_index=document_id_seed,
                    request_index=0,
                    context=context,
                    stop_sequence=self.stop_sequence,
                    generation_size=self.generation_size,
                )
            ]
        if self.has_metric_category[MetricCategory.MULTICHOICE]:
            requests[RequestType.LOGLIKELIHOOD] += [
                LoglikelihoodRequest(
                    task_name=current_task_name,
                    example_index=document_id_seed,
                    request_index=i,
                    context=context,
                    choice=choice,
                )
                for i, choice in enumerate(formatted_doc.choices)
            ]
        if self.has_metric_category[MetricCategory.MULTICHOICE_ONE_TOKEN]:
            requests[RequestType.LOGLIKELIHOOD_SINGLE_TOKEN] += [
                LoglikelihoodSingleTokenRequest(
                    task_name=current_task_name,
                    example_index=document_id_seed,
                    request_index=0,
                    context=context,
                    choices=formatted_doc.choices,
                )
            ]
        if self.has_metric_category[MetricCategory.GENERATIVE_MULTI_TURN]:
            requests[RequestType.GREEDY_UNTIL_MULTI_TURN] += [
                GreedyUntilMultiTurnRequest(
                    task_name=current_task_name,
                    example_index=document_id_seed,
                    request_index=0,
                    context=context,
                    stop_sequence=self.stop_sequence,
                    generation_size=self.generation_size,
                )
            ]

        return requests

    def process_results(self, formatted_doc: Doc, results: list[ModelReturn]) -> dict[str, float]:
        """
        Processes the results of the task, and stores them in the output dict.

        Args:
            formatted_doc (Doc): formatted document of the task.
            results (list[ModelReturn]): results of the task, returned by the model class after evaluation.

        Returns:
            dict[str, float]: output dictionary containing the results of the task.
        """
        # Metrics management is done in metrics.__init__
        outputs = {}
        if self.has_metric_category[MetricCategory.TARGET_PERPLEXITY]:
            results, cur_outputs = apply_target_perplexity_metric(
                results=results, formatted_doc=formatted_doc, metrics=self.metrics
            )
            outputs.update(cur_outputs)
        if self.has_metric_category[MetricCategory.PERPLEXITY]:
            results, cur_outputs = apply_perplexity_metric(
                results=results, formatted_doc=formatted_doc, metrics=self.metrics
            )
            outputs.update(cur_outputs)
        if self.has_metric_category[MetricCategory.GENERATIVE]:
            results, cur_outputs = apply_generative_metric(
                results=results, formatted_doc=formatted_doc, metrics=self.metrics, output_regex=self.output_regex
            )
            outputs.update(cur_outputs)
        if self.has_metric_category[MetricCategory.GENERATIVE_LOGPROB]:
            results, cur_outputs = apply_generative_logprob_metric(
                results=results, formatted_doc=formatted_doc, metrics=self.metrics
            )
            outputs.update(cur_outputs)
        if self.has_metric_category[MetricCategory.MULTICHOICE]:
            results, cur_outputs = apply_multichoice_metric(
                results=results, formatted_doc=formatted_doc, metrics=self.metrics
            )
            outputs.update(cur_outputs)
        if self.has_metric_category[MetricCategory.MULTICHOICE_ONE_TOKEN]:
            results, cur_outputs = apply_multichoice_metric_one_token(
                results=results, formatted_doc=formatted_doc, metrics=self.metrics
            )
            outputs.update(cur_outputs)
        if self.has_metric_category[MetricCategory.GENERATIVE_MULTI_TURN]:
            results, cur_outputs = apply_generative_multi_turn_metric(
                results=results, formatted_doc=formatted_doc, metrics=self.metrics
            )
            outputs.update(cur_outputs)

        return outputs

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
                download_dataset_worker((task.dataset_path, task.dataset_config_name, task.trust_dataset))
                for task in tasks
            ]
        else:
            with Pool(processes=dataset_loading_processes) as pool:
                datasets = pool.map(
                    download_dataset_worker,
                    [(task.dataset_path, task.dataset_config_name, task.trust_dataset) for task in tasks],
                )

        for task, dataset in zip(tasks, datasets):
            task.dataset = dataset


def download_dataset_worker(args):
    """
    Worker function to download a dataset from the HuggingFace Hub.
    Used for parallel dataset loading.
    """
    dataset_path, dataset_config_name, trust_dataset = args
    dataset = load_dataset(
        path=dataset_path,
        name=dataset_config_name,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        trust_remote_code=trust_dataset,
    )
    return dataset


def create_requests_from_tasks(  # noqa: C901
    task_dict: dict[str, LightevalTask],
    fewshot_dict: dict[str, list[Tuple[int, bool]]],
    num_fewshot_seeds: int,
    lm: BaseModel,
    max_samples: int,
    evaluation_tracker: "EvaluationTracker",
    use_chat_template: bool,
    system_prompt: str,
) -> Tuple[dict[RequestType, list[Request]], dict[TaskExampleId, Doc]]:
    """
    Takes a task dict and a fewshot dict and returns a dict of requests, a dict
    of docs, and a dict of requests origins.  The construction of prompts and
    thus the managing of few shots is done here.

    Args:
        task_dict (dict[str, LightevalTask]): A dictionary of tasks.
        fewshot_dict (dict[str, list[Tuple[int, bool]]]): A dictionary of few
            shot examples.
        num_fewshot_seeds (int): number of few shot seeds.
        lm (BaseModel): language model class that will be used to eventually
            truncate the few shot examples (we need the maximum input size of the
            model)
        max_samples (int): maximum number of samples.
        evaluation_tracker (EvaluationTracker): evaluation tracker.
        use_chat_template (bool): Whether to use the chat template.

    Raises:
        NotImplementedError: If the request type is not implemented for the
            task.

    Returns:
        Tuple[dict[RequestType, list[Request]], dict[TaskExampleId, Doc]]: A
            tuple containing the requests and the documents.
    """
    docs: dict[TaskExampleId, Doc] = {}
    requests: dict[RequestType, list[Request]] = collections.defaultdict(list)

    # Filter out tasks that don't have any docs
    task_dict_items = [(name, task) for name, task in task_dict.items() if len(task.eval_docs()) > 0]

    # Get lists of each type of request
    for task_name, task in task_dict_items:
        req_types = task.get_request_type()
        task_docs = list(task.eval_docs())
        n_samples = min(max_samples, len(task_docs)) if max_samples else len(task_docs)
        evaluation_tracker.task_config_logger.log_num_docs(task_name, len(task_docs), n_samples)

        # logs out the diferent versions of the tasks for every few shot
        for num_fewshot, _ in fewshot_dict[task_name]:
            cur_task_name = f"{task_name}|{num_fewshot}"
            evaluation_tracker.versions_logger.log(cur_task_name, task.VERSION)

        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(task_docs)

        seeds = task.fewshot_sampler.get_fewshot_seeds(num_fewshot_seeds)

        # We can do several round of fewshots sampling to get some variance informations
        for seed in seeds:
            for doc_id in range(n_samples):
                doc_id_seed = f"{doc_id}_{seed}"  # if we do several rounds of few shot sampling we have several seeds
                for num_fewshot, truncate_few_shots in fewshot_dict[task_name]:
                    # @clefourrier this mechanism does not work if we have the same task n times with different few shot numbers
                    # to fix!!
                    cur_task_name = f"{task_name}|{num_fewshot}"
                    doc = task_docs[doc_id]
                    is_multi_turn = doc.specific is not None and len(doc.specific.get("multi_turn_queries", [])) > 0

                    if is_multi_turn:
                        ctx, num_effective_few_shots = task.fewshot_sampler.create_multi_turn_contexts(
                            doc, use_chat_template, system_prompt, lm.tokenizer
                        )
                        doc.specific["multi_turn_queries_context"] = ctx
                    else:
                        ctx, num_effective_few_shots = task.fewshot_sampler.fewshot_context(
                            task=task,
                            doc=doc,
                            num_fewshot=num_fewshot,
                            seed=seed,
                            truncate_few_shots=truncate_few_shots,
                            max_model_length=lm.max_length,
                            sampler=rnd,
                            tokenizer=lm.tokenizer,
                            use_chat_template=use_chat_template,
                            system_prompt=system_prompt,
                        )

                    doc.num_effective_few_shots = num_effective_few_shots
                    doc.num_asked_few_shots = num_fewshot
                    doc.ctx = ctx

                    # Constructing the requests
                    docs[TaskExampleId(cur_task_name, doc_id_seed)] = doc
                    reqs = task.construct_requests(doc, ctx, doc_id_seed, cur_task_name)
                    for req_type in req_types:
                        requests[req_type].extend(reqs[req_type])

    return requests, docs
