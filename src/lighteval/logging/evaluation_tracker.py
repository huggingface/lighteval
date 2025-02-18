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

import json
import logging
import os
import re
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from datasets.utils.metadata import MetadataConfigs
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, HFSummaryWriter, hf_hub_url

from lighteval.logging.info_loggers import (
    DetailsLogger,
    GeneralConfigLogger,
    MetricsLogger,
    TaskConfigLogger,
    VersionsLogger,
)
from lighteval.utils.imports import NO_TENSORBOARDX_WARN_MSG, is_nanotron_available, is_tensorboardX_available
from lighteval.utils.utils import obj_to_markdown


logger = logging.getLogger(__name__)

if is_nanotron_available():
    from nanotron.config import GeneralArgs  # type: ignore

try:
    from fsspec import url_to_fs
except ImportError:
    from fsspec.core import url_to_fs


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Provides a proper json encoding for the loggers and trackers json dumps.
    Notably manages the json encoding of dataclasses.
    """

    def default(self, o):
        if is_dataclass(o):
            try:
                return asdict(o)  # type: ignore
            except Exception:
                return str(o)
        if callable(o):
            if hasattr(o, "__name__"):
                return o.__name__
            # https://stackoverflow.com/questions/20594193/dynamically-created-method-and-decorator-got-error-functools-partial-object-h
            # partial functions don't have __name__ so we have to unwrap the wrapped function
            elif hasattr(o, "func"):
                return o.func.__name__
        if isinstance(o, torch.dtype):
            return str(o)
        if isinstance(o, Enum):
            return o.name
        return super().default(o)


class EvaluationTracker:
    """Keeps track of the overall evaluation process and relevant information.

    The [`~logging.evaluation_tracker.EvaluationTracker`] contains specific loggers for experiments details
    ([`~logging.evaluation_tracker.DetailsLogger`]), metrics ([`~logging.evaluation_tracker.MetricsLogger`]), task versions
    ([`~logging.evaluation_tracker.VersionsLogger`]) as well as for the general configurations of both the
    specific task ([`~logging.evaluation_tracker.TaskConfigLogger`]) and overall evaluation run
    ([`~logging.evaluation_tracker.GeneralConfigLogger`]).  It compiles the data from these loggers and
    writes it to files, which can be published to the Hugging Face hub if
    requested.

    Args:
        output_dir (`str`): Local folder path where you want results to be saved.
        save_details (`bool`, defaults to True): If True, details are saved to the `output_dir`.
        push_to_hub (`bool`, defaults to False): If True, details are pushed to the hub.
            Results are pushed to `{hub_results_org}/details__{sanitized model_name}` for the model `model_name`, a public dataset,
            if `public` is True else `{hub_results_org}/details__{sanitized model_name}_private`, a private dataset.
        push_to_tensorboard (`bool`, defaults to False): If True, will create and push the results for a tensorboard folder on the hub.
        hub_results_org (`str`, *optional*): The organisation to push the results to.
            See more details about the datasets organisation in [`EvaluationTracker.save`].
        tensorboard_metric_prefix (`str`, defaults to "eval"): Prefix for the metrics in the tensorboard logs.
        public (`bool`, defaults to False): If True, results and details are pushed to public orgs.
        nanotron_run_info ([`~nanotron.config.GeneralArgs`], *optional*): Reference to information about Nanotron models runs.

    **Attributes**:
        - **details_logger** ([`~logging.info_loggers.DetailsLogger`]) -- Logger for experiment details.
        - **metrics_logger** ([`~logging.info_loggers.MetricsLogger`]) -- Logger for experiment metrics.
        - **versions_logger** ([`~logging.info_loggers.VersionsLogger`]) -- Logger for task versions.
        - **general_config_logger** ([`~logging.info_loggers.GeneralConfigLogger`]) -- Logger for general configuration.
        - **task_config_logger** ([`~logging.info_loggers.TaskConfigLogger`]) -- Logger for task configuration.
    """

    def __init__(
        self,
        output_dir: str,
        save_details: bool = True,
        push_to_hub: bool = False,
        push_to_tensorboard: bool = False,
        hub_results_org: str | None = "",
        tensorboard_metric_prefix: str = "eval",
        public: bool = False,
        nanotron_run_info: "GeneralArgs" = None,
    ) -> None:
        """Creates all the necessary loggers for evaluation tracking."""
        self.details_logger = DetailsLogger()
        self.metrics_logger = MetricsLogger()
        self.versions_logger = VersionsLogger()
        self.general_config_logger = GeneralConfigLogger()
        self.task_config_logger = TaskConfigLogger()

        self.api = HfApi()
        self.fs, self.output_dir = url_to_fs(output_dir)

        self.hub_results_org = hub_results_org  # will also contain tensorboard results
        if hub_results_org in ["", None] and any([push_to_hub, push_to_tensorboard]):
            raise Exception(
                "You need to select which org to push to, using `--results_org`, if you want to save information to the hub."
            )

        self.should_push_to_hub = push_to_hub
        self.should_save_details = save_details

        self.should_push_results_to_tensorboard = push_to_tensorboard
        self.tensorboard_repo = f"{hub_results_org}/tensorboard_logs"
        self.tensorboard_metric_prefix = tensorboard_metric_prefix
        self.nanotron_run_info = nanotron_run_info

        self.public = public

    @property
    def results(self):
        config_general = asdict(self.general_config_logger)
        # We remove the config from logging, which contains context/accelerator objects
        config_general.pop("config")
        results = {
            "config_general": config_general,
            "results": self.metrics_logger.metric_aggregated,
            "versions": self.versions_logger.versions,
            "config_tasks": self.task_config_logger.tasks_configs,
            "summary_tasks": self.details_logger.compiled_details,
            "summary_general": asdict(self.details_logger.compiled_details_over_all_tasks),
        }
        return results

    @property
    def details(self):
        return {
            task_name: [asdict(detail) for detail in task_details]
            for task_name, task_details in self.details_logger.details.items()
        }

    def save(self) -> None:
        """Saves the experiment information and results to files, and to the hub if requested."""
        logger.info("Saving experiment tracker")
        date_id = datetime.now().isoformat().replace(":", "-")

        # We first prepare data to save
        config_general = asdict(self.general_config_logger)
        # We remove the config from logging, which contains context/accelerator objects
        config_general.pop("config")

        results_dict = {
            "config_general": config_general,
            "results": self.metrics_logger.metric_aggregated,
            "versions": self.versions_logger.versions,
            "config_tasks": self.task_config_logger.tasks_configs,
            "summary_tasks": self.details_logger.compiled_details,
            "summary_general": asdict(self.details_logger.compiled_details_over_all_tasks),
        }

        # Create the details datasets for later upload
        details_datasets: dict[str, Dataset] = {}
        for task_name, task_details in self.details_logger.details.items():
            # Create a dataset from the dictionary - we force cast to str to avoid formatting problems for nested objects
            dataset = Dataset.from_list([asdict(detail) for detail in task_details])

            # We don't keep 'id' around if it's there
            column_names = dataset.column_names
            if "id" in dataset.column_names:
                column_names = [t for t in dataset.column_names if t != "id"]

            # Sort column names to make it easier later
            dataset = dataset.select_columns(sorted(column_names))
            details_datasets[task_name] = dataset

        # We save results at every case
        self.save_results(date_id, results_dict)

        if self.should_save_details:
            self.save_details(date_id, details_datasets)

        if self.should_push_to_hub:
            self.push_to_hub(
                date_id=date_id,
                details=details_datasets,
                results_dict=results_dict,
            )

        if self.should_push_results_to_tensorboard:
            self.push_to_tensorboard(
                results=self.metrics_logger.metric_aggregated, details=self.details_logger.compiled_details
            )

    def save_results(self, date_id: str, results_dict: dict):
        output_dir_results = Path(self.output_dir) / "results" / self.general_config_logger.model_name
        self.fs.mkdirs(output_dir_results, exist_ok=True)
        output_results_file = output_dir_results / f"results_{date_id}.json"
        logger.info(f"Saving results to {output_results_file}")
        with self.fs.open(output_results_file, "w") as f:
            f.write(json.dumps(results_dict, cls=EnhancedJSONEncoder, indent=2, ensure_ascii=False))

    def _get_details_sub_folder(self, date_id: str):
        output_dir_details = Path(self.output_dir) / "details" / self.general_config_logger.model_name
        if date_id in ["first", "last"]:
            # Get all folders in output_dir_details
            if not self.fs.exists(output_dir_details):
                raise FileNotFoundError(f"Details directory {output_dir_details} does not exist")

            # List all folders and filter out files
            folders = [f["name"] for f in self.fs.listdir(output_dir_details) if f["type"] == "directory"]

            if not folders:
                raise FileNotFoundError(f"No timestamp folders found in {output_dir_details}")

            # Parse timestamps and get first or last
            date_id = max(folders) if date_id == "last" else min(folders)
        return output_dir_details / date_id

    def load_details_datasets(self, date_id: str, task_names: list[str]) -> dict[str, Dataset]:
        output_dir_details_sub_folder = self._get_details_sub_folder(date_id)
        logger.info(f"Loading details from {output_dir_details_sub_folder}")
        date_id = output_dir_details_sub_folder.name  # Overwrite date_id in case of latest
        details_datasets = {}
        for file in self.fs.glob(str(output_dir_details_sub_folder / f"details_*_{date_id}.parquet")):
            task_name = Path(file).stem.replace("details_", "").replace(f"_{date_id}", "")
            if "|".join(task_name.split("|")[:-1]) not in task_names:
                logger.info(f"Skipping {task_name} because it is not in the task_names list")
                continue
            dataset = load_dataset("parquet", data_files=file, split="train")
            details_datasets[task_name] = dataset

        for task_name in task_names:
            if not any(task_name.startswith(task_name) for task_name in details_datasets.keys()):
                raise ValueError(
                    f"Task {task_name} not found in details datasets. Check the tasks to be evaluated or the date_id used to load the details ({date_id})."
                )
        return details_datasets

    def save_details(self, date_id: str, details_datasets: dict[str, Dataset]):
        output_dir_details_sub_folder = self._get_details_sub_folder(date_id)
        self.fs.mkdirs(output_dir_details_sub_folder, exist_ok=True)
        logger.info(f"Saving details to {output_dir_details_sub_folder}")
        for task_name, dataset in details_datasets.items():
            output_file_details = output_dir_details_sub_folder / f"details_{task_name}_{date_id}.parquet"
            with self.fs.open(str(output_file_details), "wb") as f:
                dataset.to_parquet(f)

    def generate_final_dict(self) -> dict:
        """Aggregates and returns all the logger's experiment information in a dictionary.

        This function should be used to gather and display said information at the end of an evaluation run.
        """
        to_dump = {
            "config_general": asdict(self.general_config_logger),
            "results": self.metrics_logger.metric_aggregated,
            "versions": self.versions_logger.versions,
            "config_tasks": self.task_config_logger.tasks_configs,
            "summary_tasks": self.details_logger.compiled_details,
            "summary_general": asdict(self.details_logger.compiled_details_over_all_tasks),
        }

        final_dict = {
            k: {eval_name.replace("|", ":"): eval_score for eval_name, eval_score in v.items()}
            for k, v in to_dump.items()
        }

        return final_dict

    def push_to_hub(
        self,
        date_id: str,
        details: dict[str, Dataset],
        results_dict: dict,
    ) -> None:
        """Pushes the experiment details (all the model predictions for every step) to the hub."""
        sanitized_model_name = self.general_config_logger.model_name.replace("/", "__")

        # "Default" detail names are the public detail names (same as results vs private-results)
        repo_id = f"{self.hub_results_org}/details_{sanitized_model_name}"
        if not self.public:  # if not public, we add `_private`
            repo_id = f"{repo_id}_private"

        fsspec_repo_uri = f"hf://datasets/{repo_id}"

        if not self.api.repo_exists(repo_id):
            self.api.create_repo(repo_id, private=not (self.public), repo_type="dataset", exist_ok=True)
            logger.info(f"Repository {repo_id} not found, creating it.")

        # We upload it both as a json and a parquet file
        result_file_base_name = f"results_{date_id}"
        results_json = json.dumps(results_dict, cls=EnhancedJSONEncoder, indent=2, ensure_ascii=False)
        url = self.api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=BytesIO(results_json.encode("utf-8")),
            path_in_repo=f"{result_file_base_name}.json",
            repo_type="dataset",
        )
        logger.info(f"Uploaded evaluation details to {url}")

        results_dataset = Dataset.from_dict(
            {key: [json.dumps(v, cls=EnhancedJSONEncoder, indent=2)] for key, v in results_dict.items()}
        )
        results_dataset.to_parquet(f"{fsspec_repo_uri}/{result_file_base_name}.parquet")

        for task_name, dataset in details.items():
            output_file_details = Path(date_id) / f"details_{task_name}_{date_id}.parquet"
            dataset.to_parquet(f"{fsspec_repo_uri}/{output_file_details}")

        self.recreate_metadata_card(repo_id)

    def push_results_to_hub(self, repo_id: str, path_in_repo: str, private: bool | None = None):
        repo_id = repo_id if "/" in repo_id else f"{self.hub_results_org}/{repo_id}"
        private = private if private is not None else not self.public
        self.api.create_repo(repo_id, private=private, repo_type="dataset", exist_ok=True)
        results_json = json.dumps(self.results, cls=EnhancedJSONEncoder, indent=2, ensure_ascii=False)
        self.api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=results_json.encode(),
            path_in_repo=path_in_repo,
            repo_type="dataset",
        )

    def push_details_to_hub(self, repo_id: str, path_in_repo: str, private: bool | None = None):
        repo_id = repo_id if "/" in repo_id else f"{self.hub_results_org}/{repo_id}"
        private = private if private is not None else not self.public
        self.api.create_repo(repo_id, private=private, repo_type="dataset", exist_ok=True)
        for task_name, details in self.details:
            details_json = "\n".join([json.dumps(detail) for detail in details])
            self.api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=details_json.encode(),
                path_in_repo=path_in_repo.format(task_name=task_name),
                repo_type="dataset",
            )

    def recreate_metadata_card(self, repo_id: str) -> None:  # noqa: C901
        """Fully updates the details repository metadata card for the currently evaluated model

        Args:
            repo_id (str): Details dataset repository path on the hub (`org/dataset`)
        """
        # Add a nice dataset card and the configuration YAML
        files_in_repo = self.api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        results_files = [f for f in files_in_repo if ".json" in f]
        parquet_files = [f for f in files_in_repo if ".parquet" in f]

        details_file_regex = re.compile(r"details_(?P<task_name>.*?)_(?P<date>\d+-\d+-\d+T.*)\.parquet$")
        multiple_results = len(results_files) > 1

        # Get last eval results date for each task (evals might be non overlapping)
        last_eval_date_results = {}
        for sub_file in parquet_files:
            # We focus on details only
            if "results_" in sub_file:
                continue

            # subfile have this general format:
            # `2023-09-03T10-57-04.203304/details_harness|hendrycksTest-us_foreign_policy|5_2023-09-03T10-57-04.203304.parquet`
            # in the iso date, the `:` are replaced by `-` because windows does not allow `:` in their filenames
            task_name = (
                details_file_regex.match(os.path.basename(sub_file)).group("task_name")  # type: ignore
            )
            # task_name is then equal to `leaderboard|mmlu:us_foreign_policy|5`

            # to be able to parse the filename as iso dates, we need to re-replace the `-` with `:`
            # iso_date[13] = iso_date[16] = ':'
            dir_name = os.path.dirname(sub_file)
            iso_date = ":".join(dir_name.rsplit("-", 2))
            eval_date = datetime.fromisoformat(iso_date)

            last_eval_date_results[task_name] = (
                max(last_eval_date_results[task_name], eval_date) if task_name in last_eval_date_results else eval_date
            )

        max_last_eval_date_results = list(last_eval_date_results.values())[0]
        # Now we convert them in iso-format
        for task in last_eval_date_results:
            if max_last_eval_date_results < last_eval_date_results[task]:
                max_last_eval_date_results = last_eval_date_results[task]
            last_eval_date_results[task] = last_eval_date_results[task].isoformat()
        max_last_eval_date_results = max_last_eval_date_results.isoformat()

        # Add the YAML for the configs
        card_metadata = MetadataConfigs()

        # Add the results config and add the result file as a parquet file
        for sub_file in parquet_files:
            if "results_" in sub_file:
                eval_date = os.path.basename(sub_file).replace("results_", "").replace(".parquet", "")
                sanitized_task = "results"
                sanitized_last_eval_date_results = re.sub(r"[^\w\.]", "_", max_last_eval_date_results)
                repo_file_name = os.path.basename(sub_file)
            else:
                filename = os.path.basename(sub_file)

                task_name_match = details_file_regex.match(filename)  # type: ignore
                if not task_name_match:
                    raise ValueError(f"Could not parse task name from filename: {filename}")
                task_name = task_name_match.group("task_name")
                eval_date = task_name_match.group("date")

                sanitized_task = re.sub(r"\W", "_", task_name)
                sanitized_last_eval_date_results = re.sub(r"[^\w\.]", "_", last_eval_date_results[task_name])
                repo_file_name = os.path.join("**", os.path.basename(sub_file))

            sanitized_eval_date = re.sub(r"[^\w\.]", "_", eval_date)

            if multiple_results:
                if sanitized_task not in card_metadata:
                    card_metadata[sanitized_task] = {
                        "data_files": [{"split": sanitized_eval_date, "path": [repo_file_name]}]
                    }
                else:
                    former_entry = card_metadata[sanitized_task]
                    card_metadata[sanitized_task] = {
                        "data_files": former_entry["data_files"]
                        + [{"split": sanitized_eval_date, "path": [repo_file_name]}]
                    }
            else:
                if sanitized_task in card_metadata:
                    raise ValueError(
                        f"Entry for {sanitized_task} already exists in {former_entry} for repo {repo_id} and file {sub_file}"
                    )
                card_metadata[sanitized_task] = {
                    "data_files": [{"split": sanitized_eval_date, "path": [repo_file_name]}]
                }

            if sanitized_eval_date == sanitized_last_eval_date_results:
                all_entry = card_metadata[sanitized_task]["data_files"]
                card_metadata[sanitized_task] = {
                    "data_files": all_entry + [{"split": "latest", "path": [repo_file_name]}]
                }

            if "results_" in sub_file:
                continue

            # Special case for MMLU with a single split covering it all
            # We add another config with all MMLU splits results together for easy inspection
            SPECIAL_TASKS = [
                "lighteval|mmlu",
                "original|mmlu",
            ]
            for special_task in SPECIAL_TASKS:
                sanitized_special_task = re.sub(r"\W", "_", special_task)
                if sanitized_special_task in sanitized_task:
                    task_info = task_name.split("|")
                    # We have few-shot infos, let's keep them in our special task name
                    if len(task_info) == 3:
                        sanitized_special_task += f"_{task_info[-1]}"
                    elif len(task_info) == 4:
                        sanitized_special_task += f"_{task_info[-2]}_{task_info[-1]}"
                    if sanitized_special_task not in card_metadata:
                        card_metadata[sanitized_special_task] = {
                            "data_files": [{"split": sanitized_eval_date, "path": [repo_file_name]}]
                        }
                    else:
                        former_entry = card_metadata[sanitized_special_task]["data_files"]
                        # Any entry for this split already?
                        try:
                            split_index = next(
                                index
                                for index, dictionary in enumerate(former_entry)
                                if dictionary.get("split", None) == sanitized_eval_date
                            )
                        except StopIteration:
                            split_index = None
                        if split_index is None:
                            card_metadata[sanitized_special_task] = {
                                "data_files": former_entry + [{"split": sanitized_eval_date, "path": [repo_file_name]}]
                            }
                        else:
                            former_entry[split_index]["path"] += [repo_file_name]
                            card_metadata[sanitized_special_task] = {"data_files": former_entry}

                    if sanitized_eval_date == sanitized_last_eval_date_results:
                        former_entry = card_metadata[sanitized_special_task]["data_files"]
                        try:
                            split_index = next(
                                index
                                for index, dictionary in enumerate(former_entry)
                                if dictionary.get("split", None) == "latest"
                            )
                        except StopIteration:
                            split_index = None
                        if split_index is None:
                            card_metadata[sanitized_special_task] = {
                                "data_files": former_entry + [{"split": "latest", "path": [repo_file_name]}]
                            }
                        else:
                            former_entry[split_index]["path"] += [repo_file_name]
                            card_metadata[sanitized_special_task] = {"data_files": former_entry}

        # Cleanup a little the dataset card
        # Get the top results
        last_results_file = [f for f in results_files if max_last_eval_date_results.replace(":", "-") in f][0]
        last_results_file_path = hf_hub_url(repo_id=repo_id, filename=last_results_file, repo_type="dataset")
        f: Dataset = load_dataset("json", data_files=last_results_file_path, split="train")  # type: ignore
        results_dict = f["results"][0]
        new_dictionary = {"all": results_dict}
        new_dictionary.update(results_dict)
        results_string = json.dumps(new_dictionary, indent=4)

        # If we are pushing to the Oppen LLM Leaderboard, we'll store specific data in the model card.
        is_open_llm_leaderboard = repo_id.split("/")[0] == "open-llm-leaderboard"
        if is_open_llm_leaderboard:
            org_string = (
                "on the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)."
            )
            leaderboard_url = "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard"
            point_of_contact = "clementine@hf.co"
        else:
            org_string = ""
            leaderboard_url = None
            point_of_contact = None

        card_data = DatasetCardData(
            dataset_summary=f"Dataset automatically created during the evaluation run of model "
            f"[{self.general_config_logger.model_name}](https://huggingface.co/{self.general_config_logger.model_name})"
            f"{org_string}.\n\n"
            f"The dataset is composed of {len(card_metadata) - 1} configuration, each one coresponding to one of the evaluated task.\n\n"
            f"The dataset has been created from {len(results_files)} run(s). Each run can be found as a specific split in each "
            f'configuration, the split being named using the timestamp of the run.The "train" split is always pointing to the latest results.\n\n'
            f'An additional configuration "results" store all the aggregated results of the run.\n\n'
            f"To load the details from a run, you can for instance do the following:\n"
            f'```python\nfrom datasets import load_dataset\ndata = load_dataset("{repo_id}",\n\t"{sanitized_task}",\n\tsplit="train")\n```\n\n'
            f"## Latest results\n\n"
            f'These are the [latest results from run {max_last_eval_date_results}]({last_results_file_path.replace("/resolve/", "/blob/")})'
            f"(note that their might be results for other tasks in the repos if successive evals didn't cover the same tasks. "
            f'You find each in the results and the "latest" split for each eval):\n\n'
            f"```python\n{results_string}\n```",
            repo_url=f"https://huggingface.co/{self.general_config_logger.model_name}",
            pretty_name=f"Evaluation run of {self.general_config_logger.model_name}",
            leaderboard_url=leaderboard_url,
            point_of_contact=point_of_contact,
        )

        card_metadata.to_dataset_card_data(card_data)
        card = DatasetCard.from_template(
            card_data,
            pretty_name=card_data.pretty_name,
        )
        card.push_to_hub(repo_id, repo_type="dataset")

    def push_to_tensorboard(  # noqa: C901
        self, results: dict[str, dict[str, float]], details: dict[str, DetailsLogger.CompiledDetail]
    ):
        if not is_tensorboardX_available:
            logger.warning(NO_TENSORBOARDX_WARN_MSG)
            return

        if not is_nanotron_available():
            logger.warning("You cannot push results to tensorboard without having nanotron installed. Skipping")
            return

        prefix = self.tensorboard_metric_prefix

        if self.nanotron_run_info is not None:
            global_step = self.nanotron_run_info.step
            run = f"{self.nanotron_run_info.run}_{prefix}"
        else:
            global_step = 0
            run = prefix

        output_dir_tb = Path(self.output_dir) / "tb" / run
        output_dir_tb.mkdir(parents=True, exist_ok=True)

        tb_context = HFSummaryWriter(
            logdir=str(output_dir_tb),
            repo_id=self.tensorboard_repo,
            repo_private=True,
            path_in_repo="tb",
            commit_every=6000,  # Very long time so that we can change our files names and trigger push ourselves (see below)
        )
        bench_averages = {}
        for name, values in results.items():
            splited_name = name.split("|")
            if len(splited_name) == 3:
                _, task_name, _ = splited_name
            else:
                task_name = name
            bench_suite = None
            if ":" in task_name:
                bench_suite = task_name.split(":")[0]  # e.g. MMLU
                logger.info(f"bench_suite {bench_suite} in {task_name}")
                for metric, value in values.items():
                    if "stderr" in metric:
                        continue
                    if bench_suite not in bench_averages:
                        bench_averages[bench_suite] = {}
                    bench_averages[bench_suite][metric] = bench_averages[bench_suite].get(metric, []) + [float(value)]
            logger.info(f"Pushing {task_name} {values} to tensorboard")
            for metric, value in values.items():
                if "stderr" in metric:
                    tb_context.add_scalar(f"stderr_{prefix}/{task_name}/{metric}", value, global_step=global_step)
                elif bench_suite is not None:
                    tb_context.add_scalar(
                        f"{prefix}_{bench_suite}/{task_name}/{metric}", value, global_step=global_step
                    )
                else:
                    tb_context.add_scalar(f"{prefix}/{task_name}/{metric}", value, global_step=global_step)
        # Tasks with subtasks
        for name, values in bench_averages.items():
            for metric, values in values.items():
                logger.info(f"Pushing average {name} {metric} {sum(values) / len(values)} to tensorboard")
                tb_context.add_scalar(f"{prefix}/{name}/{metric}", sum(values) / len(values), global_step=global_step)

        tb_context.add_text("eval_config", obj_to_markdown(results), global_step=global_step)

        for task_name, task_details in details.items():
            tb_context.add_text(
                f"eval_details_{task_name}",
                obj_to_markdown({"0": task_details}),
                global_step=global_step,
            )

        # We are doing parallel evaluations of multiple checkpoints and recording the steps not in order
        # This messes up with tensorboard, so the easiest is to rename files in the order of the checkpoints
        # See: https://github.com/tensorflow/tensorboard/issues/5958
        # But tensorboardX don't let us control the prefix of the files (only the suffix), so we need to do it ourselves before commiting the files

        # tb_context.close()  # flushes the unfinished write operations
        time.sleep(5)
        files = os.listdir(output_dir_tb)
        for file in files:
            os.rename(os.path.join(output_dir_tb, file), os.path.join(output_dir_tb, f"{global_step:07d}_{file}"))

        # Now we can push to the hub
        tb_context.scheduler.trigger()
        logger.info(
            f"Pushed to tensorboard at https://huggingface.co/{self.tensorboard_repo}/{output_dir_tb}/tensorboard"
            f" at global_step {global_step}"
        )
