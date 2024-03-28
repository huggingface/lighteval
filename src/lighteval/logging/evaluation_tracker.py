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

import copy
import json
import os
import re
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

from datasets import Dataset, load_dataset
from datasets.utils.metadata import MetadataConfigs
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, HFSummaryWriter, hf_hub_url

from lighteval.logging.hierarchical_logger import hlog, hlog_warn
from lighteval.logging.info_loggers import (
    DetailsLogger,
    GeneralConfigLogger,
    MetricsLogger,
    TaskConfigLogger,
    VersionsLogger,
)
from lighteval.utils import is_nanotron_available, obj_to_markdown


if is_nanotron_available():
    from nanotron.config import Config


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Provides a proper json encoding for the loggers and trackers json dumps.
    Notably manages the json encoding of dataclasses.
    """

    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


class EvaluationTracker:
    """
    Keeps track of the overall evaluation process and relevant informations.

    The [`EvaluationTracker`] contains specific loggers for experiments details
    ([`DetailsLogger`]), metrics ([`MetricsLogger`]), task versions
    ([`VersionsLogger`]) as well as for the general configurations of both the
    specific task ([`TaskConfigLogger`]) and overall evaluation run
    ([`GeneralConfigLogger`]).  It compiles the data from these loggers and
    writes it to files, which can be published to the Hugging Face hub if
    requested.
    """

    details_logger: DetailsLogger
    metrics_logger: MetricsLogger
    versions_logger: VersionsLogger
    general_config_logger: GeneralConfigLogger
    task_config_logger: TaskConfigLogger
    hub_results_org: str

    def __init__(self, hub_results_org: str = "", token: str = "") -> None:
        """
        Creates all the necessary loggers for evaluation tracking.

        Args:
            hub_results_org (str): The organisation to push the results to. See
                more details about the datasets organisation in
                [`EvaluationTracker.save`]
            token (str): Token to use when pushing to the hub. This token should
                have write access to `hub_results_org`.
        """
        self.details_logger = DetailsLogger()
        self.metrics_logger = MetricsLogger()
        self.versions_logger = VersionsLogger()
        self.general_config_logger = GeneralConfigLogger()
        self.task_config_logger = TaskConfigLogger()
        self.hub_results_org = hub_results_org
        self.hub_results_repo = f"{hub_results_org}/results"
        self.hub_private_results_repo = f"{hub_results_org}/private-results"
        self.api = HfApi(token=token)

    def save(
        self,
        output_dir: str,
        push_results_to_hub: bool,
        push_details_to_hub: bool,
        public: bool,
        push_results_to_tensorboard: bool = False,
    ) -> None:
        """Saves the experiment information and results to files, and to the hub if requested.

        Note:
            In case of save failure, this function will only print a warning, with the error message.

        Args:
            output_dir (str): Local folder path where you want results to be saved
            push_results_to_hub (bool): If True, results are pushed to the hub.
                Results will be pushed either to `{hub_results_org}/results`, a public dataset, if `public` is True else to `{hub_results_org}/private-results`, a private dataset.
            push_details_to_hub (bool): If True, details are pushed to the hub.
                Results are pushed to `{hub_results_org}/details__{sanitized model_name}` for the model `model_name`, a public dataset,
                if `public` is True else `{hub_results_org}/details__{sanitized model_name}_private`, a private dataset.
            public (bool): If True, results and details are pushed in private orgs

        """
        hlog("Saving experiment tracker")
        # try:
        date_id = datetime.now().isoformat().replace(":", "-")

        output_dir_results = Path(output_dir) / "results" / self.general_config_logger.model_name
        output_dir_details = Path(output_dir) / "details" / self.general_config_logger.model_name
        output_dir_details_sub_folder = output_dir_details / date_id
        output_dir_results.mkdir(parents=True, exist_ok=True)
        output_dir_details_sub_folder.mkdir(parents=True, exist_ok=True)

        output_results_file = output_dir_results / f"results_{date_id}.json"
        output_results_in_details_file = output_dir_details / f"results_{date_id}.json"

        hlog(f"Saving results to {output_results_file} and {output_results_in_details_file}")

        config_general = copy.deepcopy(self.general_config_logger)
        config_general.config = (
            config_general.config.as_dict() if is_dataclass(config_general.config) else config_general.config
        )
        config_general = asdict(config_general)

        to_dump = {
            "config_general": config_general,
            "results": self.metrics_logger.metric_aggregated,
            "versions": self.versions_logger.versions,
            "config_tasks": self.task_config_logger.tasks_configs,
            "summary_tasks": self.details_logger.compiled_details,
            "summary_general": asdict(self.details_logger.compiled_details_over_all_tasks),
        }
        dumped = json.dumps(to_dump, cls=EnhancedJSONEncoder, indent=2)

        with open(output_results_file, "w") as f:
            f.write(dumped)

        with open(output_results_in_details_file, "w") as f:
            f.write(dumped)

        for task_name, task_details in self.details_logger.details.items():
            output_file_details = output_dir_details_sub_folder / f"details_{task_name}_{date_id}.parquet"
            # Create a dataset from the dictionary
            try:
                dataset = Dataset.from_list([asdict(detail) for detail in task_details])
            except Exception:
                # We force cast to str to avoid formatting problems for nested objects
                dataset = Dataset.from_list(
                    [{k: str(v) for k, v in asdict(detail).items()} for detail in task_details]
                )

            # We don't keep 'id' around if it's there
            column_names = dataset.column_names
            if "id" in dataset.column_names:
                column_names = [t for t in dataset.column_names if t != "id"]

            # Sort column names to make it easier later
            dataset = dataset.select_columns(sorted(column_names))
            # Save the dataset to a Parquet file
            dataset.to_parquet(output_file_details.as_posix())

        if push_results_to_hub:
            self.api.upload_folder(
                repo_id=self.hub_results_repo if public else self.hub_private_results_repo,
                folder_path=output_dir_results,
                path_in_repo=self.general_config_logger.model_name,
                repo_type="dataset",
                commit_message=f"Updating model {self.general_config_logger.model_name}",
            )

        if push_details_to_hub:
            self.details_to_hub(
                model_name=self.general_config_logger.model_name,
                results_file_path=output_results_in_details_file,
                details_folder_path=output_dir_details_sub_folder,
                push_as_public=public,
            )

        if push_results_to_tensorboard:
            self.push_results_to_tensorboard(
                results=self.metrics_logger.metric_aggregated, details=self.details_logger.details
            )
        # except Exception as e:
        #     hlog("WARNING: Could not save results")
        #     hlog(repr(e))

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

    def details_to_hub(
        self,
        model_name: str,
        results_file_path: Path | str,
        details_folder_path: Path | str,
        push_as_public: bool = False,
    ) -> None:
        """Pushes the experiment details (all the model predictions for every step) to the hub.

        Args:
            model_name (str): Name of the currently evaluated model
            results_file_path (str or Path): Local path of the current's experiment aggregated results individual file
            details_folder_path (str or Path): Local path of the current's experiment details folder.
                The details folder (created by [`EvaluationTracker.save`]) should contain one parquet file per task used during the evaluation run of the current model.
            push_as_public (bool, optional): If True, the results will be pushed publicly, else the datasets will be private.

        """
        results_file_path = str(results_file_path)
        details_folder_path = str(details_folder_path)

        sanitized_model_name = model_name.replace("/", "__")

        # "Default" detail names are the public detail names (same as results vs private-results)
        repo_id = f"{self.hub_results_org}/details_{sanitized_model_name}"
        if not push_as_public:  # if not public, we add `_private`
            repo_id = f"{repo_id}_private"

        sub_folder_path = os.path.basename(results_file_path).replace(".json", "").replace("results_", "")

        paths_to_check = [os.path.basename(results_file_path)]
        try:
            checked_paths = list(self.api.get_paths_info(repo_id=repo_id, paths=paths_to_check, repo_type="dataset"))
        except Exception:
            checked_paths = []

        if len(checked_paths) == 0:
            hlog(f"Repo {repo_id} not found for {results_file_path}. Creating it.")
            self.api.create_repo(repo_id, private=not (push_as_public), repo_type="dataset", exist_ok=True)

        # Create parquet version of results file as well
        results = load_dataset("json", data_files=results_file_path)
        parquet_name = os.path.basename(results_file_path).replace(".json", ".parquet")
        parquet_local_path = os.path.join(os.path.dirname(results_file_path), parquet_name)
        results["train"].to_parquet(parquet_local_path)

        # Upload results file (json and parquet) and folder
        self.api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=results_file_path,
            path_in_repo=os.path.basename(results_file_path),
            repo_type="dataset",
        )
        self.api.upload_file(
            repo_id=repo_id, path_or_fileobj=parquet_local_path, path_in_repo=parquet_name, repo_type="dataset"
        )
        self.api.upload_folder(
            repo_id=repo_id, folder_path=details_folder_path, path_in_repo=sub_folder_path, repo_type="dataset"
        )

        self.recreate_metadata_card(repo_id, model_name)

    def recreate_metadata_card(self, repo_id: str, model_name: str = None) -> None:  # noqa: C901
        """Fully updates the details repository metadata card for the currently evaluated model

        Args:
            repo_id (str): Details dataset repository path on the hub (`org/dataset`)
            model_name (str): Name of the currently evaluated model.

        """
        # Add a nice dataset card and the configuration YAML
        files_in_repo = self.api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        results_files = [f for f in files_in_repo if ".json" in f]
        parquet_results_files = [f for f in files_in_repo if ".parquet" in f and "results_" in f]
        parquet_files = [f for f in files_in_repo if ".parquet" in f and "results_" not in f]
        multiple_results = len(results_files) > 1

        # Get last eval results date for each task (evals might be non overlapping)
        last_eval_date_results = {}
        for sub_file in parquet_files:
            # subfile have this general format:
            # `2023-09-03T10-57-04.203304/details_harness|hendrycksTest-us_foreign_policy|5_2023-09-03T10-57-04.203304.parquet`
            # in the iso date, the `:` are replaced by `-` because windows does not allow `:` in their filenames

            task_name = os.path.basename(sub_file).replace("details_", "").split("_2023")[0].split("_2024")[0]
            # task_name is then equal to `leaderboard|mmlu:us_foreign_policy|5`

            iso_date = os.path.dirname(sub_file)
            # to be able to parse the filename as iso dates, we need to re-replace the `-` with `:`
            # iso_date[13] = iso_date[16] = ':'
            iso_date = iso_date[:13] + ":" + iso_date[14:16] + ":" + iso_date[17:]

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
        for sub_file in parquet_results_files:
            eval_date = os.path.basename(sub_file).replace("results_", "").replace(".parquet", "")
            sanitized_eval_date = re.sub(r"[^\w\.]", "_", eval_date)
            sanitized_last_eval_date_results = re.sub(r"[^\w\.]", "_", max_last_eval_date_results)

            repo_file_name = os.path.basename(sub_file)

            if multiple_results:
                if "results" not in card_metadata:
                    card_metadata["results"] = {
                        "data_files": [{"split": sanitized_eval_date, "path": [repo_file_name]}]
                    }
                else:
                    former_entry = card_metadata["results"]
                    card_metadata["results"] = {
                        "data_files": former_entry["data_files"]
                        + [{"split": sanitized_eval_date, "path": [repo_file_name]}]
                    }
            else:
                if "results" in card_metadata:
                    raise ValueError(
                        f"Entry for results already exists in {former_entry} for repo {repo_id} and file {sub_file}"
                    )
                card_metadata["results"] = {"data_files": [{"split": sanitized_eval_date, "path": [repo_file_name]}]}

            if sanitized_eval_date == sanitized_last_eval_date_results:
                all_entry = card_metadata["results"]["data_files"]
                card_metadata["results"] = {"data_files": all_entry + [{"split": "latest", "path": [repo_file_name]}]}

        # Add the tasks details configs
        for sub_file in parquet_files:
            task_name = os.path.basename(sub_file).replace("details_", "").split("_2023")[0].split("_2024")[0]
            sanitized_task = re.sub(r"\W", "_", task_name)
            eval_date = os.path.dirname(sub_file)
            sanitized_eval_date = re.sub(r"[^\w\.]", "_", eval_date)
            repo_file_name = os.path.join("**", os.path.basename(sub_file))
            sanitized_last_eval_date_results = re.sub(r"[^\w\.]", "_", last_eval_date_results[task_name])

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
        f = load_dataset("json", data_files=last_results_file_path, split="train")
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
            f"[{model_name}](https://huggingface.co/{model_name})"
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
            repo_url=f"https://huggingface.co/{model_name}",
            pretty_name=f"Evaluation run of {model_name}",
            leaderboard_url=leaderboard_url,
            point_of_contact=point_of_contact,
        )

        card_metadata.to_dataset_card_data(card_data)
        card = DatasetCard.from_template(
            card_data,
            pretty_name=card_data.pretty_name,
        )
        card.push_to_hub(repo_id, repo_type="dataset")

    def push_results_to_tensorboard(  # noqa: C901
        self, results: dict[str, dict[str, float]], details: dict[str, DetailsLogger.CompiledDetail]
    ):
        if not is_nanotron_available():
            hlog_warn("You cannot push results to tensorboard with having nanotron installed. Skipping")
            return
        config: Config = self.general_config_logger.config
        lighteval_config = config.lighteval
        try:
            global_step = config.general.step
        except ValueError:
            global_step = 0
        if config.lighteval.logging.tensorboard_metric_prefix is not None:
            prefix = config.lighteval.logging.tensorboard_metric_prefix
        else:
            prefix = "eval"
        output_dir_tb = Path(lighteval_config.logging.local_output_path) / "tb" / (config.general.run + "_" + prefix)
        output_dir_tb.mkdir(parents=True, exist_ok=True)
        tb_context = HFSummaryWriter(
            logdir=str(output_dir_tb),
            repo_id=lighteval_config.logging.hub_repo_tensorboard,
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
                hlog(f"bench_suite {bench_suite} in {task_name}")
                for metric, value in values.items():
                    if "stderr" in metric:
                        continue
                    if bench_suite not in bench_averages:
                        bench_averages[bench_suite] = {}
                    bench_averages[bench_suite][metric] = bench_averages[bench_suite].get(metric, []) + [float(value)]
            hlog(f"Pushing {task_name} {values} to tensorboard")
            for metric, value in values.items():
                if "stderr" in metric:
                    tb_context.add_scalar(f"stderr_{prefix}/{task_name}/{metric}", value, global_step=global_step)
                elif bench_suite is not None:
                    tb_context.add_scalar(
                        f"{prefix}_{bench_suite}/{task_name}/{metric}", value, global_step=global_step
                    )
                else:
                    tb_context.add_scalar(f"{prefix}/{task_name}/{metric}", value, global_step=global_step)
        # e.g. MMLU
        for name, values in bench_averages.items():
            for metric, values in values.items():
                hlog(f"Pushing average {name} {metric} {sum(values) / len(values)} to tensorboard")
                tb_context.add_scalar(f"{prefix}/{name}/{metric}", sum(values) / len(values), global_step=global_step)

        tb_context.add_text("eval_config", obj_to_markdown(results), global_step=global_step)
        # tb_context.add_text("eval_sizes", obj_to_markdown(sizes), global_step=global_step)

        for task_name, task_details in details.items():
            tb_context.add_text(
                f"eval_details_{task_name}",
                obj_to_markdown({"0": task_details[0], "1": task_details[1] if len(task_details) > 1 else {}}),
                global_step=global_step,
            )

        # We are doing parallel evaluations of multiple checkpoints and recording the steps not in order
        # This messes up with tensorboard, so the easiest is to rename files in the order of the checkpoints
        # See: https://github.com/tensorflow/tensorboard/issues/5958
        # But tensorboardX don't let us control the prefix of the files (only the suffix), so we need to do it ourselves before commiting the files

        tb_context.close()  # flushes the unfinished write operations
        time.sleep(5)
        files = os.listdir(output_dir_tb)
        for file in files:
            os.rename(os.path.join(output_dir_tb, file), os.path.join(output_dir_tb, f"{global_step:07d}_{file}"))

        # Now we can push to the hub
        tb_context.scheduler.trigger()
        hlog(
            f"Pushed to tensorboard at https://huggingface.co/tensorboard/{lighteval_config.logging.hub_repo_tensorboard}/"
            f" at {output_dir_tb} and global_step {global_step}"
        )
        # except Exception as e:
        #     logger.warning(f"Could not push to tensorboard\n{e}")
