import os
from pathlib import Path
from string import ascii_uppercase

import yaml
from huggingface_hub import hf_hub_download
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import FieldSpec, Sample, hf_dataset
from inspect_ai.scorer import choice, exact, match, model_graded_fact
from inspect_ai.solver import (
    chain_of_thought,
    generate,
    multiple_choice,
    prompt_template,
    system_message,
)


def load_config(yaml_path: str = None) -> dict:
    """Load and parse the YAML configuration file."""
    if yaml_path is None:
        yaml_path = os.getenv("EVAL_YAML", "eval.yaml")

    yaml_path = Path(yaml_path)
    if not yaml_path.is_absolute():
        yaml_path = Path(__file__).parent / yaml_path

    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def record_to_sample(record, field_spec: dict):
    """Convert a dataset record to a Sample based on field_spec."""
    input_text = record[field_spec["input"]]

    # Handle target - convert numeric labels to letters for multiple choice
    target_letter = ascii_uppercase[record[field_spec["target"]]]

    # Get choices if specified
    choices_list = None
    if "choices" in field_spec:
        choices_list = [record[choice_field] for choice_field in field_spec["choices"]]

    sample_kwargs = {
        "input": input_text,
        "target": target_letter,
    }
    if choices_list:
        sample_kwargs["choices"] = choices_list

    return Sample(**sample_kwargs)


def load_dataset(repo_id: str, revision: str = "main", task_config: dict = None, global_config: dict = None):
    """Load dataset based on task configuration."""
    subset = task_config.get("subset")
    split = task_config.get("splits", "test")
    field_spec = task_config["field_spec"]

    # Use custom function if choices are specified (for multiple choice with label conversion)
    if "choices" in field_spec:
        dataset = hf_dataset(
            path=repo_id,
            revision=revision,
            name=subset,
            split=split,
            sample_fields=lambda record: record_to_sample(record, field_spec),
        )
    else:
        # For non-multiple-choice, use FieldSpec
        dataset = hf_dataset(
            path=repo_id,
            revision=revision,
            name=subset,
            split=split,
            sample_fields=FieldSpec(
                input=field_spec["input"],
                target=field_spec["target"],
                **({k: v for k, v in field_spec.items() if k not in ["input", "target"]}),
            ),
        )

    return dataset


def build_solvers(task_config: dict):
    """Build solvers list from task configuration."""
    solvers = []
    solver_names = task_config.get("solvers", [])

    for solver_name in solver_names:
        if solver_name == "prompt_template":
            if "prompt_template" in task_config and task_config["prompt_template"]:
                template = task_config["prompt_template"].strip().strip('"')
                template = template.replace("{{prompt}}", "{prompt}")
                solvers.append(prompt_template(template))
        elif solver_name == "system_message":
            if "system_message" in task_config and task_config["system_message"]:
                sys_msg = task_config["system_message"].strip().strip('"')
                solvers.append(system_message(sys_msg))
        elif solver_name == "chain_of_thought":
            solvers.append(chain_of_thought())
        elif solver_name == "multiple_choice":
            solvers.append(multiple_choice())
        elif solver_name == "generate":
            solvers.append(generate())

    return solvers


def build_scorer(task_config: dict):
    """Build scorer from task configuration."""
    scorer_name = task_config.get("scorers", ["choice"])[0]

    if scorer_name == "choice":
        return choice()
    elif scorer_name == "exact":
        return exact()
    elif scorer_name == "match":
        return match()
    elif scorer_name == "model_graded_fact":
        return model_graded_fact()
    else:
        raise ValueError(f"Unknown scorer: {scorer_name}")


def create_task_from_config(
    repo_id: str, revision: str = "main", task_config: dict = None, global_config: dict = None
):
    """Create an inspect.ai Task from a task configuration."""
    dataset = load_dataset(repo_id, revision, task_config, global_config)
    solvers = build_solvers(task_config)
    scorer = build_scorer(task_config)
    epochs = task_config.get("epochs", 1)
    epochs_reducer = task_config.get("epochs_reducer", "mean")

    return Task(
        dataset=dataset,
        solver=solvers,
        scorer=scorer,
        name=task_config["name"],
        epochs=Epochs(epochs, epochs_reducer),
    )


def create_task_function(repo_id: str, revision: str = "main"):
    """Factory function to create a task function with proper closure."""
    # read yaml from hf filesystem
    yaml_path = hf_hub_download(repo_id=repo_id, filename="eval.yaml", repo_type="dataset", revision=revision)

    with open(yaml_path, "r") as f:
        global_config = yaml.safe_load(f)

    task_config = global_config["tasks"][0]

    @task
    def task_func():
        return create_task_from_config(repo_id, revision, task_config, global_config)

    return task_func
