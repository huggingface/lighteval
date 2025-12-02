from importlib import import_module
from string import ascii_uppercase

import yaml
from huggingface_hub import hf_hub_download
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import FieldSpec, Sample, hf_dataset


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


def load_dataset(repo_id: str, revision: str = "main", task_config: dict = None):
    """Load dataset based on task configuration."""
    subset = task_config.get("subset", "default")
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
                metadata=field_spec.get("metadata", []),
            ),
        )

    return dataset


def build_solvers(task_config: dict):
    """
    Build a list of solvers from the task configuration.

    task_config example:

    ```yaml
    solvers:
      - name: prompt_template
        args:
          template: >
            You are a helpful assistant.
            {prompt}
      - name: generate
        args:
          cache: true
    ```


    """
    solvers = []
    solver_configs = task_config.get("solvers", [])
    solver_module = import_module("inspect_ai.solver")

    for solver_config in solver_configs:
        solver_name = solver_config["name"]

        if not hasattr(solver_module, solver_name):
            raise ValueError(f"Unknown solver: {solver_name}")

        solver_fn = getattr(solver_module, solver_name)
        solvers.append(solver_fn(**solver_config.get("args", {})))

    return solvers


def build_scorer(task_config: dict):
    """
    Build a scorer from the task configuration.
    task_config example:

    ```yaml
    scorers:
      - name: model_graded_fact
        args:
            template: |
                grade this,

                question:
                    {question}
                criterion:
                    {criterion}
                answer:
                    {answer}
    ```
    """
    scorers = []
    scorer_configs = task_config.get("scorers", [])
    scorer_module = import_module("inspect_ai.scorer")

    for scorer_config in scorer_configs:
        scorer_name = scorer_config["name"]

        if not hasattr(scorer_module, scorer_name):
            raise ValueError(f"Unknown scorer: {scorer_name}")

        scorer_fn = getattr(scorer_module, scorer_name)
        scorers.append(scorer_fn(**scorer_config.get("args", {})))

    return scorers


@task
def create_task_from_config(repo_id: str, revision: str = "main", task_config: dict = None):
    """Create an inspect.ai Task from a task configuration."""
    dataset = load_dataset(repo_id, revision, task_config)
    solvers = build_solvers(task_config)
    scorers = build_scorer(task_config)
    epochs = task_config.get("epochs", 1)
    epochs_reducer = task_config.get("epochs_reducer", "mean")

    return Task(
        dataset=dataset,
        solver=solvers,
        scorer=scorers,
        name=task_config["name"],
        epochs=Epochs(epochs, epochs_reducer),
    )


def create_task_function(repo_id: str, revision: str = "main") -> list:
    """Factory function to create a task function with proper closure."""
    # read yaml from hf filesystem
    yaml_path = hf_hub_download(repo_id=repo_id, filename="eval.yaml", repo_type="dataset", revision=revision)

    with open(yaml_path, "r") as f:
        global_config = yaml.safe_load(f)

    task_configs = global_config["tasks"]

    tasks = []
    for task_config in task_configs:
        tasks.append(create_task_from_config(repo_id, revision, task_config))

    return tasks
