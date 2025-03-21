# MIT License

# Copyright (c) 2024 Taratra D. RAHARISON and The HuggingFace Team

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
import logging
import os
from typing import Optional

import typer
from typer import Argument, Option
from typing_extensions import Annotated


app = typer.Typer()
CACHE_DIR = os.getenv("HF_HOME")


@app.command()
def inspect(
    tasks: Annotated[str, Argument(help="Id of tasks or path to a text file with a list of tasks")],
    custom_tasks: Annotated[Optional[str], Option(help="Path to a file with custom tasks")] = None,
    num_samples: Annotated[int, Option(help="Number of samples to display")] = 10,
    show_config: Annotated[bool, Option(help="Will display the full task config")] = False,
    cache_dir: Annotated[Optional[str], Option(help="Cache directory used to store datasets and models")] = CACHE_DIR,
):
    """
    Inspect a tasks
    """
    from dataclasses import asdict
    from pprint import pformat

    from rich import print

    from lighteval.tasks.registry import Registry, taskinfo_selector

    registry = Registry(cache_dir=cache_dir, custom_tasks=custom_tasks)

    # Loading task
    task_names_list, _ = taskinfo_selector(tasks, task_registry=registry)
    task_dict = registry.get_task_dict(task_names_list)
    for name, task in task_dict.items():
        print("-" * 10, name, "-" * 10)
        if show_config:
            print("-" * 10, "CONFIG")
            task.cfg.print()
        for ix, sample in enumerate(task.eval_docs()[: int(num_samples)]):
            if ix == 0:
                print("-" * 10, "SAMPLES")
            print(f"-- sample {ix} --")
            print(pformat(asdict(sample), indent=2))


@app.command()
def list(custom_tasks: Annotated[Optional[str], Option(help="Path to a file with custom tasks")] = None):
    """
    List all tasks
    """
    from lighteval.tasks.registry import Registry

    registry = Registry(cache_dir=CACHE_DIR, custom_tasks=custom_tasks)
    registry.print_all_tasks()


@app.command()
def create(template: str, task_name: str, dataset_name: str):
    """
    Create a new task
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Creating task for dataset {dataset_name}")

    with open(template, "r") as f:
        content = f.read()

    content = content.replace("HF_TASK_NAME", task_name)
    content = content.replace("HF_DATASET_NAME", dataset_name)

    with open(f"custom_{task_name}_task.py", "w+") as f:
        f.write(content)

    logger.info(f"Task created in custom_{task_name}_task.py")
