# Offline evaluation using local data files

If you are prototyping a task based on files that are not yet hosted on the
Hub, you can take advantage of the `hf_data_files` argument to point lighteval
at local JSON/CSV resources. This makes it easy to evaluate datasets that live
in your repo or that are generated on the fly.

Internally, `hf_data_files` is passed directly to the `data_files` parameter of `datasets.load_dataset` ([docs]((https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset))).

See [adding a custom task](adding-a-custom-task) for more information on how to create a custom task.

```python
from pathlib import Path

from lighteval.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def local_prompt(line: dict, task_name: str) -> Doc:
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=line["choices"],
        gold_index=line["answer"]
    )


local_data = Path(__file__).parent / "samples" / "faq.jsonl"

local_task = LightevalTaskConfig(
    name="faq_eval",
    prompt_function=local_prompt,
    hf_repo="json",  # Built-in streaming loader for json/jsonl files
    hf_subset="default",
    hf_data_files=str(local_data),  # Can also be a dict mapping split names to paths
    evaluation_splits=["train"],
    metrics=[Metrics.ACCURACY],
)
```

Once the config is registered in `TASKS_TABLE`, running the task with
`--custom-tasks path/to/your_file.py` will automatically load the local data
files. You can also pass a dictionary to `hf_data_files` (e.g.
`{"train": "train.jsonl", "validation": "val.jsonl"}`) to expose multiple
splits.
