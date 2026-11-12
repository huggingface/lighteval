# MIT License
#
# Copyright (c) 2026 The HuggingFace Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import csv
import json
import logging
from pathlib import Path

from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.cache_management import SampleCache


logger = logging.getLogger(__name__)


class FileModelConfig(ModelConfig):
    """Configuration for scoring responses that were generated ahead of time and stored in a file.

    The file is either JSON Lines (`.jsonl`/`.json`, one object per line) or CSV (`.csv`), with
    one row per sample:

    - `id`: the sample id, i.e. the index of the row in the task's evaluation split
      (also stored as `doc.id` in lighteval's details files).
    - `response`: the stored model output. In JSON Lines files it can also be a list of
      strings, for metrics that use several samples per document.
    - `task` (optional): the task the row belongs to, e.g. `gsm8k` or `gsm8k|0`. Required
      when the file holds responses for more than one task.

    Attributes:
        model_name (str):
            Name under which the results are reported. Defaults to "file".
        responses_path (str):
            Path to the JSONL or CSV file with the stored responses.
        id_column (str):
            Name of the column holding the sample id. Defaults to "id".
        response_column (str):
            Name of the column holding the response. Defaults to "response".
        task_column (str):
            Name of the column holding the task name. Defaults to "task".

    Example:
        ```python
        config = FileModelConfig(model_name="my-model", responses_path="responses.jsonl")
        ```
    """

    model_name: str = "file"
    responses_path: str
    id_column: str = "id"
    response_column: str = "response"
    task_column: str = "task"


class FileModel(LightevalModel):
    """Returns responses read from a file instead of generating them.

    Only generative tasks can be scored this way: log-likelihood based metrics need the model
    itself. Responses are not written to the sample cache, since the file already plays
    that role and editing it must be picked up on the next run.
    """

    def __init__(self, config: FileModelConfig):
        self.config = config
        self._responses = _load_responses(config)
        self._cache = SampleCache(config)

    @property
    def tokenizer(self):
        return None

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        raise NotImplementedError("Responses loaded from a file have no maximum length.")

    def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        task_names = {doc.task_name for doc in docs}
        if None in self._responses and len(task_names) > 1:
            raise ValueError(
                f"{self.config.responses_path} has rows without a '{self.config.task_column}' column, "
                f"which is ambiguous when evaluating several tasks ({', '.join(sorted(task_names))}). "
                f"Add a '{self.config.task_column}' column to the file."
            )

        model_responses = []
        missing = []
        for doc in docs:
            texts = self._find_response(doc)
            if texts is None:
                missing.append(f"{doc.task_name} (id={doc.id})")
                continue
            model_responses.append(ModelResponse(input=doc.query, text=texts))

        if missing:
            shown = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
            raise ValueError(f"{len(missing)} sample(s) have no response in {self.config.responses_path}: {shown}")

        return model_responses

    def _find_response(self, doc: Doc) -> list[str] | None:
        # The task can be written as its full name (`gsm8k|0`), its name without the
        # few-shot count (`gsm8k`), or left out when the file only covers one task.
        base_name = doc.task_name.rsplit("|", 1)[0]
        for task in (doc.task_name, base_name, None):
            texts = self._responses.get(task, {}).get(doc.id)
            if texts is not None:
                return texts
        return None

    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError(
            "The file backend only supports generative metrics: log-likelihoods cannot be read from stored responses."
        )

    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError(
            "The file backend only supports generative metrics: log-likelihoods cannot be read from stored responses."
        )


def _read_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    with path.open(encoding="utf-8", newline="") as f:
        if suffix == ".csv":
            return list(csv.DictReader(f))
        if suffix in {".jsonl", ".json"}:
            rows = []
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_number} is not valid JSON: {e}") from e
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must be a JSON object, got {type(row).__name__}")
                rows.append(row)
            return rows
    raise ValueError(f"Unsupported responses file {path}: expected a .jsonl, .json or .csv file.")


def _load_responses(config: FileModelConfig) -> dict[str | None, dict[str, list[str]]]:
    """Read the responses file into `{task or None: {sample id: [response, ...]}}`."""
    path = Path(config.responses_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Responses file not found: {path}")

    responses: dict[str | None, dict[str, list[str]]] = {}
    for row_number, row in enumerate(_read_rows(path), start=1):
        for column in (config.id_column, config.response_column):
            if row.get(column) is None:
                raise ValueError(f"Row {row_number} of {path} has no '{column}' value.")

        task = row.get(config.task_column) or None
        sample_id = str(row[config.id_column])
        response = row[config.response_column]
        if isinstance(response, str):
            texts = [response]
        elif isinstance(response, list) and all(isinstance(text, str) for text in response):
            texts = response
        else:
            raise ValueError(
                f"Row {row_number} of {path}: '{config.response_column}' must be a string or a list of strings."
            )

        task_responses = responses.setdefault(task, {})
        if sample_id in task_responses:
            where = f"task '{task}'" if task else "the file"
            raise ValueError(f"Duplicate response for id {sample_id} in {where} ({path}).")
        task_responses[sample_id] = texts

    if not responses:
        raise ValueError(f"Responses file {path} is empty.")

    logger.info(f"Loaded {sum(len(r) for r in responses.values())} stored responses from {path}")
    return responses
