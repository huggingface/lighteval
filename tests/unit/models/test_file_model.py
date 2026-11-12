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

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lighteval.models.file.file_model import FileModel, FileModelConfig
from lighteval.models.model_loader import load_model
from lighteval.tasks.requests import Doc


def _doc(doc_id: str, task_name: str = "qa|0") -> Doc:
    return Doc(query=f"question {doc_id}", choices=["answer"], gold_index=0, id=doc_id, task_name=task_name)


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def _model(tmp_path: Path, responses_path: str, **kwargs) -> FileModel:
    config = FileModelConfig(responses_path=responses_path, cache_dir=str(tmp_path / "cache"), **kwargs)
    return FileModel(config)


def test_jsonl_responses_follow_doc_order(tmp_path):
    path = _write_jsonl(
        tmp_path / "responses.jsonl",
        [{"id": 0, "response": "zero"}, {"id": "1", "response": ["one", "uno"]}],
    )
    model = _model(tmp_path, path)

    responses = model.greedy_until([_doc("1"), _doc("0")])

    assert [r.text for r in responses] == [["one", "uno"], ["zero"]]
    assert [r.input for r in responses] == ["question 1", "question 0"]


def test_csv_responses_with_custom_columns(tmp_path):
    path = tmp_path / "responses.csv"
    path.write_text('sample,output\n0,"first, with a comma"\n1,second\n', encoding="utf-8")
    model = _model(tmp_path, str(path), id_column="sample", response_column="output")

    responses = model.greedy_until([_doc("0"), _doc("1")])

    assert [r.text for r in responses] == [["first, with a comma"], ["second"]]


def test_task_column_matches_full_and_short_task_names(tmp_path):
    path = _write_jsonl(
        tmp_path / "responses.jsonl",
        [
            {"task": "qa|0", "id": 0, "response": "from qa"},
            {"task": "math", "id": 0, "response": "from math"},
        ],
    )
    model = _model(tmp_path, path)

    responses = model.greedy_until([_doc("0", "qa|0"), _doc("0", "math|3")])

    assert [r.text for r in responses] == [["from qa"], ["from math"]]


def test_rows_without_task_are_ambiguous_for_several_tasks(tmp_path):
    path = _write_jsonl(tmp_path / "responses.jsonl", [{"id": 0, "response": "x"}])
    model = _model(tmp_path, path)

    with pytest.raises(ValueError, match="ambiguous"):
        model.greedy_until([_doc("0", "qa|0"), _doc("0", "math|0")])


def test_missing_responses_are_reported(tmp_path):
    path = _write_jsonl(tmp_path / "responses.jsonl", [{"id": 0, "response": "x"}])
    model = _model(tmp_path, path)

    with pytest.raises(ValueError, match=r"1 sample\(s\) have no response .*qa\|0 \(id=7\)"):
        model.greedy_until([_doc("0"), _doc("7")])


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ('{"id": 0, "response": "a"}\n{"id": 0, "response": "b"}\n', "Duplicate response for id 0"),
        ('{"id": 0}\n', "has no 'response' value"),
        ('{"response": "a"}\n', "has no 'id' value"),
        ('{"id": 0, "response": 3}\n', "must be a string or a list of strings"),
        ("not json\n", "is not valid JSON"),
        ("[1, 2]\n", "must be a JSON object"),
        ("\n", "is empty"),
    ],
)
def test_invalid_files_are_rejected(tmp_path, content, match):
    path = tmp_path / "responses.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _model(tmp_path, str(path))


def test_unsupported_extension_and_missing_file(tmp_path):
    path = tmp_path / "responses.txt"
    path.write_text("whatever", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported responses file"):
        _model(tmp_path, str(path))

    with pytest.raises(FileNotFoundError):
        _model(tmp_path, str(tmp_path / "missing.jsonl"))


def test_loglikelihood_is_not_supported(tmp_path):
    path = _write_jsonl(tmp_path / "responses.jsonl", [{"id": 0, "response": "x"}])
    model = _model(tmp_path, path)

    with pytest.raises(NotImplementedError):
        model.loglikelihood([_doc("0")])
    with pytest.raises(NotImplementedError):
        model.loglikelihood_rolling([_doc("0")])


def test_load_model_returns_file_model(tmp_path):
    path = _write_jsonl(tmp_path / "responses.jsonl", [{"id": 0, "response": "x"}])
    config = FileModelConfig(responses_path=path, cache_dir=str(tmp_path / "cache"))

    assert isinstance(load_model(config), FileModel)


CUSTOM_TASK_FILE = """
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def prompt(line, task_name):
    return Doc(task_name=task_name, query=line["question"], choices=[line["answer"]], gold_index=0)


TASKS_TABLE = [
    LightevalTaskConfig(
        name="local_qa",
        prompt_function=prompt,
        hf_repo="json",
        hf_subset="default",
        hf_data_files={data_file!r},
        evaluation_splits=["train"],
        metrics=[Metrics.exact_match],
        generation_size=8,
        stop_sequence=["\\n"],
    )
]
"""


def test_cli_scores_stored_responses_end_to_end(tmp_path, monkeypatch):
    # Keep the sample cache directory inside tmp_path.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    from lighteval.__main__ import app

    data_file = _write_jsonl(
        tmp_path / "data.jsonl",
        [
            {"question": "2+2?", "answer": "4"},
            {"question": "3+3?", "answer": "6"},
            {"question": "1+0?", "answer": "1"},
        ],
    )
    tasks_file = tmp_path / "local_tasks.py"
    tasks_file.write_text(CUSTOM_TASK_FILE.format(data_file=data_file), encoding="utf-8")
    # Two of the three stored responses are right; one is wrapped in reasoning tags.
    responses_file = _write_jsonl(
        tmp_path / "responses.jsonl",
        [
            {"task": "local_qa", "id": 0, "response": "<think>easy</think>4"},
            {"task": "local_qa", "id": 1, "response": "7"},
            {"task": "local_qa", "id": 2, "response": "1"},
        ],
    )
    output_dir = tmp_path / "results"

    result = CliRunner().invoke(
        app,
        [
            "file",
            responses_file,
            "local_qa|0",
            "--custom-tasks",
            str(tasks_file),
            "--model-name",
            "stored-model",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    (results_path,) = output_dir.glob("results/stored-model/*.json")
    results = json.loads(results_path.read_text(encoding="utf-8"))["results"]
    assert results["local_qa|0"]["em"] == pytest.approx(2 / 3)
