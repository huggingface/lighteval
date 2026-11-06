# MIT License

# Copyright (c) 2026 The HuggingFace Team

"""
name:
CLIcK

dataset:
EunsuKim/CLIcK

abstract:
CLIcK is a benchmark of 1,995 multiple-choice questions on Korean culture and language,
collected from official Korean exams and textbooks and grouped into eleven categories.
These tasks score every full answer string by summed log-likelihood, with one fixed Korean
prompt, the original option order and no few-shot examples, so results are not a
reproduction of the rotated-prompt setup in the paper.

languages:
korean

tags:
knowledge, language, multiple-choice

paper:
https://aclanthology.org/2024.lrec-main.296/
"""

import hashlib
import json
from collections.abc import Mapping

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


HF_REPO = "EunsuKim/CLIcK"
HF_REVISION = "d61627859645b5e6edc03fd9f835735d8226fa4e"
PROMPT_VERSION = "click_paper_example_v2"
RAW_FIELDS = ("id", "paragraph", "question", "choices", "answer")

# Explicit paths at HF_REVISION, sorted within each category. Preserve array order.
# Categories come from file paths, because original IDs are not unique.
CATEGORY_FILES = {
    "society": (
        "Dataset/Culture/Korean Society/Society_KIIP.json",
        "Dataset/Culture/Korean Society/Society_Kedu.json",
    ),
    "tradition": (
        "Dataset/Culture/Korean Tradition/Tradition_KIIP.json",
        "Dataset/Culture/Korean Tradition/Tradition_Kedu.json",
    ),
    "history": (
        "Dataset/Culture/Korean History/History_KHB.json",
        "Dataset/Culture/Korean History/History_Kedu.json",
        "Dataset/Culture/Korean History/History_PSE.json",
    ),
    "law": (
        "Dataset/Culture/Korean Law/Law_KIIP.json",
        "Dataset/Culture/Korean Law/Law_PSAT.json",
    ),
    "politics": (
        "Dataset/Culture/Korean Politics/Politics_KIIP.json",
        "Dataset/Culture/Korean Politics/Politics_Kedu.json",
    ),
    "economy": (
        "Dataset/Culture/Korean Economy/Economy_KIIP.json",
        "Dataset/Culture/Korean Economy/Economy_Kedu.json",
    ),
    "geography": (
        "Dataset/Culture/Korean Geography/Geography_CSAT.json",
        "Dataset/Culture/Korean Geography/Geography_KIIP.json",
        "Dataset/Culture/Korean Geography/Geography_Kedu.json",
    ),
    "popular": (
        "Dataset/Culture/Korean Popular/Popular_KIIP.json",
        "Dataset/Culture/Korean Popular/Popular_Kedu.json",
    ),
    "textual": (
        "Dataset/Language/Textual/Textual_CSAT.json",
        "Dataset/Language/Textual/Textual_TOPIK.json",
    ),
    "functional": (
        "Dataset/Language/Functional/Functional_CSAT.json",
        "Dataset/Language/Functional/Functional_Kedu.json",
        "Dataset/Language/Functional/Functional_PSE.json",
    ),
    "grammar": (
        "Dataset/Language/Grammar/Grammar_CSAT.json",
        "Dataset/Language/Grammar/Grammar_Kedu.json",
        "Dataset/Language/Grammar/Grammar_TOPIK.json",
    ),
}


def _validate_row(row: Mapping[str, object]) -> None:
    if not isinstance(row, Mapping):
        raise ValueError("row must be a mapping")
    for key in ("id", "paragraph", "question", "answer"):
        if not isinstance(row.get(key), str):
            raise ValueError(f"{key} must be a string")
    options = row.get("choices")
    if not isinstance(options, list) or len(options) not in (4, 5):
        raise ValueError("choices must contain four or five options")
    if any(not isinstance(option, str) or not option.strip() for option in options):
        raise ValueError("each option must be a nonempty string")
    if not row["question"].strip():
        raise ValueError("question must be nonempty")
    if options.count(row["answer"]) != 1:
        raise ValueError("answer must match exactly one option")


def build_click_payload(row: Mapping[str, object]) -> dict:
    """Build the exact single-prompt contract without normalizing source strings."""
    _validate_row(row)
    options = row["choices"]
    labels = "ABCDE"[: len(options)]
    # Section 4, Type 1/2: https://aclanthology.org/2024.lrec-main.296.pdf#page=5
    # Follow the printed bilingual field labels and comma-separated options.
    # The parenthesized English translation of the instruction is not appended.
    has_context = bool(row["paragraph"].strip())
    prefix = (
        "주어진 맥락을 천천히 읽고, 질문에 대한 적절한 정답을 "
        if has_context
        else "주어진 질문을 천천히 읽고, 적절한 정답을 "
    )
    blocks = [prefix + ", ".join(labels) + " 중에 골라 알파벳 하나로 답하시오."]
    if has_context:
        blocks.append("맥락 (Context): " + row["paragraph"])
    blocks.append("질문 (Question): " + row["question"])
    displayed_options = [f"{label}: {option}" for label, option in zip(labels, options)]
    blocks.extend(["보기 (Options):", ", ".join(displayed_options), "정답 (Answer):"])
    return {
        "query": "\n".join(blocks),
        # Scoring continuations retain the separately specified full-answer format.
        "choices": [f" {label}. {option}" for label, option in zip(labels, options)],
        "gold_index": options.index(row["answer"]),
    }


def raw_row_sha256(row: Mapping[str, object]) -> str:
    """Hash only the five original fields, with no Unicode or whitespace changes."""
    _validate_row(row)
    raw = {key: row[key] for key in RAW_FIELDS}
    encoded = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def click_prompt(line: dict, task_name: str) -> Doc:
    """Keep trace metadata separate from the model query and scored candidates."""
    payload = build_click_payload(line)
    return Doc(
        task_name=task_name,
        specific={"original_id": line["id"], "raw_row_sha256": raw_row_sha256(line)},
        **payload,
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"click:{category}",
        prompt_function=click_prompt,
        hf_repo=HF_REPO,
        hf_subset="default",
        hf_revision=HF_REVISION,
        hf_data_files={"train": list(files)},
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        num_fewshots=0,
        generation_size=None,
        metrics=[Metrics.loglikelihood_acc],
        stop_sequence=None,
        must_remove_duplicate_docs=False,
        version=0,
    )
    for category, files in CATEGORY_FILES.items()
]
