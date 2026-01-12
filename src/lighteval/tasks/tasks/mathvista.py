"""
name:
MathVista

dataset:
AI4Math/MathVista

abstract:
Large Language Models (LLMs) and Large Multimodal Models (LMMs) exhibit impressive problem-solving skills in many tasks and domains, but their ability in mathematical reasoning in visual contexts has not been systematically studied. To bridge this gap, we present MathVista, a benchmark designed to combine challenges from diverse mathematical and visual tasks. It consists of 6,141 examples, derived from 28 existing multimodal datasets involving mathematics and 3 newly created datasets (i.e., IQTest, FunctionQA, and PaperQA). Completing these tasks requires fine-grained, deep visual understanding and compositional reasoning, which all state-of-the-art foundation models find challenging.

languages:
english

tags:
math, qa, reasoning

paper:
https://arxiv.org/pdf/2310.02255

starred:
true
"""

import logging
import re
from io import BytesIO
from pathlib import Path
from string import ascii_uppercase
from typing import Any

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, ChatMessageUser, ContentImage, ContentText
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    AnswerPattern,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.scorer._pattern import match_first
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    multiple_choice,
    solver,
    system_message,
)
from PIL import Image

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


# code insperied by: https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mathvista/mathvista.py

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Answer must come after an "ANSWER:" tag.
""".strip()


def is_image_png(image_bytes: bytes) -> bool:
    return image_bytes[:8] == b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"


def is_image_webp(image_bytes: bytes) -> bool:
    return image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP"


def get_message(record: dict[str, Any]) -> Sample:
    # extract image
    IMAGE_BASE_DIR = Path("./data/mathvista_images")
    image = Path(IMAGE_BASE_DIR / record["image"])

    # images are a mix of jpg, png, and webp but all have a file extension of .jpg
    image_bytes = record["decoded_image"]["bytes"]
    if is_image_png(image_bytes):
        image = image.with_suffix(".png")
    elif is_image_webp(image_bytes):
        image = image.with_suffix(".webp")

    if not image.exists():
        logger.debug(f"Extracting {image.name}")
        # ensure parent
        image.parent.mkdir(exist_ok=True, parents=True)
        # reduce the image size
        img = Image.open(BytesIO(image_bytes))
        img.thumbnail((1024, 1024))
        # save preserving format
        img.save(image, format=img.format)

    message: list[ChatMessage] = [
        ChatMessageUser(
            content=[
                ContentText(text=record["question"])
                if record["question_type"] == "multi_choice"
                else ContentText(text=record["query"]),
                ContentImage(image=str(image)),
            ]
        )
    ]
    return message


def record_to_sample_multiple_choice(record: dict[str, Any]) -> Sample:
    message = get_message(record)
    target = ascii_uppercase[record["choices"].index(record["answer"])]

    return Sample(
        input=message,
        choices=record["choices"],
        target=target,
        id=record["pid"],
        metadata={
            "question_type": record["question_type"],
            "answer_type": record["answer_type"],
            **record["metadata"],
        },
    )


def record_to_sample_freeform(record: dict[str, Any]) -> Sample:
    message = get_message(record)
    return Sample(
        input=message,
        target=record["answer"],
        id=record["pid"],
        metadata={
            "precision": record["precision"],
            "question_type": record["question_type"],
            "answer_type": record["answer_type"],
            **record["metadata"],
        },
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    if record["question_type"] == "multi_choice":
        return record_to_sample_multiple_choice(record)
    else:
        return record_to_sample_freeform(record)


@scorer(metrics=[accuracy(), stderr()])
def mathvista_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        match = re.search(
            AnswerPattern.LETTER if state.metadata["question_type"] == "multi_choice" else AnswerPattern.LINE,
            state.output.completion,
            re.IGNORECASE,
        )
        if match:
            # scoring pattern found in response
            groups = match.groups()
            found_match = match_first(matches=groups, target=target, ignore_case=True)

            if found_match is None and len(groups) == 1:
                answer = groups[0]
            else:
                answer = found_match

            return Score(
                value=CORRECT if found_match else INCORRECT,
                answer=answer,
                explanation=state.output.completion,
            )

        else:
            # didn't find the scoring pattern
            return Score(
                value=INCORRECT,
                explanation="Scoring pattern not matched in output: " + f"{state.output.completion}",
            )

    return score


@solver
def mathvista_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.metadata["question_type"] == "multi_choice":
            return await multiple_choice()(state, generate)
        else:
            state = await system_message(SYSTEM_PROMPT)(state, generate)
            return await generate(state)

    return solve


mathvista = LightevalTaskConfig(
    name="mathvista",
    prompt_function=lambda x: x,
    hf_repo="AI4Math/MathVista",
    hf_subset="default",
    hf_filter=lambda x: x.get("question_type") == "multi_choice",
    hf_avail_splits=["testmini, test"],
    evaluation_splits=["testmini"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=512,
    metrics=[Metrics.expr_gold_metric],
    stop_sequence=None,
    version=0,
    sample_fields=record_to_sample,
    solver=[mathvista_solver()],
    scorer=mathvista_scorer(),
)

TASKS_TABLE = [
    mathvista,
]
