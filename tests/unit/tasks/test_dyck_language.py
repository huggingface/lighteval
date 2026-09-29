import asyncio

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Target
from inspect_ai.solver import TaskState

from lighteval.tasks.tasks.dyck_language import dyck_language_2


@pytest.mark.parametrize(
    ("answer", "target", "expected"),
    [
        (") ] }", ") ] }", CORRECT),
        ("\n ) ] }  ", ") ] }", CORRECT),
        ("} ] )", ") ] }", INCORRECT),
        (") ] ] }", ") ] }", INCORRECT),
    ],
)
def test_dyck_scorer_preserves_bracket_sequence(answer: str, target: str, expected: str) -> None:
    state = TaskState(
        model="mockllm/model",
        sample_id=1,
        epoch=1,
        input="",
        messages=[],
        output=ModelOutput.from_content(model="mockllm/model", content=answer),
    )

    score = asyncio.run(dyck_language_2.scorer(state, Target(target)))

    assert score.value == expected
