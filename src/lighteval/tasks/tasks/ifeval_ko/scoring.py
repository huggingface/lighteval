# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent scorer ported from Allganize IFEval-Ko's ``ifeval_ko/utils.py``.

Reference revision: 54199e3801116897697babf341865741dcd06fc8
https://huggingface.co/datasets/allganize/IFEval-Ko/blob/54199e3801116897697babf341865741dcd06fc8/ifeval_ko/utils.py

The harness InputExample/OutputExample wrappers are replaced by explicit inputs
and PhaseResult. Checker construction is shared locally; the Ko registry import
is package-relative and lazy. No English scorer or registry is modified.
"""

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict


class ScoringResult(TypedDict):
    prompt_level_strict_acc: bool
    inst_level_strict_acc: list[bool]
    prompt_level_loose_acc: bool
    inst_level_loose_acc: list[bool]


@dataclass
class PhaseResult:
    """One phase's ordered instruction flags and their conjunction."""

    follow_instruction_list: list[bool]
    follow_all_instructions: bool


def _build_instruction(instruction_id: str, all_kwargs: Sequence[Mapping], index: int, prompt: str):
    # Loading unrelated tasks must not require the optional Ko dependencies.
    from . import instructions_registry

    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)
    kwargs = {k: v for k, v in all_kwargs[index].items() if v}
    instruction.build_description(**kwargs)
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=prompt)
    return instruction


def score_strict(
    prompt: str, instruction_id_list: Sequence[str], kwargs: Sequence[Mapping], response: str
) -> PhaseResult:
    """Evaluate every instruction on the NFC response, preserving reference errors."""
    response = unicodedata.normalize("NFC", response)
    flags = []
    for index, instruction_id in enumerate(instruction_id_list):
        instruction = _build_instruction(instruction_id, kwargs, index, prompt)
        if response.strip() and instruction.check_following(response):
            flags.append(True)
        else:
            flags.append(False)
    return PhaseResult(flags, all(flags))


def score_loose(
    prompt: str, instruction_id_list: Sequence[str], kwargs: Sequence[Mapping], response: str
) -> PhaseResult:
    """Evaluate each instruction's OR over the reference's eight NFC candidates."""
    response = unicodedata.normalize("NFC", response)
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    all_responses = [
        response,
        response.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]
    flags = []
    for index, instruction_id in enumerate(instruction_id_list):
        instruction = _build_instruction(instruction_id, kwargs, index, prompt)
        is_following = False
        for candidate in all_responses:
            if candidate.strip() and instruction.check_following(candidate):
                is_following = True
                break
        flags.append(is_following)
    return PhaseResult(flags, all(flags))


def score_response(
    prompt: str, instruction_id_list: Sequence[str], kwargs: Sequence[Mapping], response: str
) -> ScoringResult:
    """Return the four native metric values after all strict, then all loose checks.

    Pass the original Doc.query, Doc.specific["instructions_id_list"],
    Doc.specific["kwargs"], and ModelResponse.final_text[0], respectively.
    Duplicate IDs retain their index and corresponding kwargs. Only the response
    is normalized; the prompt and kwargs are forwarded unchanged. Checker errors
    propagate, including errors during construction for an empty response.
    """
    strict = score_strict(prompt, instruction_id_list, kwargs, response)
    loose = score_loose(prompt, instruction_id_list, kwargs, response)
    return {
        "prompt_level_strict_acc": strict.follow_all_instructions,
        "inst_level_strict_acc": strict.follow_instruction_list,
        "prompt_level_loose_acc": loose.follow_all_instructions,
        "inst_level_loose_acc": loose.follow_instruction_list,
    }
