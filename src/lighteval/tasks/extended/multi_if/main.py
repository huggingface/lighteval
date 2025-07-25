# MIT License

# Copyright (c) 2024 The HuggingFace Team

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


import json
import numpy as np
from aenum import extend_enum

import lighteval.tasks.extended.ifeval.instructions_registry as instructions_registry
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.metric_utils import (
    SampleLevelMetricGrouping,
)
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.utils import remove_reasoning_tags


def multi_if_prompt(line, task_name: str = ""):
    """
    Multi-IF prompt function for multi-turn instruction following evaluation.
    Each sample contains 3 turns with cumulative instruction constraints.
    """
    # Parse the turn prompts from JSON strings
    turn_1_prompt = json.loads(line["turn_1_prompt"])["content"]
    turn_2_prompt = json.loads(line["turn_2_prompt"])["content"] 
    turn_3_prompt = json.loads(line["turn_3_prompt"])["content"]
    
    # Combine all turns into a conversation
    conversation = f"Turn 1: {turn_1_prompt}\n\nTurn 2: {turn_2_prompt}\n\nTurn 3: {turn_3_prompt}"
    
    # Parse instruction lists and kwargs for all turns
    turn_1_instructions = json.loads(line["turn_1_instruction_id_list"])
    turn_1_kwargs = json.loads(line["turn_1_kwargs"])
    
    turn_2_instructions = json.loads(line["turn_2_instruction_id_list"])
    turn_2_kwargs = json.loads(line["turn_2_kwargs"])
    
    turn_3_instructions = json.loads(line["turn_3_instruction_id_list"])
    turn_3_kwargs = json.loads(line["turn_3_kwargs"])
    
    return Doc(
        task_name=task_name,
        query=conversation,
        choices=[""],
        gold_index=0,
        instruction="",
        specific={
            "turn_1_instructions": turn_1_instructions,
            "turn_1_kwargs": turn_1_kwargs,
            "turn_2_instructions": turn_2_instructions,
            "turn_2_kwargs": turn_2_kwargs,
            "turn_3_instructions": turn_3_instructions,
            "turn_3_kwargs": turn_3_kwargs,
            "language": line["language"],
            "key": line["key"]
        },
    )


submetric_names = [
    "turn_1_strict_acc",
    "turn_2_strict_acc", 
    "turn_3_strict_acc",
    "turn_1_loose_acc",
    "turn_2_loose_acc",
    "turn_3_loose_acc",
    "avg_strict_acc",
    "avg_loose_acc"
]

REASONING_TAG_PAIRS = [
    ("<think>", "</think>"),
]


def multi_if_metric(doc: Doc, model_response: ModelResponse, **kwargs) -> dict:
    """
    Multi-IF metric that evaluates instruction following across 3 conversation turns.
    Each turn builds on previous constraints, requiring cumulative constraint satisfaction.
    """
    response = model_response.text[0]
    # Remove reasoning tags to avoid false negatives
    response = remove_reasoning_tags(response, REASONING_TAG_PAIRS)
    
    # Get turn-specific instruction data
    turn_1_instructions = doc.specific["turn_1_instructions"]
    turn_1_kwargs = doc.specific["turn_1_kwargs"]
    
    turn_2_instructions = doc.specific["turn_2_instructions"] 
    turn_2_kwargs = doc.specific["turn_2_kwargs"]
    
    turn_3_instructions = doc.specific["turn_3_instructions"]
    turn_3_kwargs = doc.specific["turn_3_kwargs"]
    
    # Prepare response variants for loose evaluation (similar to IFEval)
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    
    # Evaluate each turn's constraints
    def evaluate_turn_constraints(instructions, kwargs_list, responses, strict=True):
        if not instructions:
            return True
            
        responses_to_check = [responses[0]] if strict else responses
        
        for response_variant in responses_to_check:
            turn_satisfied = True
            for instruction_id, kwargs_str in zip(instructions, kwargs_list):
                kwargs_dict = json.loads(kwargs_str) if kwargs_str.strip() != "{}" else {}
                
                try:
                    is_satisfied = instructions_registry.INSTRUCTION_DICT[instruction_id].check_following(response_variant, **kwargs_dict)
                    if not is_satisfied:
                        turn_satisfied = False
                        break
                except Exception:
                    turn_satisfied = False
                    break
                    
            if turn_satisfied:
                return True
        return False
    
    # Evaluate strict accuracy for each turn
    turn_1_strict = evaluate_turn_constraints(turn_1_instructions, turn_1_kwargs, all_responses, strict=True)
    turn_2_strict = evaluate_turn_constraints(turn_2_instructions, turn_2_kwargs, all_responses, strict=True)
    turn_3_strict = evaluate_turn_constraints(turn_3_instructions, turn_3_kwargs, all_responses, strict=True)
    
    # Evaluate loose accuracy for each turn
    turn_1_loose = evaluate_turn_constraints(turn_1_instructions, turn_1_kwargs, all_responses, strict=False)
    turn_2_loose = evaluate_turn_constraints(turn_2_instructions, turn_2_kwargs, all_responses, strict=False)
    turn_3_loose = evaluate_turn_constraints(turn_3_instructions, turn_3_kwargs, all_responses, strict=False)
    
    # Calculate averages
    avg_strict = np.mean([turn_1_strict, turn_2_strict, turn_3_strict])
    avg_loose = np.mean([turn_1_loose, turn_2_loose, turn_3_loose])
    
    return {
        "turn_1_strict_acc": turn_1_strict,
        "turn_2_strict_acc": turn_2_strict,
        "turn_3_strict_acc": turn_3_strict,
        "turn_1_loose_acc": turn_1_loose,
        "turn_2_loose_acc": turn_2_loose,
        "turn_3_loose_acc": turn_3_loose,
        "avg_strict_acc": avg_strict,
        "avg_loose_acc": avg_loose,
    }


# Create metric enum entries
for submetric_name in submetric_names:
    extend_enum(Metrics, submetric_name, (multi_if_metric, SampleLevelMetricGrouping.SINGLE))


# Task configuration
multi_if_config = LightevalTaskConfig(
    name="multi_if",
    suite=["extended"],
    prompt_function=multi_if_prompt,
    hf_repo="facebook/Multi-IF",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1024,  # Allow longer responses to accommodate multi-turn constraints
    metrics=[getattr(Metrics, submetric_name) for submetric_name in submetric_names],
    stop_sequence=None,
    trust_dataset=True,
    version=0,
)


MULTI_IF_TASKS = [multi_if_config]