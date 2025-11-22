"""
name:
MultiChallenge

dataset:
nmayorga7/multichallenge

abstract:
MultiChallenge is a realistic multi-turn conversation evaluation benchmark challenging
to frontier LLMs. It identifies four categories of challenges in multi-turn conversations
that require accurate instruction-following, context allocation, and in-context reasoning
simultaneously. All frontier models achieve less than 50% accuracy on MultiChallenge.

languages:
english

tags:
conversational, generation, multi-turn, instruction-following

paper:
https://arxiv.org/abs/2501.17399
"""

import re

from inspect_ai.dataset import Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


JUDGE_PROMPT = """You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{}
</CRITERIA>

Print your reasoning followed by your verdict, either "YES" or "NO"."""


def format_conversation(conversation: list[dict]) -> str:
    """Format conversation messages into a single string for model input."""
    formatted_messages = []
    for msg in conversation:
        role = msg["role"].upper()
        content = msg["content"]
        formatted_messages.append(f"{role}:\n{content}")

    return "\n\n".join(formatted_messages)


def multi_challenge_prompt(line, task_name: str = None):
    """Convert dataset to Doc object"""

    conversation = line["CONVERSATION"]
    formatted_conv = format_conversation(conversation)
    return Doc(
        task_name=task_name,
        query=formatted_conv,
        instruction=None,
        specific={
            "question_id": line["QUESTION_ID"],
            "axis": line["AXIS"],
            "target_question": line["TARGET_QUESTION"],
            "pass_criteria": line["PASS_CRITERIA"],
            "conversation": conversation,
        },
    )


@scorer(metrics=[accuracy(), stderr()])
def multi_challenge_scorer():
    async def score(state: TaskState, target: Target):
        response = state.output.completion

        target_question = target.text
        pass_criteria = state.metadata.get("pass_criteria", "YES")

        if not target_question:
            return Score(
                value="I",
                answer=response,
                explanation="Target question not found.",
            )

        try:
            judge_model = get_model("openai/gpt-4o-2024-08-06")
            judge_prompt = JUDGE_PROMPT.format(response, target_question)

            judge_result = await judge_model.generate(judge_prompt)
            judge_output = judge_result.completion

            verdict_match = re.search(r"\b(YES|NO)\b", judge_output, re.IGNORECASE)

            if not verdict_match:
                return Score(
                    value="I",
                    answer=response,
                    explanation=f"Could not extract verdict from judge output: {judge_output}.",
                )

            judge_verdict = verdict_match.group(1).upper()
            passed = judge_verdict == pass_criteria

            return Score(
                value="C" if passed else "I",
                answer=response,
                explanation=f"Judge verdict: {judge_verdict}, Expected: {pass_criteria}, Response: {response}.",
            )

        except Exception as e:
            return Score(
                value="I",
                answer=response,
                explanation=f"Error during judge evaluation: {str(e)}.",
            )

    return score


def record_to_sample(record: dict) -> Sample:
    """Convert dataset record to inspect-ai Sample object."""
    conversation = record["CONVERSATION"]
    formatted_conv = format_conversation(conversation)

    return Sample(
        input=formatted_conv,
        target=record["TARGET_QUESTION"],
        metadata={
            "question_id": record["QUESTION_ID"],
            "axis": record["AXIS"],
            "pass_criteria": record["PASS_CRITERIA"],
            "conversation": conversation,
        },
    )


multi_challenge = LightevalTaskConfig(
    name="multi_challenge",
    prompt_function=multi_challenge_prompt,
    hf_repo="nmayorga7/multichallenge",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    stop_sequence=[],
    version=0,
    sample_fields=record_to_sample,
    metrics=[],  # Metrics are defined in the scorer decorator for inspect-ai tasks
    solver=[generate(cache=True)],
    scorer=multi_challenge_scorer(),
)

TASKS_TABLE = [multi_challenge]
