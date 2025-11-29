"""
name:
MultiChallenge

dataset:
nmayorga7/multichallenge

abstract:
MultiChallenge evaluates large language models (LLMs) on their ability to
conduct multi-turn conversations with human users.
The model is given a target question belonging to one or
more axes (categories) and must provide a free-form answer.
The evaluation uses a secondary judge model to determine if the
answer satisfies the pass criteria for that question.

languages:
english

tags:
conversational, generation, instruction-following

paper:
https://arxiv.org/abs/2501.17399
"""

from inspect_ai.dataset import Sample
from inspect_ai.model._chat_message import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import Score, Target, accuracy, model_graded_fact, scorer, stderr
from inspect_ai.solver import Generate, TaskState, generate, solver

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


JUDGE_PROMPT = """You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{answer}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{criterion}
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


base_scorer = model_graded_fact(
    template=JUDGE_PROMPT,
    grade_pattern=r"\b(YES|NO)\b",
    model="openai/gpt-4o-2024-08-06",
)


@scorer(metrics=[accuracy(), stderr()])
def multi_challenge_scorer():
    async def score(state: TaskState, target: Target):
        score = await base_scorer(state, target)
        judge_verdict = score.value.upper() if score.value else None

        if not judge_verdict or judge_verdict not in ["YES", "NO"]:
            return Score(
                value="I",
                answer=score.answer,
                explanation=f"Could not extract valid verdict from judge output: {score.explanation}",
            )

        pass_criteria = state.metadata.get("pass_criteria", "YES")
        passed = judge_verdict == pass_criteria

        return Score(
            value="C" if passed else "I",
            answer=score.answer,
            explanation=score.explanation,
        )

    return score


@solver
def conversation_solver():
    """Solver that builds conversation history from metadata."""

    async def solve(state: TaskState, generate: Generate):
        conversation = state.metadata.get("conversation", [])

        state.messages = []

        for msg in conversation:
            role = msg["role"].lower()
            content = msg["content"]

            if role == "user":
                state.messages.append(ChatMessageUser(content=content))
            elif role == "assistant":
                state.messages.append(ChatMessageAssistant(content=content))

        return state

    return solve


def record_to_sample(record: dict) -> Sample:
    """Convert dataset record to inspect-ai Sample object."""
    conversation = record["CONVERSATION"]

    last_msg = None
    for msg in reversed(conversation):
        if msg["role"] == "user":
            last_msg = msg["content"]
            break

    return Sample(
        input=last_msg or "",
        target=record["TARGET_QUESTION"],
        metadata={
            "question_id": record["QUESTION_ID"],
            "axis": record["AXIS"],
            "pass_criteria": record["PASS_CRITERIA"],
            "conversation": conversation,
            "length": len(conversation),
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
    solver=[conversation_solver(), generate(cache=True)],
    scorer=multi_challenge_scorer(),
)

TASKS_TABLE = [multi_challenge]
