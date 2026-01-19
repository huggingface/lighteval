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

starred:
true
"""

from inspect_ai.dataset import Sample
from inspect_ai.model._chat_message import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import Score, Target, accuracy, model_graded_fact, scorer, stderr
from inspect_ai.solver import TaskState, generate

from lighteval.tasks.lighteval_task import LightevalTaskConfig


# NOTE: ChatMessageAssistant and ChatMessageUser are imported from a private module.


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


@scorer(metrics=[accuracy(), stderr()])
def multi_challenge_scorer():
    base_scorer = model_graded_fact(
        template=JUDGE_PROMPT,
        grade_pattern=r"\b(YES|NO)\b",
    )

    async def score(state: TaskState, target: Target):
        score = await base_scorer(state, target)
        judge_verdict = score.value.upper() if score.value else None

        if not judge_verdict or judge_verdict not in ["YES", "NO"]:
            return Score(
                value="I",
                answer=score.answer,
                explanation=f"Could not extract valid verdict from judge output: {score.explanation}",
            )

        pass_criteria = state.metadata.get("pass_criteria", "")
        if pass_criteria not in ["YES", "NO"]:
            return Score(
                value="I",
                answer=score.answer,
                explanation=f"Invalid pass criteria: {pass_criteria}",
            )

        passed = judge_verdict == pass_criteria

        return Score(
            value="C" if passed else "I",
            answer=score.answer,
            explanation=score.explanation,
        )

    return score


def record_to_sample(record: dict) -> Sample:
    """Convert dataset record to inspect-ai Sample object."""
    conversation = record["CONVERSATION"]
    messages = []
    for msg in conversation:
        if msg["role"] == "user":
            messages.append(ChatMessageUser(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(ChatMessageAssistant(content=msg["content"]))

    return Sample(
        input=messages,
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
    prompt_function=lambda line, task_name: line,
    hf_repo="nmayorga7/multichallenge",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    stop_sequence=[],
    version=0,
    metrics=[],  # Metrics are defined in the scorer decorator for inspect-ai tasks
    sample_fields=record_to_sample,
    solver=[generate(cache=True)],
    scorer=multi_challenge_scorer(),
)

TASKS_TABLE = [multi_challenge]
