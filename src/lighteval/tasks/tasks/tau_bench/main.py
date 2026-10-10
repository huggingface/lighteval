"""
name:
TauBench (retail)

dataset:
Vendored from https://github.com/sierra-research/tau-bench (tau_bench/envs/retail)

abstract:
TauBench is an agent benchmark evaluating tool-agent-user interaction in
real-world domains. The model acts as a customer-service agent that must
authenticate users, answer questions with read-only tools, and perform
consequential database updates (returns, exchanges, order modifications)
while respecting the domain policy. A user simulator LLM (by default the
evaluated model itself, configurable through the inspect `user` model role)
plays the customer. The reward checks that the final database state matches
the state reached by replaying the gold actions on a fresh database.

languages:
english

tags:
agentic, tool-calling, multi-turn, conversational

paper:
https://arxiv.org/abs/2406.12045

starred:
true
"""

import functools
from inspect import Parameter, Signature
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolDef, ToolParams

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

from .environment import (
    RULES,
    TERMINATE_TOOLS,
    WIKI,
    Action,
    LLMUserSimulator,
    RetailEnvironment,
)


AGENT_SYSTEM_PROMPT = (
    "# Instructions\n\n" + WIKI + "\n\n" + "\n".join(f"{i + 1}. {rule}" for i, rule in enumerate(RULES))
)

AGENT_GREETING = "Hi! How can I help you today?"


def record_to_sample(record: dict, max_steps: int = 40) -> Sample:
    """Map one vendored TauBench task to an inspect-ai sample."""
    return Sample(
        input=AGENT_GREETING,  # placeholder: the solver replaces the message list
        target="",  # reward is computed against the gold actions, not a text target
        metadata={
            "user_id": record["user_id"],
            "instruction": record["instruction"],
            "gold_actions": record["actions"],
            "outputs": record.get("outputs", []),
            "task_index": record.get("task_index"),
            "max_steps": max_steps,
        },
    )


def make_inspect_tools(env: RetailEnvironment) -> list:
    """Expose the vendored TauBench tools to inspect-ai.

    Tool names, descriptions and parameter schemas are taken verbatim from the
    vendored upstream definitions; each call is routed through
    `RetailEnvironment.execute` so that every action is recorded for scoring.
    """

    def make_call(tool_name: str, param_names: list[str]):
        # A factory keeps `tool_name` bound to this tool only, and inspect-ai
        # validates tool call arguments against the callable's signature
        # annotations, so build an explicit keyword-only signature mirroring
        # the tool schema.
        async def call(**kwargs) -> str:
            return env.execute(Action(name=tool_name, kwargs=kwargs))

        call.__annotations__ = {**dict.fromkeys(param_names, Any), "return": str}
        call.__signature__ = Signature(
            [Parameter(p, Parameter.KEYWORD_ONLY, annotation=Any) for p in param_names],
            return_annotation=str,
        )
        return call

    tools = []
    for tool_class in env.tools_map.values():
        info = tool_class.get_info()["function"]
        call = make_call(info["name"], list(info["parameters"].get("properties", {}).keys()))

        tools.append(
            ToolDef(
                call,
                name=info["name"],
                description=info["description"],
                parameters=ToolParams.model_validate(info["parameters"]),
            ).as_tool()
        )
    return tools


@solver
def tau_bench_agent(max_steps: int = 40) -> Solver:
    """Tool-calling agent loop interacting with a simulated user.

    Mirrors tau-bench's `tool-calling` agent strategy: the agent alternates
    between executing tools and sending messages to the (simulated) user until
    the user signals `###STOP###`, the agent transfers to a human, or
    `max_steps` user turns have been consumed.
    """

    async def solve(state: TaskState, solv_generate: Generate) -> TaskState:
        task = state.metadata
        env = RetailEnvironment(
            task={
                "user_id": task["user_id"],
                "instruction": task["instruction"],
                "actions": task["gold_actions"],
                "outputs": task.get("outputs", []),
            }
        )

        # The simulated user can be a different (stronger) model by configuring
        # the inspect "user" model role, e.g.
        #   lighteval inspect ... --model-roles '{"user": "openai/gpt-4o"}'
        # otherwise it falls back to the model under evaluation.
        try:
            user_model = get_model(role="user")
        except Exception:
            user_model = get_model()
        user_sim = LLMUserSimulator(model=user_model, instruction=task["instruction"])

        state.tools = make_inspect_tools(env)
        first_user_message = await user_sim.respond(AGENT_GREETING)
        state.messages = [
            ChatMessageSystem(content=AGENT_SYSTEM_PROMPT),
            ChatMessageUser(content=first_user_message),
        ]

        n_user_turns = 0
        while n_user_turns < task["max_steps"]:
            # Agent turn(s): execute any number of consecutive tool calls,
            # until the agent produces a plain message addressed to the user.
            state = await solv_generate(state, tool_calls="loop")
            agent_message = state.output.completion

            if any(action.name in TERMINATE_TOOLS for action in env.actions):
                break

            if not agent_message:
                break

            user_reply = await user_sim.respond(agent_message)
            n_user_turns += 1
            if "###STOP###" in user_reply:
                break
            state.messages.append(ChatMessageUser(content=user_reply))

        reward, reward_info = env.calculate_reward()
        state.metadata["reward"] = reward
        state.metadata["reward_info"] = reward_info
        state.metadata["episode_actions"] = [
            {"name": action.name, "arguments": action.kwargs} for action in env.actions
        ]
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def tau_bench_reward() -> Scorer:
    """Reports the TauBench reward computed at the end of the episode."""

    async def score(state: TaskState, target: Target) -> Score:
        reward = state.metadata.get("reward", 0.0)
        reward_info = state.metadata.get("reward_info", {})
        actions = state.metadata.get("episode_actions", [])
        return Score(
            value=float(reward),
            answer=", ".join(a["name"] for a in actions),
            explanation=str(reward_info),
        )

    return score


tau_bench_retail = LightevalTaskConfig(
    name="tau_bench_retail",
    prompt_function=lambda line, task_name: line,  # agent benchmark: only used by the inspect backend
    sample_fields=functools.partial(record_to_sample),
    solver=[tau_bench_agent()],
    scorer=[tau_bench_reward()],
    hf_repo="sierra-research/tau-bench",  # upstream dataset; data is vendored via hf_data_files
    hf_subset="retail",
    hf_data_files={"test": str(Path(__file__).parent / "data" / "tasks.json")},
    hf_avail_splits=["test"],
    evaluation_splits=("test",),
    metrics=[Metrics.exact_match],
)

TASKS_TABLE = [tau_bench_retail]
