# Copyright Sierra
# Copyright 2024 The HuggingFace Team
#
# Minimal, self-contained port of the TauBench retail environment, vendored
# from https://github.com/sierra-research/tau-bench (MIT License) so that the
# benchmark can run inside lighteval without adding a tau-bench dependency.
#
# Adapted pieces (behavior preserved):
# - `calculate_reward` from tau_bench/envs/base.py
# - `LLMUserSimulationEnv` from tau_bench/envs/user.py (re-implemented on top
#   of an inspect-ai Model instead of a litellm completion call)
# - `RULES` from tau_bench/envs/retail/rules.py

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser

from .tools_retail import RETAIL_TOOLS


DATA_DIR = Path(__file__).parent / "data"
RESPOND_ACTION_NAME = "respond"
TERMINATE_TOOLS = ["transfer_to_human_agents"]


def load_data() -> dict[str, Any]:
    """Load a fresh copy of the retail database (users, orders, products)."""
    data = {}
    for name in ["users", "orders", "products"]:
        with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
            data[name] = json.load(f)
    return data


RULES = [
    "You are a customer service representative for an online retail company. You are chatting with a customer, and you can call tools or respond to the user.",
    "The agent should always first confirm the user id by email or name+zip before proceeding with any task.",
    "The agent should not proceed with any task if the user id is not found.",
    "For any change to the backend database, e.g., address update, refund, or order cancellation, the agent must confirm the transaction details with the user and ask for permission, and get explicit authorization (yes) to proceed.",
    "The agent should solve the user task given the tools, without transferring to a human agent.",
    "The agent should not make up any information or knowledge not provided from the user or the tools.",
    "The agent should at most make one tool call at a time, and if the agent makes a tool call, it does not respond to the user at the same time.",
]

with open(DATA_DIR / "wiki.md", encoding="utf-8") as f:
    WIKI = f.read()


@dataclass
class Action:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


class RetailEnvironment:
    """Retail-domain TauBench environment.

    Holds the database state for one evaluation episode and replays agent
    actions. The reward follows the original TauBench logic: replay the gold
    `task["actions"]` on a fresh copy of the database and compare the resulting
    database hashes with the hash reached by the agent.
    """

    def __init__(self, task: dict[str, Any]):
        self.task = task
        self.tools_map = {tool.get_info()["function"]["name"]: tool for tool in RETAIL_TOOLS}
        self.data = load_data()
        self.actions: list[Action] = []

    # -- tools exposed to the model ----------------------------------------
    @property
    def tools_info(self) -> list[dict[str, Any]]:
        return [tool.get_info() for tool in RETAIL_TOOLS]

    def reset(self) -> None:
        self.data = load_data()
        self.actions = []

    # -- episode ------------------------------------------------------------
    def execute(self, action: Action) -> str:
        """Execute one action, recording it for the final reward."""
        self.actions.append(action)
        if action.name in self.tools_map:
            try:
                observation = self.tools_map[action.name].invoke(data=self.data, **action.kwargs)
            except Exception as e:  # noqa: BLE001 - mirror upstream behavior
                observation = f"Error: {e}"
        else:
            observation = f"Unknown action {action.name}"
        return observation

    @staticmethod
    def _hashable(item: Any) -> Any:
        if isinstance(item, dict):
            return tuple((key, RetailEnvironment._hashable(value)) for key, value in sorted(item.items()))
        if isinstance(item, list):
            return tuple(RetailEnvironment._hashable(element) for element in item)
        if isinstance(item, set):
            return tuple(sorted(RetailEnvironment._hashable(element) for element in item))
        return item

    def _data_hash(self) -> str:
        return hashlib.sha256(str(RetailEnvironment._hashable(self.data)).encode("utf-8")).hexdigest()

    def calculate_reward(self) -> tuple[float, dict[str, Any]]:
        """Replay gold actions on a fresh DB and compare database hashes."""
        info: dict[str, Any] = {}
        reward = 1.0
        data_hash = self._data_hash()

        gold_env = RetailEnvironment(task=self.task)
        for gold_action in self.task["actions"]:
            gold_env.execute(Action(name=gold_action["name"], kwargs=gold_action["arguments"]))
        gt_data_hash = gold_env._data_hash()

        info["r_actions"] = data_hash == gt_data_hash
        info["gt_data_hash"] = gt_data_hash
        if not info["r_actions"]:
            reward = 0.0

        # Optional outputs the agent must have communicated to the user.
        outputs = self.task.get("outputs") or []
        if outputs:
            respond_contents = [
                action.kwargs.get("content", "").lower().replace(",", "")
                for action in self.actions
                if action.name == RESPOND_ACTION_NAME
            ]
            found = {output: any(output.lower() in content for content in respond_contents) for output in outputs}
            info["r_outputs"] = all(found.values())
            info["outputs"] = found
            if not info["r_outputs"]:
                reward = 0.0

        return reward, info


class LLMUserSimulator:
    """LLM-based user simulator, following tau-bench's `LLMUserSimulationEnv`.

    Uses an inspect-ai `Model` (by default a configurable role, falling back to
    the evaluated model itself) instead of a litellm call.
    """

    def __init__(self, model, instruction: str):
        self.model = model
        self.instruction = instruction
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]

    def _build_system_prompt(self) -> str:
        return f"""You are a user interacting with an agent.

Instruction: {self.instruction}

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""

    async def respond(self, agent_message: str) -> str:
        self.messages.append({"role": "assistant", "content": agent_message})
        # Convert the plain-dict history into inspect chat message objects.
        chat_messages = [
            ChatMessageSystem(content=m["content"])
            if m["role"] == "system"
            else ChatMessageUser(content=m["content"])
            if m["role"] == "user"
            else ChatMessageAssistant(content=m["content"])
            for m in self.messages
        ]
        output = await self.model.generate(chat_messages)
        reply = output.completion
        self.messages.append({"role": "user", "content": reply})
        return reply
