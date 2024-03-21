import ast
import json
import re
from abc import ABC

from openai import OpenAI


# Abstract class for a judge
class Judge(ABC):
    def evaluate_answer(answers, questions, references) -> tuple[str, list[dict[str, str]], str]:
        pass


class Judge_OpenAI(Judge):
    def __init__(self, model: str, seed: int, temperature: float, templates_path: str):
        self.client = OpenAI()
        self.model = model
        self.seed = seed
        self.temperature = temperature

        data = []
        with open(templates_path, "r") as f:
            for line in f:
                tmp = json.loads(line)
                data.append(tmp)

        self.templates = {d["name"]: d for d in data}

        self.one_score_pattern = re.compile(r"\[\[(\d+\.?\d*)\]\]")
        self.one_score_pattern_backup = re.compile(r"\[(\d+\.?\d*)\]")

    def evaluate_answer(self, questions, answers, references, single_turn: bool):
        if single_turn:
            score, messages, answer = self.__single_turn_evaluate(
                questions[0], answers[0], references[0] if len(references) > 0 else None
            )
        else:
            score, messages, answer = self.__multi_turn_evaluate(questions, answers, references)
        return score, messages, answer

    def __single_turn_evaluate(self, question, answer, reference):
        if reference is None or len(reference) == 0:
            system_prompt = {"role": "system", "content": self.templates["single-v1"]["system_prompt"]}
            user_prompt_str = self.templates["single-v1"]["prompt_template"].format(question=question, answer=answer)
        else:
            system_prompt = {"role": "system", "content": self.templates["single-math-v1"]["system_prompt"]}
            user_prompt_str = self.templates["single-math-v1"]["prompt_template"].format(
                question=question, answer=answer, ref_answer_1=reference
            )

        user_prompt = {"role": "user", "content": user_prompt_str}
        messages = [system_prompt, user_prompt]
        response = self.client.chat.completions.create(
            model=self.model,
            seed=self.seed,
            temperature=self.temperature,
            messages=messages,
        )
        judgment = response.choices[0].message.content
        return self.__process_judge_response(judgment), messages, judgment

    def __multi_turn_evaluate(self, questions, answers, references):
        if references is None or len(references) == 0:
            system_prompt = {"role": "system", "content": self.templates["single-v1-multi-turn"]["system_prompt"]}
            user_prompt_str = self.templates["single-v1-multi-turn"]["prompt_template"].format(
                question_1=questions[0], answer_1=answers[0], question_2=questions[1], answer_2=answers[1]
            )
        else:
            system_prompt = {"role": "system", "content": self.templates["single-math-v1-multi-turn"]["system_prompt"]}
            user_prompt_str = self.templates["single-math-v1-multi-turn"]["prompt_template"].format(
                question_1=questions[0],
                answer_1=answers[0],
                ref_answer_1=references[0],
                question_2=questions[1],
                answer_2=answers[1],
                ref_answer_2=references[1],
            )
        user_prompt = {"role": "user", "content": user_prompt_str}
        messages = [system_prompt, user_prompt]
        response = self.client.chat.completions.create(
            model=self.model,
            seed=self.seed,
            temperature=self.temperature,
            messages=messages,
        )
        judgment = response.choices[0].message.content
        return self.__process_judge_response(judgment), messages, judgment

    def __process_judge_response(self, judgment: str) -> int:
        match = re.search(self.one_score_pattern, judgment)
        if not match:
            match = re.search(self.one_score_pattern_backup, judgment)
        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1

        return rating
