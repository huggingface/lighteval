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


import ast
import json
import re
from typing import Any, Optional


class JudgeLM:
    """
    A class representing a judge for evaluating answers using the Transformers library.

    Args:
        model (str): The name of the model to use.
        templates_path (str): The path to the JSON file containing the templates for prompts.
        multi_turn (bool): Whether to use multi-turn prompts

    Attributes:
        model (str): The name of the model.
        templates (dict): A dictionary containing the templates for prompts.
        one_score_pattern (re.Pattern): A regular expression pattern for extracting scores from the response.
        one_score_pattern_backup (re.Pattern): A backup regular expression pattern for extracting scores.

    Methods:
        evaluate_answer: Evaluates an answer using the OpenAI API or Transformers library.
        __get_prompts_multi_turn: Generates prompts for multi-turn conversations.
        __get_prompts_single_turn: Generates prompts for single-turn conversations.
        __process_judge_response: Processes the judge's response and extracts the score.
    """

    def __init__(
        self,
        model: str,
        templates_path: str,
        multi_turn: bool = False,
    ):
        self.multi_turn = multi_turn
        self.pipe = None
        self.model = model

        data = []
        with open(templates_path, "r") as f:
            for line in f:
                tmp = json.loads(line)
                data.append(tmp)

        self.templates = {d["name"]: d for d in data}

        # Patterns for extracting scores from the response
        # The first pattern is for the default case: [[score]],
        # the second is for the backup case: [score]
        self.one_score_pattern = re.compile(r"\[\[(\d+\.?\d*)\]\]")
        self.one_score_pattern_backup = re.compile(r"\[(\d+\.?\d*)\]")

    def evaluate_answer(
        self, questions: list[str], answers: list[str], references: list[str]
    ) -> tuple[list[int], list[list[dict[str, str]]], list[str | None | Any]]:
        """
        Evaluates an answer using Transformers.

        Args:
            questions (list[str]): A list of questions (can be a list because of multi-turn conversations)
            answers (list[str]): A list of answers, one for each question.
            references (list[str]): A list of reference answers, one for each question (sometimes not available)

        Returns:
            A tuple containing the score, prompts, and judgment.
        """
        # lazy loading of the pipeline
        if self.pipe is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            transformers_model = AutoModelForCausalLM.from_pretrained(
                self.model, torch_dtype=torch.bfloat16, trust_remote_code=False, device_map="cuda"
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.pipe = pipeline(
                "text-generation",
                model=transformers_model,
                tokenizer=tokenizer,
                max_new_tokens=50,
            )

        prompts = [
            self.__get_prompts_single_turn(
                questions[0], answers[0], references[0] if references and len(references) > 0 else None
            )
        ]

        if self.multi_turn:
            prompts_multi_turn = self.__get_prompts_multi_turn(
                questions, answers, references if len(references) > 1 else None
            )
            prompts.append(prompts_multi_turn)

        judgments = []
        for prompt in prompts:
            response = self.pipe(prompt)[0]["generated_text"]
            response = response[-1]["content"]
            judgments.append(response)

        scores = [self.__process_judge_response(judgment) for judgment in judgments]

        return scores, prompts, judgments

    def __get_prompts_multi_turn(
        self, questions: list[str], answers: list[str], references: Optional[list[str]]
    ) -> list[dict[str, str]]:
        """
        Generates prompts for multi-turn conversations. The prompts are generated based on the templates.
        The prompt is different for the case where reference answers are available.

        Args:
            questions (list[str]): A list of questions.
            answers (list[str]): A list of answers.
            references (Optional[list[str]]): A list of reference answers.

        Returns:
            A list of prompts.
        """
        if references is None:
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
        return [system_prompt, user_prompt]

    def __get_prompts_single_turn(self, question: str, answer: str, reference: Optional[str]) -> list[dict[str, str]]:
        """
        Generates prompts for single-turn conversations. The prompts are generated based on the templates.
        The prompt is different for the case where a reference answer is available.

        Args:
            question (str): The question.
            answer (str): The answer.
            reference (Optional[str]): The reference answer.

        Returns:
            A list of prompts.
        """
        if reference is None:
            system_prompt = {"role": "system", "content": self.templates["single-v1"]["system_prompt"]}
            user_prompt_str = self.templates["single-v1"]["prompt_template"].format(question=question, answer=answer)
        else:
            system_prompt = {"role": "system", "content": self.templates["single-math-v1"]["system_prompt"]}
            user_prompt_str = self.templates["single-math-v1"]["prompt_template"].format(
                question=question, answer=answer, ref_answer_1=reference
            )
        user_prompt = {"role": "user", "content": user_prompt_str}
        return [system_prompt, user_prompt]

    def __process_judge_response(self, judgment: str) -> int:
        """
        Processes the judge's response and extracts the score.
        Returns -1 if the score cannot be extracted.

        Args:
            judgment (str): The judge's response.

        Returns:
            The extracted score.
        """
        match = re.search(self.one_score_pattern, judgment)
        if not match:
            match = re.search(self.one_score_pattern_backup, judgment)
        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1

        return rating
