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


import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Literal

from pydantic import BaseModel
from tqdm import tqdm

from lighteval.utils.imports import is_litellm_available, is_openai_available, is_vllm_available
from lighteval.utils.utils import as_list


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


DEFAULT_FORMAT = {"type": "text"}


class JudgeLM:
    """
    A class representing a judge for evaluating answers using either the OpenAI or Transformers library.

    Args:
        model (str): The name of the model.
        templates (Callable): A function taking into account the question, options, answer, and gold and returning the judge prompt.
        process_judge_response (Callable): A function for processing the judge's response.
        judge_backend (Literal["openai", "transformers", "tgi", "vllm"]): The backend for the judge.
        url (str | None): The URL for the OpenAI API.
        api_key (str | None): The API key for the OpenAI API (either OpenAI or HF key).

    Attributes:
        model (str): The name of the model.
        template (Callable): A function taking into account the question, options, answer, and gold and returning the judge prompt.
        API_MAX_RETRY (int): The maximum number of retries for the API.
        API_RETRY_SLEEP (int): The time to sleep between retries.
        client (OpenAI | None): The OpenAI client.
        pipe (LLM | AutoModel | None): The Transformers or vllm pipeline.
        process_judge_response (Callable): A function for processing the judge's response.
        url (str | None): The URL for the OpenAI API.
        api_key (str | None): The API key for the OpenAI API (either OpenAI or HF key).
        backend (Literal["openai", "transformers", "tgi", "vllm"]): The backend for the judge

    Methods:
        evaluate_answer: Evaluates an answer using the OpenAI API or Transformers library.
        __lazy_load_client: Lazy loads the OpenAI client or Transformers pipeline.
        __call_api: Calls the API to get the judge's response.
        __call_transformers: Calls the Transformers pipeline to get the judge's response.
        __call_vllm: Calls the VLLM pipeline to get the judge's response.
    """

    def __init__(
        self,
        model: str,
        templates: Callable,
        process_judge_response: Callable,
        judge_backend: Literal["litellm", "openai", "transformers", "tgi", "vllm"],
        url: str | None = None,
        api_key: str | None = None,
        response_format: BaseModel = None,
    ):
        self.model = model
        self.template = templates

        self.API_MAX_RETRY = 3
        self.API_RETRY_SLEEP = 1

        self.client = None
        self.pipe = None
        self.process_judge_response = process_judge_response

        self.url = url
        self.api_key = api_key
        self.backend = judge_backend

        self.response_format = response_format if not None else DEFAULT_FORMAT

    def __lazy_load_client(self):
        match self.backend:
            # Wether we use openai or TGI models, we go through the openai API
            # to route to the endpoint
            case "openai" | "tgi" if is_openai_available():
                if self.client is None:
                    from openai import OpenAI

                    if self.url is None:
                        self.client = OpenAI(api_key=self.api_key)
                    else:
                        self.client = OpenAI(base_url=self.url, api_key=self.api_key)
                return self.__call_api_parallel
            case "litellm" if is_litellm_available():
                return self.__call_litellm
            case "vllm" if is_vllm_available():
                if self.pipe is None:
                    from vllm import LLM, SamplingParams
                    from vllm.transformers_utils.tokenizer import get_tokenizer

                    self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
                    self.tokenizer = get_tokenizer(self.model, tokenizer_mode="auto")
                    self.pipe = LLM(model=self.model, max_model_len=2048, gpu_memory_utilization=0.5, dtype="float16")
                return self.__call_vllm
            case "transformers":
                if self.pipe is None:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

                    transformers_model = AutoModelForCausalLM.from_pretrained(
                        self.model, torch_dtype=torch.float16, trust_remote_code=False, device_map="cuda"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(self.model)
                    self.pipe = pipeline(
                        "text-generation",
                        model=transformers_model,
                        tokenizer=tokenizer,
                        max_new_tokens=256,
                    )
                return self.__call_transformers
            case _:
                return lambda x: x

    def dict_of_lists_to_list_of_dicts(self, dict_of_lists):
        """
        Transform a dictionary of lists into a list of dictionaries.

        Each dictionary in the output list will contain one element from each list in the input dictionary,
        with the same keys as the input dictionary.

        Args:
            dict_of_lists: A dictionary where each value is a list.
                           All lists are expected to have the same length.

        Returns:
            A list of dictionaries.

        Example:
            >>> dict_of_lists_to_list_of_dicts({'k': [1, 2, 3], 'k2': ['a', 'b', 'c']})
            [{'k': 1, 'k2': 'a'}, {'k': 2, 'k2': 'b'}, {'k': 3, 'k2': 'c'}]
        """
        # Check if input is empty
        if not dict_of_lists:
            return None

        # Get all list lengths to ensure they match
        list_lengths = [len(values) for values in dict_of_lists.values()]

        # Ensure all lists have the same length
        if len(set(list_lengths)) > 1:
            raise ValueError("All lists in the input dictionary must have the same length")

        # Get the length of the lists
        n = list_lengths[0] if list_lengths else 0

        # Create list of dictionaries
        result = []
        for i in range(n):
            new_dict = {key: values[i] for key, values in dict_of_lists.items()}
            result.append(new_dict)

        return result

    def evaluate_answer_batch(
        self,
        questions: list[str],
        answers: list[str],
        options: list[list[str]] | list[None],
        golds: list[str] | list[None],
        **kwargs,
    ):
        judge_function = self.__lazy_load_client()

        kwargss = self.dict_of_lists_to_list_of_dicts(kwargs)
        if kwargss is None:
            kwargss = [{} for _ in range(len(questions))]

        # enumerate over questions answers options and golds to make the
        prompts = [
            self.template(question=q, answer=a, options=o, gold=g, **k)
            for q, a, o, g, k in zip(questions, answers, options, golds, kwargss)
        ]
        responses = judge_function(prompts)
        scores = [self.process_judge_response(response) for response in responses]

        # clean up the vllm pipeline and free up memory
        if self.pipe is not None and self.backend == "vllm":
            del self.pipe
            self.pipe = None

        return scores, prompts, responses

    def evaluate_answer(self, question: str, answer: str, options: list[str] | None = None, gold: str | None = None):
        """
        Evaluates an answer using either Transformers or OpenAI API.

        Args:
            questions (list[str]): The prompt asked to the evaluated model
            answers (list[str]): Answer given by the evaluated model
            references (list[str]): A list of reference answers

        Returns:
            A tuple containing the score, prompts, and judgment.
        """
        # lazy loading of the pipeline
        judge_function = self.__lazy_load_client()
        prompt = self.template(question=question, options=options, answer=answer, gold=gold)
        response = judge_function(prompt)
        score = self.process_judge_response(response)

        return score, prompt, response

    def __call_transformers(self, prompt):
        response = self.pipe(prompt)[0]["generated_text"]
        response = response[-1]["content"]
        return response

    def __call_vllm(self, prompt):
        tokenized = [self.tokenizer.apply_chat_template(p) for p in prompt]
        output = self.pipe.generate(prompt_token_ids=tokenized, sampling_params=self.sampling_params, use_tqdm=True)
        outputs = [output.outputs[0].text for output in output]
        return outputs

    def __call_litellm(self, prompts):
        import litellm

        def __call_api(prompt):
            error_message = "ERROR: Failed to get response from the API."
            for _ in range(self.API_MAX_RETRY):
                try:
                    max_new_tokens = 512
                    if "o1" in self.model or "o3" in self.model or "R1" in self.model:
                        max_new_tokens = min(max_new_tokens * 10, 32000)

                    kwargs = {
                        "model": self.model,
                        "messages": prompt,
                        "max_tokens": max_new_tokens,
                        "n": 1,
                        "caching": True,
                    }

                    response = litellm.completion(**kwargs)
                    text = response.choices[0].message.content
                    if not text or text == error_message:
                        kwargs["caching"] = False
                        response = litellm.completion(**kwargs)
                        text = response.choices[0].message.content
                        if not text or text == error_message:
                            # Just return an error response if the second attempt fails too
                            logger.error(f"Failed to get response from the API for prompt: {prompt}")
                            return error_message
                    return text
                except Exception as e:
                    logger.warning(f"{type(e), e}")
                    time.sleep(self.API_RETRY_SLEEP)
            return error_message

        results = []
        with ThreadPoolExecutor(100) as executor:
            for entry in tqdm(executor.map(__call_api, prompts), total=len(prompts)):
                results.append(entry)

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")

        return results

    def __call_api_parallel(self, prompts):
        results = []
        with ThreadPoolExecutor(10) as executor:
            for entry in tqdm(executor.map(self.__call_api, prompts), total=len(prompts)):
                results.append(entry)

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")

        return results

    def __call_api(self, prompt):
        for _ in range(self.API_MAX_RETRY):
            try:
                # Base model
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=as_list(prompt),
                    response_format=self.response_format,
                    max_tokens=4096,
                    temperature=0.0,
                    n=1,
                )
                answer = response.choices[0].message.parsed
                return answer
            except TypeError:
                try:
                    # Finetune
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=as_list(prompt),
                        response_format=self.response_format,
                        max_tokens=512,
                        n=1,
                    )
                    text = response.choices[0].message.content
                    return text
                except Exception as e:
                    logger.warning(f"{type(e), e}")
                    time.sleep(self.API_RETRY_SLEEP)
            except Exception as e:
                logger.warning(f"{type(e), e}")
                time.sleep(self.API_RETRY_SLEEP)

        raise Exception("Failed to get response from the API")
