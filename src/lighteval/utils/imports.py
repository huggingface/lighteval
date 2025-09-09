# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import enum
import importlib
from functools import lru_cache


class Extras(enum.Enum):
    MULTILINGUAL = "multilingual"
    EXTENDED = "extended"


@lru_cache()
def is_package_available(package_name: str):
    if package_name == Extras.MULTILINGUAL:
        return all(importlib.util.find_spec(package) is not None for package in ["stanza", "spacy", "langcodes"])
    if package_name == Extras.EXTENDED:
        return all(importlib.util.find_spec(package) is not None for package in ["spacy"])
    else:
        return importlib.util.find_spec(package_name) is not None


@lru_cache()
def is_multilingual_package_available(language: str):
    imports = []
    packages = ["spacy", "stanza"]
    if language == "vi":
        packages.append("pyvi")
    elif language == "zh":
        packages.append("jieba")

    for package in packages:
        imports.append(importlib.util.find_spec(package))

    return all(cur_import is not None for cur_import in imports)


def raise_if_package_not_available(package_name: str | Extras, *, language: str = None):
    if package_name == Extras.MULTILINGUAL and not is_multilingual_package_available(language):
        raise ImportError(not_installed_error_message(package_name))

    if not is_package_available(package_name):
        raise ImportError(not_installed_error_message(package_name))


def not_installed_error_message(package_name: str | Extras) -> str:
    if package_name == Extras.MULTILINGUAL:
        return "You are trying to run an evaluation requiring multilingual capabilities. Please install the required extra: `pip install lighteval[multilingual]`"
    elif package_name == Extras.EXTENDED:
        return "You are trying to run an evaluation requiring additional extensions. Please install the required extra: `pip install lighteval[extended] "
    elif package_name == "text_generation":
        return "You are trying to start a text generation inference endpoint, but TGI is not present in your local environement. Please install it using pip."
    elif package_name in ["bitsandbytes", "auto-gptq"]:
        return f"You are trying to load a model quantized with `{package_name}`, which is not available in your local environement. Please install it using pip."
    elif package_name == "peft":
        return "You are trying to use adapter weights models, for which you need `peft`, which is not available in your environment. Please install it using pip."
    elif package_name == "openai":
        return "You are trying to use an Open AI LLM as a judge, for which you need `openai`, which is not available in your environment. Please install it using pip."

    return f"You requested the use of `{package_name}` for this evaluation, but it is not available in your current environement. Please install it using pip."


def requires(package_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            raise_if_package_not_available(package_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator
