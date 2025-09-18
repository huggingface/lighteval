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
import functools
import importlib
import inspect
import re
from collections import defaultdict
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, metadata, version
from typing import Dict, List, Tuple

from packaging.requirements import Requirement
from packaging.version import Version


# These extras should exist in the pyproject.toml file
class Extra(enum.Enum):
    MULTILINGUAL = "multilingual"
    EXTENDED = "extended"
    TGI = "tgi"


@lru_cache()
def is_package_available(package: str | Requirement | Extra):
    """
    Check if a package is installed in the environment. Returns True if that's the case, False otherwise.
    """
    deps, deps_by_extra = required_dependencies()

    # If the package is a string, it will get the potential required version from the pyproject.toml
    if isinstance(package, str):
        if package not in deps:
            raise RuntimeError(
                f"Package {package} was tested against, but isn't specified in the pyproject.toml file. Please specify"
                f"it as a potential dependency or an extra for it to be checked."
            )
        package = deps[package]

    # If the specified package is an "Extra", we will iterate through each required dependency of that extra
    # and their version and check their existence.
    if isinstance(package, Extra):
        dependencies = deps_by_extra[package.value]
        return all(is_package_available(_package) for _package in dependencies)
    else:
        try:
            installed = Version(version(package.name))
        except PackageNotFoundError:
            return False

        # No version constraint → any installed version is OK
        if not package.specifier:
            return True

        return installed in package.specifier


@lru_cache()
def required_dependencies() -> Tuple[Dict[str, Requirement], Dict[str, List[Requirement]]]:
    """
    Parse the pyproject.toml file and return a dictionary mapping package names to requirements.
    """
    md = metadata("lighteval")
    requires_dist = md.get_all("Requires-Dist") or []
    deps_by_extra = defaultdict(list)
    deps = {}

    for dep in requires_dist:
        extra = None
        if ";" in dep:
            dep, marker = dep.split(";", 1)

            # The `metadata` function prints requirements as follows
            # 'vllm<0.10.2,>=0.10.0; extra == "vllm"'
            # The regex searches for "extra == <MARKER>" in order to parse the marker.
            match = re.search(r'extra\s*==\s*"(.*?)"', marker)
            extra = match.group(1) if match else None
        requirement = Requirement(dep.strip())
        deps_by_extra[extra].append(requirement)
        deps[requirement.name] = requirement

    return deps, deps_by_extra


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


def raise_if_package_not_available(package: Requirement | Extra, *, language: str = None, object_name: str = None):
    prefix = "You" if object_name is None else f"Through the use of {object_name}, you"

    if package == Extra.MULTILINGUAL and ((language is not None) or (not is_multilingual_package_available(language))):
        raise ImportError(prefix + not_installed_error_message(package)[3:])

    if not is_package_available(package):
        raise ImportError(prefix + not_installed_error_message(package)[3:])


def not_installed_error_message(package: Requirement) -> str:
    """
    Custom error messages if need be.
    """

    if package == Extra.MULTILINGUAL.value:
        return "You are trying to run an evaluation requiring multilingual capabilities. Please install the required extra: `pip install lighteval[multilingual]`"
    elif package == Extra.EXTENDED.value:
        return "You are trying to run an evaluation requiring additional extensions. Please install the required extra: `pip install lighteval[extended] "
    elif package == "text_generation":
        return "You are trying to start a text generation inference endpoint, but TGI is not present in your local environment. Please install it using pip."
    elif package == "peft":
        return "You are trying to use adapter weights models, for which you need `peft`, which is not available in your environment. Please install it using pip."
    elif package == "openai":
        return "You are trying to use an Open AI LLM as a judge, for which you need `openai`, which is not available in your environment. Please install it using pip."

    if isinstance(package, Extra):
        return f"You are trying to run an evaluation requiring {package.value} capabilities. Please install the required extra: `pip install lighteval[{package.value}]`"
    else:
        return f"You requested the use of `{package}` for this evaluation, but it is not available in your current environment. Please install it using pip."


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    is_dummy = True

    def __getattribute__(cls, key):
        if (key.startswith("_") and key != "_from_config") or key == "is_dummy" or key == "mro" or key == "call":
            return super().__getattribute__(key)

        for backend in cls._backends:
            raise_if_package_not_available(backend)

        return super().__getattribute__(key)


def parse_specified_backends(specified_backends):
    requirements, _ = required_dependencies()
    applied_backends = []

    for backend in specified_backends:
        if isinstance(backend, Extra):
            applied_backends.append(backend if isinstance(backend, Extra) else requirements[backend])
        elif backend not in requirements:
            raise RuntimeError(
                "A dependency was specified with @requires, but it is not defined in the possible dependencies "
                f"defined in the pyproject.toml: `{backend}`."
                f""
                f"If editing the pyproject.toml to add a new dependency, remember to reinstall lighteval for the"
                f"update to take effect."
            )
        else:
            applied_backends.append(requirements[backend])

    return applied_backends


def requires(*specified_backends):
    """
    Decorator to raise an ImportError if the decorated object (function or class) requires a dependency
    which is not installed.
    """

    applied_backends = parse_specified_backends(specified_backends)

    def inner_fn(_object):
        _object._backends = applied_backends

        if inspect.isclass(_object):

            class Placeholder(metaclass=DummyObject):
                _backends = applied_backends

                def __init__(self, *args, **kwargs):
                    for backend in self._backends:
                        raise_if_package_not_available(backend, object_name=_object.__name__)

            Placeholder.__name__ = _object.__name__
            Placeholder.__module__ = _object.__module__

            return _object if all(is_package_available(backend) for backend in applied_backends) else Placeholder
        else:

            @functools.wraps(_object)
            def wrapper(*args, **kwargs):
                for backend in _object._backends:
                    raise_if_package_not_available(backend, object_name=_object.__name__)
                return _object(*args, **kwargs)

            return wrapper

    return inner_fn
