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


from typing import Any, Callable, Mapping, TypeVar


AdapterReturnTypeVar = TypeVar("AdapterReturnTypeVar")


def create_adapter_from_dict(
    adapter: Mapping[str, Any] | Callable[[dict], AdapterReturnTypeVar],
) -> Callable[[dict], AdapterReturnTypeVar]:
    """
    Creates adapter function for the template input from a dict.
    Args:
        adapter: Dict of the form {key: value} where value is key in the input dict to get.

    """

    if not isinstance(adapter, Mapping):
        return adapter

    def adapter_fn(line: dict):
        return {key: line[value] for key, value in adapter.items()}

    return adapter_fn  # type: ignore
