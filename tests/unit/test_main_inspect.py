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

import inspect

import pytest
from inspect_ai.model import GenerateConfig, ResponseSchema

from lighteval.main_inspect import _parse_response_schema, eval
from lighteval.models.abstract_model import InspectAIModelConfig


def test_generation_params_are_valid_generate_config_fields():
    """Every generation parameter we forward must exist on inspect-ai's `GenerateConfig`.

    Current `GenerateConfig` rejects unknown fields, so a misspelled name here crashes
    `lighteval eval` at startup for every model and task.
    """
    unknown = set(InspectAIModelConfig.model_fields) - set(GenerateConfig.model_fields)
    assert not unknown, f"Unknown GenerateConfig field(s): {sorted(unknown)}"


def test_generation_params_are_exposed_on_the_cli():
    """The `eval` command and `InspectAIModelConfig` must stay in sync."""
    cli_params = set(inspect.signature(eval).parameters)
    missing = set(InspectAIModelConfig.model_fields) - cli_params
    assert not missing, f"Generation parameter(s) missing from `lighteval eval`: {sorted(missing)}"


def test_parse_response_schema_none():
    assert _parse_response_schema(None) is None


def test_parse_response_schema_wraps_a_bare_json_schema():
    schema = _parse_response_schema('{"type": "object", "properties": {"answer": {"type": "string"}}}')
    assert isinstance(schema, ResponseSchema)
    assert schema.name == "response"
    assert schema.json_schema.type == "object"


def test_parse_response_schema_accepts_a_full_response_schema():
    schema = _parse_response_schema('{"name": "answer", "json_schema": {"type": "string"}, "strict": true}')
    assert schema.name == "answer"
    assert schema.json_schema.type == "string"
    assert schema.strict is True


@pytest.mark.parametrize("value", ["not json", '"a string"', "[1, 2]"])
def test_parse_response_schema_rejects_invalid_input(value):
    with pytest.raises(ValueError):
        _parse_response_schema(value)
