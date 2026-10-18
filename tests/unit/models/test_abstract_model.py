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

from transformers import AutoTokenizer

from lighteval.models.abstract_model import InspectAIModelConfig, ModelConfig
from lighteval.models.dummy.dummy_model import DummyModel, DummyModelConfig


def test_tok_encode_pair():
    model = DummyModel(config=DummyModelConfig(seed=42))
    model._tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M")
    context = "答案："
    continuation = ["1"]
    non_pairwise_tokens = model.tok_encode_pair(context, continuation, pairwise=False)
    pairwise_tokens = model.tok_encode_pair(context, continuation, pairwise=True)
    # Non-pairwise merged "：1" to one token
    assert non_pairwise_tokens == ([[6, 47873]], [[34871]])
    # Pairwise separated "：" and "1"
    assert pairwise_tokens == ([[6, 47873, 13]], [[82]])


def test_tok_encode_pair_move_trailing_context_space():
    model = DummyModel(config=DummyModelConfig(seed=42))
    # BPE tokenizer where " Paris" ("ĠParis") differs from "Paris".
    model._tokenizer = AutoTokenizer.from_pretrained("gpt2")
    context = "Answer: "  # trailing space
    continuation = ["Paris"]
    bare = [model.tok_encode("Paris", add_special_tokens=False)]

    # Default: the trailing space is moved onto the continuation, so the scored
    # continuation is no longer the bare gold.
    model.move_trailing_context_space = True
    _, cont_moved = model.tok_encode_pair(context, continuation, pairwise=True)
    assert cont_moved != bare

    # Opt-out: the space stays in the context and the continuation is exactly the
    # gold string (needed so answer-only bits-per-byte matches the byte normalization).
    model.move_trailing_context_space = False
    _, cont_kept = model.tok_encode_pair(context, continuation, pairwise=True)
    assert cont_kept == bare


# ---------------------------------------------------------------------------
# ModelConfig._parse_args / from_args — CLI model-args string parsing.
#
# Regression coverage for issue #844: an unquoted comma inside a string
# value (e.g. a system prompt) used to silently truncate the value and turn
# the remainder into a spurious boolean flag. Quoting the value is now the
# documented escape hatch: `system_prompt="Be helpful, concise"`.
# ---------------------------------------------------------------------------


class TestModelConfigParseArgs:
    def test_simple_key_value_pairs(self):
        assert ModelConfig._parse_args("model_name=gpt2,max_length=100") == {
            "model_name": "gpt2",
            "max_length": "100",
        }

    def test_bare_flag_becomes_true(self):
        assert ModelConfig._parse_args("model_name=gpt2,use_cache") == {
            "model_name": "gpt2",
            "use_cache": True,
        }

    def test_generation_parameters_dict_still_parsed(self):
        result = ModelConfig._parse_args("model_name=gpt2,generation_parameters={temperature:0.7,top_p:0.9}")
        assert result == {
            "model_name": "gpt2",
            "generation_parameters": {"temperature": 0.7, "top_p": 0.9},
        }

    def test_unquoted_comma_in_value_still_splits(self):
        """Documented limitation: without quotes there's no way to tell a field
        boundary from a comma inside the value — this is why quoting exists.
        """
        result = ModelConfig._parse_args("model_name=gpt2,system_prompt=Be helpful, concise")
        assert result == {"model_name": "gpt2", "system_prompt": "Be helpful", "concise": True}

    def test_double_quoted_value_with_comma_is_preserved(self):
        result = ModelConfig._parse_args('model_name=gpt2,system_prompt="Be helpful, concise and friendly"')
        assert result == {"model_name": "gpt2", "system_prompt": "Be helpful, concise and friendly"}

    def test_single_quoted_value_with_comma_is_preserved(self):
        result = ModelConfig._parse_args("model_name=gpt2,system_prompt='Be helpful, concise and friendly'")
        assert result == {"model_name": "gpt2", "system_prompt": "Be helpful, concise and friendly"}

    def test_quoted_value_followed_by_more_fields(self):
        result = ModelConfig._parse_args('model_name=gpt2,system_prompt="Be helpful, concise",max_length=100')
        assert result == {
            "model_name": "gpt2",
            "system_prompt": "Be helpful, concise",
            "max_length": "100",
        }

    def test_quoted_value_with_no_comma_still_unquoted(self):
        result = ModelConfig._parse_args('model_name=gpt2,system_prompt="Be helpful"')
        assert result == {"model_name": "gpt2", "system_prompt": "Be helpful"}

    def test_from_args_end_to_end_with_quoted_system_prompt(self):
        """Full from_args -> pydantic construction, not just the dict parsing step."""
        config = ModelConfig.from_args('model_name=gpt2,system_prompt="Be helpful, concise, and friendly"')
        assert config.model_name == "gpt2"
        assert config.system_prompt == "Be helpful, concise, and friendly"


class TestSplitTopLevelArgs:
    def test_no_quotes(self):
        assert ModelConfig._split_top_level_args("a=1,b=2,c=3") == ["a=1", "b=2", "c=3"]

    def test_comma_inside_double_quotes_preserved(self):
        assert ModelConfig._split_top_level_args('a=1,b="x,y,z",c=3') == ["a=1", 'b="x,y,z"', "c=3"]

    def test_comma_inside_single_quotes_preserved(self):
        assert ModelConfig._split_top_level_args("a=1,b='x,y,z',c=3") == ["a=1", "b='x,y,z'", "c=3"]

    def test_empty_parts_are_dropped(self):
        assert ModelConfig._split_top_level_args("a=1,,b=2") == ["a=1", "b=2"]

    def test_empty_string_returns_empty_list(self):
        assert ModelConfig._split_top_level_args("") == []


class TestStripMatchingQuotes:
    def test_double_quoted(self):
        assert ModelConfig._strip_matching_quotes('"hello"') == "hello"

    def test_single_quoted(self):
        assert ModelConfig._strip_matching_quotes("'hello'") == "hello"

    def test_unquoted_unchanged(self):
        assert ModelConfig._strip_matching_quotes("hello") == "hello"

    def test_mismatched_quotes_unchanged(self):
        assert ModelConfig._strip_matching_quotes("'hello\"") == "'hello\""


class TestInspectAIModelConfigParseArgs:
    def test_simple_key_value(self):
        assert InspectAIModelConfig._parse_args("max_tokens=100") == {"max_tokens": "100"}

    def test_quoted_system_message_with_comma_is_preserved(self):
        result = InspectAIModelConfig._parse_args('system_message="Be helpful, concise",max_tokens=100')
        assert result == {"system_message": "Be helpful, concise", "max_tokens": "100"}

    def test_from_args_end_to_end_with_quoted_system_message(self):
        config = InspectAIModelConfig.from_args('system_message="Be helpful, concise, and friendly",max_tokens=100')
        assert config.system_message == "Be helpful, concise, and friendly"
        assert config.max_tokens == 100
