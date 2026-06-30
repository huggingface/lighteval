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

"""Unit tests for the LiteLLM loglikelihood implementation.

All litellm API calls are mocked — no network requests are made.
Async helpers are exercised via asyncio.run() to remain dependency-free
(no pytest-asyncio required).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from lighteval.models.model_input import GenerationParameters
from lighteval.utils.imports import is_package_available


pytestmark = pytest.mark.skipif(
    not is_package_available("litellm"),
    reason="litellm not installed — run `pip install lighteval[litellm]` to enable these tests",
)

from lighteval.models.endpoints.litellm_model import LiteLLMClient  # noqa: E402
from lighteval.models.model_output import ModelResponse  # noqa: E402
from lighteval.tasks.requests import Doc  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — build fake litellm text_completion response objects
# ---------------------------------------------------------------------------


def make_logprobs(tokens, token_logprobs, top_logprobs=None, text_offset=None):
    """Return a SimpleNamespace mimicking litellm's logprobs object."""
    return SimpleNamespace(
        tokens=tokens,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs if top_logprobs is not None else [None] * len(tokens),
        text_offset=text_offset,
    )


def make_response(tokens, token_logprobs, top_logprobs=None, text_offset=None):
    """Return a SimpleNamespace mimicking a litellm text_completion response."""
    lp = make_logprobs(tokens, token_logprobs, top_logprobs, text_offset)
    choice = SimpleNamespace(logprobs=lp)
    return SimpleNamespace(choices=[choice])


def make_doc(query, choices, gold_index=0, task_name="test_task", doc_id="0"):
    doc = Doc(query=query, choices=choices, gold_index=gold_index, task_name=task_name)
    doc.id = doc_id
    return doc


def make_bare_client(
    model="gpt-3.5-turbo-instruct",
    concurrent_requests=10,
    api_max_retry=3,
    api_retry_sleep=0.0,  # instant retries in tests
    api_retry_multiplier=1.0,
):
    """Construct a LiteLLMClient instance bypassing __init__ to avoid real API calls."""
    client = object.__new__(LiteLLMClient)
    client.model = model
    client.provider = "openai"
    client.base_url = None
    client.api_key = None
    client.timeout = None
    client.concurrent_requests = concurrent_requests
    client.API_MAX_RETRY = api_max_retry
    client.API_RETRY_SLEEP = api_retry_sleep
    client.API_RETRY_MULTIPLIER = api_retry_multiplier
    client._max_length = 4096
    client.generation_parameters = GenerationParameters()
    # disable_tqdm is a read-only @property on LightevalModel (returns False).
    # We leave it as-is; tqdm output in tests is harmless.

    # Minimal PromptManager stub: _prepare_plain_text returns doc.query directly
    pm = MagicMock()
    pm._prepare_plain_text = lambda doc: doc.query
    client.prompt_manager = pm

    # Disable the @cached decorator by setting _cache = None
    client._cache = None
    return client


# ---------------------------------------------------------------------------
# 1. _find_continuation_start — Token Alignment Engine unit tests
# ---------------------------------------------------------------------------


class TestFindContinuationStart:
    """Pure function tests — no mocking needed."""

    def test_text_offset_layer1_exact_boundary(self):
        """Continuation starts exactly at len(context_str) characters."""
        # context = "Q:" (2 chars), continuation = " A"
        context_str = "Q:"
        lp = make_logprobs(
            tokens=["Q", ":", " A", "_gen"],
            token_logprobs=[None, -0.1, -0.5, -0.9],
            text_offset=[0, 1, 2, 4],  # " A" starts at offset 2 == len("Q:")
        )
        result = LiteLLMClient._find_continuation_start(lp, context_str, "gpt-3.5-turbo-instruct")
        assert result == 2

    def test_text_offset_layer1_midpoint(self):
        """Works when the context ends mid-word and text_offset values are larger."""
        context_str = "Hello world "  # 12 chars
        lp = make_logprobs(
            tokens=["Hello", " world", " ", "foo"],
            token_logprobs=[None, -0.2, -0.1, -0.3],
            text_offset=[0, 5, 11, 12],  # "foo" starts at 12 == len(context)
        )
        result = LiteLLMClient._find_continuation_start(lp, context_str, "gpt-3.5-turbo-instruct")
        assert result == 3

    def test_text_offset_all_context_empty_continuation(self):
        """All tokens belong to context (empty continuation) → returns len(tokens)."""
        context_str = "ABCD"
        lp = make_logprobs(
            tokens=["A", "B", "C", "D"],
            token_logprobs=[None] * 4,
            text_offset=[0, 1, 2, 3],  # no token starts at offset >= 4
        )
        result = LiteLLMClient._find_continuation_start(lp, context_str, "gpt-3.5-turbo-instruct")
        assert result == 4  # == len(tokens), signals empty continuation

    def test_tiktoken_fallback_called_when_no_text_offset(self):
        """Layer 2: litellm.encode is called when text_offset is absent."""
        context_str = "Hello world"
        lp = make_logprobs(
            tokens=["Hello", " world", " foo"],
            token_logprobs=[None, -0.5, -0.3],
            text_offset=None,  # force fallback
        )
        with patch("lighteval.models.endpoints.litellm_model.encode", return_value=[1, 2]) as mock_enc:
            result = LiteLLMClient._find_continuation_start(lp, context_str, "gpt-3.5-turbo-instruct")

        mock_enc.assert_called_once_with("gpt-3.5-turbo-instruct", context_str)
        assert result == 2  # len([1, 2])

    def test_tiktoken_fallback_called_when_text_offset_is_empty_list(self):
        """Empty text_offset list is falsy → Layer 2 fallback."""
        context_str = "ctx"
        lp = make_logprobs(["ctx", "cont"], [None, -0.3], text_offset=[])
        with patch("lighteval.models.endpoints.litellm_model.encode", return_value=[99]) as mock_enc:
            result = LiteLLMClient._find_continuation_start(lp, context_str, "gpt-3.5-turbo-instruct")
        assert result == 1
        mock_enc.assert_called_once()

    def test_tiktoken_fallback_encode_failure_returns_zero(self):
        """If encode raises, we fall back gracefully to 0 (log a warning)."""
        context_str = "ctx"
        lp = make_logprobs(["ctx", "cont"], [None, -0.3], text_offset=None)
        with patch("lighteval.models.endpoints.litellm_model.encode", side_effect=RuntimeError("tiktoken unavailable")):
            result = LiteLLMClient._find_continuation_start(lp, context_str, "unknown-model")
        assert result == 0


# ---------------------------------------------------------------------------
# 2. _check_argmax — Argmax unit tests
# ---------------------------------------------------------------------------


class TestCheckArgmax:
    """Mirrors vLLM's `rank == 1` semantics. Last token is excluded (max_tokens=1 artifact)."""

    def test_all_continuation_tokens_match_top1(self):
        # tokens: [ctx, contA, contB, generated]
        # cont_start=1, cont_end=3 → check positions 1 and 2
        tokens = ["ctx", " A", " B", "_gen"]
        top_logprobs = [
            {"ctx": -0.1},
            {" A": -0.5},   # actual token " A" IS top-1 ✓
            {" B": -0.3},   # actual token " B" IS top-1 ✓
            {"_gen": -0.9},
        ]
        result = LiteLLMClient._check_argmax(tokens, [], top_logprobs, cont_start=1)
        assert result is True

    def test_first_continuation_token_does_not_match(self):
        tokens = ["ctx", " A", "_gen"]
        top_logprobs = [
            {"ctx": -0.1},
            {" B": -0.2},   # top-1 is " B" but actual is " A" ✗
            {"_gen": -0.9},
        ]
        result = LiteLLMClient._check_argmax(tokens, [], top_logprobs, cont_start=1)
        assert result is False

    def test_partial_match_fails_overall(self):
        # Two continuation tokens; first matches, second does not
        tokens = ["ctx", " A", " C", "_gen"]
        top_logprobs = [
            {"ctx": -0.1},
            {" A": -0.5},   # ✓
            {" B": -0.3},   # top-1 is " B" but actual is " C" ✗
            {"_gen": -0.9},
        ]
        result = LiteLLMClient._check_argmax(tokens, [], top_logprobs, cont_start=1)
        assert result is False

    def test_empty_continuation_returns_true(self):
        # cont_start == len(tokens) - 1 means no continuation tokens
        tokens = ["ctx", "_gen"]
        result = LiteLLMClient._check_argmax(tokens, [], [{"ctx": -0.1}, {"_gen": -0.9}], cont_start=1)
        assert result is True

    def test_empty_top_logprobs_returns_false(self):
        result = LiteLLMClient._check_argmax(["a", "b", "_gen"], [], [], cont_start=0)
        assert result is False

    def test_none_top_dict_at_position_returns_false(self):
        tokens = ["ctx", " A", "_gen"]
        top_logprobs = [{"ctx": -0.1}, None, {"_gen": -0.5}]
        result = LiteLLMClient._check_argmax(tokens, [], top_logprobs, cont_start=1)
        assert result is False

    def test_cont_start_beyond_tokens_length_returns_false(self):
        tokens = ["a"]
        top_logprobs = [{"a": -0.1}]
        # cont_start == len(tokens) - 1 means empty continuation → True
        # cont_start > len(tokens) - 1 means start >= end → True
        result = LiteLLMClient._check_argmax(tokens, [], top_logprobs, cont_start=5)
        assert result is True


# ---------------------------------------------------------------------------
# 3. _call_api_text_completion_async — backoff and retry tests
# ---------------------------------------------------------------------------


class TestCallApiTextCompletionAsync:
    """Tests the async API caller: success, 429 backoff, total failure, semaphore."""

    # Custom exception standing in for litellm.RateLimitError in tests
    class _FakeRateLimitError(Exception):
        pass

    class _FakeGenericError(Exception):
        pass

    def _run_async(self, coro):
        return asyncio.run(coro)

    def test_success_on_first_attempt(self):
        client = make_bare_client()
        fake_resp = make_response(["Hello", " world", "_gen"], [None, -0.5, -0.1])

        async def run():
            sem = asyncio.Semaphore(10)
            with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
                mock_lit.RateLimitError = self._FakeRateLimitError
                mock_lit.atext_completion = AsyncMock(return_value=fake_resp)
                return await client._call_api_text_completion_async("Hello world", sem)

        result = self._run_async(run())
        assert result is fake_resp

    def test_rate_limit_429_then_success(self):
        """First call raises RateLimitError; second call succeeds."""
        client = make_bare_client(api_max_retry=3)
        fake_resp = make_response(["tok", "_gen"], [None, -0.3])

        async def run():
            sem = asyncio.Semaphore(10)
            with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
                mock_lit.RateLimitError = self._FakeRateLimitError
                mock_lit.atext_completion = AsyncMock(
                    side_effect=[self._FakeRateLimitError("429 rate limited"), fake_resp]
                )
                return await client._call_api_text_completion_async("test prompt", sem)

        result = self._run_async(run())
        assert result is fake_resp

    def test_generic_error_then_success(self):
        """Non-rate-limit transient error is also retried with backoff."""
        client = make_bare_client(api_max_retry=3)
        fake_resp = make_response(["t"], [None])

        async def run():
            sem = asyncio.Semaphore(10)
            with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
                mock_lit.RateLimitError = self._FakeRateLimitError
                mock_lit.atext_completion = AsyncMock(
                    side_effect=[self._FakeGenericError("timeout"), fake_resp]
                )
                return await client._call_api_text_completion_async("test", sem)

        result = self._run_async(run())
        assert result is fake_resp

    def test_all_retries_exhausted_returns_none(self):
        """All API_MAX_RETRY attempts fail → returns None gracefully."""
        client = make_bare_client(api_max_retry=3)

        async def run():
            sem = asyncio.Semaphore(10)
            with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
                mock_lit.RateLimitError = self._FakeRateLimitError
                mock_lit.atext_completion = AsyncMock(
                    side_effect=self._FakeGenericError("permanent failure")
                )
                return await client._call_api_text_completion_async("test", sem)

        result = self._run_async(run())
        assert result is None

    def test_rate_limit_all_retries_exhausted_returns_none(self):
        """Persistent 429 across all retries → None, not an exception."""
        client = make_bare_client(api_max_retry=2)

        async def run():
            sem = asyncio.Semaphore(10)
            with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
                mock_lit.RateLimitError = self._FakeRateLimitError
                mock_lit.atext_completion = AsyncMock(
                    side_effect=self._FakeRateLimitError("perpetual 429")
                )
                return await client._call_api_text_completion_async("test", sem)

        result = self._run_async(run())
        assert result is None

    def test_semaphore_limits_concurrency(self):
        """Semaphore(1) causes calls to serialise; all results are still returned."""
        client = make_bare_client(concurrent_requests=1)
        call_order = []

        async def fake_atext(*args, **kwargs):
            call_order.append(kwargs.get("prompt", "?"))
            return make_response(["t"], [None])

        async def run():
            sem = asyncio.Semaphore(1)
            with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
                mock_lit.RateLimitError = self._FakeRateLimitError
                mock_lit.atext_completion = fake_atext
                results = await asyncio.gather(
                    client._call_api_text_completion_async("A", sem),
                    client._call_api_text_completion_async("B", sem),
                    client._call_api_text_completion_async("C", sem),
                )
            return results

        results = asyncio.run(run())
        assert len(results) == 3
        assert all(r is not None for r in results)
        assert len(call_order) == 3

    def test_correct_api_parameters_passed(self):
        """Verifies echo=True, logprobs=1, max_tokens=1, temperature=0.0 are sent."""
        client = make_bare_client()
        client.model = "gpt-3.5-turbo-instruct"
        client.api_key = "sk-test"
        fake_resp = make_response(["t"], [None])

        async def run():
            sem = asyncio.Semaphore(10)
            with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
                mock_lit.RateLimitError = Exception
                mock_lit.atext_completion = AsyncMock(return_value=fake_resp)
                await client._call_api_text_completion_async("the prompt", sem)
                call_kwargs = mock_lit.atext_completion.call_args.kwargs
            return call_kwargs

        kw = asyncio.run(run())
        assert kw["echo"] is True
        assert kw["logprobs"] == 1
        assert kw["max_tokens"] == 1
        assert kw["temperature"] == 0.0
        assert kw["prompt"] == "the prompt"
        assert kw["model"] == "gpt-3.5-turbo-instruct"
        assert kw["api_key"] == "sk-test"


# ---------------------------------------------------------------------------
# 4. _process_doc_loglikelihood_async — per-doc processing tests
# ---------------------------------------------------------------------------


class TestProcessDocLoglikelihoodAsync:
    """Tests the per-doc async processor using patched _call_api_text_completion_async."""

    def _run_async(self, coro):
        return asyncio.run(coro)

    def _make_doc_with_two_choices(self):
        return make_doc("Q:", [" A", " B"], gold_index=0)

    def test_basic_two_choices_correct_logprobs(self):
        """Correct logprob sums and argmax booleans for a 2-choice doc."""
        client = make_bare_client()

        # context = "Q:" (2 chars), text_offset: " A" starts at offset 2
        resp_a = make_response(
            tokens=["Q", ":", " A", "_gen"],
            token_logprobs=[None, -0.1, -0.5, -0.9],
            top_logprobs=[{"Q": -0.05}, {":": -0.1}, {" A": -0.5}, {"_gen": -0.9}],
            text_offset=[0, 1, 2, 4],
        )
        # For choice B: " B" is NOT the top-1 (top-1 would be " A")
        resp_b = make_response(
            tokens=["Q", ":", " B", "_gen"],
            token_logprobs=[None, -0.1, -2.0, -0.9],
            top_logprobs=[{"Q": -0.05}, {":": -0.1}, {" A": -0.5}, {"_gen": -0.9}],
            text_offset=[0, 1, 2, 4],
        )

        async def run():
            sem = asyncio.Semaphore(10)
            with patch.object(
                client,
                "_call_api_text_completion_async",
                AsyncMock(side_effect=[resp_a, resp_b]),
            ):
                return await client._process_doc_loglikelihood_async(
                    self._make_doc_with_two_choices(), "Q:", sem
                )

        result = self._run_async(run())

        assert isinstance(result, ModelResponse)
        assert len(result.logprobs) == 2
        assert result.logprobs[0] == pytest.approx(-0.5)   # only continuation token for A
        assert result.logprobs[1] == pytest.approx(-2.0)   # only continuation token for B
        assert result.argmax_logits_eq_gold[0] is True     # " A" was top-1
        assert result.argmax_logits_eq_gold[1] is False    # " B" was NOT top-1

    def test_multi_token_continuation_sums_correctly(self):
        """Continuation with two tokens: sum of both logprobs."""
        client = make_bare_client()

        resp = make_response(
            tokens=["Q", ":", " yes", " sir", "_gen"],
            token_logprobs=[None, -0.1, -0.4, -0.6, -0.9],
            top_logprobs=[
                {"Q": -0.05}, {":": -0.1},
                {" yes": -0.4}, {" sir": -0.6}, {"_gen": -0.9},
            ],
            text_offset=[0, 1, 2, 6, 10],  # context "Q:" ends at char 2
        )

        doc = make_doc("Q:", [" yes sir"], gold_index=0)

        async def run():
            sem = asyncio.Semaphore(10)
            with patch.object(client, "_call_api_text_completion_async", AsyncMock(return_value=resp)):
                return await client._process_doc_loglikelihood_async(doc, "Q:", sem)

        result = self._run_async(run())
        # logprobs for " yes" and " sir" → -0.4 + -0.6 = -1.0
        assert result.logprobs[0] == pytest.approx(-1.0)
        assert result.argmax_logits_eq_gold[0] is True

    def test_failed_api_call_returns_neg_inf_sentinel(self):
        """If the API call returns None, logprob = -inf, argmax = False."""
        client = make_bare_client()
        doc = self._make_doc_with_two_choices()

        async def run():
            sem = asyncio.Semaphore(10)
            with patch.object(
                client, "_call_api_text_completion_async", AsyncMock(return_value=None)
            ):
                return await client._process_doc_loglikelihood_async(doc, "Q:", sem)

        result = self._run_async(run())
        assert result.logprobs == [float("-inf"), float("-inf")]
        assert result.argmax_logits_eq_gold == [False, False]

    def test_none_logprobs_object_in_response_returns_sentinel(self):
        """Response with logprobs=None → sentinel values."""
        client = make_bare_client()
        doc = make_doc("Q:", [" A"], gold_index=0)

        bad_resp = SimpleNamespace(choices=[SimpleNamespace(logprobs=None)])

        async def run():
            sem = asyncio.Semaphore(10)
            with patch.object(
                client, "_call_api_text_completion_async", AsyncMock(return_value=bad_resp)
            ):
                return await client._process_doc_loglikelihood_async(doc, "Q:", sem)

        result = self._run_async(run())
        assert result.logprobs == [float("-inf")]
        assert result.argmax_logits_eq_gold == [False]

    def test_empty_choices_list_returns_empty_response(self):
        """Doc with no choices → empty lists (no API calls fired)."""
        client = make_bare_client()
        doc = make_doc("Q:", choices=[])

        async def run():
            sem = asyncio.Semaphore(10)
            # We verify that atext_completion is never called when choices is empty
            with patch.object(
                client, "_call_api_text_completion_async", AsyncMock()
            ) as mock_call:
                result = await client._process_doc_loglikelihood_async(doc, "Q:", sem)
                assert mock_call.call_count == 0
            return result

        result = self._run_async(run())
        assert result.logprobs == []
        assert result.argmax_logits_eq_gold == []

    def test_context_is_prepended_to_each_choice(self):
        """Verifies full_text = context + choice is sent for each choice."""
        client = make_bare_client()
        doc = make_doc("CTX", [" X", " Y"], gold_index=0)
        fake_resp = make_response(["C", "T", "X", " X", "_gen"], [None, -0.1, -0.1, -0.5, -0.9],
                                  text_offset=[0, 1, 2, 3, 5])

        captured_prompts = []

        async def fake_call(full_text, semaphore):
            captured_prompts.append(full_text)
            return fake_resp

        async def run():
            sem = asyncio.Semaphore(10)
            with patch.object(client, "_call_api_text_completion_async", side_effect=fake_call):
                return await client._process_doc_loglikelihood_async(doc, "CTX", sem)

        self._run_async(run())
        assert captured_prompts == ["CTX X", "CTX Y"]

    def test_result_order_matches_choice_order(self):
        """asyncio.gather preserves order; results align 1-to-1 with choices."""
        client = make_bare_client()
        doc = make_doc("Q:", [" A", " B", " C"], gold_index=1)

        # Each choice has a distinct logprob so we can verify ordering.
        # actual_tok is always " A" (a valid string); top_tok is " A" when
        # is_top=True (so actual==top → argmax True) or " Z" otherwise.
        def make_choice_resp(choice_lp, is_top):
            actual_tok = " A"
            top_tok = " A" if is_top else " Z"
            return make_response(
                tokens=["Q", ":", actual_tok, "_gen"],
                token_logprobs=[None, -0.1, float(choice_lp), -0.9],
                top_logprobs=[{"Q": -0.1}, {":": -0.1}, {top_tok: float(choice_lp)}, {"_gen": -0.9}],
                text_offset=[0, 1, 2, 3],
            )

        resps = [make_choice_resp(-1.0, True), make_choice_resp(-2.0, False), make_choice_resp(-3.0, False)]

        async def run():
            sem = asyncio.Semaphore(10)
            with patch.object(
                client, "_call_api_text_completion_async", AsyncMock(side_effect=resps)
            ):
                return await client._process_doc_loglikelihood_async(doc, "Q:", sem)

        result = self._run_async(run())
        assert result.logprobs[0] == pytest.approx(-1.0)
        assert result.logprobs[1] == pytest.approx(-2.0)
        assert result.logprobs[2] == pytest.approx(-3.0)
        assert result.argmax_logits_eq_gold == [True, False, False]


# ---------------------------------------------------------------------------
# 5. loglikelihood (full integration) — end-to-end with fully mocked pipeline
# ---------------------------------------------------------------------------


class TestLoglikelihoodIntegration:
    """Integration tests: exercise loglikelihood() top to bottom.

    All async work is mocked at _process_doc_loglikelihood_async so we validate
    the orchestration layer (LoglikelihoodDataset, original ordering, output shape)
    without needing live API credentials.
    """

    def _make_known_response(self, logprob_vals, argmax_vals, context="ctx"):
        """Convenience: build a ModelResponse with explicit per-choice values."""
        return ModelResponse(
            input=context,
            logprobs=list(logprob_vals),
            argmax_logits_eq_gold=list(argmax_vals),
        )

    def test_two_docs_output_shape_and_order(self):
        """Two docs returned in original input order."""
        client = make_bare_client()

        docs = [
            make_doc("Question 1:", [" A", " B", " C", " D"], gold_index=0, doc_id="0"),
            make_doc("Q2:", [" W", " X"], gold_index=1, doc_id="1"),
        ]

        resp0 = self._make_known_response([-0.3, -1.5, -2.0, -3.0], [True, False, False, False])
        resp1 = self._make_known_response([-1.8, -0.2], [False, True])

        # Map doc_id → pre-built response
        responses_by_id = {"0": resp0, "1": resp1}

        async def fake_process_doc(doc, context_str, semaphore):
            return responses_by_id[doc.id]

        with patch.object(client, "_process_doc_loglikelihood_async", side_effect=fake_process_doc), \
             patch.object(type(client), "disable_tqdm", new_callable=PropertyMock, return_value=True):
            results = client.loglikelihood(docs)

        assert len(results) == 2

        # Original order preserved (doc "0" first, "1" second)
        assert results[0].logprobs == pytest.approx([-0.3, -1.5, -2.0, -3.0])
        assert results[1].logprobs == pytest.approx([-1.8, -0.2])
        assert results[0].argmax_logits_eq_gold == [True, False, False, False]
        assert results[1].argmax_logits_eq_gold == [False, True]

    def test_single_doc_four_choices(self):
        """Single doc, 4 choices, correct output shape."""
        client = make_bare_client()
        doc = make_doc("Q:", [" A", " B", " C", " D"], gold_index=2, doc_id="0")

        pre_resp = self._make_known_response([-2.0, -1.0, -0.5, -3.0], [False, False, True, False])

        async def fake_process(doc, context_str, semaphore):
            return pre_resp

        with patch.object(client, "_process_doc_loglikelihood_async", side_effect=fake_process), \
             patch.object(type(client), "disable_tqdm", new_callable=PropertyMock, return_value=True):
            results = client.loglikelihood([doc])

        assert len(results) == 1
        assert len(results[0].logprobs) == 4
        # Choice index 2 has the highest (least negative) logprob
        assert results[0].logprobs.index(max(results[0].logprobs)) == 2

    def test_original_order_restored_after_dataset_sorting(self):
        """LoglikelihoodDataset sorts by prompt length; loglikelihood must un-sort."""
        client = make_bare_client()

        # Deliberately create docs with different query lengths so the dataset
        # reorders them. The short query will be sorted first by LoglikelihoodDataset.
        short_doc = make_doc("Q?", [" Y", " N"], gold_index=0, doc_id="short")
        long_doc = make_doc(
            "This is a much longer question that triggers reordering:",
            [" A", " B"],
            gold_index=1,
            doc_id="long",
        )

        resp_short = self._make_known_response([-0.1, -1.0], [True, False], context="Q?")
        resp_long = self._make_known_response([-2.0, -0.3], [False, True])

        responses_by_id = {"short": resp_short, "long": resp_long}

        async def fake_process(doc, context_str, semaphore):
            return responses_by_id[doc.id]

        # Input order: [short, long]. Dataset sorts long first. loglikelihood must
        # restore original order → results[0] is short_doc's response.
        with patch.object(client, "_process_doc_loglikelihood_async", side_effect=fake_process), \
             patch.object(type(client), "disable_tqdm", new_callable=PropertyMock, return_value=True):
            results = client.loglikelihood([short_doc, long_doc])

        assert len(results) == 2
        # results[0] must correspond to short_doc (original position 0)
        assert results[0].logprobs == pytest.approx([-0.1, -1.0])
        # results[1] must correspond to long_doc (original position 1)
        assert results[1].logprobs == pytest.approx([-2.0, -0.3])

    def test_all_api_failures_return_neg_inf_per_choice(self):
        """Graceful degradation: all docs return -inf when API is completely down."""
        client = make_bare_client()
        docs = [make_doc("Q:", [" A", " B"], gold_index=0, doc_id=str(i)) for i in range(3)]

        async def fake_process(doc, context_str, semaphore):
            return ModelResponse(
                input=context_str,
                logprobs=[float("-inf"), float("-inf")],
                argmax_logits_eq_gold=[False, False],
            )

        with patch.object(client, "_process_doc_loglikelihood_async", side_effect=fake_process), \
             patch.object(type(client), "disable_tqdm", new_callable=PropertyMock, return_value=True):
            results = client.loglikelihood(docs)

        assert len(results) == 3
        for r in results:
            assert r.logprobs == [float("-inf"), float("-inf")]
            assert r.argmax_logits_eq_gold == [False, False]


# ---------------------------------------------------------------------------
# 6. _check_text_completion_support — provider guard tests
# ---------------------------------------------------------------------------


class TestCheckTextCompletionSupport:
    """The guard should warn when 'echo' is absent from litellm's param list and
    stay silent (no exception) when 'echo' is present or the check itself fails."""

    def _make_client(self, model="gpt-3.5-turbo-instruct", provider="openai"):
        client = make_bare_client(model=model)
        client.provider = provider
        return client

    def test_no_warning_when_echo_supported(self, caplog):
        client = self._make_client()
        with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
            mock_lit.get_supported_openai_params = MagicMock(return_value=["echo", "logprobs", "max_tokens"])
            import logging
            with caplog.at_level(logging.WARNING, logger="lighteval.models.endpoints.litellm_model"):
                client._check_text_completion_support()
        assert "echo" not in caplog.text or "does not list" not in caplog.text

    def test_warning_emitted_when_echo_not_supported(self, caplog):
        client = self._make_client(model="gpt-4o", provider="openai")
        with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
            mock_lit.get_model_info = MagicMock(return_value={"mode": "chat"})
            import logging
            with caplog.at_level(logging.WARNING, logger="lighteval.models.endpoints.litellm_model"):
                client._check_text_completion_support()
        assert "chat-only" in caplog.text

    def test_no_crash_when_registry_lookup_raises(self):
        """If litellm's param registry explodes, the guard must stay silent."""
        client = self._make_client()
        with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
            mock_lit.get_supported_openai_params = MagicMock(side_effect=RuntimeError("registry unavailable"))
            client._check_text_completion_support()  # must not raise

    def test_no_crash_when_params_returns_none(self):
        client = self._make_client()
        with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
            mock_lit.get_supported_openai_params = MagicMock(return_value=None)
            client._check_text_completion_support()  # None → treated as empty list, no crash

    def test_warning_contains_model_name(self, caplog):
        client = self._make_client(model="claude-3-opus", provider="anthropic")
        with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
            mock_lit.get_model_info = MagicMock(return_value={"mode": "chat"})
            import logging
            with caplog.at_level(logging.WARNING, logger="lighteval.models.endpoints.litellm_model"):
                client._check_text_completion_support()
        assert "claude-3-opus" in caplog.text

    def test_loglikelihood_calls_guard(self):
        """loglikelihood() must call _check_text_completion_support before processing."""
        client = make_bare_client()
        guard_called = {"n": 0}

        def fake_guard(self_inner):
            guard_called["n"] += 1

        async def fake_process(doc, context_str, semaphore):
            return ModelResponse(input=context_str, logprobs=[-0.5], argmax_logits_eq_gold=[True])

        with patch.object(LiteLLMClient, "_check_text_completion_support", fake_guard), \
             patch.object(client, "_process_doc_loglikelihood_async", side_effect=fake_process), \
             patch.object(type(client), "disable_tqdm", new_callable=PropertyMock, return_value=True):
            client.loglikelihood([make_doc("Q:", [" A"], gold_index=0, doc_id="0")])

        assert guard_called["n"] == 1

    def test_loglikelihood_rolling_calls_guard(self):
        """loglikelihood_rolling() must also call _check_text_completion_support."""
        client = make_bare_client()
        guard_called = {"n": 0}

        def fake_guard(self_inner):
            guard_called["n"] += 1

        async def fake_process(doc, semaphore):
            return ModelResponse(input=doc.query, logprobs=[-0.1, -0.2])

        with patch.object(LiteLLMClient, "_check_text_completion_support", fake_guard), \
             patch.object(client, "_process_doc_rolling_async", side_effect=fake_process), \
             patch.object(type(client), "disable_tqdm", new_callable=PropertyMock, return_value=True):
            client.loglikelihood_rolling([make_doc("Hello world", choices=[], gold_index=0, doc_id="0")])

        assert guard_called["n"] == 1


# ---------------------------------------------------------------------------
# 7. _process_doc_rolling_async — per-token perplexity tests
# ---------------------------------------------------------------------------


class TestProcessDocRollingAsync:
    """Tests the per-document rolling log-likelihood computation."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_basic_rolling_sums_all_token_logprobs(self):
        """token_logprobs[1:-1] are the valid rolling logprobs."""
        client = make_bare_client()
        doc = make_doc("Hello world", choices=[], gold_index=0)

        # 5 tokens: [null, -0.3, -0.5, -0.2, generated]
        # rolling = [-0.3, -0.5, -0.2]  (indices 1..3, excluding last)
        resp = make_response(
            tokens=["Hello", " world", " foo", " bar", "_gen"],
            token_logprobs=[None, -0.3, -0.5, -0.2, -0.9],
        )

        async def run():
            sem = asyncio.Semaphore(10)
            with patch.object(client, "_call_api_text_completion_async", AsyncMock(return_value=resp)):
                return await client._process_doc_rolling_async(doc, sem)

        result = self._run(run())
        assert isinstance(result, ModelResponse)
        assert result.logprobs == pytest.approx([-0.3, -0.5, -0.2])

    def test_single_token_doc_returns_empty_logprobs(self):
        """A 1-token document: token_logprobs = [None, generated] → nothing to sum."""
        client = make_bare_client()
        doc = make_doc("Hi", choices=[], gold_index=0)

        resp = make_response(
            tokens=["Hi", "_gen"],
            token_logprobs=[None, -0.9],
        )

        async def run():
            sem = asyncio.Semaphore(10)
            with patch.object(client, "_call_api_text_completion_async", AsyncMock(return_value=resp)):
                return await client._process_doc_rolling_async(doc, sem)

        result = self._run(run())
        assert result.logprobs == []

    def test_failed_api_call_returns_neg_inf(self):
        client = make_bare_client()
        doc = make_doc("Hello", choices=[], gold_index=0)

        async def run():
            sem = asyncio.Semaphore(10)
            with patch.object(client, "_call_api_text_completion_async", AsyncMock(return_value=None)):
                return await client._process_doc_rolling_async(doc, sem)

        result = self._run(run())
        assert result.logprobs == [float("-inf")]

    def test_null_logprobs_in_response(self):
        client = make_bare_client()
        doc = make_doc("Hello", choices=[], gold_index=0)
        bad_resp = SimpleNamespace(choices=[SimpleNamespace(logprobs=None)])

        async def run():
            sem = asyncio.Semaphore(10)
            with patch.object(client, "_call_api_text_completion_async", AsyncMock(return_value=bad_resp)):
                return await client._process_doc_rolling_async(doc, sem)

        result = self._run(run())
        assert result.logprobs == [float("-inf")]

    def test_none_values_in_token_logprobs_are_filtered(self):
        """Unexpected None values mid-sequence are skipped gracefully."""
        client = make_bare_client()
        doc = make_doc("A B C", choices=[], gold_index=0)

        resp = make_response(
            tokens=["A", " B", " C", "_gen"],
            token_logprobs=[None, -0.4, None, -0.9],  # middle None is unusual but handled
        )

        async def run():
            sem = asyncio.Semaphore(10)
            with patch.object(client, "_call_api_text_completion_async", AsyncMock(return_value=resp)):
                return await client._process_doc_rolling_async(doc, sem)

        result = self._run(run())
        # token_logprobs[1:-1] = [-0.4, None] → filter None → [-0.4]
        assert result.logprobs == pytest.approx([-0.4])

    def test_correct_prompt_sent_to_api(self):
        """The full plain-text doc is sent as the prompt."""
        client = make_bare_client()
        doc = make_doc("The quick brown fox", choices=[], gold_index=0)
        captured = {}

        async def fake_call(full_text, semaphore):
            captured["prompt"] = full_text
            return make_response(["The", " quick", "_gen"], [None, -0.3, -0.9])

        async def run():
            sem = asyncio.Semaphore(10)
            with patch.object(client, "_call_api_text_completion_async", side_effect=fake_call):
                return await client._process_doc_rolling_async(doc, sem)

        self._run(run())
        assert captured["prompt"] == "The quick brown fox"


# ---------------------------------------------------------------------------
# 8. loglikelihood_rolling integration
# ---------------------------------------------------------------------------


class TestLoglikelihoodRollingIntegration:
    """End-to-end tests for loglikelihood_rolling() orchestration."""

    def test_three_docs_correct_shape_and_order(self):
        client = make_bare_client()
        docs = [
            make_doc("Doc one text", choices=[], gold_index=0, doc_id="0"),
            make_doc("A much longer second document for sorting test", choices=[], gold_index=0, doc_id="1"),
            make_doc("Short", choices=[], gold_index=0, doc_id="2"),
        ]

        # Pre-built per-doc responses keyed by doc id
        responses = {
            "0": ModelResponse(input="Doc one text", logprobs=[-0.3, -0.5]),
            "1": ModelResponse(input="...", logprobs=[-0.1, -0.2, -0.4]),
            "2": ModelResponse(input="Short", logprobs=[-0.8]),
        }

        async def fake_rolling(doc, semaphore):
            return responses[doc.id]

        with patch.object(client, "_process_doc_rolling_async", side_effect=fake_rolling), \
             patch.object(type(client), "disable_tqdm", new_callable=PropertyMock, return_value=True):
            results = client.loglikelihood_rolling(docs)

        assert len(results) == 3
        # Original order must be preserved (LoglikelihoodDataset sorts internally)
        assert results[0].logprobs == pytest.approx([-0.3, -0.5])
        assert results[1].logprobs == pytest.approx([-0.1, -0.2, -0.4])
        assert results[2].logprobs == pytest.approx([-0.8])

    def test_perplexity_sum_compatible(self):
        """np.sum(result.logprobs) must give the total document log-likelihood."""
        import numpy as np

        client = make_bare_client()
        doc = make_doc("Hello world", choices=[], gold_index=0, doc_id="0")

        async def fake_rolling(doc, semaphore):
            return ModelResponse(input=doc.query, logprobs=[-0.3, -0.5, -0.2])

        with patch.object(client, "_process_doc_rolling_async", side_effect=fake_rolling), \
             patch.object(type(client), "disable_tqdm", new_callable=PropertyMock, return_value=True):
            results = client.loglikelihood_rolling([doc])

        total_logprob = float(np.sum(results[0].logprobs))
        assert total_logprob == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 9. Provider guard — improved mode-based detection
# ---------------------------------------------------------------------------


class TestCheckTextCompletionSupportModeDetection:
    """The guard uses litellm.get_model_info() 'mode' field, not the params list."""

    def _make_client(self, model="gpt-3.5-turbo-instruct", provider="openai"):
        client = make_bare_client(model=model)
        client.provider = provider
        return client

    def test_no_warning_for_completion_mode_model(self, caplog):
        """mode='completion' → no warning (correct: model supports text_completion)."""
        client = self._make_client("gpt-3.5-turbo-instruct")
        with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
            mock_lit.get_model_info = MagicMock(return_value={"mode": "completion"})
            import logging
            with caplog.at_level(logging.WARNING, logger="lighteval.models.endpoints.litellm_model"):
                client._check_text_completion_support()
        assert "chat-only" not in caplog.text

    def test_warning_for_chat_mode_model(self, caplog):
        """mode='chat' → warning emitted."""
        client = self._make_client("gpt-4o", "openai")
        with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
            mock_lit.get_model_info = MagicMock(return_value={"mode": "chat"})
            import logging
            with caplog.at_level(logging.WARNING, logger="lighteval.models.endpoints.litellm_model"):
                client._check_text_completion_support()
        assert "chat-only" in caplog.text
        assert "gpt-4o" in caplog.text

    def test_no_warning_when_mode_field_absent(self, caplog):
        """mode not present in model_info → unknown model, proceed silently."""
        client = self._make_client("custom-model")
        with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
            mock_lit.get_model_info = MagicMock(return_value={"max_tokens": 4096})
            import logging
            with caplog.at_level(logging.WARNING, logger="lighteval.models.endpoints.litellm_model"):
                client._check_text_completion_support()
        assert "chat-only" not in caplog.text

    def test_no_crash_when_get_model_info_raises(self):
        """Any exception in model_info lookup must not propagate."""
        client = self._make_client()
        with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
            mock_lit.get_model_info = MagicMock(side_effect=KeyError("model not in registry"))
            client._check_text_completion_support()  # must not raise

    def test_no_crash_when_get_model_info_returns_none(self):
        client = self._make_client()
        with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
            mock_lit.get_model_info = MagicMock(return_value=None)
            client._check_text_completion_support()  # None → treated as {} → no warning


# ---------------------------------------------------------------------------
# 10. Seed forwarding
# ---------------------------------------------------------------------------


class TestSeedForwarding:
    """generation_parameters.seed must be passed to every text_completion call."""

    class _FakeRateLimitError(Exception):
        pass

    def test_seed_forwarded_via_text_completion_dict(self):
        """seed from generation_parameters flows into the API call via to_litellm_text_completion_dict."""
        client = make_bare_client()
        client.generation_parameters = GenerationParameters(seed=42)

        fake_resp = make_response(["t"], [None])

        async def run():
            sem = asyncio.Semaphore(10)
            with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
                mock_lit.RateLimitError = self._FakeRateLimitError
                mock_lit.atext_completion = AsyncMock(return_value=fake_resp)
                await client._call_api_text_completion_async("hello", sem)
                kw = mock_lit.atext_completion.call_args.kwargs
            return kw

        kw = asyncio.run(run())
        assert kw["seed"] == 42

    def test_no_seed_key_when_seed_not_set(self):
        """When seed is None, to_litellm_text_completion_dict omits it entirely
        (litellm.drop_params handles the rest)."""
        client = make_bare_client()
        client.generation_parameters = GenerationParameters()  # seed=None by default

        fake_resp = make_response(["t"], [None])

        async def run():
            sem = asyncio.Semaphore(10)
            with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
                mock_lit.RateLimitError = self._FakeRateLimitError
                mock_lit.atext_completion = AsyncMock(return_value=fake_resp)
                await client._call_api_text_completion_async("hello", sem)
                kw = mock_lit.atext_completion.call_args.kwargs
            return kw

        kw = asyncio.run(run())
        # seed was not set → not in the dict (omitted by to_litellm_text_completion_dict)
        assert "seed" not in kw

    def test_stop_tokens_forwarded(self):
        """stop_tokens from generation_parameters flows through to the API call."""
        client = make_bare_client()
        client.generation_parameters = GenerationParameters(stop_tokens=["\n", "END"])
        fake_resp = make_response(["t"], [None])

        async def run():
            sem = asyncio.Semaphore(10)
            with patch("lighteval.models.endpoints.litellm_model.litellm") as mock_lit:
                mock_lit.RateLimitError = self._FakeRateLimitError
                mock_lit.atext_completion = AsyncMock(return_value=fake_resp)
                await client._call_api_text_completion_async("hello", sem)
                return mock_lit.atext_completion.call_args.kwargs

        kw = asyncio.run(run())
        assert kw["stop"] == ["\n", "END"]


# ---------------------------------------------------------------------------
# 11. Input length guard
# ---------------------------------------------------------------------------


class TestWarnIfTooLong:
    """_warn_if_too_long emits a WARNING when encode() returns more tokens than max_length."""

    def _make_client(self, max_length=10):
        client = make_bare_client()
        client._max_length = max_length
        return client

    def test_warning_when_over_limit(self, caplog):
        client = self._make_client(max_length=3)
        with patch("lighteval.models.endpoints.litellm_model.encode", return_value=list(range(5))):
            import logging
            with caplog.at_level(logging.WARNING, logger="lighteval.models.endpoints.litellm_model"):
                client._warn_if_too_long("some long text", label="test")
        assert "exceeds max_length" in caplog.text
        assert "5" in caplog.text   # token count shown
        assert "3" in caplog.text   # max_length shown

    def test_no_warning_when_within_limit(self, caplog):
        client = self._make_client(max_length=100)
        with patch("lighteval.models.endpoints.litellm_model.encode", return_value=list(range(5))):
            import logging
            with caplog.at_level(logging.WARNING, logger="lighteval.models.endpoints.litellm_model"):
                client._warn_if_too_long("short text")
        assert "exceeds" not in caplog.text

    def test_no_crash_when_encode_raises(self):
        client = self._make_client()
        with patch("lighteval.models.endpoints.litellm_model.encode", side_effect=RuntimeError("tiktoken missing")):
            client._warn_if_too_long("text")  # must not raise

    def test_no_warning_when_max_length_is_none(self, caplog):
        """Unknown context window → skip silently."""
        client = make_bare_client()
        client._max_length = None
        with patch("lighteval.models.endpoints.litellm_model.encode", return_value=list(range(999))):
            import logging
            with caplog.at_level(logging.WARNING, logger="lighteval.models.endpoints.litellm_model"):
                client._warn_if_too_long("very long text")
        assert "exceeds" not in caplog.text

    def test_label_appears_in_warning(self, caplog):
        client = self._make_client(max_length=1)
        with patch("lighteval.models.endpoints.litellm_model.encode", return_value=[1, 2, 3]):
            import logging
            with caplog.at_level(logging.WARNING, logger="lighteval.models.endpoints.litellm_model"):
                client._warn_if_too_long("text", label="doc '42' longest choice")
        assert "doc '42'" in caplog.text

    def test_length_guard_called_in_process_doc_loglikelihood(self):
        """_warn_if_too_long is called once per doc using the longest choice."""
        client = make_bare_client()
        warn_calls = []

        def fake_warn(text, label=""):
            warn_calls.append((text, label))

        # Use choices of clearly different lengths so max(key=len) is deterministic
        doc = make_doc("Q:", [" A", " longer_choice"], gold_index=0)
        fake_resp = make_response(["Q", ":", " A", "_gen"], [None, -0.1, -0.5, -0.9],
                                  text_offset=[0, 1, 2, 4])

        async def run():
            sem = asyncio.Semaphore(10)
            with patch.object(client, "_warn_if_too_long", side_effect=fake_warn), \
                 patch.object(client, "_call_api_text_completion_async",
                              AsyncMock(return_value=fake_resp)):
                return await client._process_doc_loglikelihood_async(doc, "Q:", sem)

        asyncio.run(run())
        assert len(warn_calls) == 1                      # called exactly once per doc
        assert "longer_choice" in warn_calls[0][0]       # longest choice was used
        assert warn_calls[0][0].startswith("Q:")         # context is prepended

    def test_length_guard_called_in_process_doc_rolling(self):
        """_warn_if_too_long is called in _process_doc_rolling_async."""
        client = make_bare_client()
        warn_calls = []

        def fake_warn(text, label=""):
            warn_calls.append(text)

        doc = make_doc("Hello world", choices=[], gold_index=0)
        fake_resp = make_response(["Hello", " world", "_gen"], [None, -0.3, -0.9])

        async def run():
            sem = asyncio.Semaphore(10)
            with patch.object(client, "_warn_if_too_long", side_effect=fake_warn), \
                 patch.object(client, "_call_api_text_completion_async",
                              AsyncMock(return_value=fake_resp)):
                return await client._process_doc_rolling_async(doc, sem)

        asyncio.run(run())
        assert len(warn_calls) == 1
        assert warn_calls[0] == "Hello world"


# ---------------------------------------------------------------------------
# Regression: PR #1192 — greedy_until iterates split, not full dataset
# ---------------------------------------------------------------------------


class TestGreedyUntilSplitFix:
    """Regression test: greedy_until must build contexts from the current split only.

    The bug: `contexts = [prepare_prompt_api(doc) for doc in dataset]` iterated
    the entire dataset on every split iteration, causing each doc to be processed
    `num_splits` times instead of once.  The fix uses `for doc in split`.
    """

    def _make_greedy_doc(self, query, generation_size=32, doc_id="0"):
        doc = Doc(query=query, choices=[], gold_index=0, task_name="test", generation_size=generation_size)
        doc.id = doc_id
        return doc

    def _make_chat_response(self, content="answer"):
        choice = MagicMock()
        choice.message.content = content
        choice.message.reasoning_content = None
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_each_doc_prepared_exactly_once_across_two_splits(self):
        """Docs with different generation_size land in separate splits.
        prepare_prompt_api must be called once per doc, not once per split × N docs.
        """
        client = make_bare_client()

        # Different generation_size → GenerativeTaskDataset puts them in different splits
        doc_a = self._make_greedy_doc("Prompt A", generation_size=16, doc_id="a")
        doc_b = self._make_greedy_doc("Prompt B", generation_size=32, doc_id="b")
        docs = [doc_a, doc_b]

        prepared = []

        def tracking_prepare(doc):
            prepared.append(doc.query)
            return [{"role": "user", "content": doc.query}]

        client.prompt_manager.prepare_prompt_api.side_effect = tracking_prepare

        def fake_parallel(contexts, *args, **kwargs):
            return [self._make_chat_response() for _ in contexts]

        with patch.object(
            client, "_LiteLLMClient__call_api_parallel", side_effect=fake_parallel
        ), patch.object(type(client), "disable_tqdm", new_callable=PropertyMock, return_value=True):
            results = client.greedy_until(docs)

        assert len(prepared) == 2, (
            f"Expected 2 prepare_prompt_api calls (one per doc), got {len(prepared)}. "
            "Regression: greedy_until iterated full dataset instead of current split."
        )
        assert set(prepared) == {"Prompt A", "Prompt B"}
        assert len(results) == 2

    def test_single_split_all_docs_processed(self):
        """Sanity: one split (same generation_size) — all docs processed correctly."""
        client = make_bare_client()

        docs = [
            self._make_greedy_doc("Prompt A", generation_size=32, doc_id="a"),
            self._make_greedy_doc("Prompt B", generation_size=32, doc_id="b"),
            self._make_greedy_doc("Prompt C", generation_size=32, doc_id="c"),
        ]

        prepared = []

        def tracking_prepare(doc):
            prepared.append(doc.query)
            return [{"role": "user", "content": doc.query}]

        client.prompt_manager.prepare_prompt_api.side_effect = tracking_prepare

        def fake_parallel(contexts, *args, **kwargs):
            return [self._make_chat_response() for _ in contexts]

        with patch.object(
            client, "_LiteLLMClient__call_api_parallel", side_effect=fake_parallel
        ), patch.object(type(client), "disable_tqdm", new_callable=PropertyMock, return_value=True):
            results = client.greedy_until(docs)

        assert len(prepared) == 3
        assert set(prepared) == {"Prompt A", "Prompt B", "Prompt C"}
        assert len(results) == 3
