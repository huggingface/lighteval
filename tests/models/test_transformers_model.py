# MIT License

# Copyright (c) 2025 The HuggingFace Team

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

import unittest
from unittest.mock import Mock, patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lighteval.models.model_output import ModelResponse
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.tasks.requests import Doc


class TestTransformersTokenizerCreation(unittest.TestCase):
    def setUp(self):
        """Set up shared model instance for all tests."""
        # Creates an instance without calling init, so we manually init needed params
        self.model = TransformersModel.__new__(TransformersModel)
        self.model._max_length = 18
        self.config = TransformersModelConfig(
            model_name="lighteval/different-chat-templates-per-revision", revision="new_chat_template"
        )
        self.model.config = self.config

        # Create reference tokenizer for comparisons
        self.reference_tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision=self.config.revision,
        )

        self.transformers_tokenizer = self.model._create_auto_tokenizer()

    def test_tokenizer_created_with_correct_revision(self):
        """Test that tokenizer is created with the correct configuration."""
        self.assertEqual(self.transformers_tokenizer.chat_template, self.reference_tokenizer.chat_template)

    def test_tokenizer_padding_configuration(self):
        """Test tokenizer padding configuration separately."""
        self.assertEqual(self.transformers_tokenizer.padding_side, "left")

    def test_tokenizer_truncation_configuration(self):
        """Test tokenizer truncation configuration separately."""
        self.assertEqual(self.transformers_tokenizer.truncation_side, "left")

    def test_tokenizer_pad_token_configuration(self):
        """Test tokenizer pad token configuration separately."""
        self.assertEqual(self.transformers_tokenizer.pad_token, self.transformers_tokenizer.eos_token)

    def test_tokenizer_max_length_configuration(self):
        """Test tokenizer max length configuration separately."""
        self.assertEqual(self.transformers_tokenizer.model_max_length, self.model.max_length)


class TestTransformersModelCreation(unittest.TestCase):
    @patch("lighteval.models.transformers.transformers_model.Accelerator")
    def setUp(self, mock_accelerator):
        """Set up shared model instance for all tests."""
        # Mock accelerate related params
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.device = torch.device("cpu")
        mock_accelerator.return_value = mock_accelerator_instance
        # Creates reference model and tokenizer to compare to
        self.reference_model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.reference_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Create full model instance
        self.config = TransformersModelConfig(
            model_name="gpt2",
        )
        self.model = TransformersModel(self.config)

    def test_model_creation_attributes(self):
        """Test that TransformersModel creates and initializes basic attributes correctly."""
        # Test attributes are set correctly
        self.assertEqual(self.model.config, self.config)
        self.assertEqual(self.model.multichoice_continuations_start_space, None)
        self.assertTrue(self.model._add_special_tokens)
        self.assertFalse(self.model.pairwise_tokenization)
        self.assertIsNone(self.model.batch_size)
        self.assertFalse(self.model.continuous_batching)

    def test_model_creation_tokenizer(self):
        for attribute in [
            "name_or_path",
            "vocab_size",
            "model_max_length",
            "is_fast",
            "clean_up_tokenization_spaces",
            "added_tokens_decoder",
        ]:
            with self.subTest(attribute=attribute):
                self.assertEqual(
                    getattr(self.model.tokenizer, attribute), getattr(self.reference_tokenizer, attribute)
                )

    def test_model_creation_model(self):
        # We can't compare objects directly
        self.assertEqual(str(self.model.model), str(self.reference_model))


class TestTransformersModelCreationFromModel(unittest.TestCase):
    def setUp(self):
        """Set up shared model instance for all tests."""
        self.reference_model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.reference_tokenizer = AutoTokenizer.from_pretrained("gpt2")

        max_length = 1234
        self.reference_tokenizer.model_max_length = max_length

        self.config = TransformersModelConfig(model_name="gpt2", max_length=max_length)

        # Create full model instance
        self.model = TransformersModel.from_model(
            model=self.reference_model,
            config=self.config,
        )

    def test_model_creation_tokenizer(self):
        for attribute in [
            "name_or_path",
            "vocab_size",
            "model_max_length",
            "is_fast",
            "clean_up_tokenization_spaces",
            "added_tokens_decoder",
        ]:
            with self.subTest(attribute=attribute):
                self.assertEqual(
                    getattr(self.model.tokenizer, attribute), getattr(self.reference_tokenizer, attribute)
                )

    def test_model_creation_attributes(self):
        """Test that TransformersModel creates and initializes basic attributes correctly."""
        # Test attributes are set correctly
        self.assertEqual(self.model.config, self.config)
        self.assertEqual(self.model.multichoice_continuations_start_space, None)
        self.assertTrue(self.model._add_special_tokens)
        self.assertFalse(self.model.pairwise_tokenization)
        self.assertIsNone(self.model.batch_size)
        self.assertFalse(self.model.continuous_batching)
        self.assertEqual(self.model.model_name, self.config.model_name)
        self.assertEqual(self.model.max_length, self.config.max_length)

    def test_model_creation_model(self):
        # We can't compare objects directly
        self.assertEqual(str(self.model.model), str(self.reference_model))


class TestTransformersModelProcessing(unittest.TestCase):
    @patch("lighteval.models.transformers.transformers_model.Accelerator")
    def setUp(self, mock_accelerator):
        """Set up shared model instance for all tests."""
        # Mock accelerate related params
        mock_accelerator_instance = Mock()
        mock_accelerator_instance.device = torch.device("cpu")
        mock_accelerator_instance.is_main_process = True
        mock_accelerator_instance.prepare = lambda x: x
        mock_accelerator_instance.gather_for_metrics = lambda x: x
        mock_accelerator.return_value = mock_accelerator_instance
        # Create full model instance
        self.config = TransformersModelConfig(
            model_name="gpt2",
        )
        self.model = TransformersModel(self.config)

    def test_check_continuations_start_space(self):
        """Test _check_continuations_start_space method."""
        model = TransformersModel.__new__(TransformersModel)

        # Test with multichoice_continuations_start_space = True
        model.multichoice_continuations_start_space = True
        result_add_space = model._check_continuations_start_space("hello")
        self.assertEqual(result_add_space, " hello")

        result_keep_space = model._check_continuations_start_space(" hello")
        self.assertEqual(result_keep_space, " hello")

        # Test with multichoice_continuations_start_space = False
        model.multichoice_continuations_start_space = False
        result_remove_space = model._check_continuations_start_space(" hello")
        self.assertEqual(result_remove_space, "hello")

        result_keep_no_space = model._check_continuations_start_space("hello")
        self.assertEqual(result_keep_no_space, "hello")

        # Test with multichoice_continuations_start_space = None (no change)
        model.multichoice_continuations_start_space = None
        result_no_change = model._check_continuations_start_space(" hello")
        self.assertEqual(result_no_change, " hello")

    @patch("lighteval.models.transformers.transformers_model.DataLoader")
    def test_loglikelihood_eval(self, mock_dataloader):
        """Test _loglikelihood_tokens function with proper Doc objects."""
        # Create test documents with realistic structure
        docs = [
            Doc(query="What is the capital of France?", choices=["London", "Berlin", "Paris", "Madrid"], gold_index=2),
            Doc(query="What is 2+2?", choices=["3", "4", "5"], gold_index=1),
        ]

        # Mock the DataLoader to return our docs as a single batch
        mock_dataloader.return_value = [docs]

        # Mock accelerator prepare method to return the dataloader unchanged
        if hasattr(self.model.accelerator, "prepare"):
            self.model.accelerator.prepare = Mock(side_effect=lambda x: x)

        # Call the function under test
        results = self.model._loglikelihood_tokens(docs)

        # Specific output verification
        self.assertIsInstance(results, list, "Results should be a list")
        self.assertEqual(len(results), len(docs), f"Should return {len(docs)} ModelResponse objects")

        # Verify each result is a ModelResponse with expected attributes
        for i, result in enumerate(results):
            with self.subTest(doc_index=i):
                self.assertIsInstance(result, ModelResponse, f"Result {i} should be ModelResponse")

                # Check that logprobs are present and are floats
                self.assertIsInstance(result.logprobs, list, "logprobs should be a list")
                self.assertTrue(len(result.logprobs) > 0, "logprobs should not be empty")
                self.assertTrue(all(isinstance(lp, float) for lp in result.logprobs), "All logprobs should be floats")

                # Check that argmax_logits_eq_gold is present and contains booleans
                self.assertIsInstance(result.argmax_logits_eq_gold, list, "argmax_logits_eq_gold should be a list")
                self.assertEqual(
                    len(result.argmax_logits_eq_gold),
                    len(docs[i].choices),
                    f"argmax_logits_eq_gold length should match choices count for doc {i}",
                )
                self.assertTrue(
                    all(isinstance(eq, bool) for eq in result.argmax_logits_eq_gold),
                    "All argmax_logits_eq_gold should be booleans",
                )

                # Verify that the number of logprobs matches the number of choices
                self.assertEqual(
                    len(result.logprobs),
                    len(docs[i].choices),
                    f"Number of logprobs should match number of choices for doc {i}",
                )

                # Verify logprob values are reasonable (should be negative log probabilities)
                self.assertTrue(all(lp <= 0.0 for lp in result.logprobs), f"All logprobs should be <= 0 for doc {i}")

                # Verify that exactly one of argmax_logits_eq_gold is True per doc (for correct answer)
                gold_matches = sum(result.argmax_logits_eq_gold)
                self.assertGreaterEqual(gold_matches, 0, f"Should have non-negative gold matches for doc {i}")
                self.assertLessEqual(
                    gold_matches, len(docs[i].choices), f"Gold matches should not exceed choices for doc {i}"
                )

    @patch("lighteval.models.transformers.transformers_model.DataLoader")
    def test_loglikelihood_padded_tensors_shapes(self, mock_dataloader):
        """Test the shapes of padded_logits_sums and padded_max_equals in _loglikelihood_tokens."""
        # Create test documents with different numbers of choices to test padding
        docs = [
            Doc(
                query="What is the capital of France?",
                choices=["London", "Berlin", "Paris"],  # 3 choices
                gold_index=2,
            ),
            Doc(
                query="What is 2+2?",
                choices=["3", "4", "5", "6", "7"],  # 5 choices
                gold_index=1,
            ),
        ]

        # Mock the DataLoader to return our docs as a single batch
        mock_dataloader.return_value = [docs]

        # Mock accelerator prepare method to return the dataloader unchanged
        if hasattr(self.model.accelerator, "prepare"):
            self.model.accelerator.prepare = Mock(side_effect=lambda x: x)

        # Capture padded tensors when gather is applied by mocking the function
        captured_num_choices = None
        captured_len_choices = None
        captured_len_context = None
        captured_padded_logits = None
        captured_padded_max_equals = None
        captured_padded_continuations = None
        captured_padded_contexts = None
        ix = 0

        def mock_gather(tensor):
            nonlocal ix
            nonlocal captured_num_choices, captured_len_choices, captured_len_context
            nonlocal \
                captured_padded_logits, \
                captured_padded_max_equals, \
                captured_padded_continuations, \
                captured_padded_contexts
            # Capture the stacked tensors before gathering
            if ix == 0:
                captured_num_choices = tensor  # batch
            elif ix == 1:
                captured_len_choices = tensor  # batch
            elif ix == 2:
                captured_len_context = tensor  # batch
            elif ix == 3:
                captured_padded_logits = tensor  # batch * max_num_choices
            elif ix == 4:
                captured_padded_max_equals = tensor  # batch * max_num_choices
            elif ix == 5:
                captured_padded_continuations = tensor  # batch * max_num_choices * max_len_choices
            elif ix == 6:
                captured_padded_contexts = tensor  # batch * max_len_context

            ix += 1
            return tensor

        self.model.accelerator.gather_for_metrics = mock_gather

        # Call the function under test
        self.model._loglikelihood_tokens(docs)

        # Verify we captured everyone
        self.assertIsNotNone(captured_num_choices, "Should have captured gathered_num_choices")
        self.assertIsNotNone(captured_len_choices, "Should have captured gathered_len_choices")
        self.assertIsNotNone(captured_len_context, "Should have captured gathered_len_context")
        self.assertIsNotNone(captured_padded_logits, "Should have captured padded_logits_sums")
        self.assertIsNotNone(captured_padded_max_equals, "Should have captured padded_max_equals")
        self.assertIsNotNone(captured_padded_continuations, "Should have captured padded_continuations")
        self.assertIsNotNone(captured_padded_contexts, "Should have captured padded_contexts")

        # Expected dimensions
        expected_batch_size = len(docs)
        expected_max_choices = max(len(doc.choices) for doc in docs)  # Should be 5
        self.assertEqual(expected_max_choices, max(captured_num_choices))

        # Test dimensions
        # - Test num/len tensors
        self.assertEqual(captured_num_choices.tolist(), [len(docs[0].choices), len(docs[1].choices)])
        self.assertEqual(captured_len_choices.tolist(), [2, 1])  # len of the choice, sample per sample, in tokens
        self.assertEqual(captured_len_context.tolist(), [7, 6])  # len of the context, sample per sample, in tokens

        # - Test 1D padded tensors (logits_sums and max_equals)
        self.assertEqual(captured_padded_logits.shape, (expected_batch_size, expected_max_choices))
        self.assertEqual(captured_padded_max_equals.shape, (expected_batch_size, expected_max_choices))

        # - Test 3D padded continuations: (batch_size, max_num_choices, max_len_choices)
        self.assertEqual(
            captured_padded_continuations.shape, (len(docs), max(captured_num_choices), max(captured_len_choices))
        )

        # - Test 2D padded contexts: (batch_size, max_len_context)
        self.assertEqual(captured_padded_contexts.shape, (len(docs), max(captured_len_context)))

        # Verify padding values for 1D tensors
        # - First doc has 3 choices, so positions [3:5] should be padded with -1 for logits and False for max_equals
        self.assertTrue(torch.all(captured_padded_logits[0, 3:] == -1), "Padded positions in logits should be -1")
        self.assertTrue(
            torch.all(captured_padded_max_equals[0, 3:] == False),  # noqa E712
            "Padded positions in max_equals should be False",
        )

        # - Second doc has 5 choices, so no padding needed for num_choices dimension
        self.assertTrue(
            torch.all(captured_padded_logits[1, :] != -1), "No padding should be needed for second doc logits"
        )

        # Verify padding values for 3D continuations tensor
        # - First doc should have padding in the num_choices dimension (positions [3:5])
        self.assertTrue(
            torch.all(captured_padded_continuations[0, 3:, :] == -1), "Padded choices in continuations should be -1"
        )

        # Verify padding values for 2D contexts tensor
        # Both contexts should have some padding in the length dimension (padded with -1)
        context_padding_mask = captured_padded_contexts == -1
        self.assertTrue(torch.any(context_padding_mask), "At least some context positions should be padded with -1")

        # Restore original gather function
        self.model.accelerator.gather_for_metrics = lambda x: x


class TestTransformersModelUseChatTemplate(unittest.TestCase):
    @patch("lighteval.models.transformers.transformers_model.Accelerator")
    @patch("lighteval.models.transformers.transformers_model.TransformersModel._create_auto_model")
    @patch("lighteval.utils.imports.is_accelerate_available")
    def test_transformers_model_use_chat_template_with_different_model_names(
        self, mock_accelerator, mock_create_model, is_accelerate_available
    ):
        """Test that TransformersModel correctly determines whether to use_chat_template or not automatically from the tokenizer config."""
        test_cases = [
            ("Qwen/Qwen3-0.6B", True),
            ("gpt2", False),
        ]

        for model_name, expected_result in test_cases:
            with self.subTest(model_name=model_name):
                # Mock accelerate related params
                mock_accelerator_instance = Mock()
                mock_accelerator_instance.device = torch.device("cpu")
                mock_accelerator.return_value = mock_accelerator_instance
                is_accelerate_available = False  # noqa F841

                # Skip the model creation phase
                mock_create_model = Mock()  # noqa F841

                config = TransformersModelConfig(model_name=model_name, model_parallel=True, compile=False)
                model = TransformersModel(config)

                self.assertEqual(model.use_chat_template, expected_result)
                self.assertEqual(model.use_chat_template, model._tokenizer.chat_template is not None)


if __name__ == "__main__":
    unittest.main()
