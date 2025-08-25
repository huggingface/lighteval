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

# ruff: noqa: F405, F403, F401
"""Emotion Classification Task with Grammar Constraints using LightEval

This module demonstrates how to create a classification task in LightEval with JSON grammar-constrained generation for structured responses.


The task performs emotion classification on the 'emotion' dataset from HuggingFace Hub,
classifying text into one of six emotion categories: sadness, joy, love, anger, fear, surprise.

Example usage:
    TGI endpoint evaluation:
    ```bash
    uv run --active --extra litellm --extra tgi lighteval endpoint tgi examples/model_configs/tgi_model.yaml "custom|emotion_classification|0|0"
    --custom-tasks examples/custom_tasks_templates/custom_task_classification_grammar_task.py
    --output-dir results
    --save-details
    --no-public-run
    ```

Dataset:
    The task uses the 'emotion' dataset from HuggingFace Hub, which contains
    English Twitter messages labeled with one of six emotions. The dataset
    includes train/validation/test splits with the following distribution:
    - Total samples: ~416k (train: ~16k, validation: ~2k, test: ~2k)
    - Labels: sadness, joy, love, anger, fear, surprise
    - Text format: Short social media posts in English

Customization:
    To adapt this task for other classification problems:
    1. Update EMOTION_LABELS with your target labels
    2. Modify prompt_emotion_classification() for your use case
    3. Update the grammar schema in get_emotion_classification_grammar()
    4. Adjust the HuggingFace dataset reference in EMOTION_CLASSIFICATION_TASK
    5. Update metric calculations in emotion_classification_metric() if needed
"""

import json
import logging
from typing import Any

import numpy as np

from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import (
    LightevalTaskConfig,
    TextGenerationInputGrammarType,
)
from lighteval.tasks.requests import Doc, SamplingMethod


logger = logging.getLogger(__name__)

# Emotion labels for the emotion dataset from HuggingFace Hub
# These correspond to the 6-class emotion classification task with the following mapping:
# 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def parse_emotion_response(response: str | dict) -> dict[str, Any]:
    """Parse the model's response into a standardized format.

    This function handles both JSON string and dictionary inputs, providing robust
    parsing with validation against the predefined emotion labels. Invalid predictions
    are automatically mapped to 'unknown' with appropriate logging.

    Args:
        response (str | dict): The model's response, either as a JSON string
            containing {"classification": "emotion_label"} or as a dictionary
            with the same structure.

    Returns:
        dict[str, Any]: Standardized dictionary containing:
            - classification (str): The predicted emotion label, validated against
              EMOTION_LABELS or 'unknown' if invalid/unparseable

    Examples:
        >>> parse_emotion_response('{"classification": "joy"}')
        {'classification': 'joy'}

        >>> parse_emotion_response({'classification': 'ANGER'})
        {'classification': 'anger'}

        >>> parse_emotion_response('{"classification": "invalid_emotion"}')
        {'classification': 'unknown'}  # with warning logged

        >>> parse_emotion_response('malformed json')
        {'classification': 'unknown'}  # with error logged

    Note:
        - Case-insensitive matching: 'ANGER' and 'Anger' are normalized to 'anger'
        - Whitespace is automatically stripped from predictions
        - All parsing errors result in 'unknown' classification with detailed logging
    """
    try:
        # Handle dictionary input (already parsed JSON)
        if isinstance(response, dict):
            result = response
        # Handle string input (JSON string that needs parsing)
        else:
            result = json.loads(response.strip())

        # Extract and normalize the predicted emotion
        predicted_emotion = result["classification"].lower().strip()

        # Validate that the prediction is one of the valid emotion labels
        if predicted_emotion not in EMOTION_LABELS:
            logger.warning(
                f"Invalid emotion prediction: '{predicted_emotion}'. "
                f"Expected one of {EMOTION_LABELS}. Using 'unknown'."
            )
            predicted_emotion = "unknown"

        return {
            "classification": predicted_emotion,
        }
    except (json.JSONDecodeError, KeyError, AttributeError, TypeError) as e:
        # Handle specific parsing errors with detailed logging
        logger.error(
            f"Error parsing response: {str(e)}. Failed response was: {response}. Expected format: {{'classification': 'emotion_label'}}"
        )
        return {
            "classification": "unknown",
        }
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error parsing response: {str(e)}. Failed response was: {response}")
        return {
            "classification": "unknown",
        }


def emotion_classification_metric(model_response: ModelResponse, doc: Doc, **kwargs) -> dict[str, float]:
    """Evaluate emotion classification predictions at the sample level.

    This function computes evaluation metrics for a single prediction, comparing
    the model's emotion classification against the gold standard. It provides
    detailed logging for debugging and tracks prediction quality.

    Args:
        model_response (ModelResponse): The model's response containing generated text
            in the text attribute, typically containing one prediction as either a
            JSON string or dictionary with format {"classification": "emotion_label"}
        doc (Doc): The document containing the query, choices, and gold
            standard information. Must have gold_index attribute pointing to the
            correct emotion label index.
        **kwargs: Additional keyword arguments (unused but required for compatibility
            with LightEval's metric interface)

    Returns:
        dict[str, float]: Dictionary containing sample-level metrics:
            - exact_match (float): 1.0 if prediction matches gold label, 0.0 otherwise
            - unknown_prediction (float): 1.0 if prediction was 'unknown' (parsing
              failure), 0.0 otherwise
            - total_samples (float): Always 1.0 (count for this sample)

    Examples:
        >>> doc = Doc(query="I'm so happy!", gold_index=2)  # joy
        >>> model_response = ModelResponse(text=['{"classification": "joy"}'], ...)
        >>> result = emotion_classification_metric(model_response, doc)
        >>> result
        {'exact_match': 1.0, 'unknown_prediction': 0.0, 'total_samples': 1.0}

        >>> model_response = ModelResponse(text=['{"classification": "sadness"}'], ...)
        >>> result = emotion_classification_metric(model_response, doc)
        >>> result
        {'exact_match': 0.0, 'unknown_prediction': 0.0, 'total_samples': 1.0}

    Note:
        - The function expects exactly one prediction in the model_response.text list
        - Gold labels are mapped from integer indices to emotion label strings
        - All errors in prediction parsing result in 'unknown' classification
        - Detailed logging is provided for debugging classification performance
    """
    try:
        # Parse the first (and typically only) prediction
        prediction = parse_emotion_response(model_response.text[0])

        # Map the gold label index to the corresponding emotion string
        # The emotion dataset uses integer indices: 0=anger, 1=fear, 2=joy, etc.
        gold_label_idx = doc.gold_index
        expected_emotion = EMOTION_LABELS[gold_label_idx]

        # Log detailed information for debugging and analysis
        logger.info("-" * 50)
        logger.info("Processing new sample")
        logger.info(f"- Text: {doc.query}")
        logger.info(f"- Prediction: {prediction}")
        logger.info(f"- Expected: {expected_emotion} (index: {gold_label_idx})")

        # Calculate evaluation metrics
        is_exact_match = prediction["classification"] == expected_emotion
        is_unknown = prediction["classification"] == "unknown"

        metrics = {
            "exact_match": float(is_exact_match),
            "unknown_prediction": float(is_unknown),
            "total_samples": 1.0,
        }

        logger.info(f"- Metrics: {metrics}")
        if is_exact_match:
            logger.info("✓ Correct prediction")
        elif is_unknown:
            logger.info("⚠ Parsing failure (unknown prediction)")
        else:
            logger.info("✗ Incorrect prediction")
        logger.info("-" * 50)

        return metrics

    except (IndexError, KeyError) as e:
        # Handle errors related to accessing gold label or prediction structure
        logger.error(f"Error accessing gold label or prediction: {str(e)}")
        logger.error(f"Gold index: {getattr(doc, 'gold_index', 'N/A')}")
        logger.error(f"Raw prediction: {model_response.text[0] if model_response.text else 'Empty predictions'}")
        return {
            "exact_match": 0.0,
            "unknown_prediction": 1.0,
            "total_samples": 1.0,
        }
    except Exception as e:
        # Handle any other unexpected errors
        logger.error(f"Unexpected error processing prediction: {str(e)}")
        logger.error(f"Raw prediction was: {model_response.text[0] if model_response.text else 'Empty predictions'}")
        return {
            "exact_match": 0.0,
            "unknown_prediction": 1.0,
            "total_samples": 1.0,
        }


# Define the metric group for emotion classification evaluation
# This configures both sample-level and corpus-level metric calculations
emotion_classification_group = SampleLevelMetricGrouping(
    metric_name=[
        "exact_match",  # Primary accuracy metric
        "unknown_prediction",  # Tracks parsing failures
        "total_samples",  # Sample count for aggregation
    ],
    higher_is_better={
        "exact_match": True,  # Higher accuracy is better
        "unknown_prediction": False,  # Fewer parsing failures is better
        "total_samples": True,  # More samples processed is better
    },
    category=SamplingMethod.GENERATIVE,  # Classification via text generation
    sample_level_fn=emotion_classification_metric,  # Function for individual samples
    corpus_level_fn={
        "exact_match": np.mean,  # Average accuracy across all samples
        "unknown_prediction": np.mean,  # Proportion of parsing failures
        "total_samples": np.sum,  # Total number of samples processed
    },
)


def prompt_emotion_classification(line: dict[str, Any], task_name: str = None) -> Doc:
    """Format the emotion classification task with detailed prompt engineering.

    This function converts a single sample from the emotion dataset into a structured
    prompt that provides clear instructions and emotion definitions to improve
    classification accuracy. The prompt includes detailed explanations of each
    emotion category to reduce ambiguity.

    Args:
        line (dict[str, Any]): A single sample from the emotion dataset containing:
            - 'text' (str): The input text to classify
            - 'label' (int): The gold standard emotion label (0-5)
        task_name (str, optional): Name of the task for identification purposes.
            Defaults to None.

    Returns:
        Doc: A formatted document object containing:
            - task_name: Task identifier
            - query: The formatted prompt with text and emotion definitions
            - choices: List of available emotion labels
            - gold_index: The correct emotion label index
            - instruction: Empty string (instructions are embedded in query)

    Examples:
        >>> line = {'text': 'I am so excited for tomorrow!', 'label': 2}
        >>> doc = prompt_emotion_classification(line, 'emotion_test')
        >>> print(doc.query)
        Classify the emotion expressed in the following text: "I am so excited for tomorrow!"
        ...
        >>> doc.gold_index
        2
        >>> doc.choices
        ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

    Note:
        - The prompt includes detailed definitions for each emotion to improve accuracy
        - Emotion definitions are based on common psychological categorizations
        - The format is optimized for both human readability and model understanding
    """
    # Extract the text to be classified
    text = line["text"]

    # Create a comprehensive classification prompt with detailed emotion definitions
    # This approach helps models understand the subtle differences between emotions
    prompt = f"""Classify the emotion expressed in the following text: "{text}"

Available emotion labels and their meanings:
- sadness: Feeling of sorrow, grief, or unhappiness. Covers melancholy, disappointment,
  loss, or general negative emotional states related to unfortunate circumstances.
- joy: Feeling of happiness, delight, or pleasure. Encompasses positive emotions like
  excitement, satisfaction, contentment, and general well-being.
- love: Feeling of affection, care, or romantic attachment. Includes expressions of
  deep fondness, romantic interest, or strong positive feelings toward people or things.
- anger: Feeling of displeasure, hostility, or annoyance. Often involves frustration,
  irritation, or aggressive sentiments toward people, situations, or objects.
- fear: Feeling of anxiety, worry, or being afraid. Includes nervousness, concern
  about future events, or apprehension about potential threats or negative outcomes.
- surprise: Feeling of astonishment or being caught off guard. Includes unexpected
  reactions, amazement, or responses to sudden or unanticipated events.

Choose the emotion that best matches the sentiment expressed in the text."""

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=EMOTION_LABELS,  # Available emotion label options
        gold_index=line["label"],  # Gold standard emotion index (0-5)
        instruction="",  # Instructions are embedded in the query
    )


def get_emotion_classification_grammar() -> TextGenerationInputGrammarType:
    """Define the JSON schema grammar for constrained emotion classification responses.

    This function creates a strict JSON schema that constrains the model's output
    to only valid emotion labels, preventing hallucination and ensuring consistent
    response format. The grammar constraint is enforced during text generation.

    Returns:
        TextGenerationInputGrammarType: A JSON schema grammar specification that:
            - Enforces JSON object structure with required "classification" field
            - Constrains classification values to only valid emotion labels
            - Ensures consistent response parsing across different models

    Schema Structure:
        {
          "type": "object",
          "properties": {
            "classification": {
              "type": "string",
              "description": "Emotion classification",
              "enum": ["anger", "fear", "joy", "love", "sadness", "surprise"]
            }
          },
          "required": ["classification"]
        }

    Examples:
        Valid responses that match this grammar:
        - {"classification": "joy"}
        - {"classification": "anger"}

        Invalid responses that would be rejected:
        - {"emotion": "joy"}  # Wrong field name
        - {"classification": "happy"}  # Invalid emotion label
        - "joy"  # Not a JSON object

    Note:
        - This grammar constraint significantly improves response consistency
        - It prevents the model from generating invalid emotion labels
        - Compatible with grammar-enabled backends like vLLM, TGI, and others
        - The enum constraint is crucial for maintaining label consistency
    """
    return TextGenerationInputGrammarType(
        type="json",  # Specify JSON schema grammar type
        value={
            "type": "object",  # Require JSON object structure
            "properties": {
                "classification": {
                    "type": "string",  # Classification must be a string
                    "description": "Emotion classification from the provided list",
                    "enum": EMOTION_LABELS,  # Strictly constrain to valid emotion labels only
                },
            },
            "required": ["classification"],  # Classification field is mandatory
            "additionalProperties": False,  # Prevent extra fields in response
        },
    )


# Task configuration for emotion classification using the HuggingFace emotion dataset
# This configuration optimizes for accuracy while maintaining efficient resource usage
EMOTION_CLASSIFICATION_TASK = LightevalTaskConfig(
    name="emotion_classification",  # Unique task identifier
    prompt_function=prompt_emotion_classification,  # Custom prompt formatting function
    suite=["custom"],  # Classification as a community/custom task
    hf_repo="emotion",  # HuggingFace Hub dataset repository
    hf_subset=None,  # Use default subset (no subset specified)
    metrics=[emotion_classification_group],  # Evaluation metrics configuration
    generation_size=64,  # Conservative token limit for JSON responses (~30-40 tokens typical)
    generation_grammar=get_emotion_classification_grammar(),  # JSON schema constraint
    stop_sequence=["\n\n"],  # Early stopping on double newline
    evaluation_splits=["test"],  # Evaluate on test split only
    hf_avail_splits=["train", "validation", "test"],  # Available dataset splits
)

# Export the task for LightEval discovery
# This list is automatically detected by LightEval when loading custom tasks
TASKS_TABLE = [EMOTION_CLASSIFICATION_TASK]

# Development and testing utilities
if __name__ == "__main__":
    # Print available tasks for verification
    print("Available tasks:", [t.name for t in TASKS_TABLE])
    print("Total tasks:", len(TASKS_TABLE))

    # Print task configuration summary for debugging
    task = TASKS_TABLE[0]
    print("\nTask Configuration Summary:")
    print(f"  Name: {task.name}")
    print(f"  Dataset: {task.hf_repo}")
    print(f"  Splits: {task.evaluation_splits}")
    print(f"  Metrics: {[m.metric_name for m in task.metric]}")
    print(f"  Generation size: {task.generation_size}")
    print(f"  Grammar constrained: {task.generation_grammar is not None}")
    print(f"  Stop sequences: {task.stop_sequence}")

    # Verify emotion labels configuration
    print(f"\nEmotion Labels ({len(EMOTION_LABELS)}):")
    for i, label in enumerate(EMOTION_LABELS):
        print(f"  {i}: {label}")

    print("\nUsage Examples:")
    print(
        f"  TGI: uv run lighteval endpoint tgi config/tgi/tgi.yaml 'custom|{task.name}|0|0' --custom-tasks {__file__} --output-dir results --override-batch-size 1 --use-chat-template --save-details --no-public-run --max-samples 10"
    )
    print(
        f"  Full: uv run lighteval endpoint tgi config/tgi/tgi.yaml 'custom|{task.name}|5|1' --custom-tasks {__file__} --output-dir results --override-batch-size 1 --use-chat-template --save-details --no-public-run"
    )
