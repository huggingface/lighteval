# Caching System

Lighteval includes a caching system that can significantly speed up evaluations by storing and reusing model predictions.
This is especially useful when running the same evaluation multiple times, or comparing different evaluation metrics on the same model outputs.

## How It Works

The caching system caches the predictions of the model for now (we will add tokenized input caching later).
It stores model responses objects (generations, logits, probabilities) for evaluation samples.

### Cache Structure

Cached data is stored on disk using HuggingFace datasets in the following structure:

```
.cache/
└── huggingface/
    └── lighteval/
        └── predictions/
            └── {model_name}/
                └── {model_hash}/
                    └── {task_name}.parquet
```

Where:
- `model_name`: The model name (path on the hub or local path)
- `model_hash`: Hash of the model configuration to ensure cache invalidation when parameters change
- `task_name`: Name of the evaluation task

### Cache Recreation

A new cache is automatically created when:
- Model configuration changes (different parameters, quantization, etc.)
- Model weights change (different revision, checkpoint, etc.)
- Generation parameters change (temperature, max_tokens, etc.)

This ensures that cached results are always consistent with your current model setup.

## Using Caching

### Automatic Caching

All built-in model classes in Lighteval automatically support caching. No additional configuration is needed.
For custom models you need to add a cache to the model class and decorators on all functions.

## Cache Management

### Clearing Cache

To clear cache for a specific model, delete the corresponding directory:

```bash
rm -rf ~/.cache/huggingface/lighteval/predictions/{model_name}/{model_hash}/
```

To clear all caches:

```bash
rm -rf ~/.cache/huggingface/lighteval/predictions
```
