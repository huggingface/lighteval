# Metric Test Cases

This directory contains individual JSON files for each metric tested in the LightEval framework. Each file contains all test cases for a specific metric.

## Structure

Each JSON file follows this structure:

```json
{
  "name": "Metric Name Test Suite",
  "description": "Description of the test suite",
  "test_cases": [
    {
      "name": "Test Case Name",
      "metric_class": "metric_name",
      "metric_params": {},
      "doc": {
        "query": "Input query",
        "choices": ["choice1", "choice2", "choice3"],
        "gold_index": 0,
        "task_name": "test"
      },
      "model_response": {
        "text": ["model_output"],
        "logprobs": [],
        "output_tokens": []
      },
      "expected_output": {
        "metric_key": expected_value
      },
      "tolerance": 0.01,
      "description": "Test case description"
    }
  ]
}
```

## Available Test Files

All 47 metrics from the `METRIC_CLASSES` dictionary have their own JSON test files:

### Text Generation Metrics
- `exact_match.json` - Exact match metric (2 test cases)
- `f1_score.json` - F1 score metric (1 test case)
- `f1_score_macro.json` - F1 score macro metric
- `f1_score_micro.json` - F1 score micro metric
- `rouge1.json` - ROUGE1 metric (1 test case)
- `rouge2.json` - ROUGE2 metric
- `rougeL.json` - ROUGE-L metric
- `rougeLsum.json` - ROUGE-Lsum metric
- `rouge_t5.json` - ROUGE-T5 metric
- `bert_score.json` - BERT Score metric
- `bleu.json` - BLEU metric
- `bleu_1.json` - BLEU-1 metric
- `bleu_4.json` - BLEU-4 metric
- `bleurt.json` - BLEURT metric
- `chrf.json` - ChrF metric
- `chrf_plus.json` - ChrF+ metric
- `ter.json` - TER metric

### Perplexity Metrics
- `bits_per_byte.json` - Bits per byte metric
- `byte_perplexity.json` - Byte perplexity metric
- `word_perplexity.json` - Word perplexity metric
- `prediction_perplexity.json` - Prediction perplexity metric
- `target_perplexity.json` - Target perplexity metric

### Likelihood Metrics
- `loglikelihood_acc.json` - Loglikelihood accuracy metric (1 test case)
- `loglikelihood_f1.json` - Loglikelihood F1 metric
- `acc_golds_likelihood.json` - Accuracy golds likelihood metric

### Pass-at-k Metrics
- `pass_at_k.json` - Pass at k metric
- `pass_at_k_math.json` - Pass at k math metric
- `pass_at_k_letters.json` - Pass at k letters metric
- `g_pass_at_k.json` - G-pass at k metric
- `g_pass_at_k_math.json` - G-pass at k math metric
- `g_pass_at_k_latex.json` - G-pass at k latex metric
- `gpqa_instruct_pass_at_k.json` - GPQA instruct pass at k metric

### Other Metrics
- `recall_at_k.json` - Recall at k metric
- `mrr.json` - Mean Reciprocal Rank metric
- `avg_at_k.json` - Average at k metric
- `avg_at_k_math.json` - Average at k math metric
- `maj_at_k.json` - Majority at k metric
- `extractiveness.json` - Extractiveness metric
- `faithfulness.json` - Faithfulness metric
- `copyright.json` - Copyright metric
- `drop.json` - DROP metric
- `gpqa_instruct_metric.json` - GPQA instruct metric
- `expr_gold_metric.json` - Expression gold metric
- `truthfulqa_mc_metrics.json` - TruthfulQA multiple choice metrics
- `simpleqa_judge.json` - SimpleQA judge metric
- `multi_f1_numeric.json` - Multi F1 numeric metric
- `mcc.json` - Matthews Correlation Coefficient metric

## Usage

These test files can be used with the `AutomatedMetricTester` class in `test_metrics_automated.py`:

```python
tester = AutomatedMetricTester()
results = tester.run_test_suites_from_file("tests/metrics/test_cases/exact_match.json")
```

## Adding New Test Cases

To add new test cases for a metric:

1. Open the corresponding JSON file for that metric
2. Add a new test case object to the `test_cases` array
3. Follow the same structure as existing test cases
4. Ensure the `metric_class` matches the metric being tested
