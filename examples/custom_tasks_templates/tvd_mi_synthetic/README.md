# TVD-MI synthetic example (paired-response)

This folder contains a tiny, synthetic paired-response dataset intended to demonstrate how to run the `tvd_mi` metric.

## Data format

The dataset is a `.jsonl` file where each line is a JSON object with:

- `response_a` (str): first response in the pair
- `response_b` (str): second response in the pair
- `pair_label` (int): `1` if the two responses come from the same underlying item/task/source, `0` otherwise

Example line:

```json
{"response_a":"The capital of France is Paris.","response_b":"Paris is the capital of France.","pair_label":1}
````

## What this example is (and isn’t)

* ✅ A minimal, copyable example showing the expected fields for `tvd_mi`
* ✅ Useful as a template for building larger paired-response benchmarks
* ❌ Not intended to be a scientifically meaningful benchmark by itself

## Running

`tvd_mi` is an LLM-as-judge metric. To run with the OpenAI backend, set:

```bash
export OPENAI_API_KEY=...
```

You can then load this dataset as Docs and evaluate with `tvd_mi` (see the Python loader in `tvd_mi_synthetic.py`).
