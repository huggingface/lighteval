# Adding a New Metric

First, check if you can use one of the parametrized functions in
[src.lighteval.metrics.metrics_corpus]() or
[src.lighteval.metrics.metrics_sample]().

If not, you can use the `custom_task` system to register your new metric:

<Tip>
To see an example of a custom metric added along with a custom task, look at
<a href="">the IFEval custom task</a>.
</Tip>

- Create a new Python file which should contain the full logic of your metric.
- The file also needs to start with these imports

```python
from aenum import extend_enum
from lighteval.metrics import Metrics
```

You need to define a sample level metric:

```python
def custom_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> bool:
    response = predictions[0]
    return response == formatted_doc.choices[formatted_doc.gold_index]
```

Here the sample level metric only returns one metric, if you want to return multiple metrics per sample you need to return a dictionary with the metrics as keys and the values as values.

```python
def custom_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> dict:
    response = predictions[0]
    return {"accuracy": response == formatted_doc.choices[formatted_doc.gold_index], "other_metric": 0.5}
```

Then, you can define an aggreagtion function if needed, a comon aggregation function is `np.mean`.

```python
def agg_function(items):
    flat_items = [item for sublist in items for item in sublist]
    score = sum(flat_items) / len(flat_items)
    return score
```

Finally, you can define your metric. If it's a sample level metric, you can use the following code:

```python
my_custom_metric = SampleLevelMetric(
    metric_name={custom_metric_name},
    higher_is_better={either True or False},
    category={MetricCategory},
    use_case={MetricUseCase},
    sample_level_fn=custom_metric,
    corpus_level_fn=agg_function,
)
```

If your metric defines multiple metrics per sample, you can use the following code:

```python
custom_metric = SampleLevelMetricGrouping(
    metric_name={submetric_names},
    higher_is_better={n: {True or False} for n in submetric_names},
    category={MetricCategory},
    use_case={MetricUseCase},
    sample_level_fn=custom_metric,
    corpus_level_fn={
        "accuracy": np.mean,
        "other_metric": agg_function,
    },
)
```

And to end with the following, so that it adds your metric to our metrics list
when loaded as a module.

```python
# Adds the metric to the metric list!
extend_enum(Metrics, "metric_name", metric_function)
if __name__ == "__main__":
    print("Imported metric")
```

You can then give your custom metric to lighteval by using `--custom-tasks
path_to_your_file` when launching it.
