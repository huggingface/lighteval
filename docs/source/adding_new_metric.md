# Adding a New Metric

First, check if you can use one of the parametrized functions in
``src.lighteval.metrics.metrics_corpus`` or ``src.lighteval.metrics.metrics_sample``.

If not, you can use the `custom_task` system to register your new metric:

- Create a new Python file which should contain the full logic of your metric.
- The file also needs to start with these imports

```python
from aenum import extend_enum
from lighteval.metrics import Metrics

# And any other class you might need to redefine your specific metric,
# depending on whether it's a sample or corpus metric.
```

- And to end with the following, so that it adds your metric to our metrics
  list when loaded as a module.

```python
# Adds the metric to the metric list!
extend_enum(Metrics, "metric_name", metric_function)
if __name__ == "__main__":
    print("Imported metric")
```

You can then give your custom metric to lighteval by using `--custom-tasks
path_to_your_file` when launching it.

To see an example of a custom metric added along with a custom task, look at
``examples/tasks/custom_tasks_with_custom_metrics/ifeval/ifeval.py.``
