# Adding a Custom Task

To add a new task, first either open an issue, to determine whether it will be
integrated in the core evaluations of lighteval, in the extended tasks, or the
community tasks, and add its dataset on the hub.

- Core evaluations are evaluations that only require standard logic in their
  metrics and processing, and that we will add to our test suite to ensure non
  regression through time. They already see high usage in the community.
- Extended evaluations are evaluations that require custom logic in their
  metrics (complex normalisation, an LLM as a judge, ...), that we added to
  facilitate the life of users. They already see high usage in the community.
- Community evaluations are submissions by the community of new tasks.

A popular community evaluation can move to become an extended or core evaluation over time.

[`lighteval.metrics.utils.CorpusLevelMetric`]

TODO: Add code snippet to show how to add a new task to lighteval.

```python
```
