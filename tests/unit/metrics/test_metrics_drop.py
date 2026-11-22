from lighteval.metrics.harness_compatibility.drop import DropMetrics


def test_drop_em_duplicate_span_not_exact_match():
    m = DropMetrics()
    # A prediction that repeats the reference span is not an exact match
    # (matches the reference DROP evaluator in allennlp/allennlp drop_eval).
    em, _ = m._get_metrics(["Paris", "Paris"], ["Paris"])
    assert em == 0.0


def test_drop_em_single_span():
    m = DropMetrics()
    assert m._get_metrics(["Paris"], ["Paris"])[0] == 1.0
    assert m._get_metrics(["Lyon"], ["Paris"])[0] == 0.0
