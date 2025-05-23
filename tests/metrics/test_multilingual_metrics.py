from lighteval.metrics.dynamic_metrics import (
    multilingual_quasi_exact_match_metric,
    multilingual_quasi_f1_score_metric,
)
from lighteval.utils.language import Language


def test_multilingual_quasi_exact_match_happy_path():
    """Test basic functionality of exact match metric"""
    metric = multilingual_quasi_exact_match_metric(language=Language.ENGLISH)

    # Test exact match
    result = metric.sample_level_fn(
        golds=["hello world"],
        predictions=["hello world"],
    )
    assert result == 1

    # Test with different spacing/punctuation
    result = metric.sample_level_fn(
        golds=["hello world"],
        predictions=["hello, world!"],
    )
    assert result == 1

    # Test with no match
    result = metric.sample_level_fn(
        golds=["hello world"],
        predictions=["goodbye world"],
    )
    assert result == 0


def test_multilingual_quasi_exact_match_bb_extraction():
    """Test bold text extraction functionality"""
    metric = multilingual_quasi_exact_match_metric(language=Language.ENGLISH, extract_bb=True)

    # Test with single bold tag
    result = metric.sample_level_fn(
        golds=["answer"],
        predictions=["The correct answer is <b>answer</b>"],
    )
    assert result == 1

    # Test with multiple bold tags - should take last one
    result = metric.sample_level_fn(
        golds=["final answer"],
        predictions=["First <b>wrong</b> then <b>final answer</b>"],
    )
    assert result == 1

    # Test with no bold tags - should use full text
    result = metric.sample_level_fn(
        golds=["answer"],
        predictions=["answer"],
    )
    assert result == 1

    # Test with empty bold tags
    result = metric.sample_level_fn(
        golds=["answer"],
        predictions=["<b></b> answer"],
    )
    assert result == 0


def test_multilingual_quasi_f1_score_happy_path():
    """Test basic functionality of F1 score metric"""
    metric = multilingual_quasi_f1_score_metric(language=Language.ENGLISH)

    # Test perfect match
    result = metric.sample_level_fn(
        golds=["hello world"],
        predictions=["hello world"],
    )
    assert result == 1

    # Test partial match
    result = metric.sample_level_fn(
        golds=["hello beautiful world"],
        predictions=["hello world"],
    )
    assert result > 0 and result < 1

    # Test no match
    result = metric.sample_level_fn(
        golds=["hello world"],
        predictions=["goodbye moon"],
    )
    assert result == 0


def test_multilingual_quasi_f1_score_bb_extraction():
    """Test bold text extraction functionality with F1 score"""
    metric = multilingual_quasi_f1_score_metric(language=Language.ENGLISH, extract_bb=True)

    # Test with single bold tag
    result = metric.sample_level_fn(
        golds=["answer key"],
        predictions=["The correct answer is <b>answer key</b>"],
    )
    assert result == 1

    # Test with multiple bold tags - should take last one
    result = metric.sample_level_fn(
        golds=["final answer"],
        predictions=["First <b>wrong</b> then <b>final answer</b>"],
    )
    assert result == 1

    # Test with partial match in bold
    result = metric.sample_level_fn(
        golds=["complete answer key"],
        predictions=["The text contains <b>answer key</b>"],
    )
    assert result > 0 and result < 1

    # Test with no bold tags - should use full text
    result = metric.sample_level_fn(
        golds=["answer"],
        predictions=["answer"],
    )
    assert result == 1


def test_multilingual_support():
    """Test metrics work with different languages"""
    languages = [Language.ENGLISH, Language.FRENCH, Language.CHINESE]

    for lang in languages:
        # Test exact match
        em_metric = multilingual_quasi_exact_match_metric(language=lang)
        result = em_metric.sample_level_fn(
            golds=["test"],
            predictions=["test"],
        )
        assert result == 1

        # Test F1 score
        f1_metric = multilingual_quasi_f1_score_metric(language=lang)
        result = f1_metric.sample_level_fn(
            golds=["test"],
            predictions=["test"],
        )
        assert result == 1
