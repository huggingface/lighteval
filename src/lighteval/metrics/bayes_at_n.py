# MIT License

# Copyright (c) 2024 The HuggingFace Team
# Copyright (c) 2026 Mohsen Hariri, Vipin Chaudhary

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
# DEALINGS IN THE SOFTWARE.

"""Bayes@N posterior moments (https://arxiv.org/abs/2510.04265).

Native implementation of Algorithm 1, following the MIT-licensed Scorio
reference: https://github.com/mohsenhariri/scorio.
"""

import math
from collections.abc import Sequence

import numpy as np


def _as_2d_int_matrix(values: Sequence[Sequence[int]] | np.ndarray, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular 1D or 2D array.") from exc
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array.")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if matrix.dtype.kind not in "biuf":
        raise ValueError(f"{name} entries must be real numeric integer category ids.")
    if not np.all(np.isfinite(matrix)) or not np.all(matrix == np.floor(matrix)):
        raise ValueError(f"{name} entries must be finite integer category ids.")
    if matrix.size and (np.any(matrix < 0) or np.any(matrix >= 2**63)):
        raise ValueError(f"{name} entries must be non-negative integer category ids smaller than 2**63.")
    return matrix.astype(np.int64, copy=False)


def _as_weights(weights: Sequence[float] | np.ndarray | None, R: np.ndarray) -> np.ndarray:
    if weights is None:
        if not np.all((R == 0) | (R == 1)):
            raise ValueError("R contains non-binary category ids; pass weights to score multi-category outcomes.")
        return np.array([0.0, 1.0])
    try:
        array = np.asarray(weights)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be a finite non-empty 1D numeric array.") from exc
    if array.ndim != 1 or not array.size or array.dtype.kind not in "biuf":
        raise ValueError("weights must be a non-empty 1D array of real numeric values.")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("weights must contain only finite float64 values.")
    return array


def _row_bincount(matrix: np.ndarray, length: int) -> np.ndarray:
    counts = np.zeros((matrix.shape[0], length), dtype=np.int64)
    rows = np.repeat(np.arange(matrix.shape[0]), matrix.shape[1])
    np.add.at(counts, (rows, matrix.ravel()), 1)
    return counts


def _as_prior_matrix(prior, num_rows: int) -> np.ndarray:
    if prior is None:
        return np.empty((num_rows, 0), dtype=np.int64)
    else:
        try:
            prior_array = np.asarray(prior)
            if prior_array.ndim == 1:
                prior_array = prior_array.reshape(num_rows, -1)
        except (TypeError, ValueError) as exc:
            raise ValueError("prior must be rectangular, with the same number of rows as R.") from exc
        prior_outcomes = _as_2d_int_matrix(prior_array, "prior")
        if prior_outcomes.shape[0] != num_rows:
            raise ValueError("prior must have the same number of rows as R.")
    return prior_outcomes


def bayes_at_n(
    R: Sequence[Sequence[int]] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
    prior: Sequence[Sequence[int]] | np.ndarray | None = None,
) -> tuple[float, float]:
    """Return posterior moments of the equal-question mean rubric score.

    Each question has an independent Dirichlet posterior with one pseudo-count
    per category, plus its observed and prior category counts.

    Args:
        R: Non-empty rectangular ``M x N`` matrix of category ids. A 1D array
            represents one question. Boolean and integer-valued real ids are accepted.
        weights: Finite real score for each category. Defaults to ``[0, 1]``.
        prior: Optional row-aligned ``M x D`` prior outcomes, with fixed D.
            A flat array is reshaped into M rows.

    Returns:
        ``(mu, sigma)``: posterior mean and posterior standard deviation of
        the mean over this fixed question set. Neither value is a bootstrap statistic.
    """
    outcomes = _as_2d_int_matrix(R, "R")
    if outcomes.shape[1] == 0:
        raise ValueError("R must contain at least one outcome per row.")
    scores = _as_weights(weights, outcomes)
    num_rows, num_samples = outcomes.shape
    prior_outcomes = _as_prior_matrix(prior, num_rows)
    for matrix, name in ((outcomes, "R"), (prior_outcomes, "prior")):
        if np.any(matrix >= scores.size):
            raise ValueError(f"{name} entries must be integers in [0, {scores.size - 1}].")

    counts = _row_bincount(outcomes, scores.size) + _row_bincount(prior_outcomes, scores.size) + 1
    total = scores.size + num_samples + prior_outcomes.shape[1]
    offset = float(scores[0])
    with np.errstate(over="ignore", invalid="ignore"):
        centered = scores - offset
    if np.all(np.isfinite(centered)):
        scale = float(np.max(np.abs(centered)))
    else:
        offset = 0.0
        centered = scores
        scale = float(np.max(np.abs(scores)))
    if scale == 0:
        return offset, 0.0
    normalized = centered / scale
    probabilities = counts / total
    row_means = probabilities @ normalized
    row_variances = probabilities @ (normalized**2) - row_means**2
    if np.any(row_variances < -64 * np.finfo(float).eps):
        raise FloatingPointError("Bayes@N posterior variance is materially negative.")
    variance = float(np.maximum(row_variances, 0).sum()) / (num_rows**2 * (total + 1))
    return float(offset + scale * float(row_means.mean())), float(scale * math.sqrt(variance))


__all__ = ["bayes_at_n"]
