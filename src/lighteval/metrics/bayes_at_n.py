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
# DEALINGS IN THE SOFTWARE.

"""Bayes@N posterior moments for repeated categorical outcomes."""

from collections.abc import Sequence

import numpy as np


def _as_2d_int_matrix(values: Sequence[Sequence[int]] | np.ndarray, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(values)
    except ValueError as exc:
        raise ValueError(f"{name} must be a rectangular 1D or 2D array.") from exc

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array.")

    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")

    if matrix.dtype == np.dtype("bool"):
        return matrix.astype(int)
    if np.issubdtype(matrix.dtype, np.integer):
        return matrix.astype(int, copy=False)
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError(f"{name} entries must be integer category ids.")

    float_matrix = matrix.astype(float)
    if not np.all(np.isfinite(float_matrix)):
        raise ValueError(f"{name} entries must be finite integer category ids.")
    if not np.all(float_matrix == np.floor(float_matrix)):
        raise ValueError(f"{name} entries must be integer category ids.")
    return float_matrix.astype(int)


def _as_weights(weights: Sequence[float] | np.ndarray | None, R: np.ndarray) -> np.ndarray:
    if weights is None:
        unique_values = np.unique(R)
        if np.all(np.isin(unique_values, [0, 1])):
            return np.array([0.0, 1.0])

        unique_str = ", ".join(str(value) for value in unique_values)
        raise ValueError(
            f"R contains non-binary category ids ({unique_str}); pass weights to score multi-category outcomes."
        )

    weight_array = np.asarray(weights, dtype=float)
    if weight_array.ndim != 1:
        raise ValueError("weights must be a 1D array.")
    if weight_array.size == 0:
        raise ValueError("weights must contain at least one value.")
    if not np.all(np.isfinite(weight_array)):
        raise ValueError("weights must contain only finite values.")
    return weight_array


def _validate_matrix_range(matrix: np.ndarray, low: int, high: int, name: str) -> None:
    if matrix.size == 0:
        return
    if matrix.min() < low or matrix.max() > high:
        raise ValueError(f"{name} entries must be integers in [{low}, {high}].")


def _row_bincount(matrix: np.ndarray, length: int) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], length), dtype=int)

    counts = np.zeros((matrix.shape[0], length), dtype=int)
    rows = np.repeat(np.arange(matrix.shape[0]), matrix.shape[1])
    np.add.at(counts, (rows, matrix.ravel()), 1)
    return counts


def _as_prior_matrix(
    prior: Sequence[Sequence[int]] | np.ndarray | None,
    num_rows: int,
) -> np.ndarray:
    if prior is None:
        return np.zeros((num_rows, 0), dtype=int)

    prior_matrix = _as_2d_int_matrix(prior, "prior")
    if prior_matrix.ndim == 1:
        prior_matrix = prior_matrix.reshape(1, -1)
    if prior_matrix.shape[0] != num_rows:
        if prior_matrix.size % num_rows != 0:
            raise ValueError("prior must have the same number of rows as R.")
        prior_matrix = prior_matrix.reshape(num_rows, -1)
    return prior_matrix


def bayes_at_n(
    R: Sequence[Sequence[int]] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
    prior: Sequence[Sequence[int]] | np.ndarray | None = None,
) -> tuple[float, float]:
    """Return the Bayes@N posterior mean and standard deviation.

    Args:
        R: ``M x N`` matrix of integer category ids. A 1D array is treated as
            one row.
        weights: Category score weights. If omitted, ``R`` must be binary and
            weights ``[0.0, 1.0]`` are used.
        prior: Optional ``M x D`` matrix of row-aligned prior observations.

    Returns:
        ``(mu, sigma)``, where ``mu`` is the posterior mean and ``sigma`` is the
        posterior standard deviation.
    """
    outcome_matrix = _as_2d_int_matrix(R, "R")
    if outcome_matrix.shape[1] == 0:
        raise ValueError("R must contain at least one outcome per row.")

    weight_array = _as_weights(weights, outcome_matrix)
    num_rows, num_samples = outcome_matrix.shape
    max_category = weight_array.size - 1
    prior_matrix = _as_prior_matrix(prior, num_rows)

    _validate_matrix_range(outcome_matrix, 0, max_category, "R")
    _validate_matrix_range(prior_matrix, 0, max_category, "prior")

    prior_samples = prior_matrix.shape[1]
    total_count = 1 + max_category + prior_samples + num_samples

    outcome_counts = _row_bincount(outcome_matrix, max_category + 1)
    prior_counts = _row_bincount(prior_matrix, max_category + 1) + 1
    posterior_counts = outcome_counts + prior_counts

    delta_weights = weight_array - weight_array[0]
    mu = weight_array[0] + (posterior_counts @ delta_weights).sum() / (num_rows * total_count)

    posterior_probs = posterior_counts / total_count
    second_moment = (posterior_probs * (delta_weights**2)).sum(axis=1)
    squared_mean = (posterior_probs @ delta_weights) ** 2
    sigma = np.sqrt((second_moment - squared_mean).sum() / (num_rows**2 * (total_count + 1)))

    return float(mu), float(sigma)


__all__ = ["bayes_at_n"]
