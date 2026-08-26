from __future__ import annotations

from typing import Iterable

import numpy as np


def power_measure(k: int, n: int, q: float) -> float:
    """Symmetric fuzzy power measure."""
    if n <= 0:
        raise ValueError("n must be greater than zero.")
    if k < 0:
        raise ValueError("k must be greater than or equal to zero.")
    return (k / n) ** q


def choquet_power_aggregation(x: Iterable[float], q: float = 1.0) -> float:
    """Choquet-inspired aggregation using a power measure."""
    values = np.asarray(list(x), dtype=float)
    if values.size == 0:
        raise ValueError("x cannot be empty.")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("All x values must be in [0, 1].")

    x_sorted = np.sort(values)
    n = len(x_sorted)

    result = float(x_sorted[0])
    for i in range(n - 1):
        diff = float(x_sorted[i + 1] - x_sorted[i])
        k = n - (i + 1)
        result += diff * power_measure(k, n, q)

    return float(np.clip(result, 0.0, 1.0))
