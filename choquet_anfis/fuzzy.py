from __future__ import annotations

import numpy as np


def gaussian_mf(x: np.ndarray, c: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Gaussian membership function."""
    sigma = np.maximum(sigma, 1e-6)
    return np.exp(-((x - c) ** 2) / (2.0 * sigma**2))


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize weights so they sum to 1."""
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total == 0:
        return np.zeros_like(weights)
    return weights / total


def fit_feature_weights(X_train_norm: np.ndarray) -> np.ndarray:
    """Estimate feature-level rule weights from training distribution."""
    q75 = np.quantile(X_train_norm, 0.75, axis=0)
    q50 = np.quantile(X_train_norm, 0.50, axis=0)
    spread = np.std(X_train_norm, axis=0)

    rarity = np.clip(q75 - q50, 0.0, 1.0)
    weights = rarity / (spread + 1e-6)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return normalize_weights(weights + 1e-9)
