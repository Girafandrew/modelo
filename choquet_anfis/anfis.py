from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, fbeta_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler

try:
    from .choquet import choquet_power_aggregation
    from .fuzzy import fit_feature_weights, gaussian_mf, normalize_weights
except ImportError:
    from choquet import choquet_power_aggregation
    from fuzzy import fit_feature_weights, gaussian_mf, normalize_weights


@dataclass
class FuzzyFeatureParams:
    centers: np.ndarray
    sigmas: np.ndarray
    weights: np.ndarray


class ChoquetANFISScorer:
    def __init__(self, q: float = 1.0):
        self.q = float(q)
        self.scaler = MinMaxScaler()
        self.feature_names: list[str] = []
        self.params: FuzzyFeatureParams | None = None
        self.threshold_: float | None = None
        self.score_orientation_: float = 1.0

    def fit(self, X_train) -> "ChoquetANFISScorer":
        self.feature_names = list(X_train.columns)
        X_train_norm = self.scaler.fit_transform(X_train)

        q25 = np.quantile(X_train_norm, 0.25, axis=0)
        q50 = np.quantile(X_train_norm, 0.50, axis=0)
        q75 = np.quantile(X_train_norm, 0.75, axis=0)
        std = np.std(X_train_norm, axis=0)

        centers = np.stack([q25, q50, q75], axis=1)
        sigmas = np.repeat(np.maximum(std[:, None], 1e-3), 3, axis=1)
        weights = fit_feature_weights(X_train_norm)
        self.params = FuzzyFeatureParams(centers=centers, sigmas=sigmas, weights=weights)
        self.threshold_ = None
        self.score_orientation_ = 1.0
        return self

    def set_score_orientation(self, scores: np.ndarray, y_true: np.ndarray) -> float:
        scores = np.asarray(scores, dtype=float)
        y_true = np.asarray(y_true).astype(int)

        positive_scores = scores[y_true == 1]
        negative_scores = scores[y_true == 0]

        if positive_scores.size == 0 or negative_scores.size == 0:
            self.score_orientation_ = 1.0
            return self.score_orientation_

        positive_mean = float(np.mean(positive_scores))
        negative_mean = float(np.mean(negative_scores))

        self.score_orientation_ = 1.0 if positive_mean >= negative_mean else -1.0
        return self.score_orientation_

    def _orient_scores(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=float) * float(self.score_orientation_)

    def _set_threshold_by_mode(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        threshold_mode: str,
        train_quantile: float,
        threshold_beta: float,
        min_precision: float,
    ) -> None:
        if threshold_mode == "f1":
            self.fit_threshold_with_labels(scores, y_true, beta=threshold_beta)
        elif threshold_mode == "precision_floor":
            self.fit_threshold_with_precision_floor(scores, y_true, min_precision=min_precision)
        elif threshold_mode == "balanced_accuracy":
            self.fit_threshold_balanced_accuracy(scores, y_true)
        else:
            self.set_threshold_from_quantile(scores, quantile=train_quantile)

    @staticmethod
    def _metric_by_mode(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold_mode: str,
        threshold_beta: float,
        min_precision: float,
    ) -> float:
        if threshold_mode == "f1":
            return float(fbeta_score(y_true, y_pred, beta=threshold_beta, zero_division=0))
        if threshold_mode == "precision_floor":
            p = float(precision_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            return f1 if p >= min_precision else -1.0
        if threshold_mode == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred))
        return float(f1_score(y_true, y_pred, zero_division=0))

    def refine_weights_with_labels(
        self,
        X_train,
        y_train,
        epochs: int = 10,
        learning_rate: float = 0.1,
        patience: int = 0,
        min_delta: float = 0.0,
        threshold_mode: str = "f1",
        train_quantile: float = 0.95,
        threshold_beta: float = 1.0,
        min_precision: float = 0.5,
    ) -> dict[str, float]:
        if self.params is None:
            raise RuntimeError("Model must be fitted before reweighting.")
        if epochs <= 0:
            return {"best_metric": float("nan"), "epochs": 0.0, "stopped_early": 0.0}
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if patience < 0:
            raise ValueError("patience must be greater than or equal to zero.")
        if min_delta < 0:
            raise ValueError("min_delta must be greater than or equal to zero.")

        X_norm = self.scaler.transform(X_train)
        y_true = np.asarray(y_train).astype(int)

        best_metric = -1.0
        best_weights = self.params.weights.copy()
        epochs_ran = 0
        no_improvement_count = 0
        stopped_early = False

        initial_scores = self.score_matrix(X_norm)
        self.set_score_orientation(initial_scores, y_true)

        for _ in range(int(epochs)):
            epochs_ran += 1
            scores = self._orient_scores(self.score_matrix(X_norm))
            self._set_threshold_by_mode(
                scores,
                y_true,
                threshold_mode=threshold_mode,
                train_quantile=train_quantile,
                threshold_beta=threshold_beta,
                min_precision=min_precision,
            )
            y_pred = (scores >= self.threshold_).astype(int)
            metric = self._metric_by_mode(
                y_true,
                y_pred,
                threshold_mode=threshold_mode,
                threshold_beta=threshold_beta,
                min_precision=min_precision,
            )

            if metric > (best_metric + min_delta):
                best_metric = metric
                best_weights = self.params.weights.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if patience > 0 and no_improvement_count >= patience:
                stopped_early = True
                break

            # Heuristic supervised update: increase weights for contributions that help
            # false negatives and decrease for contributions that support false positives.
            contributions = self._feature_contributions(X_norm)
            error_signal = (y_true - y_pred).astype(float)
            delta = np.mean(error_signal[:, None] * contributions, axis=0)
            updated = np.clip(self.params.weights + learning_rate * delta, 0.0, None)
            self.params.weights = normalize_weights(updated + 1e-9)

        self.params.weights = best_weights

        final_scores = self._orient_scores(self.score_matrix(X_norm))
        self._set_threshold_by_mode(
            final_scores,
            y_true,
            threshold_mode=threshold_mode,
            train_quantile=train_quantile,
            threshold_beta=threshold_beta,
            min_precision=min_precision,
        )
        return {
            "best_metric": float(best_metric),
            "epochs": float(epochs_ran),
            "stopped_early": float(1 if stopped_early else 0),
        }

    def set_threshold_from_quantile(self, scores: np.ndarray, quantile: float = 0.95) -> float:
        self.threshold_ = float(np.quantile(scores, quantile))
        return self.threshold_

    def fit_threshold_with_labels(self, scores: np.ndarray, y_true: np.ndarray, beta: float = 1.0) -> float:
        if beta <= 0:
            raise ValueError("beta must be greater than zero.")

        thresholds = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 181)))
        best_f1 = -1.0
        best_threshold = float(np.median(scores))
        for threshold in thresholds:
            y_pred = (scores >= threshold).astype(int)
            if beta == 1.0:
                current_f1 = f1_score(y_true, y_pred, zero_division=0)
            else:
                current_f1 = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            if current_f1 > best_f1 or (
                np.isclose(current_f1, best_f1) and threshold > best_threshold
            ):
                best_f1 = current_f1
                best_threshold = float(threshold)
        self.threshold_ = best_threshold
        return self.threshold_

    def fit_threshold_with_precision_floor(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        min_precision: float = 0.5,
    ) -> float:
        if not (0.0 <= min_precision <= 1.0):
            raise ValueError("min_precision must be between 0 and 1.")

        thresholds = np.unique(np.quantile(scores, np.linspace(0.05, 0.995, 300)))

        best_threshold = float(np.quantile(scores, 0.99))
        best_recall = -1.0
        best_precision = 0.0
        found_feasible = False

        for threshold in thresholds:
            y_pred = (scores >= threshold).astype(int)
            p = float(precision_score(y_true, y_pred, zero_division=0))
            r = float(recall_score(y_true, y_pred, zero_division=0))

            if p >= min_precision:
                found_feasible = True
                if r > best_recall or (
                    np.isclose(r, best_recall) and threshold > best_threshold
                ):
                    best_recall = r
                    best_precision = p
                    best_threshold = float(threshold)

        # If there is no threshold that satisfies precision floor, fallback to max precision.
        if not found_feasible:
            best_precision = -1.0
            for threshold in thresholds:
                y_pred = (scores >= threshold).astype(int)
                p = float(precision_score(y_true, y_pred, zero_division=0))
                if p > best_precision or (
                    np.isclose(p, best_precision) and threshold > best_threshold
                ):
                    best_precision = p
                    best_threshold = float(threshold)

        self.threshold_ = best_threshold
        return self.threshold_

    def fit_threshold_balanced_accuracy(self, scores: np.ndarray, y_true: np.ndarray) -> float:
        thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 240)))
        best_score = -1.0
        best_threshold = float(np.median(scores))

        for threshold in thresholds:
            y_pred = (scores >= threshold).astype(int)
            ba = float(balanced_accuracy_score(y_true, y_pred))
            if ba > best_score or (
                np.isclose(ba, best_score) and threshold > best_threshold
            ):
                best_score = ba
                best_threshold = float(threshold)

        self.threshold_ = best_threshold
        return self.threshold_

    def _feature_contributions(self, X_norm: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Model must be fitted before inference.")

        low_mu = gaussian_mf(X_norm, self.params.centers[:, 0], self.params.sigmas[:, 0])
        mid_mu = gaussian_mf(X_norm, self.params.centers[:, 1], self.params.sigmas[:, 1])
        high_mu = gaussian_mf(X_norm, self.params.centers[:, 2], self.params.sigmas[:, 2])

        fuzzy_score = 0.1 * low_mu + 0.4 * mid_mu + 1.0 * high_mu
        consequent = 0.5 * X_norm + 0.5 * fuzzy_score
        contributions = consequent * self.params.weights

        return np.clip(contributions, 0.0, 1.0)

    def score_matrix(self, X_norm: np.ndarray) -> np.ndarray:
        contributions = self._feature_contributions(X_norm)
        return np.array([choquet_power_aggregation(row, q=self.q) for row in contributions], dtype=float)

    def score(self, X) -> np.ndarray:
        X_norm = self.scaler.transform(X)
        return self._orient_scores(self.score_matrix(X_norm))

    def predict(self, X) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Decision threshold is not set.")
        scores = self.score(X)
        return (scores >= self.threshold_).astype(int)
