from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# FEATURES: EXATAMENTE AS MESMAS DOS OUTROS MODELOS
# ============================================================

SELECTED_FEATURES = [
    "Max",
    "Tot size",
    "Std",
    "IAT",
    "psh_flag_number",
    "Min",
    "Header_Length",
    "UDP",
]

LABEL_CANDIDATES = (
    "label",
    "class",
    "target",
    "y",
    "attack",
    "is_attack",
)

DEFAULT_SPOOF_TRAIN_FILE = "ARP_Spoofing_train.pcap.csv"
DEFAULT_SPOOF_TEST_FILE = "ARP_Spoofing_test.pcap.csv"
DEFAULT_BENIGN_TRAIN_FILE = "Benign_train.pcap.csv"
DEFAULT_BENIGN_TEST_FILE = "Benign_test.pcap.csv"


# ============================================================
# ARGUMENTOS
# ============================================================

def parser():
    p = argparse.ArgumentParser(
        description=(
            "Improved supervised Choquet-ANFIS ensemble "
            "for ARP spoofing using exactly 8 features."
        )
    )

    p.add_argument("--spoof-train", type=Path, default=Path(DEFAULT_SPOOF_TRAIN_FILE))
    p.add_argument("--spoof-test", type=Path, default=Path(DEFAULT_SPOOF_TEST_FILE))
    p.add_argument("--benign-train", type=Path, default=Path(DEFAULT_BENIGN_TRAIN_FILE))
    p.add_argument("--benign-test", type=Path, default=Path(DEFAULT_BENIGN_TEST_FILE))

    # Ensemble cross-fitted
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=2)

    # Busca das partes que realmente exigem refit do fuzzy model
    p.add_argument("--search-iter", type=int, default=10)

    # q e blend são baratos: são testados em todas as combinações
    p.add_argument(
        "--q-grid",
        type=str,
        default="0.35,0.5,0.75,1.0,1.5,2.0",
    )
    p.add_argument(
        "--blend-grid",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
    )

    # Threshold robusto a prevalence shift
    p.add_argument(
        "--prevalence-floor-factor",
        type=float,
        default=0.50,
        help=(
            "Menor prevalencia considerada na selecao robusta do threshold, "
            "como fracao da prevalencia observada no treino."
        ),
    )
    p.add_argument(
        "--max-oof-fpr",
        type=float,
        default=0.10,
        help="FPR maximo desejado no OOF para a selecao robusta.",
    )
    p.add_argument(
        "--min-robust-precision",
        type=float,
        default=0.20,
        help=(
            "Precision minima exigida no pior caso de prevalencia "
            "considerado durante a selecao do threshold."
        ),
    )

    p.add_argument("--random-state", type=int, default=42)

    p.add_argument(
        "--output",
        type=Path,
        default=Path("predictions_anfis_improved.csv"),
    )
    p.add_argument(
        "--report-prefix",
        type=Path,
        default=Path("report_anfis_improved"),
    )

    return p


# ============================================================
# UTILITÁRIOS
# ============================================================

def parse_float_grid(raw: str) -> list[float]:
    values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Grid cannot be empty.")
    return values


def resolve_input_file(file_path: Path) -> Path:
    file_path = Path(file_path).expanduser()

    if file_path.is_absolute():
        if file_path.exists():
            return file_path.resolve()
        raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")

    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / file_path,
        script_dir / file_path,
        script_dir.parent / file_path,
    ]

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in checked:
            checked.append(candidate)
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {p}" for p in checked)
    raise FileNotFoundError(
        f"Nao foi possivel localizar '{file_path}'.\n"
        f"Locais verificados:\n{searched}"
    )


def detect_label_column(columns: Iterable[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


# ============================================================
# DADOS
# ============================================================

def load_selected_features(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"CSV vazio: {file_path}")

    label_col = detect_label_column(df.columns)
    if label_col is not None:
        df = df.drop(columns=[label_col])

    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Features ausentes em {file_path.name}: {missing}\n"
            f"Esperadas: {SELECTED_FEATURES}"
        )

    X = df[SELECTED_FEATURES].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    return X


def build_raw_split(
    spoof_file: Path,
    benign_file: Path,
) -> tuple[pd.DataFrame, np.ndarray]:

    spoof = load_selected_features(spoof_file)
    benign = load_selected_features(benign_file)

    X = pd.concat([benign, spoof], axis=0, ignore_index=True)

    y = np.concatenate([
        np.zeros(len(benign), dtype=int),
        np.ones(len(spoof), dtype=int),
    ])

    return X, y


def impute_from_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:

    medians = X_train.median(numeric_only=True).fillna(0.0)

    return (
        X_train.fillna(medians),
        X_test.fillna(medians),
        {k: float(v) for k, v in medians.items()},
    )


# ============================================================
# FUZZY / CHOQUET
# ============================================================

def gaussian_mf(x: np.ndarray, center: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-6)
    return np.exp(-((x - center) ** 2) / (2.0 * sigma**2))


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0:
        return np.ones_like(weights) / len(weights)
    return weights / total


def choquet_power_aggregation_matrix(
    values: np.ndarray,
    q: float,
) -> np.ndarray:
    """
    Integral Choquet simetrica vetorizada com medida potencia.
    """
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)

    if values.ndim != 2:
        raise ValueError("values must be 2D.")

    ordered = np.sort(values, axis=1)
    n = ordered.shape[1]

    if n == 1:
        return ordered[:, 0].copy()

    diffs = np.diff(ordered, axis=1)
    k = np.arange(n - 1, 0, -1, dtype=float)
    measure = (k / n) ** float(q)

    result = ordered[:, 0] + np.sum(diffs * measure[None, :], axis=1)

    return np.clip(result, 0.0, 1.0)


# ============================================================
# CONFIGURAÇÃO DO MODELO
# ============================================================

@dataclass(frozen=True)
class BaseConfig:
    n_mfs: int
    mf_width: float
    clip_quantile: float
    weight_power: float
    log_transform: bool


@dataclass
class FuzzyParameters:
    centers: np.ndarray
    sigmas: np.ndarray
    consequents: np.ndarray
    feature_weights: np.ndarray


# ============================================================
# MODELO FUZZY SUPERVISIONADO
# ============================================================

class ImprovedChoquetANFIS:

    def __init__(
        self,
        config: BaseConfig,
        q: float = 1.0,
        blend: float = 0.5,
        consequent_smoothing: float = 1e-3,
    ):
        self.config = config
        self.q = float(q)
        self.blend = float(blend)
        self.consequent_smoothing = float(consequent_smoothing)

        self.scaler = MinMaxScaler()
        self.params: FuzzyParameters | None = None

        self.clip_low_: np.ndarray | None = None
        self.clip_high_: np.ndarray | None = None

    # --------------------------------------------------------
    # Preprocessamento
    # --------------------------------------------------------

    @staticmethod
    def _signed_log1p(X: np.ndarray) -> np.ndarray:
        return np.sign(X) * np.log1p(np.abs(X))

    def _pre_base(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype=float)

        if self.config.log_transform:
            arr = self._signed_log1p(arr)

        return arr

    def _fit_transform_preprocessor(self, X) -> np.ndarray:
        arr = self._pre_base(X)

        q = float(self.config.clip_quantile)

        if q > 0:
            self.clip_low_ = np.quantile(arr, q, axis=0)
            self.clip_high_ = np.quantile(arr, 1.0 - q, axis=0)
            arr = np.clip(arr, self.clip_low_, self.clip_high_)
        else:
            self.clip_low_ = np.full(arr.shape[1], -np.inf)
            self.clip_high_ = np.full(arr.shape[1], np.inf)

        return self.scaler.fit_transform(arr)

    def _transform_preprocessor(self, X) -> np.ndarray:
        arr = self._pre_base(X)
        arr = np.clip(arr, self.clip_low_, self.clip_high_)
        return self.scaler.transform(arr)

    # --------------------------------------------------------
    # Memberships
    # --------------------------------------------------------

    def _membership_matrix(self, X_norm: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Model is not fitted.")

        return gaussian_mf(
            X_norm[:, :, None],
            self.params.centers[None, :, :],
            self.params.sigmas[None, :, :],
        )

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        y = np.asarray(y).astype(int)

        X_norm = self._fit_transform_preprocessor(X)

        n_features = X_norm.shape[1]
        n_mfs = int(self.config.n_mfs)

        # Centros por quantis para seguir a distribuicao real.
        quantiles = np.linspace(0.08, 0.92, n_mfs)
        centers = np.stack(
            [np.quantile(X_norm, q, axis=0) for q in quantiles],
            axis=1,
        )

        # Largura baseada em distancia entre centros + dispersao.
        if n_mfs > 1:
            spacing = np.diff(centers, axis=1)
            typical_spacing = np.median(np.maximum(spacing, 1e-4), axis=1)
        else:
            typical_spacing = np.std(X_norm, axis=0)

        global_std = np.std(X_norm, axis=0)

        base_sigma = np.maximum(
            typical_spacing,
            0.20 * global_std,
        )

        base_sigma = np.maximum(
            base_sigma * float(self.config.mf_width),
            1e-3,
        )

        sigmas = np.repeat(base_sigma[:, None], n_mfs, axis=1)

        # Parametros temporarios.
        self.params = FuzzyParameters(
            centers=centers,
            sigmas=sigmas,
            consequents=np.zeros((n_features, n_mfs), dtype=float),
            feature_weights=np.ones(n_features, dtype=float) / n_features,
        )

        mu = self._membership_matrix(X_norm)

        positive = y == 1
        negative = y == 0

        # Consequentes supervisionados e balanceados por classe.
        pos_strength = np.mean(mu[positive], axis=0)
        neg_strength = np.mean(mu[negative], axis=0)

        s = self.consequent_smoothing

        consequents = (pos_strength + s) / (
            pos_strength + neg_strength + 2.0 * s
        )

        self.params.consequents = np.clip(consequents, 0.0, 1.0)

        # Score local de cada feature.
        local = self._local_scores_from_mu(mu)

        # Pesos supervisionados:
        # combina AUC e AP-lift de cada feature.
        prevalence = float(np.mean(y))

        utilities = []

        for j in range(n_features):
            feature_score = local[:, j]

            try:
                auc = float(roc_auc_score(y, feature_score))
                auc_utility = max(0.0, 2.0 * (auc - 0.5))
            except ValueError:
                auc_utility = 0.0

            try:
                ap = float(average_precision_score(y, feature_score))
                ap_utility = max(
                    0.0,
                    (ap - prevalence) / max(1.0 - prevalence, 1e-9),
                )
            except ValueError:
                ap_utility = 0.0

            utility = 0.60 * auc_utility + 0.40 * ap_utility
            utilities.append(utility)

        utilities = np.asarray(utilities, dtype=float)

        # Piso evita zerar totalmente uma feature.
        max_u = max(float(np.max(utilities)), 1e-6)
        utilities = np.maximum(utilities, 0.02 * max_u)

        utilities = utilities ** float(self.config.weight_power)

        self.params.feature_weights = normalize_weights(utilities)

        return self

    # --------------------------------------------------------
    # Scores locais
    # --------------------------------------------------------

    def _local_scores_from_mu(self, mu: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Model is not fitted.")

        numerator = np.sum(
            mu * self.params.consequents[None, :, :],
            axis=2,
        )

        denominator = np.maximum(np.sum(mu, axis=2), 1e-12)

        return np.clip(numerator / denominator, 0.0, 1.0)

    def local_scores(self, X: pd.DataFrame) -> np.ndarray:
        X_norm = self._transform_preprocessor(X)
        mu = self._membership_matrix(X_norm)
        return self._local_scores_from_mu(mu)

    # --------------------------------------------------------
    # Agregacao
    # --------------------------------------------------------

    def aggregate_local_scores(
        self,
        local: np.ndarray,
        q: float | None = None,
        blend: float | None = None,
    ) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Model is not fitted.")

        q = self.q if q is None else float(q)
        blend = self.blend if blend is None else float(blend)

        weights = self.params.feature_weights

        weighted_mean = local @ weights

        # Choquet tambem respeita a importancia das features.
        relative_weight = weights / np.mean(weights)

        weighted_local = np.clip(
            local * relative_weight[None, :],
            0.0,
            1.0,
        )

        choquet_score = choquet_power_aggregation_matrix(
            weighted_local,
            q=q,
        )

        score = blend * weighted_mean + (1.0 - blend) * choquet_score

        return np.clip(score, 0.0, 1.0)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.aggregate_local_scores(self.local_scores(X))


# ============================================================
# THRESHOLD ROBUSTO A PREVALENCE SHIFT
# ============================================================

def robust_threshold_search(
    y_true: np.ndarray,
    scores: np.ndarray,
    prevalence_floor_factor: float,
    max_fpr: float,
    min_robust_precision: float,
) -> dict:
    """
    Seleciona o threshold de forma robusta a mudancas de prevalencia.

    Implementacao vetorizada:
    - ordena os scores uma unica vez;
    - calcula TP/FP cumulativos;
    - avalia todos os thresholds distintos sem chamar
      confusion_matrix repetidamente.

    Isso substitui a versao anterior, que fazia centenas de
    varreduras completas no dataset para cada combinacao q/blend.
    """

    y_true = np.asarray(y_true).astype(np.int8)
    scores = np.asarray(scores, dtype=np.float64)

    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same length.")

    if y_true.size == 0:
        raise ValueError("Empty arrays are not allowed.")

    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))

    if positives == 0 or negatives == 0:
        raise ValueError(
            "Both classes are required for threshold optimization."
        )

    observed_prevalence = positives / len(y_true)

    min_prevalence = max(
        1e-4,
        observed_prevalence * float(prevalence_floor_factor),
    )

    prevalence_grid = np.linspace(
        min_prevalence,
        observed_prevalence,
        9,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Ordena do maior score para o menor.
    # Para threshold = score[k], tudo de 0..k e positivo.
    # --------------------------------------------------------

    order = np.argsort(scores)[::-1]

    sorted_scores = scores[order]
    sorted_y = y_true[order]

    cumulative_tp = np.cumsum(sorted_y == 1, dtype=np.int64)
    cumulative_fp = np.cumsum(sorted_y == 0, dtype=np.int64)

    # --------------------------------------------------------
    # Avalia apenas o ultimo indice de cada grupo de scores
    # iguais, pois threshold >= valor inclui todo o grupo.
    # --------------------------------------------------------

    distinct_end = np.empty(len(sorted_scores), dtype=bool)

    if len(sorted_scores) > 1:
        distinct_end[:-1] = (
            sorted_scores[:-1] != sorted_scores[1:]
        )

    distinct_end[-1] = True

    idx = np.flatnonzero(distinct_end)

    thresholds = sorted_scores[idx]
    tp = cumulative_tp[idx].astype(np.float64)
    fp = cumulative_fp[idx].astype(np.float64)

    fn = positives - tp
    tn = negatives - fp

    # --------------------------------------------------------
    # Metricas observadas no OOF.
    # --------------------------------------------------------

    tpr = tp / positives
    fpr = fp / negatives

    predicted_positive = tp + fp

    observed_precision = np.divide(
        tp,
        predicted_positive,
        out=np.zeros_like(tp),
        where=predicted_positive > 0,
    )

    observed_f1 = np.divide(
        2.0 * tp,
        2.0 * tp + fp + fn,
        out=np.zeros_like(tp),
        where=(2.0 * tp + fp + fn) > 0,
    )

    specificity = tn / negatives
    balanced_accuracy = 0.5 * (tpr + specificity)

    # --------------------------------------------------------
    # Precision/F1 esperados para diferentes prevalencias.
    #
    # Shape:
    # thresholds x prevalence_grid
    # --------------------------------------------------------

    pi = prevalence_grid[None, :]

    tpr_matrix = tpr[:, None]
    fpr_matrix = fpr[:, None]

    denom = (
        tpr_matrix * pi
        + fpr_matrix * (1.0 - pi)
    )

    robust_precision = np.divide(
        tpr_matrix * pi,
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )

    robust_f1 = np.divide(
        2.0 * robust_precision * tpr_matrix,
        robust_precision + tpr_matrix,
        out=np.zeros_like(robust_precision),
        where=(robust_precision + tpr_matrix) > 0,
    )

    worst_precision = np.min(
        robust_precision,
        axis=1,
    )

    worst_f1 = np.min(
        robust_f1,
        axis=1,
    )

    mean_robust_f1 = np.mean(
        robust_f1,
        axis=1,
    )

    feasible = (
        (fpr <= float(max_fpr))
        & (
            worst_precision
            >= float(min_robust_precision)
        )
    )

    # --------------------------------------------------------
    # Escolha lexicografica.
    #
    # Regra principal:
    # maior worst_f1.
    #
    # Desempates:
    # maior mean robust F1,
    # maior F1 OOF,
    # menor FPR.
    # --------------------------------------------------------

    feasible_idx = np.flatnonzero(feasible)

    used_fallback = False

    if feasible_idx.size > 0:

        candidates = feasible_idx

        ranking = np.lexsort((
            fpr[candidates],
            -observed_f1[candidates],
            -mean_robust_f1[candidates],
            -worst_f1[candidates],
        ))

        best_idx = candidates[
            ranking[0]
        ]

    else:

        used_fallback = True

        objective = (
            worst_f1
            - 0.35 * fpr
        )

        ranking = np.lexsort((
            -balanced_accuracy,
            -observed_f1,
            -objective,
        ))

        best_idx = ranking[0]

    return {
        "threshold":
            float(thresholds[best_idx]),

        "tpr":
            float(tpr[best_idx]),

        "fpr":
            float(fpr[best_idx]),

        "observed_precision":
            float(
                observed_precision[best_idx]
            ),

        "observed_f1":
            float(
                observed_f1[best_idx]
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy[best_idx]
            ),

        "worst_precision":
            float(
                worst_precision[best_idx]
            ),

        "worst_f1":
            float(
                worst_f1[best_idx]
            ),

        "mean_robust_f1":
            float(
                mean_robust_f1[best_idx]
            ),

        "feasible":
            bool(
                feasible[best_idx]
            ),

        "observed_prevalence":
            float(
                observed_prevalence
            ),

        "min_assumed_prevalence":
            float(
                min_prevalence
            ),

        "used_fallback":
            bool(
                used_fallback
            ),

        "n_thresholds_evaluated":
            int(
                len(thresholds)
            ),
    }


# ============================================================
# MÉTRICAS
# ============================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def metrics_report_text(
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> str:

    m = evaluate_predictions(y_true, y_pred, y_score)

    cm = np.array([
        [m["tn"], m["fp"]],
        [m["fn"], m["tp"]],
    ])

    return "\n".join([
        title,
        f"Accuracy          : {m['accuracy']:.4f}",
        f"Balanced accuracy : {m['balanced_accuracy']:.4f}",
        f"Precision         : {m['precision']:.4f}",
        f"Recall            : {m['recall']:.4f}",
        f"Specificity       : {m['specificity']:.4f}",
        f"FPR               : {m['fpr']:.4f}",
        f"F1                : {m['f1']:.4f}",
        f"ROC AUC           : {m['roc_auc']:.4f}",
        f"PR AUC            : {m['pr_auc']:.4f}",
        "Confusion matrix:",
        str(cm),
        "Classification report:",
        classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0,
        ),
    ])


# ============================================================
# ESPAÇO DE BUSCA DOS PARÂMETROS FUZZY
# ============================================================

def all_base_configs() -> list[BaseConfig]:
    configs = []

    for n_mfs, mf_width, clip_q, weight_power, log_transform in product(
        [3, 5],
        [0.75, 1.0, 1.35, 1.70],
        [0.0, 0.0025, 0.005, 0.01],
        [0.65, 1.0, 1.5, 2.0],
        [False, True],
    ):
        configs.append(
            BaseConfig(
                n_mfs=n_mfs,
                mf_width=mf_width,
                clip_quantile=clip_q,
                weight_power=weight_power,
                log_transform=log_transform,
            )
        )

    return configs


# ============================================================
# GERA OOF PARA UMA CONFIGURAÇÃO
# ============================================================

def generate_oof_for_config(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    config: BaseConfig,
    q_grid: list[float],
    blend_grid: list[float],
    cv_folds: int,
    cv_repeats: int,
    random_state: int,
    store_models: bool = False,
):
    """
    Retorna scores OOF medios para cada (q, blend).

    Cada amostra recebe exatamente um score OOF por repeticao.
    Depois os scores sao promediados entre repeticoes.
    """

    splitter = RepeatedStratifiedKFold(
        n_splits=cv_folds,
        n_repeats=cv_repeats,
        random_state=random_state,
    )

    combinations = [
        (float(q), float(blend))
        for q in q_grid
        for blend in blend_grid
    ]

    sums = {
        combo: np.zeros(len(y_train), dtype=np.float64)
        for combo in combinations
    }

    counts = np.zeros(len(y_train), dtype=np.int16)

    models = []

    n_total_folds = cv_folds * cv_repeats

    for fold_index, (fit_idx, val_idx) in enumerate(
        splitter.split(X_train, y_train),
        start=1,
    ):
        model = ImprovedChoquetANFIS(config=config)

        model.fit(
            X_train.iloc[fit_idx],
            y_train[fit_idx],
        )

        local_val = model.local_scores(
            X_train.iloc[val_idx]
        )

        weights = model.params.feature_weights

        weighted_mean = local_val @ weights

        relative_weight = weights / np.mean(weights)

        weighted_local = np.clip(
            local_val * relative_weight[None, :],
            0.0,
            1.0,
        )

        choquet_cache = {
            float(q): choquet_power_aggregation_matrix(
                weighted_local,
                q=float(q),
            )
            for q in q_grid
        }

        for q, blend in combinations:
            score = (
                blend * weighted_mean
                + (1.0 - blend) * choquet_cache[q]
            )
            sums[(q, blend)][val_idx] += score

        counts[val_idx] += 1

        if store_models:
            models.append(model)

        print(
            f"    fold {fold_index:02d}/{n_total_folds} concluido",
            flush=True,
        )

    if np.any(counts == 0):
        raise RuntimeError("Some samples did not receive an OOF prediction.")

    oof = {
        combo: sums[combo] / counts
        for combo in combinations
    }

    return oof, models


# ============================================================
# BUSCA COMPLETA
# ============================================================

def tune_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    q_grid: list[float],
    blend_grid: list[float],
    search_iter: int,
    cv_folds: int,
    cv_repeats: int,
    random_state: int,
    prevalence_floor_factor: float,
    max_oof_fpr: float,
    min_robust_precision: float,
):
    rng = np.random.default_rng(random_state)

    configs = all_base_configs()

    search_iter = min(search_iter, len(configs))

    selected_idx = rng.choice(
        len(configs),
        size=search_iter,
        replace=False,
    )

    selected_configs = [
        configs[int(i)]
        for i in selected_idx
    ]

    best = None
    logs = []

    print("\n" + "=" * 70)
    print("BUSCA DO ANFIS")
    print("=" * 70)

    print(f"Base configs : {len(selected_configs)}")
    print(f"CV           : {cv_folds} folds x {cv_repeats} repeats")
    print(f"q            : {q_grid}")
    print(f"blend        : {blend_grid}")

    for idx, config in enumerate(selected_configs, start=1):

        print("\n" + "-" * 70)
        print(f"Config {idx}/{len(selected_configs)}")
        print(config)
        print("-" * 70)

        t0 = time.perf_counter()

        oof_map, _ = generate_oof_for_config(
            X_train=X_train,
            y_train=y_train,
            config=config,
            q_grid=q_grid,
            blend_grid=blend_grid,
            cv_folds=cv_folds,
            cv_repeats=cv_repeats,
            random_state=random_state,
            store_models=False,
        )

        config_candidates = []

        print(
            f"    avaliando {len(oof_map)} combinacoes q/blend...",
            flush=True,
        )

        combo_index = 0

        for (q, blend), score in oof_map.items():

            combo_index += 1

            threshold_info = robust_threshold_search(
                y_true=y_train,
                scores=score,
                prevalence_floor_factor=prevalence_floor_factor,
                max_fpr=max_oof_fpr,
                min_robust_precision=min_robust_precision,
            )

            roc_auc = float(
                roc_auc_score(y_train, score)
            )

            pr_auc = float(
                average_precision_score(y_train, score)
            )

            candidate = {
                "config": config,
                "q": float(q),
                "blend": float(blend),
                "oof_roc_auc": roc_auc,
                "oof_pr_auc": pr_auc,
                "threshold": threshold_info,
            }

            key = (
                threshold_info["worst_f1"],
                pr_auc,
                threshold_info["observed_f1"],
                roc_auc,
            )

            candidate["key"] = key

            config_candidates.append(candidate)

            if (
                combo_index == 1
                or combo_index % 5 == 0
                or combo_index == len(oof_map)
            ):
                print(
                    f"      combinacao "
                    f"{combo_index:02d}/{len(oof_map)} | "
                    f"q={q:.2f} | "
                    f"blend={blend:.2f} | "
                    f"robust_F1={threshold_info['worst_f1']:.4f} | "
                    f"OOF_F1={threshold_info['observed_f1']:.4f} | "
                    f"FPR={threshold_info['fpr']:.4f}",
                    flush=True,
                )

            if best is None or key > best["key"]:
                best = candidate

        elapsed = time.perf_counter() - t0

        config_best_candidate = max(
            config_candidates,
            key=lambda item: item["key"],
        )

        config_best = {
            "q":
                config_best_candidate["q"],

            "blend":
                config_best_candidate["blend"],

            "pr_auc":
                config_best_candidate["oof_pr_auc"],

            "roc_auc":
                config_best_candidate["oof_roc_auc"],

            "threshold":
                config_best_candidate["threshold"],
        }

        log = {
            "config": config.__dict__,
            "best_q": config_best["q"],
            "best_blend": config_best["blend"],
            "best_oof_pr_auc": float(config_best["pr_auc"]),
            "best_oof_roc_auc": float(config_best["roc_auc"]),
            "best_threshold": config_best["threshold"],
            "seconds": elapsed,
        }

        logs.append(log)

        print(
            f"Melhor desta config: "
            f"q={config_best['q']:.3f} | "
            f"blend={config_best['blend']:.2f} | "
            f"robust_F1={config_best['threshold']['worst_f1']:.4f} | "
            f"OOF_F1={config_best['threshold']['observed_f1']:.4f} | "
            f"PR_AUC={config_best['pr_auc']:.4f} | "
            f"FPR={config_best['threshold']['fpr']:.4f}"
        )

    return best, logs


# ============================================================
# TREINA O ENSEMBLE FINAL CROSS-FITTED
# ============================================================

def fit_final_crossfit_ensemble(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    config: BaseConfig,
    q: float,
    blend: float,
    cv_folds: int,
    cv_repeats: int,
    random_state: int,
):
    splitter = RepeatedStratifiedKFold(
        n_splits=cv_folds,
        n_repeats=cv_repeats,
        random_state=random_state,
    )

    oof_sum = np.zeros(len(y_train), dtype=np.float64)
    oof_count = np.zeros(len(y_train), dtype=np.int16)

    models = []

    weights = []

    n_total = cv_folds * cv_repeats

    for fold_index, (fit_idx, val_idx) in enumerate(
        splitter.split(X_train, y_train),
        start=1,
    ):
        model = ImprovedChoquetANFIS(
            config=config,
            q=q,
            blend=blend,
        )

        model.fit(
            X_train.iloc[fit_idx],
            y_train[fit_idx],
        )

        val_score = model.score(
            X_train.iloc[val_idx]
        )

        oof_sum[val_idx] += val_score
        oof_count[val_idx] += 1

        models.append(model)
        weights.append(model.params.feature_weights.copy())

        print(
            f"  ensemble fold {fold_index:02d}/{n_total} concluido",
            flush=True,
        )

    oof_score = oof_sum / oof_count

    mean_weights = np.mean(
        np.asarray(weights),
        axis=0,
    )

    return models, oof_score, mean_weights


# ============================================================
# ENSEMBLE SCORE
# ============================================================

def ensemble_score(
    models: list[ImprovedChoquetANFIS],
    X: pd.DataFrame,
) -> np.ndarray:

    accumulator = np.zeros(len(X), dtype=np.float64)

    for idx, model in enumerate(models, start=1):
        accumulator += model.score(X)

        print(
            f"    modelo {idx:02d}/{len(models)}",
            flush=True,
        )

    return accumulator / len(models)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parser().parse_args()

    args.spoof_train = resolve_input_file(args.spoof_train)
    args.spoof_test = resolve_input_file(args.spoof_test)
    args.benign_train = resolve_input_file(args.benign_train)
    args.benign_test = resolve_input_file(args.benign_test)

    q_grid = parse_float_grid(args.q_grid)
    blend_grid = parse_float_grid(args.blend_grid)

    print("\n" + "=" * 70)
    print("IMPROVED CROSS-FITTED CHOQUET-ANFIS")
    print("ARP SPOOFING - 8 FEATURES")
    print("=" * 70)

    print("\nFeatures:")
    for idx, feature in enumerate(SELECTED_FEATURES, start=1):
        print(f"  {idx}. {feature}")

    # ========================================================
    # 1. DADOS
    # ========================================================

    print("\n[1/7] Carregando dados...", flush=True)

    X_train, y_train = build_raw_split(
        args.spoof_train,
        args.benign_train,
    )

    X_test, y_test = build_raw_split(
        args.spoof_test,
        args.benign_test,
    )

    X_train, X_test, medians = impute_from_train(
        X_train,
        X_test,
    )

    print("[OK] Dados carregados.")
    print(f"Treino: {X_train.shape}")
    print(f"Teste : {X_test.shape}")
    print(f"Features: {X_train.shape[1]}")

    train_prevalence = float(np.mean(y_train))

    print(f"Prevalencia spoof treino: {train_prevalence:.4%}")
    print(
        "Faixa de prevalencia usada pelo threshold robusto: "
        f"{train_prevalence * args.prevalence_floor_factor:.4%} "
        f"a {train_prevalence:.4%}"
    )

    # ========================================================
    # 2. TUNING
    # ========================================================

    print(
        "\n[2/7] Buscando configuracao fuzzy, q, blend e threshold...",
        flush=True,
    )

    t0 = time.perf_counter()

    best, search_logs = tune_model(
        X_train=X_train,
        y_train=y_train,
        q_grid=q_grid,
        blend_grid=blend_grid,
        search_iter=args.search_iter,
        cv_folds=args.cv_folds,
        cv_repeats=args.cv_repeats,
        random_state=args.random_state,
        prevalence_floor_factor=args.prevalence_floor_factor,
        max_oof_fpr=args.max_oof_fpr,
        min_robust_precision=args.min_robust_precision,
    )

    tuning_time = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print("MELHOR CONFIGURAÇÃO")
    print("=" * 70)

    print(f"Base config     : {best['config']}")
    print(f"q               : {best['q']:.6f}")
    print(f"blend           : {best['blend']:.6f}")
    print(f"OOF ROC AUC     : {best['oof_roc_auc']:.6f}")
    print(f"OOF PR AUC      : {best['oof_pr_auc']:.6f}")
    print(f"Threshold       : {best['threshold']['threshold']:.6f}")
    print(f"OOF F1          : {best['threshold']['observed_f1']:.6f}")
    print(f"Robust worst F1 : {best['threshold']['worst_f1']:.6f}")
    print(f"OOF Precision   : {best['threshold']['observed_precision']:.6f}")
    print(f"OOF Recall      : {best['threshold']['tpr']:.6f}")
    print(f"OOF FPR         : {best['threshold']['fpr']:.6f}")
    print(f"Tuning seconds  : {tuning_time:.2f}")

    # ========================================================
    # 3. ENSEMBLE FINAL
    # ========================================================

    print(
        "\n[3/7] Treinando ensemble cross-fitted final...",
        flush=True,
    )

    t0 = time.perf_counter()

    models, train_oof_score, mean_feature_weights = (
        fit_final_crossfit_ensemble(
            X_train=X_train,
            y_train=y_train,
            config=best["config"],
            q=best["q"],
            blend=best["blend"],
            cv_folds=args.cv_folds,
            cv_repeats=args.cv_repeats,
            random_state=args.random_state,
        )
    )

    fit_time = time.perf_counter() - t0

    # Recalcula threshold usando o OOF do ensemble final.
    threshold_info = robust_threshold_search(
        y_true=y_train,
        scores=train_oof_score,
        prevalence_floor_factor=args.prevalence_floor_factor,
        max_fpr=args.max_oof_fpr,
        min_robust_precision=args.min_robust_precision,
    )

    threshold = float(
        threshold_info["threshold"]
    )

    print("[OK] Ensemble treinado.")
    print(f"Modelos no ensemble: {len(models)}")
    print(f"Threshold final     : {threshold:.6f}")
    print(f"OOF F1 final        : {threshold_info['observed_f1']:.6f}")
    print(f"OOF FPR final       : {threshold_info['fpr']:.6f}")
    print(f"Tempo fit ensemble  : {fit_time:.2f}s")

    print("\nPesos médios das features:")
    for feature, weight in zip(
        SELECTED_FEATURES,
        mean_feature_weights,
    ):
        print(f"  {feature:<18} {weight:.6f}")

    # ========================================================
    # 4. TEST SCORE
    # ========================================================

    print(
        "\n[4/7] Gerando scores do conjunto de teste...",
        flush=True,
    )

    t0 = time.perf_counter()

    test_score = ensemble_score(
        models,
        X_test,
    )

    infer_time = time.perf_counter() - t0

    train_pred = (
        train_oof_score >= threshold
    ).astype(int)

    test_pred = (
        test_score >= threshold
    ).astype(int)

    print("[OK] Scores gerados.")
    print(f"Tempo inferencia teste: {infer_time:.6f}s")

    # ========================================================
    # 5. MÉTRICAS
    # ========================================================

    print(
        "\n[5/7] Calculando métricas...",
        flush=True,
    )

    # Importante:
    # train metrics = OOF, portanto nao sao in-sample.
    train_metrics = evaluate_predictions(
        y_train,
        train_pred,
        train_oof_score,
    )

    test_metrics = evaluate_predictions(
        y_test,
        test_pred,
        test_score,
    )

    train_text = metrics_report_text(
        "Train OOF metrics",
        y_train,
        train_pred,
        train_oof_score,
    )

    test_text = metrics_report_text(
        "Test metrics",
        y_test,
        test_pred,
        test_score,
    )

    print("\n" + "=" * 70)
    print(train_text)
    print("=" * 70)
    print(test_text)
    print("=" * 70)

    # ========================================================
    # 6. SALVAR
    # ========================================================

    print(
        "\n[6/7] Salvando resultados...",
        flush=True,
    )

    result = X_test.copy()
    result["true_label"] = y_test
    result["score"] = test_score
    result["prediction"] = test_pred
    result.to_csv(args.output, index=False)

    txt_path = args.report_prefix.with_suffix(".txt")
    json_path = args.report_prefix.with_suffix(".json")

    txt_path.write_text(
        "\n".join([
            "MODEL: Improved Cross-Fitted Choquet-ANFIS",
            "=" * 70,
            f"Features: {SELECTED_FEATURES}",
            f"Base config: {best['config']}",
            f"q: {best['q']}",
            f"blend: {best['blend']}",
            f"threshold: {threshold}",
            "",
            train_text,
            "",
            "=" * 70,
            test_text,
        ]),
        encoding="utf-8",
    )

    payload = {
        "model": "ImprovedCrossFittedChoquetANFIS",
        "features": SELECTED_FEATURES,
        "base_config": best["config"].__dict__,
        "q": best["q"],
        "blend": best["blend"],
        "threshold": threshold,
        "threshold_info": threshold_info,
        "cv_folds": args.cv_folds,
        "cv_repeats": args.cv_repeats,
        "ensemble_size": len(models),
        "search_iterations": args.search_iter,
        "tuning_seconds": tuning_time,
        "ensemble_fit_seconds": fit_time,
        "test_inference_seconds": infer_time,
        "imputation_medians": medians,
        "mean_feature_weights": {
            feature: float(weight)
            for feature, weight in zip(
                SELECTED_FEATURES,
                mean_feature_weights,
            )
        },
        "train_oof_metrics": train_metrics,
        "test_metrics": test_metrics,
        "search_logs": search_logs,
    }

    json_path.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Predições  : {args.output}")
    print(f"Report TXT : {txt_path}")
    print(f"Report JSON: {json_path}")

    # ========================================================
    # 7. FINAL
    # ========================================================

    print("\n[7/7] Finalizando...")

    print("\n" + "=" * 70)
    print("EXECUÇÃO CONCLUÍDA")
    print("=" * 70)

    print("Modelo        : ImprovedCrossFittedChoquetANFIS")
    print(f"Features      : {len(SELECTED_FEATURES)}")
    print(f"Ensemble      : {len(models)} modelos")
    print(f"q             : {best['q']:.6f}")
    print(f"Blend         : {best['blend']:.6f}")
    print(f"Threshold     : {threshold:.6f}")
    print(f"OOF F1        : {train_metrics['f1']:.6f}")
    print(f"OOF PR AUC    : {train_metrics['pr_auc']:.6f}")
    print(f"Predições     : {args.output}")
    print(f"Report TXT    : {txt_path}")
    print(f"Report JSON   : {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
