from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# FEATURES FIXAS - MESMAS DOS OUTROS MODELOS
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
            "ANFIS supervisionado com Choquet 2-aditivo, "
            "calibracao OOF e ensemble para ARP spoofing."
        )
    )

    p.add_argument(
        "--spoof-train",
        type=Path,
        default=Path(DEFAULT_SPOOF_TRAIN_FILE),
    )

    p.add_argument(
        "--spoof-test",
        type=Path,
        default=Path(DEFAULT_SPOOF_TEST_FILE),
    )

    p.add_argument(
        "--benign-train",
        type=Path,
        default=Path(DEFAULT_BENIGN_TRAIN_FILE),
    )

    p.add_argument(
        "--benign-test",
        type=Path,
        default=Path(DEFAULT_BENIGN_TEST_FILE),
    )

    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
    )

    p.add_argument(
        "--search-iter",
        type=int,
        default=6,
        help="Numero de configuracoes fuzzy/Choquet testadas.",
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Numero de melhores configuracoes usadas no ensemble.",
    )

    p.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    # Threshold robusto
    p.add_argument(
        "--prevalence-floor-factor",
        type=float,
        default=0.50,
    )

    p.add_argument(
        "--max-oof-fpr",
        type=float,
        default=0.08,
    )

    p.add_argument(
        "--min-robust-precision",
        type=float,
        default=0.25,
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path("predictions_anfis_v4.csv"),
    )

    p.add_argument(
        "--report-prefix",
        type=Path,
        default=Path("report_anfis_v4"),
    )

    return p


# ============================================================
# UTILITÁRIOS
# ============================================================

def resolve_input_file(file_path: Path) -> Path:

    file_path = Path(file_path).expanduser()

    if file_path.is_absolute():

        if file_path.exists():
            return file_path.resolve()

        raise FileNotFoundError(
            f"Arquivo nao encontrado: {file_path}"
        )

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

    searched = "\n".join(
        f"  - {path}"
        for path in checked
    )

    raise FileNotFoundError(
        f"Nao foi possivel localizar '{file_path}'.\n"
        f"Locais verificados:\n{searched}"
    )


def detect_label_column(
    columns: Iterable[str],
) -> str | None:

    lowered = {
        column.lower(): column
        for column in columns
    }

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
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    return value


# ============================================================
# DADOS
# ============================================================

def load_selected_features(
    file_path: Path,
) -> pd.DataFrame:

    df = pd.read_csv(
        file_path
    )

    if df.empty:
        raise ValueError(
            f"CSV vazio: {file_path}"
        )

    label_col = detect_label_column(
        df.columns
    )

    if label_col is not None:
        df = df.drop(
            columns=[label_col]
        )

    missing = [
        feature
        for feature in SELECTED_FEATURES
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Features ausentes em {file_path.name}: {missing}\n"
            f"Esperadas: {SELECTED_FEATURES}"
        )

    X = df[
        SELECTED_FEATURES
    ].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return X


def build_raw_split(
    spoof_file: Path,
    benign_file: Path,
) -> tuple[
    pd.DataFrame,
    np.ndarray
]:

    spoof = load_selected_features(
        spoof_file
    )

    benign = load_selected_features(
        benign_file
    )

    X = pd.concat(
        [benign, spoof],
        axis=0,
        ignore_index=True
    )

    y = np.concatenate([
        np.zeros(
            len(benign),
            dtype=int
        ),
        np.ones(
            len(spoof),
            dtype=int
        ),
    ])

    return X, y


def impute_from_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float]
]:

    medians = (
        X_train
        .median(
            numeric_only=True
        )
        .fillna(0.0)
    )

    X_train = X_train.fillna(
        medians
    )

    X_test = X_test.fillna(
        medians
    )

    return (
        X_train,
        X_test,
        {
            key: float(value)
            for key, value
            in medians.items()
        }
    )


# ============================================================
# FUNÇÕES FUZZY
# ============================================================

def gaussian_mf(
    x: np.ndarray,
    center: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:

    sigma = np.maximum(
        sigma,
        1e-6
    )

    return np.exp(
        -(
            (x - center) ** 2
        )
        / (
            2.0
            * sigma**2
        )
    )


def normalize_weights(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=float
    )

    values = np.maximum(
        values,
        0.0
    )

    total = float(
        np.sum(values)
    )

    if total <= 0:

        return np.ones_like(
            values
        ) / len(values)

    return values / total


# ============================================================
# CONFIGURAÇÃO
# ============================================================

@dataclass(frozen=True)
class ANFISConfig:

    n_mfs: int

    clip_quantile: float

    log_transform: bool

    base_width: float

    weight_power: float

    interaction_strength: float


@dataclass
class FuzzyParameters:

    centers: np.ndarray

    sigmas: np.ndarray

    consequents: np.ndarray

    singleton_mobius: np.ndarray

    pair_indices: np.ndarray

    pair_mobius: np.ndarray


# ============================================================
# ANFIS + CHOQUET 2-ADITIVO
# ============================================================

class TwoAdditiveChoquetANFIS:
    """
    ANFIS supervisionado com capacidade Choquet 2-aditiva.

    C_mu(x) =
        sum_i m_i x_i
        + sum_{i<j} m_ij min(x_i, x_j)

    Os coeficientes de Mobius sao nao-negativos e normalizados,
    portanto definem uma capacidade monotona 2-aditiva restrita.

    Centros, sigmas e consequentes fuzzy sao refinados
    supervisionadamente usando apenas o conjunto de ajuste.
    """

    def __init__(
        self,
        config: ANFISConfig,
        consequent_smoothing: float = 1e-3,
    ):

        self.config = config

        self.consequent_smoothing = float(
            consequent_smoothing
        )

        self.scaler = MinMaxScaler()

        self.clip_low_: (
            np.ndarray
            | None
        ) = None

        self.clip_high_: (
            np.ndarray
            | None
        ) = None

        self.params: (
            FuzzyParameters
            | None
        ) = None


    # ========================================================
    # PREPROCESSAMENTO
    # ========================================================

    @staticmethod
    def _signed_log1p(
        X: np.ndarray,
    ) -> np.ndarray:

        return (
            np.sign(X)
            * np.log1p(
                np.abs(X)
            )
        )


    def _base_transform(
        self,
        X,
    ) -> np.ndarray:

        arr = np.asarray(
            X,
            dtype=float
        )

        if self.config.log_transform:

            arr = self._signed_log1p(
                arr
            )

        return arr


    def _fit_preprocess(
        self,
        X,
    ) -> np.ndarray:

        arr = self._base_transform(
            X
        )

        q = float(
            self.config.clip_quantile
        )

        if q > 0:

            self.clip_low_ = (
                np.quantile(
                    arr,
                    q,
                    axis=0
                )
            )

            self.clip_high_ = (
                np.quantile(
                    arr,
                    1.0 - q,
                    axis=0
                )
            )

        else:

            self.clip_low_ = (
                np.full(
                    arr.shape[1],
                    -np.inf
                )
            )

            self.clip_high_ = (
                np.full(
                    arr.shape[1],
                    np.inf
                )
            )

        arr = np.clip(
            arr,
            self.clip_low_,
            self.clip_high_
        )

        return (
            self.scaler
            .fit_transform(
                arr
            )
        )


    def _transform(
        self,
        X,
    ) -> np.ndarray:

        arr = self._base_transform(
            X
        )

        arr = np.clip(
            arr,
            self.clip_low_,
            self.clip_high_
        )

        return (
            self.scaler
            .transform(
                arr
            )
        )


    # ========================================================
    # CONSEQUENTES
    # ========================================================

    def _derive_consequents(
        self,
        mu: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:

        positive = (
            y == 1
        )

        negative = (
            y == 0
        )

        pos_strength = np.mean(
            mu[positive],
            axis=0
        )

        neg_strength = np.mean(
            mu[negative],
            axis=0
        )

        s = (
            self.consequent_smoothing
        )

        consequent = (
            pos_strength + s
        ) / (
            pos_strength
            + neg_strength
            + 2.0 * s
        )

        return np.clip(
            consequent,
            0.0,
            1.0
        )


    # ========================================================
    # SCORE LOCAL DE UMA FEATURE
    # ========================================================

    @staticmethod
    def _local_from_mu(
        mu: np.ndarray,
        consequents: np.ndarray,
    ) -> np.ndarray:

        numerator = np.sum(
            mu
            * consequents[None, :],
            axis=1
        )

        denominator = np.maximum(
            np.sum(
                mu,
                axis=1
            ),
            1e-12
        )

        return np.clip(
            numerator / denominator,
            0.0,
            1.0
        )


    # ========================================================
    # UTILIDADE SUPERVISIONADA
    # ========================================================

    @staticmethod
    def _score_quality(
        y: np.ndarray,
        score: np.ndarray,
    ) -> float:

        prevalence = float(
            np.mean(y)
        )

        try:

            roc = float(
                roc_auc_score(
                    y,
                    score
                )
            )

        except ValueError:

            roc = 0.5

        if roc < 0.5:

            score = 1.0 - score

            roc = 1.0 - roc

        try:

            ap = float(
                average_precision_score(
                    y,
                    score
                )
            )

        except ValueError:

            ap = prevalence

        roc_gain = max(
            0.0,
            2.0 * (
                roc - 0.5
            )
        )

        ap_gain = max(
            0.0,
            (
                ap - prevalence
            )
            / max(
                1.0 - prevalence,
                1e-9
            )
        )

        return (
            0.50 * roc_gain
            + 0.50 * ap_gain
        )


    # ========================================================
    # REFINAMENTO SUPERVISIONADO DE CENTROS/SIGMAS
    # ========================================================

    def _fit_one_feature(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float
    ]:
        """
        Busca supervisionada curta dos parametros fuzzy
        de uma feature.

        Isso torna centros e sigmas parametros efetivamente
        aprendidos, em vez de apenas estatisticas fixas.
        """

        n_mfs = int(
            self.config.n_mfs
        )

        quantile_positions = (
            np.linspace(
                0.08,
                0.92,
                n_mfs
            )
        )

        base_centers = np.quantile(
            x,
            quantile_positions
        )

        median_center = float(
            np.median(
                base_centers
            )
        )

        if n_mfs > 1:

            spacing = np.diff(
                base_centers
            )

            spacing = spacing[
                spacing > 1e-5
            ]

            if spacing.size > 0:

                base_sigma = float(
                    np.median(
                        spacing
                    )
                )

            else:

                base_sigma = float(
                    np.std(x)
                )

        else:

            base_sigma = float(
                np.std(x)
            )

        base_sigma = max(
            base_sigma
            * float(
                self.config.base_width
            ),
            1e-3
        )

        # Busca curta e supervisionada.
        center_scales = [
            0.80,
            1.00,
            1.20,
        ]

        sigma_scales = [
            0.70,
            1.00,
            1.40,
        ]

        best = None

        for center_scale in center_scales:

            centers = (
                median_center
                + center_scale
                * (
                    base_centers
                    - median_center
                )
            )

            centers = np.clip(
                centers,
                0.0,
                1.0
            )

            for sigma_scale in sigma_scales:

                sigma_value = max(
                    base_sigma
                    * sigma_scale,
                    1e-3
                )

                sigmas = np.full(
                    n_mfs,
                    sigma_value,
                    dtype=float
                )

                mu = gaussian_mf(
                    x[:, None],
                    centers[None, :],
                    sigmas[None, :]
                )

                consequents = (
                    self._derive_consequents(
                        mu,
                        y
                    )
                )

                local = (
                    self._local_from_mu(
                        mu,
                        consequents
                    )
                )

                quality = (
                    self._score_quality(
                        y,
                        local
                    )
                )

                orientation = 1.0

                try:

                    auc = roc_auc_score(
                        y,
                        local
                    )

                    if auc < 0.5:

                        orientation = -1.0

                        local = (
                            1.0
                            - local
                        )

                        consequents = (
                            1.0
                            - consequents
                        )

                except ValueError:

                    pass

                candidate = {
                    "quality":
                        quality,

                    "centers":
                        centers.copy(),

                    "sigmas":
                        sigmas.copy(),

                    "consequents":
                        consequents.copy(),

                    "local":
                        local.copy(),

                    "orientation":
                        orientation,
                }

                if (
                    best is None
                    or quality
                    > best["quality"]
                ):

                    best = candidate

        return (
            best["centers"],
            best["sigmas"],
            best["consequents"],
            best["local"],
            float(
                best["quality"]
            ),
        )


    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> "TwoAdditiveChoquetANFIS":

        y = np.asarray(
            y
        ).astype(int)

        X_norm = (
            self._fit_preprocess(
                X
            )
        )

        n_samples, n_features = (
            X_norm.shape
        )

        n_mfs = int(
            self.config.n_mfs
        )

        centers = np.zeros(
            (
                n_features,
                n_mfs
            ),
            dtype=float
        )

        sigmas = np.zeros_like(
            centers
        )

        consequents = np.zeros_like(
            centers
        )

        local_scores = np.zeros(
            (
                n_samples,
                n_features
            ),
            dtype=float
        )

        singleton_quality = np.zeros(
            n_features,
            dtype=float
        )

        # ----------------------------------------------------
        # Parametros fuzzy supervisionados por feature
        # ----------------------------------------------------

        for feature_idx in range(
            n_features
        ):

            (
                centers_i,
                sigmas_i,
                consequents_i,
                local_i,
                quality_i,

            ) = self._fit_one_feature(

                X_norm[
                    :,
                    feature_idx
                ],

                y
            )

            centers[
                feature_idx
            ] = centers_i

            sigmas[
                feature_idx
            ] = sigmas_i

            consequents[
                feature_idx
            ] = consequents_i

            local_scores[
                :,
                feature_idx
            ] = local_i

            singleton_quality[
                feature_idx
            ] = quality_i

        # ----------------------------------------------------
        # Mobius singleton
        # ----------------------------------------------------

        singleton_floor = (
            max(
                float(
                    np.max(
                        singleton_quality
                    )
                ),
                1e-6
            )
            * 0.02
        )

        singleton_raw = np.maximum(
            singleton_quality,
            singleton_floor
        )

        singleton_raw = (
            singleton_raw
            ** float(
                self.config.weight_power
            )
        )

        # ----------------------------------------------------
        # Interacoes 2-aditivas
        #
        # O termo min(x_i,x_j) e exatamente a forma
        # 2-aditiva da integral de Choquet em coordenadas
        # de Mobius.
        #
        # Mantemos m_ij >= 0 para garantir monotonicidade.
        # ----------------------------------------------------

        pair_indices = np.array(
            list(
                combinations(
                    range(n_features),
                    2
                )
            ),
            dtype=int
        )

        pair_raw = np.zeros(
            len(pair_indices),
            dtype=float
        )

        for pair_idx, (
            i,
            j
        ) in enumerate(
            pair_indices
        ):

            pair_component = np.minimum(
                local_scores[:, i],
                local_scores[:, j]
            )

            pair_quality = (
                self._score_quality(
                    y,
                    pair_component
                )
            )

            baseline = (
                0.5
                * (
                    singleton_quality[i]
                    + singleton_quality[j]
                )
            )

            lift = max(
                0.0,
                pair_quality
                - baseline
            )

            pair_raw[
                pair_idx
            ] = (
                float(
                    self.config
                    .interaction_strength
                )
                * lift
            )

        # ----------------------------------------------------
        # Normalizacao da capacidade:
        #
        # sum_i m_i + sum_ij m_ij = 1
        # ----------------------------------------------------

        total = (
            float(
                np.sum(
                    singleton_raw
                )
            )
            + float(
                np.sum(
                    pair_raw
                )
            )
        )

        if total <= 0:

            singleton_mobius = (
                np.ones(
                    n_features
                )
                / n_features
            )

            pair_mobius = np.zeros(
                len(pair_indices)
            )

        else:

            singleton_mobius = (
                singleton_raw
                / total
            )

            pair_mobius = (
                pair_raw
                / total
            )

        self.params = (
            FuzzyParameters(
                centers=centers,
                sigmas=sigmas,
                consequents=consequents,
                singleton_mobius=
                    singleton_mobius,
                pair_indices=
                    pair_indices,
                pair_mobius=
                    pair_mobius,
            )
        )

        return self


    # ========================================================
    # LOCAL SCORES
    # ========================================================

    def local_scores(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:

        if self.params is None:

            raise RuntimeError(
                "Model is not fitted."
            )

        X_norm = (
            self._transform(
                X
            )
        )

        n_samples = (
            X_norm.shape[0]
        )

        n_features = (
            X_norm.shape[1]
        )

        local = np.zeros(
            (
                n_samples,
                n_features
            ),
            dtype=float
        )

        for feature_idx in range(
            n_features
        ):

            mu = gaussian_mf(
                X_norm[
                    :,
                    feature_idx
                ][:, None],

                self.params
                .centers[
                    feature_idx
                ][None, :],

                self.params
                .sigmas[
                    feature_idx
                ][None, :]
            )

            local[
                :,
                feature_idx
            ] = (
                self._local_from_mu(

                    mu,

                    self.params
                    .consequents[
                        feature_idx
                    ]
                )
            )

        return local


    # ========================================================
    # CHOQUET 2-ADITIVO
    # ========================================================

    def score_from_local(
        self,
        local: np.ndarray,
    ) -> np.ndarray:

        if self.params is None:

            raise RuntimeError(
                "Model is not fitted."
            )

        # Singleton
        score = (
            local
            @ self.params
            .singleton_mobius
        )

        # Pairwise Mobius terms
        for pair_idx, (
            i,
            j
        ) in enumerate(
            self.params
            .pair_indices
        ):

            coefficient = (
                self.params
                .pair_mobius[
                    pair_idx
                ]
            )

            if coefficient <= 0:
                continue

            score += (
                coefficient
                * np.minimum(
                    local[:, i],
                    local[:, j]
                )
            )

        return np.clip(
            score,
            0.0,
            1.0
        )


    def score(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:

        return self.score_from_local(
            self.local_scores(
                X
            )
        )


# ============================================================
# CALIBRAÇÃO
# ============================================================

class ScoreCalibrator:

    def __init__(
        self,
        method: str,
    ):

        self.method = method

        self.model = None


    def fit(
        self,
        scores: np.ndarray,
        y: np.ndarray,
    ):

        scores = np.asarray(
            scores,
            dtype=float
        )

        y = np.asarray(
            y
        ).astype(int)

        if self.method == "sigmoid":

            model = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
            )

            model.fit(
                scores.reshape(-1, 1),
                y
            )

        elif self.method == "isotonic":

            model = IsotonicRegression(
                y_min=1e-6,
                y_max=1.0 - 1e-6,
                out_of_bounds="clip",
            )

            model.fit(
                scores,
                y
            )

        else:

            raise ValueError(
                f"Unknown calibration method: {self.method}"
            )

        self.model = model

        return self


    def predict(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:

        scores = np.asarray(
            scores,
            dtype=float
        )

        if self.method == "sigmoid":

            return (
                self.model
                .predict_proba(
                    scores.reshape(
                        -1,
                        1
                    )
                )[:, 1]
            )

        return np.asarray(
            self.model.predict(
                scores
            ),
            dtype=float
        )


def cross_fitted_calibration(
    raw_scores: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    random_state: int,
) -> tuple[
    np.ndarray,
    ScoreCalibrator,
    str,
    dict
]:
    """
    Compara sigmoid e isotonic usando calibracao cross-fitted.

    A escolha usa Brier score, nao os dados de teste.
    """

    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    method_results = {}

    for method in [
        "sigmoid",
        "isotonic",
    ]:

        calibrated_oof = np.zeros(
            len(y),
            dtype=float
        )

        for fit_idx, val_idx in (
            splitter.split(
                raw_scores,
                y
            )
        ):

            calibrator = (
                ScoreCalibrator(
                    method
                )
            )

            calibrator.fit(
                raw_scores[
                    fit_idx
                ],
                y[
                    fit_idx
                ]
            )

            calibrated_oof[
                val_idx
            ] = (
                calibrator.predict(
                    raw_scores[
                        val_idx
                    ]
                )
            )

        brier = float(
            brier_score_loss(
                y,
                calibrated_oof
            )
        )

        method_results[
            method
        ] = {
            "brier":
                brier,

            "calibrated_oof":
                calibrated_oof,
        }

    selected_method = min(
        method_results,
        key=lambda method:
            method_results[
                method
            ]["brier"]
    )

    final_calibrator = (
        ScoreCalibrator(
            selected_method
        )
    )

    final_calibrator.fit(
        raw_scores,
        y
    )

    return (
        method_results[
            selected_method
        ]["calibrated_oof"],

        final_calibrator,

        selected_method,

        {
            method: {
                "brier":
                    float(
                        info["brier"]
                    )
            }
            for method, info
            in method_results.items()
        },
    )


# ============================================================
# THRESHOLD ROBUSTO VETORIZADO
# ============================================================

def robust_threshold_search(
    y_true: np.ndarray,
    scores: np.ndarray,
    prevalence_floor_factor: float,
    max_fpr: float,
    min_robust_precision: float,
) -> dict:

    y_true = np.asarray(
        y_true
    ).astype(np.int8)

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    positives = int(
        np.sum(
            y_true == 1
        )
    )

    negatives = int(
        np.sum(
            y_true == 0
        )
    )

    observed_prevalence = (
        positives
        / len(y_true)
    )

    min_prevalence = max(
        1e-4,

        observed_prevalence
        * float(
            prevalence_floor_factor
        )
    )

    prevalence_grid = np.linspace(
        min_prevalence,
        observed_prevalence,
        9
    )

    order = np.argsort(
        scores
    )[::-1]

    sorted_scores = (
        scores[
            order
        ]
    )

    sorted_y = (
        y_true[
            order
        ]
    )

    cumulative_tp = np.cumsum(
        sorted_y == 1,
        dtype=np.int64
    )

    cumulative_fp = np.cumsum(
        sorted_y == 0,
        dtype=np.int64
    )

    distinct_end = np.empty(
        len(sorted_scores),
        dtype=bool
    )

    if len(sorted_scores) > 1:

        distinct_end[:-1] = (
            sorted_scores[:-1]
            != sorted_scores[1:]
        )

    distinct_end[-1] = True

    idx = np.flatnonzero(
        distinct_end
    )

    thresholds = (
        sorted_scores[
            idx
        ]
    )

    tp = (
        cumulative_tp[
            idx
        ]
        .astype(float)
    )

    fp = (
        cumulative_fp[
            idx
        ]
        .astype(float)
    )

    fn = positives - tp

    tn = negatives - fp

    tpr = (
        tp / positives
    )

    fpr = (
        fp / negatives
    )

    observed_precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp),
        where=(tp + fp) > 0
    )

    observed_f1 = np.divide(
        2.0 * tp,
        2.0 * tp
        + fp
        + fn,
        out=np.zeros_like(tp),
        where=(
            2.0 * tp
            + fp
            + fn
        ) > 0
    )

    specificity = (
        tn / negatives
    )

    balanced_accuracy = (
        0.5
        * (
            tpr
            + specificity
        )
    )

    pi = (
        prevalence_grid[
            None,
            :
        ]
    )

    tpr_matrix = (
        tpr[
            :,
            None
        ]
    )

    fpr_matrix = (
        fpr[
            :,
            None
        ]
    )

    denominator = (
        tpr_matrix
        * pi
        + fpr_matrix
        * (
            1.0 - pi
        )
    )

    robust_precision = np.divide(
        tpr_matrix * pi,
        denominator,
        out=np.zeros_like(
            denominator
        ),
        where=denominator > 0
    )

    robust_f1 = np.divide(
        2.0
        * robust_precision
        * tpr_matrix,

        robust_precision
        + tpr_matrix,

        out=np.zeros_like(
            robust_precision
        ),

        where=(
            robust_precision
            + tpr_matrix
        ) > 0
    )

    worst_precision = np.min(
        robust_precision,
        axis=1
    )

    worst_f1 = np.min(
        robust_f1,
        axis=1
    )

    mean_robust_f1 = np.mean(
        robust_f1,
        axis=1
    )

    feasible = (
        (
            fpr
            <= max_fpr
        )
        & (
            worst_precision
            >= min_robust_precision
        )
    )

    feasible_idx = np.flatnonzero(
        feasible
    )

    used_fallback = False

    if feasible_idx.size > 0:

        candidates = (
            feasible_idx
        )

        ranking = np.lexsort((
            fpr[
                candidates
            ],

            -observed_f1[
                candidates
            ],

            -mean_robust_f1[
                candidates
            ],

            -worst_f1[
                candidates
            ],
        ))

        best_idx = (
            candidates[
                ranking[0]
            ]
        )

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

        best_idx = (
            ranking[0]
        )

    return {
        "threshold":
            float(
                thresholds[
                    best_idx
                ]
            ),

        "tpr":
            float(
                tpr[
                    best_idx
                ]
            ),

        "fpr":
            float(
                fpr[
                    best_idx
                ]
            ),

        "observed_precision":
            float(
                observed_precision[
                    best_idx
                ]
            ),

        "observed_f1":
            float(
                observed_f1[
                    best_idx
                ]
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy[
                    best_idx
                ]
            ),

        "worst_precision":
            float(
                worst_precision[
                    best_idx
                ]
            ),

        "worst_f1":
            float(
                worst_f1[
                    best_idx
                ]
            ),

        "mean_robust_f1":
            float(
                mean_robust_f1[
                    best_idx
                ]
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
    }


# ============================================================
# MÉTRICAS
# ============================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:

    y_true = np.asarray(
        y_true
    ).astype(int)

    y_pred = np.asarray(
        y_pred
    ).astype(int)

    y_score = np.asarray(
        y_score,
        dtype=float
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )
        .ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "specificity":
            float(
                specificity
            ),

        "fpr":
            float(
                fpr
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    y_score
                )
            ),

        "pr_auc":
            float(
                average_precision_score(
                    y_true,
                    y_score
                )
            ),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    }


def metrics_report_text(
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> str:

    m = evaluate_predictions(
        y_true,
        y_pred,
        y_score
    )

    cm = np.array([
        [
            m["tn"],
            m["fp"]
        ],

        [
            m["fn"],
            m["tp"]
        ]
    ])

    return "\n".join([
        title,

        f"Accuracy          : "
        f"{m['accuracy']:.4f}",

        f"Balanced accuracy : "
        f"{m['balanced_accuracy']:.4f}",

        f"Precision         : "
        f"{m['precision']:.4f}",

        f"Recall            : "
        f"{m['recall']:.4f}",

        f"Specificity       : "
        f"{m['specificity']:.4f}",

        f"FPR               : "
        f"{m['fpr']:.4f}",

        f"F1                : "
        f"{m['f1']:.4f}",

        f"ROC AUC           : "
        f"{m['roc_auc']:.4f}",

        f"PR AUC            : "
        f"{m['pr_auc']:.4f}",

        "Confusion matrix:",

        str(cm),

        "Classification report:",

        classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0
        )
    ])


# ============================================================
# ESPAÇO DE BUSCA
# ============================================================

def all_configs() -> list[
    ANFISConfig
]:

    configs = []

    for (
        n_mfs,
        clip_quantile,
        log_transform,
        base_width,
        weight_power,
        interaction_strength,

    ) in product(

        [3, 5],

        [
            0.0025,
            0.005,
            0.01
        ],

        [
            False,
            True
        ],

        [
            0.85,
            1.15,
            1.50
        ],

        [
            0.75,
            1.25,
            1.75
        ],

        [
            0.25,
            0.75,
            1.50
        ],
    ):

        configs.append(
            ANFISConfig(

                n_mfs=
                    n_mfs,

                clip_quantile=
                    clip_quantile,

                log_transform=
                    log_transform,

                base_width=
                    base_width,

                weight_power=
                    weight_power,

                interaction_strength=
                    interaction_strength,
            )
        )

    return configs


# ============================================================
# OOF DE UMA CONFIGURAÇÃO
# ============================================================

def generate_oof_raw(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    config: ANFISConfig,
    cv_folds: int,
    random_state: int,
    X_test: pd.DataFrame | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    list[TwoAdditiveChoquetANFIS]
]:

    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    oof = np.zeros(
        len(y_train),
        dtype=float
    )

    test_accumulator = (
        np.zeros(
            len(X_test),
            dtype=float
        )
        if X_test is not None
        else None
    )

    models = []

    for fold, (
        fit_idx,
        val_idx

    ) in enumerate(

        splitter.split(
            X_train,
            y_train
        ),

        start=1
    ):

        print(
            f"      fold "
            f"{fold:02d}/{cv_folds}",
            flush=True
        )

        model = (
            TwoAdditiveChoquetANFIS(
                config=config
            )
        )

        model.fit(
            X_train.iloc[
                fit_idx
            ],
            y_train[
                fit_idx
            ]
        )

        oof[
            val_idx
        ] = model.score(
            X_train.iloc[
                val_idx
            ]
        )

        if X_test is not None:

            test_accumulator += (
                model.score(
                    X_test
                )
                / cv_folds
            )

        models.append(
            model
        )

    return (
        oof,
        test_accumulator,
        models
    )


# ============================================================
# TUNING
# ============================================================

def tune_configs(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_folds: int,
    search_iter: int,
    random_state: int,
) -> list[dict]:

    rng = np.random.default_rng(
        random_state
    )

    configs = all_configs()

    search_iter = min(
        search_iter,
        len(configs)
    )

    indexes = rng.choice(
        len(configs),
        size=search_iter,
        replace=False
    )

    selected = [
        configs[
            int(index)
        ]
        for index in indexes
    ]

    results = []

    print("\n" + "=" * 70)
    print("BUSCA DE CONFIGURAÇÕES")
    print("=" * 70)

    for index, config in enumerate(
        selected,
        start=1
    ):

        print(
            "\n" + "-" * 70
        )

        print(
            f"Config "
            f"{index}/{len(selected)}"
        )

        print(config)

        print(
            "-" * 70
        )

        t0 = time.perf_counter()

        raw_oof, _, models = (
            generate_oof_raw(
                X_train,
                y_train,
                config,
                cv_folds,
                random_state
                + index,
                X_test=None,
            )
        )

        pr_auc = float(
            average_precision_score(
                y_train,
                raw_oof
            )
        )

        roc_auc = float(
            roc_auc_score(
                y_train,
                raw_oof
            )
        )

        elapsed = (
            time.perf_counter()
            - t0
        )

        result = {
            "config":
                config,

            "raw_oof":
                raw_oof,

            "pr_auc":
                pr_auc,

            "roc_auc":
                roc_auc,

            "seconds":
                elapsed,
        }

        results.append(
            result
        )

        print(
            f"OOF PR AUC : "
            f"{pr_auc:.6f}"
        )

        print(
            f"OOF ROC AUC: "
            f"{roc_auc:.6f}"
        )

        print(
            f"Tempo      : "
            f"{elapsed:.2f}s"
        )

    results.sort(
        key=lambda item: (
            item["pr_auc"],
            item["roc_auc"]
        ),
        reverse=True
    )

    return results


# ============================================================
# PESOS DO ENSEMBLE
# ============================================================

def candidate_ensemble_weights(
    n_models: int,
) -> list[np.ndarray]:

    if n_models == 1:

        return [
            np.array(
                [1.0]
            )
        ]

    candidates = []

    # Equal
    candidates.append(
        np.ones(
            n_models
        )
        / n_models
    )

    # Modelos individuais
    for i in range(
        n_models
    ):

        w = np.zeros(
            n_models
        )

        w[i] = 1.0

        candidates.append(w)

    # Grid simples para top-2
    if n_models == 2:

        for w0 in np.linspace(
            0.0,
            1.0,
            11
        ):

            candidates.append(
                np.array([
                    w0,
                    1.0 - w0
                ])
            )

    # Grid 0.25 para top-3
    elif n_models == 3:

        for a in np.arange(
            0.0,
            1.0001,
            0.25
        ):

            for b in np.arange(
                0.0,
                1.0001 - a,
                0.25
            ):

                c = (
                    1.0
                    - a
                    - b
                )

                if c < -1e-9:
                    continue

                candidates.append(
                    np.array([
                        a,
                        b,
                        max(
                            0.0,
                            c
                        )
                    ])
                )

    # Remove duplicados
    unique = {}

    for weights in candidates:

        weights = normalize_weights(
            weights
        )

        key = tuple(
            np.round(
                weights,
                6
            )
        )

        unique[
            key
        ] = weights

    return list(
        unique.values()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = (
        parser()
        .parse_args()
    )

    args.spoof_train = (
        resolve_input_file(
            args.spoof_train
        )
    )

    args.spoof_test = (
        resolve_input_file(
            args.spoof_test
        )
    )

    args.benign_train = (
        resolve_input_file(
            args.benign_train
        )
    )

    args.benign_test = (
        resolve_input_file(
            args.benign_test
        )
    )

    print("\n" + "=" * 70)
    print("ANFIS V4 - CHOQUET 2-ADITIVO + CALIBRAÇÃO + ENSEMBLE")
    print("ARP SPOOFING - 8 FEATURES")
    print("=" * 70)

    print(
        "\nFeatures:"
    )

    for index, feature in enumerate(
        SELECTED_FEATURES,
        start=1
    ):

        print(
            f"  {index}. {feature}"
        )

    # ========================================================
    # 1. DADOS
    # ========================================================

    print(
        "\n[1/8] Carregando dados...",
        flush=True
    )

    X_train, y_train = (
        build_raw_split(
            args.spoof_train,
            args.benign_train
        )
    )

    X_test, y_test = (
        build_raw_split(
            args.spoof_test,
            args.benign_test
        )
    )

    (
        X_train,
        X_test,
        medians

    ) = impute_from_train(
        X_train,
        X_test
    )

    print(
        "[OK] Dados carregados."
    )

    print(
        f"Treino: "
        f"{X_train.shape}"
    )

    print(
        f"Teste : "
        f"{X_test.shape}"
    )

    print(
        f"Features: "
        f"{X_train.shape[1]}"
    )

    train_prevalence = float(
        np.mean(
            y_train
        )
    )

    print(
        f"Prevalência spoof treino: "
        f"{train_prevalence:.4%}"
    )

    # ========================================================
    # 2. BUSCA DE CONFIGURAÇÕES
    # ========================================================

    print(
        "\n[2/8] Buscando configurações "
        "fuzzy + Choquet 2-aditivo...",
        flush=True
    )

    tuning_start = (
        time.perf_counter()
    )

    tuned = tune_configs(
        X_train=X_train,
        y_train=y_train,
        cv_folds=args.cv_folds,
        search_iter=args.search_iter,
        random_state=args.random_state,
    )

    tuning_seconds = (
        time.perf_counter()
        - tuning_start
    )

    top_k = min(
        args.top_k,
        len(tuned)
    )

    selected = (
        tuned[
            :top_k
        ]
    )

    print("\n" + "=" * 70)
    print(f"TOP {top_k} CONFIGURAÇÕES")
    print("=" * 70)

    for index, item in enumerate(
        selected,
        start=1
    ):

        print(
            f"\n#{index}"
        )

        print(
            item["config"]
        )

        print(
            f"OOF PR AUC : "
            f"{item['pr_auc']:.6f}"
        )

        print(
            f"OOF ROC AUC: "
            f"{item['roc_auc']:.6f}"
        )

    # ========================================================
    # 3. REFIT CROSS-FITTED + TEST SCORE
    # ========================================================

    print(
        "\n[3/8] Gerando OOF e scores de teste "
        "para as melhores configurações...",
        flush=True
    )

    config_data = []

    for index, item in enumerate(
        selected,
        start=1
    ):

        print(
            f"\nConfig final "
            f"{index}/{top_k}",
            flush=True
        )

        (
            raw_oof,
            raw_test,
            models,

        ) = generate_oof_raw(

            X_train,
            y_train,

            item["config"],

            args.cv_folds,

            args.random_state
            + 100
            + index,

            X_test=X_test,
        )

        # ====================================================
        # 4. CALIBRAÇÃO
        # ====================================================

        print(
            "    calibrando scores OOF...",
            flush=True
        )

        (
            calibrated_oof,
            calibrator,
            calibration_method,
            calibration_info,

        ) = cross_fitted_calibration(

            raw_oof,

            y_train,

            cv_folds=args.cv_folds,

            random_state=
                args.random_state
                + 500
                + index,
        )

        calibrated_test = (
            calibrator.predict(
                raw_test
            )
        )

        config_data.append({
            "config":
                item["config"],

            "raw_oof":
                raw_oof,

            "raw_test":
                raw_test,

            "calibrated_oof":
                calibrated_oof,

            "calibrated_test":
                calibrated_test,

            "calibration_method":
                calibration_method,

            "calibration_info":
                calibration_info,

            "pr_auc":
                float(
                    average_precision_score(
                        y_train,
                        calibrated_oof
                    )
                ),

            "roc_auc":
                float(
                    roc_auc_score(
                        y_train,
                        calibrated_oof
                    )
                ),

            "models":
                models,
        })

        print(
            f"    método calibração: "
            f"{calibration_method}"
        )

        print(
            f"    calibrated PR AUC: "
            f"{config_data[-1]['pr_auc']:.6f}"
        )

    # ========================================================
    # 5. ENSEMBLE
    # ========================================================

    print(
        "\n[5/8] Otimizando ensemble no OOF...",
        flush=True
    )

    oof_matrix = np.column_stack([
        item[
            "calibrated_oof"
        ]
        for item in config_data
    ])

    test_matrix = np.column_stack([
        item[
            "calibrated_test"
        ]
        for item in config_data
    ])

    ensemble_candidates = (
        candidate_ensemble_weights(
            top_k
        )
    )

    best_ensemble = None

    for weights in (
        ensemble_candidates
    ):

        ensemble_oof = (
            oof_matrix
            @ weights
        )

        threshold_info = (
            robust_threshold_search(
                y_true=y_train,
                scores=ensemble_oof,

                prevalence_floor_factor=
                    args.prevalence_floor_factor,

                max_fpr=
                    args.max_oof_fpr,

                min_robust_precision=
                    args.min_robust_precision,
            )
        )

        pr_auc = float(
            average_precision_score(
                y_train,
                ensemble_oof
            )
        )

        roc_auc = float(
            roc_auc_score(
                y_train,
                ensemble_oof
            )
        )

        key = (
            threshold_info[
                "worst_f1"
            ],

            pr_auc,

            threshold_info[
                "observed_f1"
            ],

            roc_auc,
        )

        candidate = {
            "weights":
                weights,

            "oof":
                ensemble_oof,

            "threshold":
                threshold_info,

            "pr_auc":
                pr_auc,

            "roc_auc":
                roc_auc,

            "key":
                key,
        }

        if (
            best_ensemble is None
            or key
            > best_ensemble["key"]
        ):

            best_ensemble = (
                candidate
            )

    weights = (
        best_ensemble[
            "weights"
        ]
    )

    ensemble_oof = (
        best_ensemble[
            "oof"
        ]
    )

    ensemble_test = (
        test_matrix
        @ weights
    )

    threshold = float(
        best_ensemble[
            "threshold"
        ][
            "threshold"
        ]
    )

    print(
        "\nPesos do ensemble:"
    )

    for index, weight in enumerate(
        weights,
        start=1
    ):

        print(
            f"  Config {index}: "
            f"{weight:.4f}"
        )

    print(
        f"Threshold OOF robusto: "
        f"{threshold:.6f}"
    )

    print(
        f"OOF PR AUC: "
        f"{best_ensemble['pr_auc']:.6f}"
    )

    print(
        f"OOF ROC AUC: "
        f"{best_ensemble['roc_auc']:.6f}"
    )

    # ========================================================
    # 6. PREDIÇÕES
    # ========================================================

    print(
        "\n[6/8] Gerando predições finais...",
        flush=True
    )

    train_pred = (
        ensemble_oof
        >= threshold
    ).astype(int)

    test_pred = (
        ensemble_test
        >= threshold
    ).astype(int)

    # ========================================================
    # 7. MÉTRICAS
    # ========================================================

    print(
        "\n[7/8] Calculando métricas...",
        flush=True
    )

    train_metrics = (
        evaluate_predictions(
            y_train,
            train_pred,
            ensemble_oof
        )
    )

    test_metrics = (
        evaluate_predictions(
            y_test,
            test_pred,
            ensemble_test
        )
    )

    train_text = (
        metrics_report_text(
            "Train OOF metrics",
            y_train,
            train_pred,
            ensemble_oof
        )
    )

    test_text = (
        metrics_report_text(
            "Test metrics",
            y_test,
            test_pred,
            ensemble_test
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        train_text
    )

    print(
        "=" * 70
    )

    print(
        test_text
    )

    print(
        "=" * 70
    )

    # ========================================================
    # 8. SALVAR
    # ========================================================

    print(
        "\n[8/8] Salvando resultados...",
        flush=True
    )

    result = (
        X_test.copy()
    )

    result[
        "true_label"
    ] = y_test

    result[
        "score"
    ] = ensemble_test

    result[
        "prediction"
    ] = test_pred

    result.to_csv(
        args.output,
        index=False
    )

    txt_path = (
        args.report_prefix
        .with_suffix(".txt")
    )

    json_path = (
        args.report_prefix
        .with_suffix(".json")
    )

    txt_path.write_text(
        "\n".join([
            "MODEL: ANFIS V4 - 2-Additive Choquet + Calibration + Ensemble",
            "=" * 70,
            f"Features: {SELECTED_FEATURES}",
            f"Ensemble weights: {weights.tolist()}",
            f"Threshold: {threshold}",
            "",
            train_text,
            "",
            "=" * 70,
            test_text,
        ]),
        encoding="utf-8"
    )

    # Resumo das capacidades do primeiro modelo de cada config
    capacity_summary = []

    for config_index, item in enumerate(
        config_data,
        start=1
    ):

        model = (
            item["models"][0]
        )

        singleton = {
            feature:
                float(weight)

            for feature, weight
            in zip(
                SELECTED_FEATURES,
                model.params
                .singleton_mobius
            )
        }

        interactions = []

        for (
            pair,
            coefficient

        ) in zip(

            model.params
            .pair_indices,

            model.params
            .pair_mobius
        ):

            if coefficient <= 0:
                continue

            i, j = (
                int(pair[0]),
                int(pair[1])
            )

            interactions.append({
                "feature_1":
                    SELECTED_FEATURES[i],

                "feature_2":
                    SELECTED_FEATURES[j],

                "mobius":
                    float(
                        coefficient
                    ),
            })

        interactions.sort(
            key=lambda item:
                item["mobius"],
            reverse=True
        )

        capacity_summary.append({
            "config_index":
                config_index,

            "config":
                item["config"]
                .__dict__,

            "calibration_method":
                item[
                    "calibration_method"
                ],

            "calibration_info":
                item[
                    "calibration_info"
                ],

            "singleton_mobius":
                singleton,

            "top_interactions":
                interactions[
                    :10
                ],
        })

    payload = {
        "model":
            "ANFIS_V4_TwoAdditiveChoquet",

        "features":
            SELECTED_FEATURES,

        "cv_folds":
            args.cv_folds,

        "search_iter":
            args.search_iter,

        "top_k":
            top_k,

        "tuning_seconds":
            float(
                tuning_seconds
            ),

        "ensemble_weights":
            weights,

        "threshold":
            threshold,

        "threshold_info":
            best_ensemble[
                "threshold"
            ],

        "imputation_medians":
            medians,

        "train_oof_metrics":
            train_metrics,

        "test_metrics":
            test_metrics,

        "selected_configs":
            [
                {
                    "config":
                        item["config"]
                        .__dict__,

                    "tuning_pr_auc":
                        item[
                            "pr_auc"
                        ],

                    "tuning_roc_auc":
                        item[
                            "roc_auc"
                        ],
                }

                for item
                in selected
            ],

        "capacity_summary":
            capacity_summary,
    }

    json_path.write_text(
        json.dumps(
            json_safe(
                payload
            ),
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EXECUÇÃO CONCLUÍDA"
    )

    print(
        "=" * 70
    )

    print(
        "Modelo        : "
        "ANFIS V4 2-Additive Choquet"
    )

    print(
        f"Features      : "
        f"{len(SELECTED_FEATURES)}"
    )

    print(
        f"Top configs   : "
        f"{top_k}"
    )

    print(
        f"Threshold     : "
        f"{threshold:.6f}"
    )

    print(
        f"OOF F1        : "
        f"{train_metrics['f1']:.6f}"
    )

    print(
        f"OOF PR AUC    : "
        f"{train_metrics['pr_auc']:.6f}"
    )

    print(
        f"OOF ROC AUC   : "
        f"{train_metrics['roc_auc']:.6f}"
    )

    print(
        f"Predições     : "
        f"{args.output}"
    )

    print(
        f"Report TXT    : "
        f"{txt_path}"
    )

    print(
        f"Report JSON   : "
        f"{json_path}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
