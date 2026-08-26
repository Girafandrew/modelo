from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LABEL_CANDIDATES = ("label", "class", "target", "y", "attack", "is_attack")

DEFAULT_SPOOF_TRAIN_FILE = "ARP_Spoofing_train.pcap.csv"
DEFAULT_SPOOF_TEST_FILE = "ARP_Spoofing_test.pcap.csv"
DEFAULT_BENIGN_TRAIN_FILE = "Benign_train.pcap.csv"
DEFAULT_BENIGN_TEST_FILE = "Benign_test.pcap.csv"


def detect_label_column(columns: Iterable[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_csv_dataset(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Empty CSV file: {file_path}")
    return df


def prepare_features(df: pd.DataFrame, label_column: str | None = None) -> tuple[pd.DataFrame, np.ndarray | None]:
    if label_column is None:
        label_column = detect_label_column(df.columns)

    labels = None
    if label_column is not None and label_column in df.columns:
        labels = df[label_column].to_numpy()
        features = df.drop(columns=[label_column]).copy()
    else:
        features = df.copy()

    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True))
    return features, labels


def build_labeled_split(spoof_file: Path, benign_file: Path) -> tuple[pd.DataFrame, np.ndarray]:
    spoof_df = load_csv_dataset(spoof_file)
    benign_df = load_csv_dataset(benign_file)

    spoof_features, _ = prepare_features(spoof_df)
    benign_features, _ = prepare_features(benign_df)

    common_features = [column for column in spoof_features.columns if column in benign_features.columns]
    if not common_features:
        raise ValueError("No common features between benign and spoofing files.")

    spoof_features = spoof_features[common_features]
    benign_features = benign_features[common_features]

    X = pd.concat([benign_features, spoof_features], axis=0, ignore_index=True)
    y = np.concatenate([
        np.zeros(len(benign_features), dtype=int),
        np.ones(len(spoof_features), dtype=int),
    ])
    return X, y


def align_train_test_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    common_features = [column for column in X_train.columns if column in X_test.columns]
    if not common_features:
        raise ValueError("No common features between training and testing sets.")
    return X_train[common_features], X_test[common_features], common_features
