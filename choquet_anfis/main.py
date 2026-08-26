from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from .data import (  # noqa: E402
        DEFAULT_BENIGN_TEST_FILE,
        DEFAULT_BENIGN_TRAIN_FILE,
        DEFAULT_SPOOF_TEST_FILE,
        DEFAULT_SPOOF_TRAIN_FILE,
        align_train_test_features,
        build_labeled_split,
    )
    from .evaluate import print_evaluation  # noqa: E402
    from .train import (  # noqa: E402
        describe_score_distribution,
        print_threshold_report,
        threshold_report,
        train_model,
        train_model_with_q_search,
    )
except ImportError:
    from data import (  # noqa: E402
        DEFAULT_BENIGN_TEST_FILE,
        DEFAULT_BENIGN_TRAIN_FILE,
        DEFAULT_SPOOF_TEST_FILE,
        DEFAULT_SPOOF_TRAIN_FILE,
        align_train_test_features,
        build_labeled_split,
    )
    from evaluate import print_evaluation  # noqa: E402
    from train import (  # noqa: E402
        describe_score_distribution,
        print_threshold_report,
        threshold_report,
        train_model,
        train_model_with_q_search,
    )


def parse_q_grid(q_grid_raw: str) -> list[float]:
    values = []

    for chunk in q_grid_raw.split(","):
        chunk = chunk.strip()

        if not chunk:
            continue

        values.append(float(chunk))

    if not values:
        raise ValueError(
            "q-grid is empty. Provide comma-separated values like 0.5,1.0,1.5"
        )

    return values


def parse_thresholds(thresholds_raw: str) -> np.ndarray:
    values = []

    for chunk in thresholds_raw.split(","):
        chunk = chunk.strip()

        if not chunk:
            continue

        values.append(float(chunk))

    if not values:
        raise ValueError(
            "thresholds is empty. Provide comma-separated values like 0.01,0.02,0.03"
        )

    return np.asarray(values, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Choquet-inspired ANFIS for benign vs ARP spoofing."
    )

    parser.add_argument(
        "--spoof-train",
        type=Path,
        default=Path(DEFAULT_SPOOF_TRAIN_FILE),
        help="Spoofing train CSV",
    )

    parser.add_argument(
        "--spoof-test",
        type=Path,
        default=Path(DEFAULT_SPOOF_TEST_FILE),
        help="Spoofing test CSV",
    )

    parser.add_argument(
        "--benign-train",
        type=Path,
        default=Path(DEFAULT_BENIGN_TRAIN_FILE),
        help="Benign train CSV",
    )

    parser.add_argument(
        "--benign-test",
        type=Path,
        default=Path(DEFAULT_BENIGN_TEST_FILE),
        help="Benign test CSV",
    )

    parser.add_argument(
        "--q",
        type=float,
        default=1.0,
        help="Power-measure parameter",
    )

    parser.add_argument(
        "--auto-q",
        action="store_true",
        help="Enable automatic q search using validation split",
    )

    parser.add_argument(
        "--q-grid",
        type=str,
        default="0.3,0.5,1.0,1.5,2.0,3.0",
        help="Comma-separated q values for search",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation split ratio for automatic q search",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for validation split",
    )

    parser.add_argument(
        "--threshold-mode",
        choices=("f1", "precision_floor", "balanced_accuracy", "quantile"),
        default="f1",
        help="Threshold strategy",
    )

    parser.add_argument(
        "--threshold-beta",
        type=float,
        default=0.5,
        help=(
            "Beta for F-beta threshold tuning when threshold-mode=f1; "
            "values below 1.0 favor precision on noisy data"
        ),
    )

    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.5,
        help="Minimum precision target when threshold-mode=precision_floor",
    )

    parser.add_argument(
        "--train-quantile",
        type=float,
        default=0.95,
        help="Quantile used when threshold-mode=quantile",
    )

    parser.add_argument(
        "--reweight-epochs",
        type=int,
        default=0,
        help="Supervised iterative reweighting epochs for feature-rule weights",
    )

    parser.add_argument(
        "--reweight-lr",
        type=float,
        default=0.1,
        help="Learning rate for supervised reweighting",
    )

    parser.add_argument(
        "--reweight-patience",
        type=int,
        default=0,
        help="Early stopping patience for reweighting. 0 disables early stopping.",
    )

    parser.add_argument(
        "--reweight-min-delta",
        type=float,
        default=0.0,
        help="Minimum metric improvement to reset patience",
    )

    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help=(
            "Fixed decision threshold. "
            "If omitted, the threshold is selected automatically using threshold-mode."
        ),
    )

    parser.add_argument(
        "--threshold-report",
        action="store_true",
        help="Print threshold report on the test set.",
    )

    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.005,0.0075,0.010,0.012,0.015,0.0175,0.020,0.0225,0.025,0.030,0.035,0.040,0.050,0.060,0.070,0.080,0.090,0.100,0.110,0.120",
        help="Comma-separated thresholds used when --threshold-report is enabled.",
    )

    parser.add_argument(
        "--describe-scores",
        action="store_true",
        help="Print score distribution by class for train and test sets.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Output CSV with scores and predictions",
    )

    args = parser.parse_args()

    X_train, y_train = build_labeled_split(
        args.spoof_train,
        args.benign_train,
    )

    X_test, y_test = build_labeled_split(
        args.spoof_test,
        args.benign_test,
    )

    X_train, X_test, common_features = align_train_test_features(
        X_train,
        X_test,
    )

    training_meta = None

    if args.auto_q:
        model, train_scores, train_pred, training_meta = train_model_with_q_search(
            X_train,
            y_train,
            q_grid=parse_q_grid(args.q_grid),
            threshold_mode=args.threshold_mode,
            train_quantile=args.train_quantile,
            threshold_beta=args.threshold_beta,
            min_precision=args.min_precision,
            val_size=args.val_size,
            random_state=args.random_state,
            reweight_epochs=args.reweight_epochs,
            reweight_lr=args.reweight_lr,
            reweight_patience=args.reweight_patience,
            reweight_min_delta=args.reweight_min_delta,
            decision_threshold=args.decision_threshold,
        )

    else:
        model, train_scores, train_pred = train_model(
            X_train,
            y_train,
            q=args.q,
            threshold_mode=args.threshold_mode,
            train_quantile=args.train_quantile,
            threshold_beta=args.threshold_beta,
            min_precision=args.min_precision,
            reweight_epochs=args.reweight_epochs,
            reweight_lr=args.reweight_lr,
            reweight_patience=args.reweight_patience,
            reweight_min_delta=args.reweight_min_delta,
            decision_threshold=args.decision_threshold,
        )

    test_scores = model.score(X_test)
    test_pred = model.predict(X_test)

    result_df = X_test.copy()
    result_df["true_label"] = y_test
    result_df["score"] = test_scores
    result_df["prediction"] = test_pred
    result_df.to_csv(args.output, index=False)

    print(f"Features used: {len(common_features)}")

    if training_meta is not None:
        print(f"Selected q (auto): {training_meta['selected_q']:.4f}")
        print(f"Validation metric: {training_meta['validation_metric']:.4f}")
        print(f"Validation precision: {training_meta['validation_precision']:.4f}")
        print(f"Validation recall   : {training_meta['validation_recall']:.4f}")
        print(f"Validation F1       : {training_meta['validation_f1']:.4f}")
        print(
            f"Validation bal. acc.: "
            f"{training_meta['validation_balanced_accuracy']:.4f}"
        )
        print(f"Validation threshold: {training_meta['validation_threshold']:.6f}")
        print(f"Threshold beta      : {training_meta['threshold_beta']:.2f}")
        print(f"Min precision       : {training_meta['min_precision']:.2f}")
        print(f"Reweight epochs     : {training_meta['reweight_epochs']:.0f}")
        print(f"Reweight LR         : {training_meta['reweight_lr']:.4f}")
        print(f"Reweight patience   : {training_meta['reweight_patience']:.0f}")
        print(f"Reweight min dlt    : {training_meta['reweight_min_delta']:.6f}")

        if training_meta.get("decision_threshold") is not None:
            print(
                f"Fixed decision threshold: "
                f"{training_meta['decision_threshold']:.6f}"
            )

        q_logs = training_meta.get("q_search_logs", {})

        if q_logs:
            print("\nq-search summary:")
            for key, info in q_logs.items():
                marker = " ✓" if info["q"] == training_meta["selected_q"] else ""
                early_stop_str = " (early stopped)" if info["stopped_early"] else ""

                print(
                    f"  q={info['q']:.2f}: "
                    f"metric={info['validation_metric']:.4f}, "
                    f"precision={info['validation_precision']:.4f}, "
                    f"recall={info['validation_recall']:.4f}, "
                    f"f1={info['validation_f1']:.4f}, "
                    f"bal_acc={info['validation_balanced_accuracy']:.4f}, "
                    f"threshold={info['threshold']:.6f}, "
                    f"reweight={info['epochs_ran']}/{int(training_meta['reweight_epochs'])}"
                    f"{early_stop_str}"
                    f"{marker}"
                )

    else:
        print(f"Selected q (fixed): {args.q:.4f}")

        if args.threshold_mode == "f1":
            print(f"Threshold beta    : {args.threshold_beta:.2f}")

        if args.threshold_mode == "precision_floor":
            print(f"Min precision     : {args.min_precision:.2f}")

        print(f"Reweight epochs   : {args.reweight_epochs}")
        print(f"Reweight LR       : {args.reweight_lr:.4f}")
        print(f"Reweight patience : {args.reweight_patience}")
        print(f"Reweight min dlt  : {args.reweight_min_delta:.6f}")

        if args.decision_threshold is not None:
            print(f"Fixed decision threshold: {args.decision_threshold:.6f}")

    print(f"Decision threshold: {model.threshold_:.6f}")
    print(f"Mean train score  : {train_scores.mean():.6f}")
    print(f"Mean test score   : {test_scores.mean():.6f}")
    print(f"Saved predictions to: {args.output}")
    print()

    print_evaluation(y_train, train_pred, title="Train metrics")
    print_evaluation(y_test, test_pred, title="Test metrics")

    if args.describe_scores:
        describe_score_distribution(
            scores=train_scores,
            y_true=y_train,
            name="train scores",
        )

        describe_score_distribution(
            scores=test_scores,
            y_true=y_test,
            name="test scores",
        )

    if args.threshold_report:
        manual_thresholds = parse_thresholds(args.thresholds)

        rows = threshold_report(
            scores=test_scores,
            y_true=y_test,
            thresholds=manual_thresholds,
        )

        print("\nThreshold report:")
        print_threshold_report(rows)

        print("\nBest thresholds by F1:")
        print_threshold_report(
            rows,
            sort_by="f1",
            descending=True,
            top_k=10,
        )

        print("\nBest thresholds by balanced accuracy:")
        print_threshold_report(
            rows,
            sort_by="balanced_accuracy",
            descending=True,
            top_k=10,
        )

        print("\nBest thresholds by precision:")
        print_threshold_report(
            rows,
            sort_by="precision",
            descending=True,
            top_k=10,
        )


if __name__ == "__main__":
    main()