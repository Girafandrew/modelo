from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import train_test_split

try:
    from .anfis import ChoquetANFISScorer
except ImportError:
    from anfis import ChoquetANFISScorer


def _score_threshold_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_mode: str,
    threshold_beta: float,
    min_precision: float,
) -> float:
    """
    Calcula a métrica usada para escolher threshold e q.

    Modos:
    - f1: usa F-beta.
    - precision_floor: favorece F1, mas penaliza quando precision < min_precision.
    - balanced_accuracy: usa balanced accuracy.
    - fallback: usa F1.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    if threshold_mode == "f1":
        return float(fbeta_score(y_true, y_pred, beta=threshold_beta, zero_division=0))

    if threshold_mode == "precision_floor":
        if precision >= min_precision:
            return f1

        # Penalização suave:
        # evita transformar precision 0.49 e precision 0.01 no mesmo valor -1.
        if min_precision > 0:
            penalty = precision / min_precision
        else:
            penalty = 0.0

        return f1 * penalty

    if threshold_mode == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))

    return f1


def _calibrate_threshold_hint(
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold_hint: float,
    threshold_mode: str,
    threshold_beta: float,
    min_precision: float,
) -> float:
    """
    Calibra um threshold usando threshold_hint como pista.

    Observação:
    Esta função foi mantida, mas no fluxo principal eu recomendo usar
    decision_threshold como threshold fixo.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true).astype(int)

    score_min = float(np.min(scores))
    score_max = float(np.max(scores))

    if score_min == score_max:
        return score_min

    lower = max(score_min, float(threshold_hint) * 0.25)

    if threshold_hint > 0:
        upper = min(score_max, float(threshold_hint) * 1.75)
    else:
        upper = score_max

    if lower >= upper:
        lower, upper = score_min, score_max

    candidate_thresholds = np.unique(
        np.quantile(scores, np.linspace(0.01, 0.99, 240))
    )

    candidate_thresholds = candidate_thresholds[
        (candidate_thresholds >= lower) & (candidate_thresholds <= upper)
    ]

    if candidate_thresholds.size == 0:
        candidate_thresholds = np.linspace(lower, upper, 121)

    best_threshold = float(threshold_hint)
    best_metric = -1.0

    for threshold in candidate_thresholds:
        y_pred = (scores >= threshold).astype(int)

        metric = _score_threshold_metric(
            y_true,
            y_pred,
            threshold_mode=threshold_mode,
            threshold_beta=threshold_beta,
            min_precision=min_precision,
        )

        if metric > best_metric:
            best_metric = metric
            best_threshold = float(threshold)

    return best_threshold


def threshold_report(
    scores: np.ndarray,
    y_true: np.ndarray,
    thresholds: list[float] | np.ndarray | None = None,
) -> list[dict[str, float]]:
    """
    Gera uma tabela de desempenho para vários thresholds.

    Retorna uma lista de dicionários com:
    - threshold
    - accuracy
    - balanced_accuracy
    - precision
    - recall
    - f1
    - tn, fp, fn, tp
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true).astype(int)

    if thresholds is None:
        thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    rows: list[dict[str, float]] = []

    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        accuracy = float((tp + tn) / (tp + tn + fp + fn))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))

        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy,
                "balanced_accuracy": bal_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    return rows


def print_threshold_report(
    rows: list[dict[str, float]],
    sort_by: str | None = None,
    descending: bool = True,
    top_k: int | None = None,
) -> None:
    """
    Imprime a tabela de thresholds.

    Exemplo:
        rows = threshold_report(test_scores, y_test)
        print_threshold_report(rows, sort_by="f1", top_k=20)
    """
    if sort_by is not None:
        rows = sorted(rows, key=lambda row: row[sort_by], reverse=descending)

    if top_k is not None:
        rows = rows[:top_k]

    print(
        "threshold | acc    | bal_acc | precision | recall | f1     | "
        "TN    | FP    | FN    | TP"
    )
    print("-" * 92)

    for row in rows:
        print(
            f"{row['threshold']:.6f} | "
            f"{row['accuracy']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | "
            f"{row['precision']:.4f}    | "
            f"{row['recall']:.4f} | "
            f"{row['f1']:.4f} | "
            f"{row['tn']:5d} | "
            f"{row['fp']:5d} | "
            f"{row['fn']:5d} | "
            f"{row['tp']:5d}"
        )


def describe_score_distribution(
    scores: np.ndarray,
    y_true: np.ndarray,
    name: str = "scores",
) -> None:
    """
    Imprime estatísticas dos scores por classe.

    Isso ajuda a ver se as classes estão separadas ou se os scores
    estão muito sobrepostos.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true).astype(int)

    print(f"\nScore distribution: {name}")

    for cls in [0, 1]:
        cls_scores = scores[y_true == cls]

        if cls_scores.size == 0:
            print(f"Class {cls}: no samples")
            continue

        print(f"\nClass {cls}:")
        print(f"  count : {cls_scores.size}")
        print(f"  min   : {np.min(cls_scores):.6f}")
        print(f"  q25   : {np.quantile(cls_scores, 0.25):.6f}")
        print(f"  mean  : {np.mean(cls_scores):.6f}")
        print(f"  median: {np.median(cls_scores):.6f}")
        print(f"  q75   : {np.quantile(cls_scores, 0.75):.6f}")
        print(f"  max   : {np.max(cls_scores):.6f}")


def _fit_threshold(
    model: ChoquetANFISScorer,
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold_mode: str,
    train_quantile: float,
    threshold_beta: float,
    min_precision: float,
    decision_threshold: float | None,
) -> None:
    """
    Define o threshold do modelo.

    Se decision_threshold for informado, ele é usado como threshold fixo.
    Caso contrário, usa o modo automático escolhido por threshold_mode.
    """
    if decision_threshold is not None:
        model.threshold_ = float(decision_threshold)
        return

    if threshold_mode == "f1":
        model.fit_threshold_with_labels(
            scores,
            y_true,
            beta=threshold_beta,
        )

    elif threshold_mode == "precision_floor":
        model.fit_threshold_with_precision_floor(
            scores,
            y_true,
            min_precision=min_precision,
        )

    elif threshold_mode == "balanced_accuracy":
        model.fit_threshold_balanced_accuracy(
            scores,
            y_true,
        )

    else:
        model.set_threshold_from_quantile(
            scores,
            quantile=train_quantile,
        )


def train_model(
    X_train,
    y_train,
    q: float = 1.0,
    threshold_mode: str = "f1",
    train_quantile: float = 0.95,
    threshold_beta: float = 0.5,
    min_precision: float = 0.5,
    reweight_epochs: int = 0,
    reweight_lr: float = 0.1,
    reweight_patience: int = 0,
    reweight_min_delta: float = 0.0,
    decision_threshold: float | None = None,
) -> tuple[ChoquetANFISScorer, np.ndarray, np.ndarray]:
    """
    Treina o modelo ANFIS/Choquet e ajusta o threshold.

    decision_threshold:
    - None: threshold automático.
    - float: usa exatamente esse valor como threshold.
    """
    model = ChoquetANFISScorer(q=q)
    model.fit(X_train)

    if reweight_epochs > 0:
        model.refine_weights_with_labels(
            X_train,
            y_train,
            epochs=reweight_epochs,
            learning_rate=reweight_lr,
            patience=reweight_patience,
            min_delta=reweight_min_delta,
            threshold_mode=threshold_mode,
            train_quantile=train_quantile,
            threshold_beta=threshold_beta,
            min_precision=min_precision,
        )

    raw_train_scores = model.score(X_train)
    model.set_score_orientation(raw_train_scores, y_train)
    train_scores = model.score(X_train)

    _fit_threshold(
        model=model,
        scores=train_scores,
        y_true=y_train,
        threshold_mode=threshold_mode,
        train_quantile=train_quantile,
        threshold_beta=threshold_beta,
        min_precision=min_precision,
        decision_threshold=decision_threshold,
    )

    train_pred = model.predict(X_train)

    return model, train_scores, train_pred


def train_model_with_q_search(
    X_train,
    y_train,
    q_grid: list[float],
    threshold_mode: str = "f1",
    train_quantile: float = 0.95,
    threshold_beta: float = 0.5,
    min_precision: float = 0.5,
    val_size: float = 0.2,
    random_state: int = 42,
    reweight_epochs: int = 0,
    reweight_lr: float = 0.1,
    reweight_patience: int = 0,
    reweight_min_delta: float = 0.0,
    decision_threshold: float | None = None,
) -> tuple[ChoquetANFISScorer, np.ndarray, np.ndarray, dict[str, float]]:
    """
    Busca o melhor q usando validação estratificada.

    Agora imprime métricas de validação para cada q, o que ajuda a entender
    por que determinado q foi escolhido.
    """
    if not q_grid:
        raise ValueError("q_grid cannot be empty.")

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )

    best_q = float(q_grid[0])
    best_metric = -1.0
    best_val_precision = 0.0
    best_val_recall = 0.0
    best_val_f1 = 0.0
    best_val_bal_acc = 0.0
    best_threshold = 0.0

    q_search_logs: dict[str, dict] = {}

    for idx, q in enumerate(q_grid, 1):
        candidate = ChoquetANFISScorer(q=float(q))
        candidate.fit(X_fit)

        reweight_info = {
            "q": float(q),
            "epochs_ran": 0,
            "stopped_early": False,
        }

        if reweight_epochs > 0:
            meta = candidate.refine_weights_with_labels(
                X_fit,
                y_fit,
                epochs=reweight_epochs,
                learning_rate=reweight_lr,
                patience=reweight_patience,
                min_delta=reweight_min_delta,
                threshold_mode=threshold_mode,
                train_quantile=train_quantile,
                threshold_beta=threshold_beta,
                min_precision=min_precision,
            )

            reweight_info["epochs_ran"] = int(meta.get("epochs", 0))
            reweight_info["stopped_early"] = bool(meta.get("stopped_early", 0))

            print(
                f"  q={q:.2f} "
                f"(candidate {idx}/{len(q_grid)}): "
                f"reweight {reweight_info['epochs_ran']} epochs"
                f"{' (early stopped)' if reweight_info['stopped_early'] else ''}"
            )

        raw_fit_scores = candidate.score(X_fit)
        candidate.set_score_orientation(raw_fit_scores, y_fit)
        fit_scores = candidate.score(X_fit)

        _fit_threshold(
            model=candidate,
            scores=fit_scores,
            y_true=y_fit,
            threshold_mode=threshold_mode,
            train_quantile=train_quantile,
            threshold_beta=threshold_beta,
            min_precision=min_precision,
            decision_threshold=decision_threshold,
        )

        val_pred = candidate.predict(X_val)

        val_metric = _score_threshold_metric(
            y_val,
            val_pred,
            threshold_mode=threshold_mode,
            threshold_beta=threshold_beta,
            min_precision=min_precision,
        )

        val_precision = float(precision_score(y_val, val_pred, zero_division=0))
        val_recall = float(recall_score(y_val, val_pred, zero_division=0))
        val_f1 = float(f1_score(y_val, val_pred, zero_division=0))
        val_bal_acc = float(balanced_accuracy_score(y_val, val_pred))
        threshold_value = float(candidate.threshold_)

        print(
            f"  q={q:.2f}: "
            f"val_metric={val_metric:.4f}, "
            f"precision={val_precision:.4f}, "
            f"recall={val_recall:.4f}, "
            f"f1={val_f1:.4f}, "
            f"bal_acc={val_bal_acc:.4f}, "
            f"threshold={threshold_value:.6f}"
        )

        q_search_logs[f"q_{q:.2f}"] = {
            **reweight_info,
            "validation_metric": val_metric,
            "validation_precision": val_precision,
            "validation_recall": val_recall,
            "validation_f1": val_f1,
            "validation_balanced_accuracy": val_bal_acc,
            "threshold": threshold_value,
        }

        # Critério principal: maior val_metric.
        # Desempate: maior F1, depois maior balanced accuracy.
        is_better = False

        if val_metric > best_metric:
            is_better = True
        elif np.isclose(val_metric, best_metric):
            if val_f1 > best_val_f1:
                is_better = True
            elif np.isclose(val_f1, best_val_f1) and val_bal_acc > best_val_bal_acc:
                is_better = True

        if is_better:
            best_metric = val_metric
            best_q = float(q)
            best_val_precision = val_precision
            best_val_recall = val_recall
            best_val_f1 = val_f1
            best_val_bal_acc = val_bal_acc
            best_threshold = threshold_value

    model, train_scores, train_pred = train_model(
        X_train,
        y_train,
        q=best_q,
        threshold_mode=threshold_mode,
        train_quantile=train_quantile,
        threshold_beta=threshold_beta,
        min_precision=min_precision,
        reweight_epochs=reweight_epochs,
        reweight_lr=reweight_lr,
        reweight_patience=reweight_patience,
        reweight_min_delta=reweight_min_delta,
        decision_threshold=decision_threshold,
    )

    meta = {
        "selected_q": best_q,
        "validation_metric": best_metric,
        "validation_precision": best_val_precision,
        "validation_recall": best_val_recall,
        "validation_f1": best_val_f1,
        "validation_balanced_accuracy": best_val_bal_acc,
        "validation_threshold": best_threshold,
        "threshold_beta": float(threshold_beta),
        "min_precision": float(min_precision),
        "reweight_epochs": float(reweight_epochs),
        "reweight_lr": float(reweight_lr),
        "reweight_patience": float(reweight_patience),
        "reweight_min_delta": float(reweight_min_delta),
        "decision_threshold": float(decision_threshold)
        if decision_threshold is not None
        else None,
        "q_search_logs": q_search_logs,
    }

    return model, train_scores, train_pred, meta