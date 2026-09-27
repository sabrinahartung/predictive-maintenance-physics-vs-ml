"""Cross-validated evaluation for imbalanced binary classification."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

N_SPLITS = 5
RANDOM_STATE = 0


def make_cv(n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """The shared cross-validation splitter: stratified, shuffled, with a fixed seed."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def oof_proba(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, cv=None) -> np.ndarray:
    """Out-of-fold failure probabilities: every row is scored by a model that never saw it."""
    cv = cv if cv is not None else make_cv()
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def best_f1_threshold(y_true, proba) -> tuple[float, float]:
    """Return (threshold, f1) that maximises F1 on the given predictions.

    ``precision_recall_curve`` returns one more precision/recall value than thresholds:
    ``precision[i]`` and ``recall[i]`` belong to ``thresholds[i]``, and the final point
    (precision=1, recall=0) has no threshold, so it is excluded.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Precision, recall, F1 and confusion counts for hard 0/1 predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def score(y_true, proba, threshold: float | None = None) -> dict[str, float]:
    """Ranking metrics plus threshold metrics (at the best-F1 threshold unless given)."""
    if threshold is None:
        threshold, _ = best_f1_threshold(y_true, proba)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "threshold": threshold,
        **classification_metrics(y_true, y_pred),
    }


def compare_models(
    models: dict[str, BaseEstimator], X: pd.DataFrame, y: pd.Series, cv=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate each model with the same CV splits.

    Returns a results table (one row per model, sorted by PR-AUC) and a frame with the
    out-of-fold probabilities (one column per model, indexed like ``X``).
    """
    cv = cv if cv is not None else make_cv()
    probas = pd.DataFrame(
        {name: oof_proba(model, X, y, cv) for name, model in models.items()}, index=X.index
    )
    results = pd.DataFrame({name: score(y, probas[name]) for name in probas}).T
    results.index.name = "model"
    return results.sort_values("pr_auc", ascending=False), probas
