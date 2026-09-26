import numpy as np
import pytest
from sklearn.metrics import f1_score

from pdm.evaluate import best_f1_threshold, classification_metrics


def test_best_threshold_perfect_separation():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    threshold, f1 = best_f1_threshold(y, proba)
    assert threshold == pytest.approx(0.7)
    assert f1 == pytest.approx(1.0)


def test_best_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.3 + rng.normal(0.35, 0.2, 500), 0, 1)
    threshold, f1 = best_f1_threshold(y, proba)

    brute = max(f1_score(y, (proba >= t).astype(int)) for t in np.unique(proba))
    assert f1 == pytest.approx(brute)
    assert f1_score(y, (proba >= threshold).astype(int)) == pytest.approx(f1)


def test_classification_metrics_no_positive_predictions():
    metrics = classification_metrics([0, 1, 1], [0, 0, 0])
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["fn"] == 2
