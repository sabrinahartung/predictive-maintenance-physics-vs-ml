import numpy as np
import pandas as pd
import pytest

from pdm.cost import (
    RULE_SCORE,
    best_threshold,
    cost_curve,
    cv_policy_cost,
    policy_cost,
    tool_change_score,
)
from pdm.data import FEATURES, load_data
from pdm.evaluate import make_cv
from pdm.rules import rule_flags


def test_policy_cost():
    result = policy_cost([1, 1, 0, 0], [1, 0, 1, 0], cost_ratio=10)
    assert result == {"alarms": 2, "tp": 1, "fp": 1, "fn": 1, "cost": 12}


def test_cost_curve_matches_brute_force():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    scores = np.round(rng.random(300), 2)  # rounding creates ties
    curve = cost_curve(y, scores, cost_ratio=7)
    for _, row in curve.iterrows():
        expected = policy_cost(y, scores >= row["threshold"], cost_ratio=7)
        assert row["alarms"] == expected["alarms"]
        assert row["fn"] == expected["fn"]
        assert row["cost"] == pytest.approx(expected["cost"])


def test_cost_curve_includes_never_alarm():
    curve = cost_curve([1, 0, 1], [0.9, 0.2, 0.4], cost_ratio=5)
    never = curve.iloc[-1]
    assert np.isinf(never["threshold"])
    assert never["alarms"] == 0
    assert never["cost"] == 10


def test_best_threshold_depends_on_cost_ratio():
    y = [1, 0, 0, 0, 1]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    # Cheap failures: never alarm. Expensive failures: alarm on everything down to 0.5.
    assert np.isinf(best_threshold(y, scores, cost_ratio=0.5))
    assert best_threshold(y, scores, cost_ratio=100) == 0.5


def test_cv_policy_cost_uses_one_threshold_per_fold():
    df = load_data()
    y = df["machine_failure"]
    splits = list(make_cv().split(df, y))
    result = cv_policy_cost(y, df["tool_wear_min"], cost_ratio=10, splits=splits)
    assert len(result["thresholds"]) == len(splits)
    assert result["tp"] + result["fn"] == y.sum()
    assert result["tp"] + result["fp"] == result["alarms"]


def test_tool_change_score_ranks_rule_rows_first():
    df = load_data()
    X = df[FEATURES]
    scores = tool_change_score(X)
    fired = rule_flags(X).any(axis=1).to_numpy()
    assert (scores[fired] == RULE_SCORE).all()
    pd.testing.assert_series_equal(
        pd.Series(scores[~fired]), pd.Series(X["tool_wear_min"].to_numpy(float)[~fired])
    )
