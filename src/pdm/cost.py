"""Expected maintenance cost of alarm policies.

Cost model (in units of one inspection):

- every alarm triggers an inspection that costs ``alarm_cost`` (default 1)
- every missed failure causes an unplanned breakdown that costs ``cost_ratio`` inspections

A policy raises an alarm when its score is at or above a threshold. Scores can be probabilities
or any other quantity where higher means riskier, such as tool wear.
"""

import numpy as np
import pandas as pd

from pdm.rules import rule_flags

# Score given to rows where a physics rule fires, so they rank above any tool wear value
RULE_SCORE = 1e6


def policy_cost(y_true, alarm, cost_ratio: float, alarm_cost: float = 1.0) -> dict[str, float]:
    """Cost and confusion counts of a fixed set of alarms."""
    y_true, alarm = np.asarray(y_true).astype(bool), np.asarray(alarm).astype(bool)
    fn = int((y_true & ~alarm).sum())
    return {
        "alarms": int(alarm.sum()),
        "tp": int((y_true & alarm).sum()),
        "fp": int((~y_true & alarm).sum()),
        "fn": fn,
        "cost": alarm_cost * alarm.sum() + cost_ratio * fn,
    }


def cost_curve(y_true, scores, cost_ratio: float, alarm_cost: float = 1.0) -> pd.DataFrame:
    """Cost for every distinct threshold (alarm if ``score >= threshold``).

    The last row uses an infinite threshold, i.e. never raising an alarm.
    """
    y_true, scores = np.asarray(y_true).astype(bool), np.asarray(scores, dtype=float)
    thresholds = np.append(np.unique(scores), np.inf)
    sorted_all = np.sort(scores)
    sorted_pos = np.sort(scores[y_true])
    alarms = len(scores) - np.searchsorted(sorted_all, thresholds, side="left")
    fn = np.searchsorted(sorted_pos, thresholds, side="left")
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "alarms": alarms,
            "fn": fn,
            "cost": alarm_cost * alarms + cost_ratio * fn,
        }
    )


def best_threshold(y_true, scores, cost_ratio: float, alarm_cost: float = 1.0) -> float:
    """Threshold with the lowest total cost (the highest one if several tie)."""
    curve = cost_curve(y_true, scores, cost_ratio, alarm_cost)
    best = curve.loc[curve["cost"] == curve["cost"].min(), "threshold"]
    return float(best.iloc[-1])


def cv_policy_cost(
    y_true, scores, cost_ratio: float, splits, alarm_cost: float = 1.0
) -> dict[str, float]:
    """Honest cost estimate: for each fold, pick the threshold on the other folds, apply it here.

    ``splits`` is a list of (train_idx, test_idx) pairs, e.g. ``list(make_cv().split(X, y))``.
    """
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    alarm = np.zeros(len(scores), dtype=bool)
    thresholds = []
    for train, test in splits:
        t = best_threshold(y_true[train], scores[train], cost_ratio, alarm_cost)
        thresholds.append(t)
        alarm[test] = scores[test] >= t
    return {
        **policy_cost(y_true, alarm, cost_ratio, alarm_cost),
        "thresholds": thresholds,
    }


def tool_change_score(X: pd.DataFrame) -> np.ndarray:
    """Score for the policy "physics rules + preventive tool change".

    Rows where a rule fires get ``RULE_SCORE``; all other rows are scored by tool wear, so a
    threshold of e.g. 210 means: alarm if a rule fires or the tool has been used for 210+ minutes.
    """
    fired = rule_flags(X).any(axis=1).to_numpy()
    return np.where(fired, RULE_SCORE, X["tool_wear_min"].to_numpy(dtype=float))
