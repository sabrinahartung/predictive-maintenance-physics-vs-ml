"""Train the final model and write its evaluation metrics.

Usage::

    uv run pdm-train                  # train, cross-validate, write model + metrics
    uv run pdm-train --cost-ratio 20  # different cost of a missed failure
    uv run pdm-train --skip-cv        # train only (fast)

Outputs ``models/model.joblib`` (the fitted calibrated hybrid) and ``reports/metrics.json``.
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from pdm.config import MODELS_DIR, PROJECT_ROOT
from pdm.cost import RULE_SCORE, best_threshold, cv_policy_cost, tool_change_score
from pdm.data import FEATURES, TARGET, load_data
from pdm.evaluate import classification_metrics, make_cv, oof_proba
from pdm.models import hybrid_model
from pdm.rules import PhysicsRuleClassifier

DEFAULT_COST_RATIO = 10.0
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"


def tool_change_minutes(X: pd.DataFrame, y: pd.Series, cost_ratio: float) -> float | None:
    """Cost-optimal tool change age for the "rules + tool change" policy (None = never)."""
    threshold = best_threshold(y, tool_change_score(X), cost_ratio)
    return None if threshold >= RULE_SCORE else float(threshold)


def cross_validate(X: pd.DataFrame, y: pd.Series, cost_ratio: float) -> dict:
    """Out-of-fold metrics of the final model and CV costs of the main policies."""
    splits = list(make_cv().split(X, y))
    proba = oof_proba(hybrid_model(), X, y, make_cv())
    alarm = proba > 1 / cost_ratio

    policy_scores = {
        "rules_only": oof_proba(PhysicsRuleClassifier(), X, y, make_cv()),
        "hybrid_calibrated": proba,
        "rules_plus_tool_change": tool_change_score(X),
    }
    policies = {}
    for name, scores in policy_scores.items():
        result = cv_policy_cost(y, scores, cost_ratio, splits)
        policies[name] = {k: v for k, v in result.items() if k != "thresholds"}
    never = float(y.sum() * cost_ratio)
    for result in policies.values():
        result["savings_vs_no_maintenance"] = 1 - result["cost"] / never

    return {
        "hybrid": {
            "pr_auc": average_precision_score(y, proba),
            "roc_auc": roc_auc_score(y, proba),
            "brier": brier_score_loss(y, proba),
            "log_loss": log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6)),
            "at_threshold_1_over_r": classification_metrics(y, alarm),
        },
        "policies": policies,
        "no_maintenance_cost": never,
    }


def _to_builtin(obj):
    """Make numpy scalars JSON-serialisable and round floats for readable diffs."""
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 6)
    return obj


def run(
    cost_ratio: float = DEFAULT_COST_RATIO,
    skip_cv: bool = False,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict:
    df = load_data()
    X, y = df[FEATURES], df[TARGET]

    model = hybrid_model().fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metrics = {
        "data": {"rows": len(df), "failures": int(y.sum())},
        "cost_ratio": cost_ratio,
        "alarm_threshold": 1 / cost_ratio,
        "tool_change_at_min": tool_change_minutes(X, y, cost_ratio),
        "versions": {pkg: version(pkg) for pkg in ["pdm", "scikit-learn", "pandas", "numpy"]},
    }
    if not skip_cv:
        metrics["cross_validation"] = cross_validate(X, y, cost_ratio)

    metrics = _to_builtin(metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cost-ratio",
        type=float,
        default=DEFAULT_COST_RATIO,
        help="cost of a missed failure in units of one inspection (default: %(default)s)",
    )
    parser.add_argument("--skip-cv", action="store_true", help="train only, no cross-validation")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    args = parser.parse_args(argv)

    metrics = run(args.cost_ratio, args.skip_cv, args.model_path, args.metrics_path)

    print(f"Model saved to {args.model_path}")
    print(f"Metrics saved to {args.metrics_path}")
    print(
        f"Cost ratio R = {metrics['cost_ratio']:g}: alarm if p > {metrics['alarm_threshold']:.3f}"
    )
    tool_change = metrics["tool_change_at_min"]
    print(
        "Tool change: "
        + (f"at {tool_change:.0f} min" if tool_change is not None else "not worth it")
    )
    if "cross_validation" in metrics:
        cv = metrics["cross_validation"]
        print(f"Hybrid PR-AUC {cv['hybrid']['pr_auc']:.3f}, Brier {cv['hybrid']['brier']:.4f}")
        for name, p in cv["policies"].items():
            savings = p["savings_vs_no_maintenance"]
            print(f"  {name:<24} cost {p['cost']:>7,.0f}  savings {savings:.1%}")


if __name__ == "__main__":
    main()
