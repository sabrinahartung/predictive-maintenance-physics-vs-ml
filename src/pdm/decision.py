"""Turn one process reading into a maintenance recommendation.

The recommended action follows the cheapest policy from notebook 05: physics rules plus a
preventive tool change. The calibrated hybrid probability is reported alongside as a risk score.
"""

from dataclasses import dataclass, field

import pandas as pd

from pdm.data import FEATURES, MACHINE_TYPES, NUMERIC_FEATURES
from pdm.features import add_physics_features
from pdm.rules import (
    HEAT_RPM_MAX,
    HEAT_TEMP_DIFF_MAX_K,
    POWER_MAX_W,
    POWER_MIN_W,
    STRAIN_LIMIT_MIN_NM,
    rule_flags,
)

ACTION_STOP = "Stop: failure condition"
ACTION_CHANGE_TOOL = "Change tool"
ACTION_INSPECT = "Inspect"
ACTION_OK = "Continue"


@dataclass
class Check:
    """One physics rule evaluated for a single process."""

    name: str
    fired: bool
    value: str
    limit: str


@dataclass
class Assessment:
    probability: float
    risk_alarm: bool
    action: str
    checks: list[Check]
    tool_change_at: float | None
    tool_change_due: bool
    physics: dict[str, float] = field(default_factory=dict)


def physics_checks(row: pd.Series) -> list[Check]:
    """Evaluate the three physics rules for one row of raw features."""
    f = add_physics_features(row.to_frame().T.astype({"type": str}))
    fired = rule_flags(f).iloc[0]
    f = f.iloc[0]
    strain_limit = STRAIN_LIMIT_MIN_NM[str(f["type"])]
    return [
        Check(
            "Power",
            bool(fired["pwf"]),
            f"{f['power_w']:,.0f} W",
            f"{POWER_MIN_W:,}–{POWER_MAX_W:,} W",
        ),
        Check(
            "Overstrain",
            bool(fired["osf"]),
            f"{f['strain_min_nm']:,.0f} min·Nm",
            f"≤ {strain_limit:,} min·Nm (type {f['type']})",
        ),
        Check(
            "Heat dissipation",
            bool(fired["hdf"]),
            f"ΔT {f['temp_diff_k']:.1f} K at {f['rpm']:,.0f} rpm",
            f"fails if ΔT < {HEAT_TEMP_DIFF_MAX_K} K and rpm < {HEAT_RPM_MAX:,}",
        ),
    ]


def row_to_frame(row: pd.Series) -> pd.DataFrame:
    """One reading of raw features as a single-row frame with the dtypes of ``load_data()``."""
    X = row[FEATURES].to_frame().T
    X["type"] = pd.Categorical(X["type"].astype(str), categories=MACHINE_TYPES, ordered=True)
    for col in NUMERIC_FEATURES:
        X[col] = X[col].astype(float)
    return X


def assess(model, row: pd.Series, cost_ratio: float, tool_change_at: float | None) -> Assessment:
    """Assess one process.

    ``model`` is a fitted hybrid (``pdm.models.hybrid_model``), ``row`` holds the raw features,
    and ``tool_change_at`` is the cost-optimal tool age for ``cost_ratio`` (None = never).
    """
    X = row_to_frame(row)
    probability = float(model.predict_proba(X)[0, 1])
    checks = physics_checks(row)
    tool_change_due = tool_change_at is not None and row["tool_wear_min"] >= tool_change_at
    risk_alarm = probability > 1 / cost_ratio

    if any(c.fired for c in checks):
        action = ACTION_STOP
    elif tool_change_due:
        action = ACTION_CHANGE_TOOL
    elif risk_alarm:
        action = ACTION_INSPECT
    else:
        action = ACTION_OK

    physics = add_physics_features(X).iloc[0][["temp_diff_k", "power_w", "strain_min_nm"]]
    return Assessment(
        probability=probability,
        risk_alarm=risk_alarm,
        action=action,
        checks=checks,
        tool_change_at=tool_change_at,
        tool_change_due=bool(tool_change_due),
        physics={k: float(v) for k, v in physics.items()},
    )
