"""Physics-based features derived from the raw sensor readings."""

import numpy as np
import pandas as pd

PHYSICS_FEATURES = ["temp_diff_k", "power_w", "strain_min_nm"]


def power_w(torque_nm, rpm):
    """Mechanical power P = torque · angular velocity, in watts."""
    return torque_nm * rpm * 2 * np.pi / 60


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the physics features added.

    - ``temp_diff_k``: process minus air temperature; a small difference means poor heat dissipation
    - ``power_w``: mechanical power at the spindle
    - ``strain_min_nm``: tool wear × torque, a proxy for accumulated load on the tool
    """
    out = df.copy()
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["power_w"] = power_w(out["torque_nm"], out["rpm"])
    out["strain_min_nm"] = out["tool_wear_min"] * out["torque_nm"]
    return out
