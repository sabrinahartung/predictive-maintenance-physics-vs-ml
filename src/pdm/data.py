"""Load the AI4I 2020 dataset with clean, code-friendly column names."""

from pathlib import Path

import pandas as pd

from pdm.config import DATA_RAW

COLUMN_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

TARGET = "machine_failure"
CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

FAILURE_MODES = {
    "twf": "Tool wear failure",
    "hdf": "Heat dissipation failure",
    "pwf": "Power failure",
    "osf": "Overstrain failure",
    "rnf": "Random failure",
}

# Machine quality variants: L = low (50% of products), M = medium (30%), H = high (20%)
MACHINE_TYPES = ["L", "M", "H"]


def load_data(path: Path = DATA_RAW) -> pd.DataFrame:
    """Read the raw CSV and return it with snake_case column names.

    ``type`` is returned as an ordered categorical (L < M < H).
    """
    df = pd.read_csv(path, encoding="utf-8-sig").rename(columns=COLUMN_MAP)
    missing = set(COLUMN_MAP.values()) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected dataset schema, missing columns: {sorted(missing)}")
    df["type"] = pd.Categorical(df["type"], categories=MACHINE_TYPES, ordered=True)
    return df
