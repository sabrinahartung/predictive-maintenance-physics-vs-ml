"""Physics rules for the three deterministic failure modes, as sklearn-compatible classifiers.

The limits come from the dataset documentation (Matzka, 2020). In a real plant they would come from
machine specifications. Notebook 04 shows that each limit also lies inside the interval that
separates failing from non-failing rows in the data.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from pdm.features import add_physics_features

# Power failure: power outside this window
POWER_MIN_W = 3500
POWER_MAX_W = 9000

# Overstrain failure: tool wear × torque above a limit that depends on machine type
STRAIN_LIMIT_MIN_NM = {"L": 11_000, "M": 12_000, "H": 13_000}

# Heat dissipation failure: small temperature difference AND slow spindle.
# Temperatures have one decimal, so 27 rows have a difference of exactly 8.6 K. Their labels follow
# the floating-point result of the subtraction (8.5999… fails, 8.6000… does not), so the rule must
# use the unrounded difference to reproduce the labels. See notebook 04.
HEAT_TEMP_DIFF_MAX_K = 8.6
HEAT_RPM_MAX = 1380

RULE_MODES = {
    "pwf": "Power failure",
    "osf": "Overstrain failure",
    "hdf": "Heat dissipation failure",
}


def rule_flags(X: pd.DataFrame) -> pd.DataFrame:
    """Evaluate each rule on raw features. Returns one boolean column per failure mode."""
    f = add_physics_features(X)
    strain_limit = f["type"].astype(str).map(STRAIN_LIMIT_MIN_NM)
    return pd.DataFrame(
        {
            "pwf": (f["power_w"] < POWER_MIN_W) | (f["power_w"] > POWER_MAX_W),
            "osf": f["strain_min_nm"] > strain_limit,
            "hdf": (f["temp_diff_k"] < HEAT_TEMP_DIFF_MAX_K) & (f["rpm"] < HEAT_RPM_MAX),
        },
        index=X.index,
    )


class PhysicsRuleClassifier(ClassifierMixin, BaseEstimator):
    """Predicts a failure (probability 1) if any physics rule fires, otherwise 0.

    There is nothing to learn: ``fit`` only records the classes.
    """

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = rule_flags(X).any(axis=1).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RulesThenModel(ClassifierMixin, BaseEstimator):
    """Hybrid: if a rule fires, predict failure with probability 1; otherwise use ``model``.

    The rules cover the failure modes that are fully explained by physics, and the model
    handles everything else.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model_ = clone(self.model).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        p = self.model_.predict_proba(X)[:, 1]
        p = np.where(rule_flags(X).any(axis=1), 1.0, p)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
