import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier

from pdm.data import FEATURES, TARGET, load_data
from pdm.features import PHYSICS_FEATURES, add_physics_features, power_w
from pdm.rules import RULE_MODES, PhysicsRuleClassifier, RulesThenModel, rule_flags


@pytest.fixture(scope="module")
def df():
    return load_data()


def test_power_formula():
    # 60 Nm at 1000 rpm: 60 * 1000 * 2π / 60 = 2000π W
    assert power_w(60, 1000) == pytest.approx(2000 * np.pi)


def test_add_physics_features_does_not_modify_input(df):
    out = add_physics_features(df.head())
    assert set(PHYSICS_FEATURES) <= set(out.columns)
    assert not set(PHYSICS_FEATURES) & set(df.columns)


@pytest.mark.parametrize("mode", list(RULE_MODES))
def test_rules_reproduce_failure_mode_labels(df, mode):
    flags = rule_flags(df[FEATURES])
    pd.testing.assert_series_equal(flags[mode].astype(int), df[mode], check_names=False)


def test_rule_classifier_predicts_only_rule_failures(df):
    X, y = df[FEATURES], df[TARGET]
    pred = clone(PhysicsRuleClassifier()).fit(X, y).predict(X)
    assert ((pred == 1) <= (y == 1)).all(), "every rule alarm is a real failure"
    assert pred.sum() == df[list(RULE_MODES)].any(axis=1).sum()


def test_hybrid_uses_model_when_no_rule_fires(df):
    X, y = df[FEATURES], df[TARGET]
    hybrid = RulesThenModel(DummyClassifier(strategy="prior")).fit(X, y)
    proba = hybrid.predict_proba(X)[:, 1]
    fired = rule_flags(X).any(axis=1).to_numpy()
    assert (proba[fired] == 1).all()
    np.testing.assert_allclose(proba[~fired], y.mean())
