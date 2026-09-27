import numpy as np
import pytest

from pdm.data import FEATURES, TARGET, load_data
from pdm.explain import explain, explain_hybrid
from pdm.features import PHYSICS_FEATURES
from pdm.models import baseline_models, hybrid_model


@pytest.fixture(scope="module")
def fitted():
    df = load_data().sample(3000, random_state=0)
    X, y = df[FEATURES], df[TARGET]
    pipe = baseline_models()["HistGradientBoosting"].fit(X, y)
    return pipe, X


def test_one_value_per_input_column(fitted):
    pipe, X = fitted
    exp = explain(pipe, X.head(50))
    assert exp.values.shape == (50, len(FEATURES))
    assert exp.feature_names == FEATURES


def test_shap_values_add_up_to_model_output(fitted):
    pipe, X = fitted
    sample = X.head(50)
    exp = explain(pipe, sample)
    log_odds = pipe.decision_function(sample)
    np.testing.assert_allclose(exp.values.sum(axis=1) + exp.base_values, log_odds, atol=1e-4)


def test_data_is_in_raw_units(fitted):
    pipe, X = fitted
    exp = explain(pipe, X.head(5))
    rpm = FEATURES.index("rpm")
    np.testing.assert_allclose(exp.data[:, rpm], X["rpm"].head(5))


def test_explain_includes_physics_features():
    df = load_data().sample(2000, random_state=1)
    X, y = df[FEATURES], df[TARGET]
    pipe = baseline_models(physics=True)["HistGradientBoosting"].fit(X, y)
    exp = explain(pipe, X.head(20))
    assert exp.feature_names == [*FEATURES, *PHYSICS_FEATURES]
    np.testing.assert_allclose(
        exp.values.sum(axis=1) + exp.base_values, pipe.decision_function(X.head(20)), atol=1e-4
    )


def test_explain_hybrid_averages_calibration_folds():
    df = load_data().sample(3000, random_state=2)
    X, y = df[FEATURES], df[TARGET]
    hybrid = hybrid_model().fit(X, y)
    sample = X.head(10)
    exp = explain_hybrid(hybrid, sample)
    folds = hybrid.model_.calibrated_classifiers_
    mean_log_odds = np.mean([cc.estimator.decision_function(sample) for cc in folds], axis=0)
    assert exp.feature_names == [*FEATURES, *PHYSICS_FEATURES]
    np.testing.assert_allclose(exp.values.sum(axis=1) + exp.base_values, mean_log_odds, atol=1e-4)
