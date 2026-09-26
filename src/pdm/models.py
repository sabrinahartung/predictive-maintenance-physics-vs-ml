"""Model pipelines: preprocessing + classifier."""

from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pdm.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

RANDOM_STATE = 0


def make_preprocessor(
    numeric: list[str] = NUMERIC_FEATURES,
    categorical: list[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical ones."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(
    classifier: ClassifierMixin,
    numeric: list[str] = NUMERIC_FEATURES,
    categorical: list[str] = CATEGORICAL_FEATURES,
) -> Pipeline:
    return Pipeline([("pre", make_preprocessor(numeric, categorical)), ("clf", classifier)])


def baseline_models(
    numeric: list[str] = NUMERIC_FEATURES,
    categorical: list[str] = CATEGORICAL_FEATURES,
) -> dict[str, Pipeline]:
    """Baseline candidates, from a linear model to gradient boosting."""
    classifiers = {
        "LogReg (balanced)": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest (balanced)": RandomForestClassifier(
            n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    return {name: make_pipeline(clf, numeric, categorical) for name, clf in classifiers.items()}
