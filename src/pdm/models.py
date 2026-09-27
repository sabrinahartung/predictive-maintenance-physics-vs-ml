"""Model pipelines: preprocessing + classifier."""

from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from pdm.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from pdm.features import PHYSICS_FEATURES, add_physics_features

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
    physics: bool = False,
) -> Pipeline:
    """Preprocessing + classifier. With ``physics=True`` the pipeline takes raw features, derives
    the physics features itself and uses them in addition to ``numeric``."""
    steps = []
    if physics:
        steps.append(("physics", FunctionTransformer(add_physics_features)))
        numeric = [*numeric, *PHYSICS_FEATURES]
    steps += [("pre", make_preprocessor(numeric, categorical)), ("clf", classifier)]
    return Pipeline(steps)


def baseline_models(
    numeric: list[str] = NUMERIC_FEATURES,
    categorical: list[str] = CATEGORICAL_FEATURES,
    physics: bool = False,
) -> dict[str, Pipeline]:
    """Model candidates, from a linear model to gradient boosting."""
    classifiers = {
        "LogReg (balanced)": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest (balanced)": RandomForestClassifier(
            n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    return {
        name: make_pipeline(clf, numeric, categorical, physics) for name, clf in classifiers.items()
    }
