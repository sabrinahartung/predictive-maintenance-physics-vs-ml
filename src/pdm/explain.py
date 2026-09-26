"""SHAP explanations in the units of the original features."""

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _output_columns(pre: ColumnTransformer) -> dict[str, list[int]]:
    """Map each input column to the positions of the columns it produces after preprocessing."""
    groups, pos = {}, 0
    for _, transformer, columns in pre.transformers_:
        if transformer == "drop" or len(columns) == 0:
            continue
        if isinstance(transformer, OneHotEncoder):
            sizes = [len(categories) for categories in transformer.categories_]
        else:
            sizes = [1] * len(columns)
        for column, size in zip(columns, sizes, strict=True):
            groups[column] = list(range(pos, pos + size))
            pos += size
    return groups


def explain(pipeline: Pipeline, X: pd.DataFrame) -> shap.Explanation:
    """SHAP values of a fitted ``pre`` + tree classifier pipeline for the rows in ``X``.

    Values are in log-odds of failure. Contributions of one-hot encoded columns are summed back
    into their original column, so there is exactly one value per column of ``X``. ``data`` holds
    the raw (unscaled) feature values; categorical columns are given as category codes, with the
    labels in ``display_data``.
    """
    pre, clf = pipeline.named_steps["pre"], pipeline[-1]
    raw = shap.TreeExplainer(clf)(pre.transform(X))

    groups = _output_columns(pre)
    values = np.column_stack([raw.values[:, groups[col]].sum(axis=1) for col in X.columns])
    data = np.column_stack(
        [
            X[col].cat.codes if isinstance(X[col].dtype, pd.CategoricalDtype) else X[col]
            for col in X.columns
        ]
    ).astype(float)

    return shap.Explanation(
        values=values,
        base_values=np.broadcast_to(np.ravel(raw.base_values)[0], len(X)).copy(),
        data=data,
        display_data=X.to_numpy(dtype=object),
        feature_names=list(X.columns),
    )
