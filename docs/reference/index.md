# API reference

The `pdm` package contains all the logic used by the notebooks, the training CLI and the app. This
reference is generated from the docstrings in `src/pdm/`.

| Module | Responsibility |
|---|---|
| [`pdm.data`](data.md) | Load the dataset with clean column names |
| [`pdm.features`](features.md) | Physics features |
| [`pdm.rules`](rules.md) | Physics rules and the rules-then-model hybrid |
| [`pdm.models`](models.md) | Model pipelines and the final hybrid |
| [`pdm.evaluate`](evaluate.md) | Cross-validation and metrics |
| [`pdm.explain`](explain.md) | SHAP explanations |
| [`pdm.cost`](cost.md) | Cost-based decisions |
| [`pdm.decision`](decision.md) | Recommendation for one reading |
| [`pdm.train`](train.md) | Training CLI |

Paths and constants shared by all modules are in `pdm.config`.
