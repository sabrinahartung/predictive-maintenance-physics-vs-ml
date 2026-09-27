# Physics vs. Black Box: Predictive Maintenance on AI4I 2020

[![CI](https://github.com/sabrinahartung/predictive-maintenance-physics-vs-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/sabrinahartung/predictive-maintenance-physics-vs-ml/actions/workflows/ci.yml)

> Work in progress. See [ROADMAP.md](ROADMAP.md) for the plan.

Can a machine learning model predict milling machine failures better than a few lines of physics?
This project compares black-box classifiers, explainability (SHAP), physics-informed features and
simple rules, then turns the result into a cost-aware maintenance decision and an interactive demo.

## Quickstart

```bash
uv sync
uv run pdm-train     # train the final model, write reports/metrics.json
uv run pytest        # run the tests
uv run jupyter lab   # explore the notebooks
```

## Project structure

```
data/raw/        AI4I 2020 dataset
notebooks/       Analysis notebooks (EDA → models → explainability → physics → cost)
src/pdm/         Reusable Python package
app/             Streamlit demo
tests/           Unit tests
reports/         Figures and metrics
```

## Dataset

Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository.
https://doi.org/10.24432/C5HS5C — licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## License

Code: MIT (see [LICENSE](LICENSE)). Dataset: CC BY 4.0, see above.
