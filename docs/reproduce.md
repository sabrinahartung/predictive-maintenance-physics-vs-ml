# Run it yourself

Everything in this project can be reproduced from the repository with one tool: [uv](https://docs.astral.sh/uv/).
It installs the right Python version (3.12) and the exact dependency versions from `uv.lock`.

```bash
git clone https://github.com/sabrinahartung/predictive-maintenance-physics-vs-ml.git
cd predictive-maintenance-physics-vs-ml
uv sync
```

## Common tasks

| Task | Command |
|---|---|
| Run the app locally | `uv run streamlit run app/streamlit_app.py` |
| Train the final model and write `reports/metrics.json` | `uv run pdm-train` |
| Train for a different cost ratio | `uv run pdm-train --cost-ratio 20` |
| Run all tests | `uv run pytest` |
| Lint and format | `uv run ruff check .` and `uv run ruff format .` |
| Explore the notebooks | `uv run jupyter lab` |
| Re-run a notebook from scratch | `uv run jupyter nbconvert --execute --to notebook --inplace notebooks/04_physics_features.ipynb` |
| Preview this documentation | `uv run --group docs mkdocs serve` |

The notebooks build on each other's saved results in `reports/`. To reproduce everything from
scratch, run them in order, 01 to 05.

## Training output

`uv run pdm-train` prints a summary like this:

```text
Cost ratio R = 10: alarm if p > 0.100
Tool change: at 225 min
Hybrid PR-AUC 0.913, Brier 0.0050
  rules_only               cost     807  savings 76.2%
  hybrid_calibrated        cost     798  savings 76.5%
  rules_plus_tool_change   cost     763  savings 77.5%
```

It saves the fitted model to `models/model.joblib` (not committed) and all metrics to
`reports/metrics.json`. Use `--skip-cv` to only train, which takes a few seconds.

## Using the model in your own code

```python
import pandas as pd

from pdm.decision import assess
from pdm.data import FEATURES, TARGET, load_data
from pdm.models import hybrid_model

df = load_data()
model = hybrid_model().fit(df[FEATURES], df[TARGET])

reading = pd.Series(
    {
        "type": "M",
        "air_temp_k": 299.0,
        "process_temp_k": 309.5,
        "rpm": 1550,
        "torque_nm": 38.0,
        "tool_wear_min": 230,
    }
)
result = assess(model, reading, cost_ratio=10, tool_change_at=225)
print(result.action, f"{result.probability:.1%}")  # Change tool 3.7%
```
