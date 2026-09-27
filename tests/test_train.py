import json

import joblib
import pytest

from pdm.data import FEATURES, load_data
from pdm.train import main, run, tool_change_minutes


def test_train_writes_model_and_metrics(tmp_path):
    model_path, metrics_path = tmp_path / "model.joblib", tmp_path / "metrics.json"
    metrics = run(cost_ratio=10, skip_cv=True, model_path=model_path, metrics_path=metrics_path)

    assert json.loads(metrics_path.read_text()) == metrics
    assert metrics["alarm_threshold"] == pytest.approx(0.1)
    assert metrics["data"] == {"rows": 10_000, "failures": 339}

    model = joblib.load(model_path)
    proba = model.predict_proba(load_data()[FEATURES].head(100))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_tool_change_depends_on_cost_ratio():
    df = load_data()
    X, y = df[FEATURES], df["machine_failure"]
    assert tool_change_minutes(X, y, cost_ratio=2) is None  # cheap failures: never worth it
    assert 195 <= tool_change_minutes(X, y, cost_ratio=100) <= 230


def test_cli_runs(tmp_path, capsys):
    main(
        [
            "--skip-cv",
            "--cost-ratio",
            "20",
            "--model-path",
            str(tmp_path / "m.joblib"),
            "--metrics-path",
            str(tmp_path / "metrics.json"),
        ]
    )
    out = capsys.readouterr().out
    assert "alarm if p > 0.050" in out
    assert "Tool change: at" in out
