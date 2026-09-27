"""Smoke tests for the Streamlit app, run headless with Streamlit's testing API."""

import pytest
from streamlit.testing.v1 import AppTest

from pdm.config import PROJECT_ROOT

APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    return at


def test_default_preset_is_healthy(app):
    assert app.success[0].value.startswith("**Continue:**")


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("Power failure (overload)", "Power limit exceeded"),
        ("Overstrain", "Overstrain limit exceeded"),
        ("Heat dissipation", "Heat dissipation limit exceeded"),
    ],
)
def test_failure_presets_stop(app, preset, expected):
    app.selectbox(key="preset").set_value(preset).run()
    assert not app.exception
    assert expected in app.error[0].value


def test_worn_tool_preset_recommends_tool_change(app):
    app.selectbox(key="preset").set_value("Worn tool").run()
    assert not app.exception
    assert app.warning[0].value.startswith("**Change tool:**")
