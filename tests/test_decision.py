import pandas as pd
import pytest

from pdm.data import FEATURES, TARGET, load_data
from pdm.decision import ACTION_CHANGE_TOOL, ACTION_OK, ACTION_STOP, assess, physics_checks
from pdm.models import hybrid_model


def make_row(**overrides):
    row = {
        "type": "M",
        "air_temp_k": 298.5,
        "process_temp_k": 309.0,
        "rpm": 1500,
        "torque_nm": 40.0,
        "tool_wear_min": 50,
    }
    return pd.Series(row | overrides)


@pytest.fixture(scope="module")
def model():
    df = load_data()
    return hybrid_model().fit(df[FEATURES], df[TARGET])


def fired(row):
    return {c.name for c in physics_checks(row) if c.fired}


def test_physics_checks():
    assert fired(make_row()) == set()
    assert fired(make_row(rpm=1300, torque_nm=70)) == {"Power"}
    assert fired(make_row(type="L", torque_nm=55, tool_wear_min=210)) == {"Overstrain"}
    assert fired(make_row(air_temp_k=302.5, process_temp_k=310.8, rpm=1340)) == {"Heat dissipation"}


def test_overstrain_limit_depends_on_type():
    worn = {"torque_nm": 50.0, "tool_wear_min": 230}  # strain 11,500
    assert fired(make_row(type="L", **worn)) == {"Overstrain"}
    assert fired(make_row(type="M", **worn)) == set()


def test_assess_actions(model):
    healthy = assess(model, make_row(), cost_ratio=10, tool_change_at=225)
    assert healthy.action == ACTION_OK
    assert healthy.probability < 0.1

    power = assess(model, make_row(rpm=1300, torque_nm=70), cost_ratio=10, tool_change_at=225)
    assert power.action == ACTION_STOP
    assert power.probability == 1.0

    worn = assess(model, make_row(tool_wear_min=230), cost_ratio=10, tool_change_at=225)
    assert worn.action == ACTION_CHANGE_TOOL
    assert worn.tool_change_due


def test_no_tool_change_when_not_worth_it(model):
    result = assess(model, make_row(tool_wear_min=240), cost_ratio=2, tool_change_at=None)
    assert not result.tool_change_due
    assert result.action != ACTION_CHANGE_TOOL
