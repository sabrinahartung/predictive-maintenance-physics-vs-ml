"""Streamlit demo: physics rules vs. machine learning for predictive maintenance.

Run locally with ``uv run streamlit run app/streamlit_app.py``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from pdm.cost import RULE_SCORE, best_threshold, tool_change_score
from pdm.data import FEATURES, MACHINE_TYPES, TARGET, load_data
from pdm.decision import (
    ACTION_CHANGE_TOOL,
    ACTION_INSPECT,
    ACTION_STOP,
    assess,
    row_to_frame,
)
from pdm.explain import explain_hybrid
from pdm.features import add_physics_features
from pdm.models import hybrid_model
from pdm.rules import (
    HEAT_RPM_MAX,
    HEAT_TEMP_DIFF_MAX_K,
    POWER_MAX_W,
    POWER_MIN_W,
    STRAIN_LIMIT_MIN_NM,
)

REPO_URL = "https://github.com/sabrinahartung/predictive-maintenance-physics-vs-ml"
COST_RATIOS = [2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]
GREY, RED, BLUE = "#8d99ae", "#d1495b", "#1d3557"

PRESETS = {
    "Healthy process": {
        "type": "M", "air_temp_k": 298.5, "process_temp_k": 309.0,
        "rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 50,
    },
    "Power failure (overload)": {
        "type": "L", "air_temp_k": 300.0, "process_temp_k": 310.5,
        "rpm": 1300, "torque_nm": 70.0, "tool_wear_min": 30,
    },
    "Overstrain": {
        "type": "L", "air_temp_k": 300.0, "process_temp_k": 310.5,
        "rpm": 1450, "torque_nm": 55.0, "tool_wear_min": 210,
    },
    "Heat dissipation": {
        "type": "L", "air_temp_k": 302.5, "process_temp_k": 310.8,
        "rpm": 1340, "torque_nm": 45.0, "tool_wear_min": 80,
    },
    "Worn tool": {
        "type": "M", "air_temp_k": 299.0, "process_temp_k": 309.5,
        "rpm": 1550, "torque_nm": 38.0, "tool_wear_min": 230,
    },
}  # fmt: skip

st.set_page_config(
    page_title="Predictive Maintenance: Physics vs. ML", page_icon="⚙️", layout="wide"
)


@st.cache_resource(show_spinner="Training the model on the AI4I 2020 dataset…")
def load_model():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    model = hybrid_model().fit(X, y)
    scores = tool_change_score(X)
    tool_change = {}
    for r in COST_RATIOS:
        t = best_threshold(y, scores, r)
        tool_change[r] = None if t >= RULE_SCORE else float(t)
    return add_physics_features(df), model, tool_change


def apply_preset():
    st.session_state.update(PRESETS[st.session_state["preset"]])


if "type" not in st.session_state:
    # A preset can be linked directly, e.g. ?preset=Overstrain
    requested = st.query_params.get("preset", "Healthy process")
    st.session_state["preset"] = requested if requested in PRESETS else "Healthy process"
    apply_preset()

data, model, tool_change = load_model()

# --- Sidebar: inputs ------------------------------------------------------------------------
with st.sidebar:
    st.header("Process reading")
    st.selectbox("Load an example", list(PRESETS), key="preset", on_change=apply_preset)
    st.radio(
        "Machine type (quality variant)",
        MACHINE_TYPES,
        key="type",
        horizontal=True,
        help="L = low, M = medium, H = high quality. Better machines tolerate more strain.",
    )
    st.slider("Air temperature [K]", 295.0, 305.0, step=0.1, key="air_temp_k")
    st.slider("Process temperature [K]", 305.0, 314.0, step=0.1, key="process_temp_k")
    st.slider("Rotational speed [rpm]", 1150, 2900, step=10, key="rpm")
    st.slider("Torque [Nm]", 3.0, 77.0, step=0.5, key="torque_nm")
    st.slider("Tool wear [min]", 0, 255, step=1, key="tool_wear_min")

    st.header("Costs")
    cost_ratio = st.select_slider(
        "Cost of a missed failure (in inspections)",
        COST_RATIOS,
        value=10,
        help="How much more an unplanned breakdown costs than one planned inspection or tool "
        "change. It sets the alarm level (1/R) and the cost-optimal tool change age.",
    )

row = pd.Series({f: st.session_state[f] for f in FEATURES})
X_row = row_to_frame(row)
result = assess(model, row, cost_ratio, tool_change[cost_ratio])
fired = [c.name for c in result.checks if c.fired]

# --- Header and recommendation --------------------------------------------------------------
st.title("Predictive maintenance: physics vs. black box")
st.caption(
    "Milling machine data from the AI4I 2020 dataset. Three physics rules explain 85% of all "
    "failures with 100% precision. A calibrated ML model scores the rest. "
    f"[Code and analysis on GitHub]({REPO_URL})"
)

if result.action == ACTION_STOP:
    st.error(f"**{ACTION_STOP}:** {', '.join(fired)} limit exceeded. This process will fail.")
elif result.action == ACTION_CHANGE_TOOL:
    st.warning(
        f"**{ACTION_CHANGE_TOOL}:** the tool has run {row['tool_wear_min']:.0f} min. At a cost "
        f"ratio of {cost_ratio}, replacing it from {result.tool_change_at:.0f} min on is cheaper "
        "than risking a tool wear failure."
    )
elif result.action == ACTION_INSPECT:
    st.warning(
        f"**{ACTION_INSPECT}:** no physics limit is exceeded, but the model's failure risk is "
        f"above the alarm level of {1 / cost_ratio:.0%}."
    )
else:
    st.success(
        "**Continue:** no physics limit exceeded, tool below the change age, and failure risk "
        "below the alarm level."
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Failure risk", f"{result.probability:.1%}", help="Calibrated hybrid model")
c2.metric("Alarm level (1/R)", f"{1 / cost_ratio:.1%}", help="Alarm if the risk is above this")
c3.metric(
    "Change tool at",
    f"{result.tool_change_at:.0f} min" if result.tool_change_at is not None else "not worth it",
    help="Cost-optimal tool age for the chosen cost ratio (notebook 05)",
)
c4.metric("Tool wear", f"{row['tool_wear_min']:.0f} min")

st.subheader("Physics checks")
for col, check in zip(st.columns(3), result.checks, strict=True):
    with col.container(border=True):
        st.markdown(
            f"**{check.name}** {'⛔ limit exceeded' if check.fired else '✅ within limits'}"
        )
        st.markdown(f"{check.value}  \nLimit: {check.limit}")

# --- Details --------------------------------------------------------------------------------
tab_point, tab_why, tab_about = st.tabs(["Operating point", "Why? (model explanation)", "About"])

with tab_point:
    p = result.physics

    def point_plot(x, y, xlabel, ylabel, title, current, draw_limits, **limits):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(data[x], data[y], s=3, alpha=0.2, color=GREY)
        draw_limits(ax)
        ax.scatter(*current, s=250, marker="*", color=RED, edgecolor="black", zorder=3)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title, **limits)
        fig.tight_layout()
        return fig

    def power_limits(ax):
        rpm_grid = np.linspace(1150, 2900, 200)
        for limit in (POWER_MIN_W, POWER_MAX_W):
            ax.plot(rpm_grid, limit / (rpm_grid * 2 * np.pi / 60), "--", color="black", lw=1)

    strain_limit = STRAIN_LIMIT_MIN_NM[row["type"]]

    def strain_limits(ax):
        wear_grid = np.linspace(60, 255, 200)
        ax.plot(wear_grid, strain_limit / wear_grid, "--", color="black", lw=1)
        if result.tool_change_at is not None:
            ax.axvline(result.tool_change_at, color=BLUE, lw=1.5, label="tool change age")
            ax.legend(loc="lower left")

    def heat_limits(ax):
        ax.plot(
            [7.4, HEAT_TEMP_DIFF_MAX_K, HEAT_TEMP_DIFF_MAX_K],
            [HEAT_RPM_MAX, HEAT_RPM_MAX, 1150],
            "--",
            color="black",
            lw=1,
        )

    figures = [
        point_plot(
            "rpm", "torque_nm", "rpm", "torque [Nm]",
            f"Power: {p['power_w']:,.0f} W\n(limits 3,500–9,000 W)",
            (row["rpm"], row["torque_nm"]), power_limits, ylim=(0, 80),
        ),
        point_plot(
            "tool_wear_min", "torque_nm", "tool wear [min]", "torque [Nm]",
            f"Strain: {p['strain_min_nm']:,.0f} min·Nm\n"
            f"(limit {strain_limit:,} for type {row['type']})",
            (row["tool_wear_min"], row["torque_nm"]), strain_limits, ylim=(0, 80),
        ),
        point_plot(
            "temp_diff_k", "rpm", "process − air temperature [K]", "rpm",
            f"Heat: ΔT {p['temp_diff_k']:.1f} K at {row['rpm']:,} rpm\n"
            "(fails in the lower-left box)",
            (p["temp_diff_k"], row["rpm"]), heat_limits, xlim=(7.4, 12.2), ylim=(1150, 2200),
        ),
    ]  # fmt: skip
    for col, fig in zip(st.columns(3), figures, strict=True):
        col.pyplot(fig, clear_figure=True)
    st.caption(
        "Grey: all 10,000 processes in the dataset. Red star: current reading. Dashed lines: "
        "physics limits; crossing one means a failure."
    )

with tab_why:
    if fired:
        st.info(
            f"The {', '.join(fired)} rule decides this case. The chart below shows the ML score "
            "for reference only; the rule overrides it."
        )
    exp = explain_hybrid(model, X_row)[0]
    shap.plots.waterfall(exp, max_display=len(exp.values), show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 4.5)
    st.pyplot(fig, clear_figure=True)
    st.caption(
        "SHAP contributions to the gradient boosting score (log-odds, before calibration). Red "
        "bars push towards failure, blue bars away from it. The model is trained only on "
        "processes where no physics rule fires."
    )

with tab_about:
    st.markdown(
        f"""
**What is this?** A portfolio project on the [AI4I 2020 dataset](https://doi.org/10.24432/C5HS5C)
comparing black-box machine learning with physics-informed rules.

| Approach | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| ML on raw sensor data | 0.835 | 0.78 | 80% | 76% |
| Physics rules only (3 lines) | 0.852 | 0.92 | 100% | 85% |
| Hybrid: rules + calibrated ML | 0.913 | 0.92 | 100% | 85% |

**How the decision works**

1. If a physics rule fires (power outside 3.5–9 kW, tool wear × torque above the type limit,
   or poor heat dissipation at low speed), the process will fail: **stop**.
2. Otherwise, if the tool has reached the cost-optimal change age, **change the tool**.
   Tool wear failures happen at a random time between about 200 and 240 minutes, so no sensor
   can predict the exact moment; replacing the tool in time is the cheapest answer.
3. Otherwise, if the calibrated failure risk is above 1/R, **inspect**.

With a cost ratio of 10, this policy saves about 77% of maintenance cost compared with running
machines until they fail.

Full analysis in the [notebooks on GitHub]({REPO_URL}/tree/main/notebooks).
"""
    )
