# 3. Explainability

Notebook: [03_explainability.ipynb][nb03] · Code: [`pdm.explain`](../reference/explain.md)

The best baseline clearly learned *something*. This step used **SHAP** to find out *what*, and to
turn it into precise, testable hypotheses.

## SHAP in one paragraph

For every prediction, SHAP splits the model's output into one contribution per feature. The
contributions add up exactly to the output, starting from the model's average (the base value).
Here the output is in **log-odds of failure**: positive contributions push towards failure, negative
ones away from it. The same chart appears in the app's *Why?* tab
([how to read it](../app/guide.md#why-model-explanation)).

## What matters overall

![Feature importance](../figures/03_importance.png){ width="480" }

![SHAP beeswarm](../figures/03_beeswarm.png)

Tool wear, torque and speed dominate. The long tails to the right of the beeswarm are the key
observation: most rows get small negative contributions, and a few get very large positive ones.
The model has learned **sharp failure regions**, not smooth trends.

## Three interactions become three hypotheses

Plotting the *combined* contribution of two features against a physically meaningful quantity
reveals the structure.

### H1: power

![Joint SHAP of speed and torque with lines of equal power](../figures/03_h1_power.png){ width="560" }

The high-risk regions follow the dotted **lines of equal power** (torque × speed), not vertical or
horizontal lines. The joint contribution is near zero between about 4 and 8 kW and jumps outside
roughly 3.5–9 kW.

### H2: overstrain

![Joint SHAP of tool wear and torque](../figures/03_h2_strain.png){ width="560" }

Risk rises with the **product** tool wear × torque and jumps above about 11,000 min·Nm. The picture
is noisier because torque also carries the power effect.

### H3: heat dissipation

![Temperature effect against the temperature difference](../figures/03_h3_heat.png){ width="560" }

Plotted against the temperature **difference**, the two temperature contributions collapse onto a
clear pattern: they rise sharply below about 8.5 K, **but only for slow spindles** (dark points).
That is an AND condition between two quantities.

| # | Failure mode | Quantity | Hypothesis |
|---|---|---|---|
| H1 | Power | torque · rpm · 2π/60 | fails if power < ~3.5 kW or > ~9 kW |
| H2 | Overstrain | tool wear · torque | fails if strain > ~11,000 min·Nm, maybe depending on type |
| H3 | Heat dissipation | process − air temperature | fails if ΔT < ~8.5 K **and** rpm < ~1,400 |

## A lesson about explaining single predictions

The notebook also explains one missed tool wear failure. The first attempt used the model trained
on **all** rows, which gave this row a 68% failure probability and a convincing explanation. The
model that actually missed it had never seen the row and gave it about 0%.

!!! tip "Lesson learned"
    A model explained on its own training data can tell a convincing but **false** story, because it
    has partly memorised the labels. Always explain the prediction that was actually made. The
    notebook explains each example with the cross-validation model that never saw it.
