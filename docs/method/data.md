# 1. Data exploration

Notebook: [01_eda.ipynb][nb01]

## The dataset

The [AI4I 2020 Predictive Maintenance Dataset][dataset] (Matzka, 2020) simulates a milling machine.
Each of its 10,000 rows is one machining process with:

- **5 sensor readings:** air temperature, process temperature, rotational speed, torque, tool wear
- **the machine type:** quality variant L, M or H (60% / 30% / 10% of rows)
- **the target:** did the process end in a machine failure (0/1)?
- **5 failure-mode labels** explaining *why* it failed

The data is complete: no missing values, no duplicates, and all ranges are physically plausible.
Every row has its own product ID, so the rows are **independent snapshots**, not a time series.
The task is therefore classification ("will this process fail?"), not forecasting the remaining
useful life.

!!! warning "Failure-mode labels are not features"
    The five mode labels are only known after a failure happened. Using them as inputs would leak
    the answer into the model. They are used only for analysis.

## Failures are rare

![Class balance](../figures/01_class_balance.png){ width="420" }

Only **339 of 10,000 processes (3.4%)** fail. Consequences for the whole project:

- Accuracy is useless as a metric, so the project uses **PR-AUC** and F1 instead.
- **Stratified cross-validation** keeps the failure share equal in every fold.
- No resampling to 50/50: class weights and a deliberately chosen threshold keep the probabilities
  meaningful.

## Five failure modes

![Failure modes](../figures/01_failure_modes.png){ width="520" }

| Mode | Count | Notes |
|---|---|---|
| Heat dissipation (HDF) | 115 | most common |
| Overstrain (OSF) | 98 | 87 of them on type L machines |
| Power (PWF) | 95 | |
| Tool wear (TWF) | 46 | happen between about 200 and 240 minutes of wear |
| Random (RNF) | 19 | only 1 of them counts as a machine failure |

Type L machines fail almost twice as often as type H (3.9% vs. 2.1%), mainly through overstrain.

## Single sensors vs. combinations

![Feature distributions](../figures/01_feature_distributions.png)

The distributions of failing and healthy processes overlap heavily for every single sensor. No
single threshold separates them. Looking at **pairs** of sensors changes the picture:

![Feature interactions](../figures/01_interactions.png)

- **Speed vs. torque:** failures sit at the two ends of the speed–torque band, at high torque with low
  speed and at low torque with very high speed.
- **Tool wear vs. torque:** failures pile up above about 200 minutes of wear, across almost all
  torques.

The linear correlation of every sensor with the target is weak (at most 0.19). Failures are driven
by **non-linear combinations**, which is the first hint towards the physics rules found later.

## Label quality

Not every label is consistent:

- **9 failures have no failure mode.** Nothing in the data explains them.
- **18 processes have a failure mode but no failure.** All of them are random failures.
- **24 processes have more than one mode.**

The target is used as given. These inconsistencies cap the best achievable score, which is
quantified in [step 4](physics.md#the-performance-ceiling).
