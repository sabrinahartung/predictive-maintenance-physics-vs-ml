# 4. Physics features and rules

Notebook: [04_physics_features.ipynb][nb04] · Code: [`pdm.features`](../reference/features.md),
[`pdm.rules`](../reference/rules.md)

This is the core of the project: do the three hypotheses from SHAP hold, and do they beat the black
box?

## Three physics features

| Feature | Formula | Physical meaning |
|---|---|---|
| `temp_diff_k` | process temperature − air temperature | driving force for heat flowing out of the process |
| `power_w` | torque · rpm · 2π / 60 | mechanical power at the spindle |
| `strain_min_nm` | tool wear · torque | load on an already worn tool |

With `physics=True`, the model pipelines compute these automatically from the raw inputs.

## Finding the exact thresholds

For each rule, the notebook looks for the **gap** between failing and non-failing rows: for example,
the highest power of any low-power failure and the lowest power of any other row. Any threshold
inside the gap classifies every row correctly.

| Rule | Gap in the data | Documented limit | Inside? |
|---|---|---|---|
| Power, lower | 3,477 – 3,515 W | 3,500 W | ✅ |
| Power, upper | 8,998 – 9,004 W | 9,000 W | ✅ |
| Overstrain, type L | 10,994 – 11,003 min·Nm | 11,000 | ✅ |
| Overstrain, type M | 11,920 – 12,337 min·Nm | 12,000 | ✅ |
| Overstrain, type H | 11,873 – 13,235 min·Nm | 13,000 | ✅ |
| Heat, speed | 1,379 – 1,381 rpm | 1,380 rpm | ✅ |
| Heat, temperature difference | 8.6 – 8.6 K | 8.6 K | ✅ (see below) |

Every limit from the dataset documentation lies inside the gap, so **all three hypotheses held**.
The overstrain limit really depends on the machine type: better machines tolerate more strain.

![The three rules](../figures/04_rules.png)

Each rule reproduces its failure-mode label on **every one of the 10,000 rows**, and unit tests in
`tests/test_rules.py` lock this in.

!!! example "A labelling detail: floating-point rounding decides 27 rows"
    Temperatures are stored with one decimal, so 27 slow-spindle processes have a temperature
    difference of exactly 8.6 K. Their labels still split 12 to 15. Whether a row counts as a heat
    dissipation failure depends on how the subtraction rounds in binary: `310.9 − 302.3 = 8.5999…`
    fails, while `311.0 − 302.4 = 8.6000…` does not.

    The rule reproduces these labels only because it performs the same subtraction. **100% agreement
    can mean you reproduced an artefact of how the labels were made**, not physics.

## Physics beats the black box

All approaches use the same cross-validation:

![Comparison of approaches](../figures/04_comparison.png)

| Approach | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| ML, raw features (gradient boosting) | 0.835 | 0.779 | 80% | 76% |
| ML + physics (logistic regression) | 0.445 | 0.466 | 44% | 49% |
| ML + physics (gradient boosting) | 0.888 | 0.851 | 94% | 78% |
| ML + physics (random forest) | 0.891 | 0.877 | 94% | 82% |
| **Rules only** | 0.852 | **0.917** | **100%** | 85% |
| **Hybrid: rules + gradient boosting** | **0.906** | **0.917** | **100%** | 85% |

- **Three lines of physics beat every model on F1**, with every alarm being a real failure.
- **Physics features help every tree model**, but none of them reaches the rules.
- **Logistic regression stays poor** even with the physics features, because the rules are windows
  and AND conditions, which a linear model cannot express.
- **The hybrid** matches the rules and adds a ranking for the remaining rows (best PR-AUC).
  The final app uses an improved version of it ([step 5](decisions.md#calibration)).

## Why gradient boosting misses power failures

Even with `power_w` as a feature, gradient boosting misses 18 of 95 power failures. The reason is
**histogram binning**: `HistGradientBoostingClassifier` groups each feature into at most 255 bins at
quantiles and can only split between bins. Power failures are rare extremes, so the tail bins are
wide. The 3,500 W limit falls inside a bin spanning about 1,148–3,538 W, and the 9,000 W limit inside
one spanning about 8,908–9,160 W. No number of trees can split exactly at the limit.

## The performance ceiling

![What explains the failures](../figures/04_failure_breakdown.png){ width="560" }

- **287 failures (84.7%)** are explained by the three rules.
- **43** are tool wear failures, which happen at a random moment between about 200 and 240 minutes.
- **9** have no failure mode at all.

Flagging the whole tool wear window would catch most tool wear failures, but at the cost of 671
false alarms (F1 drops to 0.49). The best achievable result on this data is therefore essentially
what the rules deliver: **100% precision at 84.7% recall, F1 ≈ 0.92**.

That is the ceiling for *prediction*. The next step shows that a *cost-based decision* can still do
better with tool wear failures, by replacing old tools in time.
