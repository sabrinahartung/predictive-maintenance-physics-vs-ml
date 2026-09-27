# 5. From predictions to decisions

Notebook: [05_cost_threshold.ipynb][nb05] · Code: [`pdm.cost`](../reference/cost.md),
[`pdm.models.hybrid_model`](../reference/models.md)

F1 treats a missed failure and a false alarm as equally bad. In maintenance they are not. This step
chooses what to do based on what errors **cost**.

## 1. The cost model

All costs are in units of one inspection:

| Event | Cost |
|---|---|
| Alarm (inspection, repair or tool change if needed) | 1 |
| Missed failure (unplanned breakdown) | **R** |
| No alarm, no failure | 0 |

**Total cost = alarms + R × missed failures.** The repair itself is needed either way, so only the
difference between planned and unplanned handling counts. The default is **R = 10**. Running
machines until they fail costs 339 × R (3,390 at R = 10).

## 2. Five policies

| Policy | Alarm when… |
|---|---|
| ML, raw features | gradient boosting probability ≥ threshold |
| ML + physics | same, with physics features |
| Rules only | any physics rule fires |
| Hybrid (calibrated) | a rule fires, or the calibrated ML probability ≥ threshold |
| **Rules + tool change** | a rule fires, or the tool has been used for ≥ W minutes |

The last policy is classic reliability engineering (**age-based replacement**). Tool wear failures
happen at a random moment, and no sensor predicts *when*, so the answer is to replace the tool
before it reaches the risky age. W is chosen from the costs, like any other threshold.

![Cost curves at R = 10](../figures/05_cost_curves.png)

## 3. Honest cost estimate

For each fold, the threshold is chosen on the other four folds and then applied to the held-out
fold. At R = 10:

| Policy | Cost | Savings vs. running to failure | Alarms | Missed failures |
|---|---|---|---|---|
| **Rules + tool change at ~225 min** | **763** | **77.5%** | 353 | 41 |
| Hybrid (calibrated) | 798 | 76.5% | 318 | 48 |
| Rules only | 807 | 76.2% | 287 | 52 |
| ML + physics | 933 | 72.5% | 403 | 53 |
| ML, raw features | 1,055 | 68.9% | 495 | 56 |

Replacing tools at about 225 minutes catches 11 tool wear failures at the price of 66 extra alarms.

## 4. How the best policy depends on R

![Savings by cost ratio](../figures/05_sensitivity.png)

| R | Best policy | Tool change at | Savings |
|---|---|---|---|
| 2–7 | Rules only | (late or never) | 42–73% |
| 10 | Rules + tool change | 225 min | 77.5% |
| 20 | Rules + tool change | 205 min | 83.6% |
| 100 | Rules + tool change | ~200 min | 94.0% |

The more expensive failures are, the earlier the tool change and the bigger its lead. At R = 100,
the rules alone save 84% and rules + tool change 94%. **The hybrid ML model never beats the simple
policies**, even though it has the best PR-AUC. A single metric cannot tell you which policy to deploy.

## 5. Why the simple policy beats the ML model

![Tool wear failures caught per extra alarm](../figures/05_tool_wear_capture.png){ width="520" }

Among processes where no rule fires, ranking by **tool wear alone** catches tool wear failures much
faster than the ML probability. The model sees the 43 tool wear failures as isolated points among
hundreds of similar processes, and partly learns their noise. Tool wear is monotone in risk: an
older tool is always closer to its random failure time. When the only real signal is "how old is
the tool", the simplest score is the best.

## 6. Calibration { #calibration }

The app shows a failure **probability**, so that number has to mean what it says. Three versions of
the hybrid were compared:

![Reliability of the three hybrid variants](../figures/05_calibration.png){ width="460" }

| Variant | PR-AUC | Brier score | log loss |
|---|---|---|---|
| ML part trained on all rows, uncalibrated | 0.905 | 0.0065 | 0.039 |
| ML part trained on all rows, calibrated | 0.905 | 0.0058 | 0.029 |
| **ML part trained on rows without a rule, calibrated** | **0.913** | **0.0050** | **0.027** |

Calibration alone was not enough: rows predicted at about 30% still failed only about 4% of the
time. The reason was a mismatch between training and use. In the hybrid, the ML model only ever
scores rows where **no rule fires**, but it was trained on all rows, where a high score usually meant
a rule failure. Training (and calibrating) it only on the rows it will actually see fixed this:
predicted and observed failure rates now agree.

With calibrated probabilities, decision theory gives the alarm threshold directly: alarm when
*p × R > 1*, so **p > 1/R**. This untuned threshold stays within about 5% of the cross-validated one,
which is why the app needs only one slider for R.

## Recommendation

| Situation | Policy |
|---|---|
| Breakdowns are cheap (R ≤ 7) | Physics rules only |
| Default (R ≈ 10) | Physics rules + change tools at about **225 minutes** |
| Breakdowns are expensive (R ≥ 20) | Physics rules + change tools at about **200–205 minutes** |
| Showing a risk to operators | Calibrated hybrid probability, alarm if p > 1/R |

!!! warning "Limitations"
    - **Static evaluation.** After deployment, no process would run with an older tool, so the data
      would look different. The estimate shows the direction and size of the benefit, not an exact
      number.
    - **One alarm cost.** Inspections and tool changes are assumed to cost the same. If a tool
      change costs more, the optimal age moves later.
    - **R is an assumption** and should come from the plant's own downtime and repair costs.
