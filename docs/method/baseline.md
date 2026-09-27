# 2. Baseline models

Notebook: [02_baseline_models.ipynb][nb02] · Code: [`pdm.models`](../reference/models.md),
[`pdm.evaluate`](../reference/evaluate.md)

## Evaluation protocol

All results in this project use the same protocol:

- **Stratified 5-fold cross-validation** with fixed splits. Every model sees exactly the same folds.
- **Out-of-fold predictions:** each row is scored by a model that did not see it during training.
- **PR-AUC** (average precision) as the main metric. A random guess scores 0.034, a perfect model 1.0.
- **F1 at the best threshold** for a readable operating point. This is slightly optimistic because
  the threshold is tuned on the same predictions; [step 5](decisions.md) replaces it with an honest,
  cost-based choice.

!!! bug "A bug fixed from the original analysis"
    `sklearn.metrics.precision_recall_curve` returns one more precision/recall value than
    thresholds. The original notebook picked `thresholds[best_idx - 1]`, the neighbouring threshold
    instead of the best one. [`best_f1_threshold`](../reference/evaluate.md) uses the correct index
    and is covered by unit tests, including a brute-force comparison.

## Results on raw sensor data

| Model | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| **Gradient boosting** | **0.835** | **0.779** | 80% | 76% |
| Random forest (class-weighted) | 0.766 | 0.722 | 78% | 68% |
| Logistic regression (class-weighted) | 0.424 | 0.447 | 41% | 50% |

![Precision-recall curves](../figures/02_pr_curves.png){ width="480" }

- **Gradient boosting** is the best baseline.
- **Logistic regression** fails because failure regions like "high torque *and* low speed" cannot be
  represented by a linear boundary.
- **ROC-AUC** is 0.90–0.98 for all three models, even logistic regression. On imbalanced data it
  looks good for almost any model, which is why the project relies on PR-AUC.

## Which failures are missed?

![Recall by failure mode](../figures/02_recall_by_mode.png){ width="520" }

Heat dissipation, overstrain and power failures are caught 83–94% of the time. **Tool wear failures
are caught only 13% of the time**, and random failures and failures without a mode not at all.

## One derived feature already helps

The original version of this analysis reported PR-AUC 0.858. It turned out that it had included one
derived feature, the temperature difference `process_temp − air_temp`. Adding it back:

| Model | PR-AUC raw | PR-AUC + temperature difference |
|---|---|---|
| Gradient boosting | 0.835 | 0.858 (+0.023) |
| Random forest | 0.766 | 0.831 (+0.065) |

Trees *can* approximate a difference from two raw temperatures, but only with many splits. Giving
it to them directly is more efficient. This was the first hint that **domain knowledge in the
features matters**, which the next two steps build on.
