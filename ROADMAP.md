# Roadmap: Physics vs. Black Box — Predictive Maintenance on AI4I 2020

**Pitch:** A gradient-boosted model reaches F1 ≈ 0.81 on machine failure prediction. Three physics rules
(power, overstrain, heat dissipation) reach F1 ≈ 0.92 with no ML at all. This project shows how to find
that out with explainability, how to combine domain knowledge and ML, where the honest performance
ceiling is, and how to turn it into a cost-aware decision and a live demo.

**Decisions (change here if needed)**

| Topic | Choice |
|---|---|
| Language | English (code, notebooks, README) |
| Python / tooling | Python 3.12, `uv` (pyproject + lockfile), `ruff`, `pytest` |
| ML stack | scikit-learn (HistGradientBoosting, RandomForest, LogReg), `shap` |
| Demo | Streamlit, deployed on Streamlit Community Cloud |
| Data | Commit `ai4i2020.csv` (≈0.5 MB, CC BY 4.0) with attribution |

**Target repo layout**

```
├── data/raw/ai4i2020.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_explainability.ipynb
│   ├── 04_physics_features.ipynb
│   └── 05_cost_threshold.ipynb
├── src/pdm/
│   ├── config.py      # paths, column names, constants (rule limits)
│   ├── data.py        # load + clean column names
│   ├── features.py    # temp diff, power [W], strain [min·Nm]
│   ├── rules.py       # PhysicsRuleClassifier (sklearn-compatible)
│   ├── models.py      # model/pipeline factory
│   ├── evaluate.py    # CV, OOF probabilities, metrics, best threshold
│   ├── cost.py        # expected cost vs. threshold
│   └── train.py       # CLI: train final model, write metrics.json
├── app/streamlit_app.py
├── reports/figures/
├── tests/
├── .github/workflows/ci.yml
├── pyproject.toml, uv.lock, README.md, ROADMAP.md, LICENSE
```

Each milestone ends with a commit (or PR) and a clear **Done when** check.

---

## M0 — Project setup

- [x] `git init`; first commit contains the **original** notebook + CSV unchanged (shows progression in history)
- [x] `.gitignore` (`.venv/`, `.idea/`, `.DS_Store`, `__pycache__/`, `.ipynb_checkpoints/`, `models/`)
- [x] Delete old Python 3.9 `.venv`; `uv init` with Python 3.12, `src/pdm` package layout
- [x] Dependencies: `pandas numpy scikit-learn matplotlib seaborn shap joblib streamlit`; dev: `ruff pytest jupyterlab nbconvert`
- [x] Move data to `data/raw/`, original notebook to `notebooks/archive/`
- [x] `LICENSE` (MIT for code) and README stub with dataset citation
- [x] Create GitHub repo and push (private until M8)

**Done when:** `uv sync && uv run python -c "import pdm, shap, sklearn"` works on a fresh clone.

## M1 — Reproducible EDA (`01_eda.ipynb`)

- [x] Port the existing EDA to English; fix hidden state (`df` undefined → use `load_data()` from `src/pdm/data.py`)
- [x] Clean column names once in `data.py` (e.g. `air_temp_k`, `process_temp_k`, `rpm`, `torque_nm`, `tool_wear_min`)
- [x] Keep: variable table, range sanity check, class imbalance, failure-mode counts, correlation heatmap
- [x] Add: failure rate by machine type, feature distributions split by failure, failure mode × type table
- [x] Add: label consistency check (9 failures without a mode, 18 modes without failure)
- [x] Save key figures to `reports/figures/`

**Done when:** `uv run jupyter nbconvert --execute --to notebook notebooks/01_eda.ipynb` runs top to bottom without errors.

## M2 — Baseline models (`02_baseline_models.ipynb` + `src/pdm`)

- [x] `evaluate.py`: stratified 5-fold CV, out-of-fold probabilities, PR-AUC, ROC-AUC, F1 at best threshold
- [x] Fix threshold bug: use `th[best_idx]` (guard `best_idx == len(th)`), not `th[best_idx - 1]`
- [x] `models.py`: LogReg, RandomForest, HistGradientBoosting pipelines on **raw** sensor features + type
- [x] Results table + PR curves for all models in one plot; save OOF results to `reports/`

**Done when:** notebook reproduces roughly PR-AUC ≈ 0.86 / F1 ≈ 0.81 for the best baseline, and all logic lives in `src/pdm`.

> Result: raw-feature baseline is PR-AUC 0.835 / F1 0.78 (HistGradientBoosting). The original 0.858 included a
> derived temperature-difference feature; adding it back reproduces 0.858 exactly (see notebook section 6).

## M3 — Explainability (`03_explainability.ipynb`)

- [x] SHAP (TreeExplainer) on best baseline: summary/beeswarm plot, global importance
- [x] Dependence plots: torque × rpm, tool wear × torque, process temp − air temp × rpm
- [x] Write the hypotheses these plots suggest (power window, overstrain threshold, heat dissipation condition)

**Done when:** notebook ends with 3 explicit, testable hypotheses that lead into M4.

## M4 — Physics features & rules (`04_physics_features.ipynb`) ⭐ core of the story

- [x] `features.py`: `temp_diff_k`, `power_w = torque · rpm · 2π/60`, `strain = tool_wear · torque`
- [x] `rules.py`: `PhysicsRuleClassifier` (fit/predict/predict_proba) with HDF, PWF, OSF rules; returns triggered mode
- [x] Verify each rule reproduces its failure-mode label (100%)
- [x] Compare in one table: raw-feature ML · rules only · ML + physics features · rules + ML hybrid
- [x] Per-failure-mode recall: show TWF/RNF are (near) unpredictable → theoretical ceiling
- [x] Short section "What this means on real data" (synthetic dataset caveat, why domain knowledge still matters)

**Done when:** comparison table shows rules ≈ F1 0.92 and ML + physics features ≥ rules; ceiling is quantified.

> Result: rules F1 0.917 (precision 1.0, recall 0.847); hybrid rules + HGB best PR-AUC 0.906 with the same F1.
> ML + physics features did **not** reach the rules on F1 (RF 0.877, HGB 0.851: histogram binning can't place
> the power limits). Ceiling = rules: 15% of failures carry no signal. 27 HDF labels are floating-point artefacts.

## M5 — Cost-based threshold (`05_cost_threshold.ipynb`)

- [x] `cost.py`: expected cost per threshold from a cost matrix (missed failure vs. false alarm vs. inspection)
- [x] Choose default costs (e.g. missed failure = 10× false alarm), plot cost vs. threshold, pick optimum
- [x] Sensitivity: optimal threshold for cost ratios 2×–50×
- [x] Check probability calibration (reliability plot); calibrate if needed

**Done when:** a recommended threshold with a one-paragraph business justification.

> Result: cheapest policy for R ≥ 10 is **physics rules + preventive tool change** (~225 min at R = 10, ~200 min at
> R = 100): 77.5% savings vs. no predictive maintenance at R = 10, beats every ML policy (CV-selected thresholds).
> Rules alone are optimal for R ≤ 7. Final hybrid = ML trained only on rows without a rule + isotonic calibration
> (PR-AUC 0.913, well calibrated), so the app can alarm at p > 1/R without tuning.

## M6 — Training pipeline, tests, CI

- [x] `pdm/train.py` CLI (`uv run pdm-train`): trains final model, writes `models/model.joblib` + `reports/metrics.json`
- [x] Tests: feature formulas, rules reproduce HDF/PWF/OSF labels, pipeline predicts correct shape, threshold helper edge cases
- [x] `ruff check` + `ruff format` clean
- [x] GitHub Actions: `uv sync` → `ruff check` → `pytest` on push/PR

**Done when:** CI badge is green.

## M7 — Streamlit demo (`app/streamlit_app.py`)

- [x] Inputs: machine type, air/process temp, rpm, torque, tool wear (sliders with realistic ranges)
- [x] Cost ratio slider R (default 10): alarm if calibrated hybrid probability > 1/R
- [x] Outputs: failure probability (`hybrid_model()`), decision, triggered physics rule(s), derived values (power, strain, ΔT) vs. limits
- [x] Tool change recommendation from the M5 policy (tool wear vs. cost-optimal W for the chosen R)
- [x] SHAP waterfall for the current prediction
- [x] Presets: "healthy", "overstrain", "power failure", "heat dissipation", "worn tool"
- [x] Train on startup with `st.cache_resource` (dataset is small, avoids pickle version issues)
- [ ] Deploy to Streamlit Community Cloud; add link + screenshot/GIF to README (needs repo public + Streamlit account)

**Done when:** public URL works and shows a correct explanation for each preset.

> Local app done: `uv run streamlit run app/streamlit_app.py`. Headless smoke tests in `tests/test_app.py`;
> `requirements.txt` (exported from `uv.lock`, checked in CI) for Streamlit Cloud. Startup trains the model (~20–30 s).

## M8 — README & polish

- [ ] README structure: pitch · key result table · 2–3 figures · approach · how to run · limitations · dataset citation
- [ ] Notebooks: consistent headings, short conclusions per notebook, outputs committed
- [ ] Final pass: remove dead code, check all notebooks execute, tag `v1.0`
- [ ] Pin repo on GitHub profile; optional LinkedIn/blog post

**Done when:** someone can understand the result in 60 seconds from the README alone.

## M9 — Project documentation

- [x] Decide format: MkDocs Material site on GitHub Pages
- [x] Pages: overview, app reading guide + worked examples, methodology (evaluation protocol, SHAP, physics rules, cost model), results, decisions & limitations
- [x] API reference for `src/pdm` generated from docstrings (`mkdocstrings`)
- [x] How-to pages: reproduce results, train the model, run the app locally
- [x] Build docs in CI (`mkdocs build --strict`), deploy workflow for GitHub Pages
- [ ] Enable GitHub Pages (source: GitHub Actions) once the repo is public; link from README

**Done when:** docs site is live and every public function in `src/pdm` has a docstring shown in the API reference.

## M10 — Optional extension: NASA C-MAPSS (remaining useful life)

- [ ] Separate notebook or repo: RUL regression on turbofan time series (FD001)
- [ ] Windowed features, piecewise-linear RUL target, RMSE + NASA score
- [ ] Link from main README as "harder, real time-series follow-up"

---

**Dataset citation:** Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning
Repository. https://doi.org/10.24432/C5HS5C (CC BY 4.0)
