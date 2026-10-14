# PD challengers: XGBoost and LightGBM

## Objective
Find a PD model with better discrimination than the production logistic regression with splines
(external Gini 0.4225, Brier 0.1435), using XGBoost and LightGBM tuned by cross-validation.

## Scope and constraints
- Notebook: `Notebooks/05_1_Modelizacion PD challengers.ipynb` (notebook 05 stays untouched as baseline).
- Interpretability is not a constraint. Deploy (API, Render, `artefacto_pipeline.pkl`) is out of scope.
- Same table, same 70/30 stratified split and same `StratifiedKFold(5, shuffle, 42)` as notebook 05; full train (58,275 rows).
- Same `preprocesador.joblib`, cloned inside each pipeline.
- Main metric: ROC-AUC (reported as Gini). Guardrails: Brier, log loss, decile calibration.
- Model selection by CV only; test and external validation used once, at the end.
- No MLflow (not installed). Results in `06_resultados/Modelizacion/`.
- Deferred: improved logistic, Random Forest, tuned HGB.
- Skill: `ds-09-modelizar`. TDD: off (notebook work, no test runner); checks = notebook executes end to end without errors.
- Environment: `.venv` (Python 3.13.7), lightgbm 4.7.0 installed 2026-10-14 with user approval.

## Tasks
- [x] T1 Setup: imports, data, split, CV, metrics, splines baseline by fold (inline)
- [ ] T2 Validation-curve probe for XGBoost and LightGBM (inline)
- [ ] T3 RandomizedSearchCV for XGBoost (inline)
- [ ] T4 RandomizedSearchCV for LightGBM (inline)
- [ ] T5 Ranking vs splines, per-fold paired comparison -> review stop 1 (inline)
- [ ] T6 Winner: best iteration, out-of-fold calibration, recalibration if needed (inline)
- [ ] T7 Final evaluation on test and external validation + paired bootstrap -> review stop 2 (inline)
- [ ] T8 Persist config JSON, ranking, report, decision entry (inline)

Route: inline, single notebook file built incrementally with executed outputs read back after each step.

## Acceptance criteria
- Every code cell has an explanatory markdown cell before it; key results have a `**Lectura**` with real numbers.
- Notebook executes end to end.
- Ranking table with CV mean/std, train score and train-CV gap for every model.

## Progress
- T1 done: baseline splines reproduced by fold (0.7023/0.7066/0.7131/0.7177/0.7094, mean AUC 0.7098, Brier 0.1449, log loss 0.4536, gap 0.0025). Notebook executes end to end. RDD: off (global).

## Next step
T2.
