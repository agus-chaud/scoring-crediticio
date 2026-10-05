# Preprocessor and PD modelling improvements

## Objective
Apply the 7 transformation improvements (ds-06) to `Notebooks/04_Transformacion de datos.ipynb`
and the 7 modelling improvements to `Notebooks/05_Modelizacion Clasificacion PD.ipynb`.
No class rebalancing (user agreed).

## Problem
- 04 fits encoders/scalers loosely, never persists them, groups rare categories with pandas
  outside any pipeline, uses `unknown_value=12`, MinMax on skewed variables, keeps redundant dummies,
  drops variables without evidence, and has no output validations.
- 05 fits on data already scaled with the full table (validation leakage), splits without seed or
  stratification, ignores `validacion.pkl`, uses a grid whose best C sits on the edge, reports AUC only,
  and its saved outputs come from an older 140k-row table.

## Scope
- In: notebooks 04 and 05, `05_modelos/preprocesador.joblib`, `01_Documentos/Diseño_Transformaciones.md`,
  new raw PD tablón, regenerated PD/EAD/LGD tablones.
- Out: `07_despliegue/*` (own pipeline), variable selection (ds-07), balancing (rejected).

## Constraints
- Preprocessor uses sklearn-native steps only (portable joblib, no custom functions).
- Keep the same 14 raw input features as today; discarded variables are documented, not added.
- Notebooks 06/07 must still run on the regenerated EAD/LGD tablones.

## TDD
Mode: off (no project/session configuration; notebooks have no test runner). Checks: execute notebooks
with `.venv` nbconvert; hard asserts inside the notebook (FASE 4 validations, round-trip).

## Tasks
- [ ] T1 — 04: sklearn preprocessor (OHE drop='first' + min_frequency, ordinal unknown=-1 + StandardScaler,
  Yeo-Johnson on skewed numerics, Binarizer), evidence for discarded variables, FASE 4 validations,
  persisted joblib, design matrix, regenerated tablones. Route: inline (parent already held the full audit context).
- [ ] T2 — 05: raw tablón + cloned preprocessor inside the Pipeline, stratified seeded split + StratifiedKFold,
  external evaluation on `validacion.pkl`, logspace grid, `best_estimator_`, Gini/KS/Brier/calibration with plain
  explanations, chart fixes. Route: inline.
- [ ] T3 — Smoke-run 06 and 07 on the regenerated tablones. Route: inline.

## Acceptance criteria
- 04 executes end to end; all FASE 4 asserts pass; joblib round-trips.
- 05 executes end to end; best C not on the grid edge (or flagged); metrics shown on internal test and `validacion.pkl`.

## Delivery
Strategy: ask-on-risk. Commits per task on `feat/preprocesador-y-modelizacion-pd`. RDD: off (global).

## Progress / evidence
(pending)

## Next step
T1.
