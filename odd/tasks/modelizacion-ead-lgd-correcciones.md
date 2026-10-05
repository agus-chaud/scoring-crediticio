# EAD and LGD modelling corrections

## Objective
Apply the six "red severity" fixes from the ds-09-modelizar review to
`Notebooks/06_Modelizacion Regresion EAD.ipynb` and `Notebooks/07_Modelizacion Regresion LGD.ipynb`.

## Problem
1. The final model is refit with hardcoded hyperparameters that differ from the grid winner.
2. `train_test_split` has no seed and `cv=3` does not shuffle: results change on every run.
3. No baseline: an MAE cannot be judged without the MAE of always predicting the median.
4. EAD/LGD tablones are already transformed by a preprocessor fitted on the whole training set
   (notebook 04, `preprocesador.fit_transform(x_raw)`), so the internal validation split leaks scaling
   statistics. Ridge/Lasso also run without a scaler and with a linear alpha grid of 0.1-1.
5. HGB uses `scoring='neg_mean_absolute_percentage_error'` for its automatic early stopping (MAPE explodes
   near 0) while the search optimises MAE, and it trains with squared error instead of absolute error.
6. Residuals are plotted against the real value, which always shows a positive slope by construction.

## Scope
- In: notebooks 04 (export raw EAD/LGD tablones only), 06, 07; new raw tablones
  `df_tablon_ead_sin_transformar.pkl` and `df_tablon_lgd_sin_transformar.pkl`.
- Out: yellow/green review items (more algorithms, RandomizedSearchCV, multi-metric ranking, two-stage LGD,
  external validation for EAD/LGD), notebook 05, `08_Preproduccion.ipynb`, `07_despliegue/*`.

## Constraints
- Same preprocessor contract as notebook 05: `clone(joblib.load('../05_modelos/preprocesador.joblib'))` inside the Pipeline.
- Notebook prose stays in Spanish (existing project language), neutral register; readings use real executed numbers.
- Re-running 04 must keep `preprocesador.joblib` and existing tablones equivalent (deterministic fit).

## TDD
Mode: off (no project/session configuration; notebooks have no test runner). Checks: execute each notebook
end to end with `.venv` nbconvert; asserts inside the notebook where useful.

## Tasks
- [x] T1 — 04: export raw (untransformed) EAD and LGD tablones, defaults only, same index and targets.
  Route: delegated (writer trigger: 3 non-trivial notebooks).
- [x] T2 — 06 EAD: fixes 1-6, execute, rewrite readings from real output. Route: delegated.
- [x] T3 — 07 LGD: fixes 1-6, execute, rewrite readings from real output. Route: delegated.

## Acceptance criteria
- 04, 06, 07 execute end to end without errors.
- 06/07 final model is `grid_search.best_estimator_`; split and CV are seeded and shuffled (5 folds).
- 06/07 show model MAE next to a `DummyRegressor(strategy='median')` baseline on the same split.
- 06/07 Pipeline: cloned preprocessor + scaler for Ridge/Lasso (passthrough for HGB); alpha in logspace.
- HGB has no MAPE scoring; uses `loss='absolute_error'`.
- Residual plot is residual vs predicted; markdown readings match outputs.

## Progress
- Engram mirror: pending (Engram session unavailable at creation).
- T1 done. Executed `04_Transformacion de datos.ipynb` end to end (`.venv` nbconvert); its 7 existing
  asserts plus the new raw/transformado index+target assert passed (`PD: (83250, 41) | EAD: (16568, 41) |
  LGD: (16568, 41)`). Confirmed `df_tablon_ead.pkl`, `df_tablon_lgd.pkl`, `df_tablon_pd.pkl`,
  `df_tablon_pd_sin_transformar.pkl` and `preprocesador.joblib` are content-equal to the pre-change version
  (`pd.testing.assert_frame_equal` against a backup copy — all `EQUAL`), confirming deterministic refit.
  New files `df_tablon_ead_sin_transformar.pkl` and `df_tablon_lgd_sin_transformar.pkl` written to
  `02_datos/03_Entrenamiento/`. None of the pkls under `02_datos/` are git-tracked, so only the notebook is
  committed.

- T1 commit: `cdafa9b`.
- T2 done. `06` executed end to end with `.venv` nbconvert (exit 0, no error outputs; the joblib
  `resource_tracker` KeyError in stderr is Windows temp-folder cleanup after completion). Winner HGB
  (`loss='absolute_error'`, lr 0.05, depth 10, 200 iter, l2 0.25), MAE CV 0.154 ± 0.002; internal validation
  MAE 0.159 vs median baseline 0.177 (-9.8%). Residual vs predicted: no slope (corr 0.03), dispersion drops
  from 0.23 to 0.16 across prediction terciles; predictions span only 0.55-0.95.
- T3 done. `07` executed end to end (same check). Winner HGB (lr 0.01, depth 5, 200 iter, l2 0.75),
  MAE CV 0.089 ± 0.003; internal validation MAE 0.087 vs median baseline 0.088 (-0.6%): the model barely
  beats the baseline. Predictions span 0.88-0.92 (p5-p95) vs real 0.57-1.00.
- Route evidence: T2/T3 code written by the delegated writer; executions and readings finished by the parent
  after the writer's background runs died with its session.

## Next step
Done. Follow-up (not authorized yet): LGD needs better features or a two-stage approach; yellow review items.
