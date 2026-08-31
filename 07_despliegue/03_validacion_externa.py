"""Evaluate the released artefact against the untouched validation split."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

from api.scoring import scoring_df

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_PATH = PROJECT_ROOT / "02_datos" / "02_Validacion" / "validacion.pkl"
RESULT_DIR = PROJECT_ROOT / "06_resultados" / "Validacion"
DEFAULT_STATUSES = {
    "Charged Off",
    "Does not meet the credit policy. Status:Charged Off",
    "Default",
}

validation = pd.read_pickle(VALIDATION_PATH)
validation = validation.loc[validation["ingresos"].fillna(0) <= 400000].copy()
validation["target_pd"] = validation["estado"].isin(DEFAULT_STATUSES).astype(int)
pending = validation["principal"] - validation["imp_amortizado"]
validation["target_ead"] = (pending / validation["principal"]).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
validation["target_lgd"] = (1 - validation["imp_recuperado"] / pending).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)

predictions = scoring_df(validation)
default_mask = validation["target_pd"] == 1
metrics = {
    "validation_rows": int(len(validation)),
    "default_rows": int(default_mask.sum()),
    "pd_roc_auc": float(roc_auc_score(validation["target_pd"], predictions["score_pd"])),
    "ead_mae_on_defaults": float(mean_absolute_error(validation.loc[default_mask, "target_ead"], predictions.loc[default_mask.to_numpy(), "score_ead"])),
    "lgd_mae_on_defaults": float(mean_absolute_error(validation.loc[default_mask, "target_lgd"], predictions.loc[default_mask.to_numpy(), "score_lgd"])),
}

RESULT_DIR.mkdir(parents=True, exist_ok=True)
(RESULT_DIR / "metricas_validacion_externa.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(RESULT_DIR / "informe_validacion_modelos.md").write_text(
    "# Validación externa del artefacto de scoring\n\n"
    "La evaluación se ejecutó sobre `02_datos/02_Validacion/validacion.pkl`, separado antes del reentrenamiento.\n\n"
    "## Resultados\n\n"
    f"- Registros evaluados: {metrics['validation_rows']}\n"
    f"- Casos de incumplimiento: {metrics['default_rows']}\n"
    f"- PD ROC-AUC: {metrics['pd_roc_auc']:.4f}\n"
    f"- EAD MAE, solo incumplimientos: {metrics['ead_mae_on_defaults']:.4f}\n"
    f"- LGD MAE, solo incumplimientos: {metrics['lgd_mae_on_defaults']:.4f}\n\n"
    "## Interpretación\n\n"
    "Estas métricas muestran comportamiento sobre datos no usados para entrenar. No sustituyen la validación de negocio, el análisis de sesgos, la calibración ni el monitoreo posterior.\n",
    encoding="utf-8",
)

print(json.dumps(metrics, indent=2))
