"""Thin, import-safe scoring engine extracted from the production script."""

from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTEFACT_PATH = BASE_DIR / "artefacto_pipeline.pkl"

with ARTEFACT_PATH.open("rb") as stream:
    ARTEFACT = cloudpickle.load(stream)


def scoring_df(df: pd.DataFrame) -> pd.DataFrame:
    """Score raw records and always return a DataFrame."""
    models = ARTEFACT["models"]
    result = pd.DataFrame(index=df.index)
    if "id_cliente" in df.columns:
        result["id_cliente"] = df["id_cliente"].values
    result["score_pd"] = models["pd"].predict_proba(df)[:, 1]
    result["score_ead"] = np.clip(models["ead"].predict(df), 0, 1)
    result["score_lgd"] = np.clip(models["lgd"].predict(df), 0, 1)
    result["perdida_esperada_relativa"] = result["score_pd"] * result["score_ead"] * result["score_lgd"]
    return result.reset_index(drop=True)
