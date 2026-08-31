"""Score raw loan applications using the released PD, EAD and LGD artefact."""

import argparse
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTEFACT_PATH = BASE_DIR / "artefacto_pipeline.pkl"


with ARTEFACT_PATH.open("rb") as stream:
    ARTEFACT = cloudpickle.load(stream)


def scoring_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return one PD, EAD, LGD and expected-loss estimate per input record."""
    models = ARTEFACT["models"]
    result = pd.DataFrame(index=df.index)
    if "id_cliente" in df.columns:
        result["id_cliente"] = df["id_cliente"].values
    result["score_pd"] = models["pd"].predict_proba(df)[:, 1]
    result["score_ead"] = np.clip(models["ead"].predict(df), 0, 1)
    result["score_lgd"] = np.clip(models["lgd"].predict(df), 0, 1)
    result["perdida_esperada_relativa"] = result["score_pd"] * result["score_ead"] * result["score_lgd"]
    return result.reset_index(drop=True)


parser = argparse.ArgumentParser(description="Score raw credit-risk records.")
parser.add_argument("--input", required=True, help="CSV with raw loan records")
parser.add_argument("--output", required=True, help="CSV path for scoring results")
args = parser.parse_args()

input_df = pd.read_csv(args.input)
scoring_df(input_df).to_csv(args.output, index=False)
