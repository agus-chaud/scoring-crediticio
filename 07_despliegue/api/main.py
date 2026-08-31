import json
import platform
from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from api.schemas import RegistroEntrada, ScoringSalida
from api.scoring import scoring_df

app = FastAPI(title="Credit risk scoring API", version="1.0.0")
BASE_DIR = Path(__file__).resolve().parent


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=list[ScoringSalida])
def predict(records: list[RegistroEntrada]) -> list[dict]:
    frame = pd.DataFrame([record.model_dump(by_alias=True) for record in records])
    return scoring_df(frame).to_dict(orient="records")


@app.get("/debug")
def debug() -> dict:
    try:
        payload = json.loads((BASE_DIR / "test_payload.json").read_text(encoding="utf-8"))
        result = scoring_df(pd.DataFrame(payload))
        return {
            "python_version": platform.python_version(),
            "status": "OK",
            "input_columns_detected": list(payload[0]),
            "output_columns_detected": list(result.columns),
            "sample_output": result.iloc[0].to_dict(),
            "engine_error": None,
        }
    except Exception as error:
        return {"python_version": platform.python_version(), "status": "ERROR", "engine_error": repr(error)}
