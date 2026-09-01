"""Cliente mínimo para consumir la API de scoring crediticio."""

import json
import os
from pathlib import Path

import requests

URL_POR_DEFECTO = "http://127.0.0.1:8000/predict"


def llamar_api_scoring(payload: list[dict], url: str) -> list[dict]:
    """Envía uno o varios casos y devuelve el resultado del scoring crediticio."""
    respuesta = requests.post(url, json=payload, timeout=30)
    respuesta.raise_for_status()
    return respuesta.json()


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    payload = json.loads((base_dir / "payload.json").read_text(encoding="utf-8-sig"))
    url = os.getenv("API_URL", URL_POR_DEFECTO)
    print("Respuesta de la API:")
    print(llamar_api_scoring(payload, url))