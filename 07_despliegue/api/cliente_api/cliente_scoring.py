from pathlib import Path
import json
import requests

payload = json.loads((Path(__file__).parent / "payload.json").read_text(encoding="utf-8"))
response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=30)
response.raise_for_status()
print(response.json())
