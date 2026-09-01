# Streamlit Credit-Risk Portfolio App

Demonstrative, accessible client for the canonical `07_despliegue/api` service. It sends raw user-selected values to the API and contains **no ML artifact, scoring logic, underwriting, offer, or policy**.

## Quick path

1. Install the UI dependencies:
   ```powershell
   .venv\Scripts\python.exe -m pip install -r 07_despliegue\app\requirements.txt
   ```
2. In terminal 1, start the API first:
   ```powershell
   cd 07_despliegue\api
   ..\..\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --app-dir ..
   ```
3. In terminal 2, start Streamlit:
   ```powershell
   .venv\Scripts\python.exe -m streamlit run 07_despliegue\app\app.py
   ```

The UI defaults to `http://127.0.0.1:8000`. Set `API_BASE_URL` for a deployed API; the API must be available before starting the UI.

## What the app does

- Analytical dashboard layout: a narrow left filter panel (25%) and a right results area (blue result card, horizontal impact bar, gauge, secondary metrics).
- Exposes 8 editable filters (`principal`, `num_cuotas`, `tipo_interes`, `imp_cuota`, `ingresos`, `dti`, `porc_uso_revolving`, `rating`). The other 6 required contract fields live in `idea/design_spec.json` under `hidden_fields` with their `test_payload.json` values and are merged into every request by `compose_payload_values` (see DEC-005 in `decisions.md`).
- Sends exactly one editable raw record as `[record]` to `POST /predict` after the form submit action.
- Shows only `score_pd`, `score_ead`, `score_lgd`, and `perdida_esperada_relativa`, plus the explanatory `PD × EAD × LGD` text. `perdida_esperada_relativa` is rendered as a percentage on a demonstrative 0–25% scale with bands at 5% (threshold) and 10%; both Plotly charts carry a text equivalent.
- Preserves the last valid response if a later request fails.
- Makes one bounded `/health` warm-up and at most one retry for cold-start-type failures.
- Explores at most six qualifying **ceteris-paribus sensitivity scenarios**. They only lower or preserve `principal` and may change `num_cuotas`; every other raw input (including `imp_cuota`) is fixed.

## Important limitations

The `≤ 0.05` classification is a configurable demonstrative threshold on relative expected loss. It is uncalibrated and must not be interpreted as an approval, offer, pricing recommendation, financial advice, or production policy. The client intentionally does not normalize `tipo_interes` or `num_cuotas`; it visibly warns when their raw convention can drift from training conventions.

## Verification evidence

| Evidence | Command / action | Scope |
|---|---|---|
| Static | `.venv\Scripts\python.exe -m compileall -q 07_despliegue` | Syntax only |
| Dependencies | `.venv\Scripts\python.exe -m pip check` | Installed environment consistency |
| Mocked | `.venv\Scripts\python.exe -m unittest discover -s 07_despliegue\app\tests -v` | Pure core and HTTP boundary behavior |
| AppTest | `.venv\Scripts\python.exe -m unittest discover -s 07_despliegue\app\tests -p test_app.py -v` | Streamlit rendering smoke only |
| Live (optional) | `Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType application/json -InFile 07_despliegue/api/test_payload.json` | Requires a running local API |
| Manual | Check 375, 768, 1024, and 1440px with keyboard-only traversal | Responsive/accessibility evidence; not automated |

Passing static, mocked, or AppTest checks does not prove a live API or browser layout.

## Render deployment sequence

Deploy the canonical API before this app. For the Streamlit service, use root directory `07_despliegue/app`, build command `pip install -r requirements.txt`, start command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`, and Python `3.13.7`. Set `API_BASE_URL` to the public API URL. Render cold starts can add latency; the UI uses bounded recovery and tells users to retry.

Dependencies are pinned in `requirements.txt`: `streamlit==1.62.0`, `requests==2.34.2`, `plotly==7.0.0` (the gauge and impact bar). The blue theme is set in `.streamlit/config.toml` (kept in the app root so it also applies on Render).
