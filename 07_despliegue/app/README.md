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
- Exposes 9 editable filters (`principal`, `num_cuotas`, `tipo_interes`, `imp_cuota`, `ingresos`, `dti`, `porc_uso_revolving`, `rating`, `sector_empleo`). The other 6 required contract fields live in `idea/design_spec.json` under `hidden_fields` with their `test_payload.json` values and are merged into every request by `compose_payload_values` (see DEC-005 in `decisions.md`).
- Sends exactly one editable raw record as `[record]` to `POST /predict` after the form submit action.
- Shows only `score_pd`, `score_ead`, `score_lgd`, and `perdida_esperada_relativa`, plus the explanatory `PD × EAD × LGD` text. `perdida_esperada_relativa` is rendered as a percentage on a demonstrative 0–25% scale with bands at 5% (threshold) and 10%; both Plotly charts carry a text equivalent.
- Preserves the last valid response if a later request fails.
- Makes one bounded `/health` warm-up and at most one retry for cold-start-type failures.
- Explores at most six qualifying **ceteris-paribus sensitivity scenarios**. They only lower or preserve `principal` and may change `num_cuotas`; every other raw input (including `imp_cuota`) is fixed.
- Includes a grounded chat that explains the current PD, EAD, LGD and relative expected loss in plain language. It never scores again, uses no applicant payload or `id_cliente`, and always states that the result is an estimate rather than credit policy.

## Important limitations

The `≤ 0.05` classification is a configurable demonstrative threshold on relative expected loss. It is uncalibrated and must not be interpreted as an approval, offer, pricing recommendation, financial advice, or production policy. The client intentionally does not normalize `tipo_interes` or `num_cuotas`; it visibly warns when their raw convention can drift from training conventions.

The explainer cannot recommend approval or rejection because the project has no authoritative net margin, default-cost, funding, operating, collection, capital or risk-appetite inputs. It explains that limit rather than inventing a threshold. A future break-even calculation must obtain those values from a reviewed business source.

## Optional LLM configuration and evals

The chat remains usable with deterministic guarded answers when `ANTHROPIC_API_KEY` is absent. To enable the Anthropic-backed explanation locally, write the key in `07_despliegue/app/.env`; the file is ignored by Git and is loaded without overriding server environment values:

```dotenv
ANTHROPIC_API_KEY=your-key
ANTHROPIC_MODEL=claude-sonnet-5-5
```

In deployment, configure the same values in the host's secret store. Never commit the key.

The provider receives a sanitized context containing only the four score metrics and fixed policy text. It has no tool access, cannot see the raw loan payload, and its answer falls back to the local explanation if it violates the safety contract. The application adds the exact metrics and mandatory disclaimer deterministically before validation.

Every local fallback carries a `fallback_reason` and writes one `WARNING` line from the `explainer` logger to the Streamlit server output (exception class, HTTP status and model only; never the key or the payload):

| `fallback_reason` | Cause | Notice shown to the user |
|---|---|---|
| `not_configured` | `ANTHROPIC_API_KEY` missing or blank | Generative assistant unavailable; validated local explanation shown |
| `provider_error` | Network, authentication, model or SDK error | Generative assistant unavailable; validated local explanation shown |
| `incomplete` | Provider stopped with `stop_reason` `max_tokens` (truncated answer) or `refusal` | Generative assistant unavailable; validated local explanation shown |
| `guardrail` | Generated percentage, credit recommendation, amount or economic assumption | Generative explanation could not be used safely; validated local explanation shown |

Set `DEBUG_MODE=true` to also show the reason code in the chat. If Streamlit is launched from a sandboxed agent shell with outbound network disabled (for example `CODEX_SANDBOX_NETWORK_DISABLED=1`), every question ends in `provider_error` even with a valid key: start Streamlit from a regular terminal.

Regression eval cases live in `evals/scoring_explainer_cases.json`. Run them with:

```powershell
Set-Location 07_despliegue\app
..\..\.venv\Scripts\python.exe evals\run_scoring_explainer_evals.py
```

## Verification evidence

| Evidence | Command / action | Scope |
|---|---|---|
| Static | `.venv\Scripts\python.exe -m compileall -q 07_despliegue` | Syntax only |
| Dependencies | `.venv\Scripts\python.exe -m pip check` | Installed environment consistency |
| Mocked | `.venv\Scripts\python.exe -m unittest discover -s 07_despliegue\app\tests -v` | Pure core and HTTP boundary behavior |
| AppTest | `.venv\Scripts\python.exe -m unittest discover -s 07_despliegue\app\tests -p test_app.py -v` | Streamlit rendering smoke only |
| Explainer evals | `.venv\Scripts\python.exe 07_despliegue\app\evals\run_scoring_explainer_evals.py` | Grounding, no-decision boundary and four named questions; calls the provider only if its server key is configured |
| Live (optional) | `Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType application/json -InFile 07_despliegue/api/test_payload.json` | Requires a running local API |
| Manual | Check 375, 768, 1024, and 1440px with keyboard-only traversal | Responsive/accessibility evidence; not automated |

Passing static, mocked, or AppTest checks does not prove a live API or browser layout.

## Render deployment sequence

Deploy the canonical API before this app. For the Streamlit service, use root directory `07_despliegue/app`, build command `pip install -r requirements.txt`, start command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`, and Python `3.13.7`. Set `API_BASE_URL` to the public API URL. Render cold starts can add latency; the UI uses bounded recovery and tells users to retry.

Dependencies are pinned in `requirements.txt`: `streamlit==1.62.0`, `requests==2.34.2`, `plotly==7.0.0` (the gauge and impact bar). The blue theme is set in `.streamlit/config.toml` (kept in the app root so it also applies on Render).
