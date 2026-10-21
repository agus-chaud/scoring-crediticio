# Guarded LLM scoring explainer

Objective: add a non-technical, grounded explanation chat to the canonical Streamlit credit-risk client.
Branch: `feat/preprocesador-y-modelizacion-pd` · delivery strategy: `ask-on-risk` · runner: `.venv\\Scripts\\python.exe -m unittest discover -s 07_despliegue\\app\\tests -v`

## Specs

S1. The explainer must answer: “Explicame esto como si no conociera finanzas.”, “¿Qué diferencia hay entre riesgo alto y pérdida alta?”, “¿Qué debería verificar antes de usar este resultado?”, and “¿Cuál métrica está aumentando más la pérdida esperada?”

S2. It must “explique porque no se anima a recomendar aprobar ni rechazar” because “no tenemos los costos ni margenes asociados a los tipos de error”.

S3. It must “mostrar siempre que es una estimacion y no una politica crediticia”, explain possible consequences of the metrics currently returned by the model, and never invent a decision.

S4. It must “agregale Evals y guardrails” and “ignora por ahora el mcp”.

S5. Any economic break-even calculation must use explicit, auditable business inputs; it cannot present invented X/Y values as real.

## Tasks

- T1 · S1-S5 · inline (large tracking: cross-session requirements; Risk: item 2 — sends model-score context to an external LLM and handles a secret) · in progress · RED/GREEN pending

## Log

- L1 · 2026-10-21 · usá gentle ai y ODD para implementar esto
- L2 · 2026-10-21 · Scope inherited from prior messages: implement a Streamlit chat that explains current scoring metrics to non-technical users, explains why it cannot approve/reject without business costs and margins, answers the four named questions, and includes evals/guardrails; MCP is excluded.
- L3 · 2026-10-21 · Evidence: canonical app is `07_despliegue/app`; it calls the canonical API and displays PD/EAD/LGD/relative expected loss. The current result labels are demonstrative only. OpenAI Python v2.11.0 official docs confirm a Responses API client can read `OPENAI_API_KEY` and produce structured outputs.
- L4 · 2026-10-21 · Design decision: deterministic fact-building and guardrails remain local; the LLM receives only a sanitized score context plus approved business assumptions and returns a typed explanation. No secrets or external calls occur when the key is absent.
