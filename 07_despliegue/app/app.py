"""Interfaz Streamlit demostrativa para la API canónica de scoring."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from app_core import (
    APPROVAL_THRESHOLD,
    AppState,
    HttpResult,
    build_payload,
    demonstrative_decision,
    score_action,
    score_record,
    sensitivity_records,
    transition,
    warm_api,
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
SCORE_ENDPOINT = "/predict"
WARMUP_ENDPOINT = "/health"
DEFAULT_TIMEOUT = 12.0
WARMUP_TIMEOUT = 4.0
ENABLE_WARMUP = True
WARMUP_RETRIES = 1
SCORING_RETRIES = 1
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
SPEC_PATH = Path(__file__).parent / "idea" / "design_spec.json"


@st.cache_data(show_spinner=False)
def load_design_spec() -> dict[str, Any]:
    """Cachea sólo metadatos estáticos; nunca resultados ni respuestas HTTP."""
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def label_for(field: str) -> str:
    return field.replace("_", " ").capitalize()


def render_field(name: str, config: dict[str, Any]) -> Any:
    label = label_for(name)
    help_text = config.get("semantic_warning")
    default = config.get("default")
    if config["widget"] == "selectbox":
        return st.selectbox(label, config["options"], index=config["options"].index(default), help=help_text)
    if config["type"] == "integer":
        return st.text_input(label, value="" if default is None else str(default), help=help_text)
    minimum = float(config.get("min", 0))
    return st.number_input(label, min_value=minimum, value=float(default), help=help_text)


def get_state() -> AppState:
    raw = st.session_state.get("credit_risk_state")
    if isinstance(raw, AppState):
        return raw
    state = AppState()
    st.session_state.credit_risk_state = state
    return state


def set_state(state: AppState) -> None:
    st.session_state.credit_risk_state = state


def render_results(result: dict[str, float], spec: dict[str, Any], last_valid: bool = False) -> None:
    if last_valid:
        st.info("Mostrando el último resultado válido; el último envío no pudo completarse.")
    st.subheader("Resultado demostrativo")
    names = {"score_pd": "PD", "score_ead": "EAD", "score_lgd": "LGD", "perdida_esperada_relativa": "Pérdida esperada relativa"}
    cols = st.columns(2)
    for index, metric in enumerate(spec["result_view"]["metrics"]):
        cols[index % 2].metric(names[metric], f"{result[metric]:.4f}")
    st.caption(f"Descomposición informativa: PD × EAD × LGD = {result['score_pd']:.4f} × {result['score_ead']:.4f} × {result['score_lgd']:.4f}.")
    qualifies, decision = demonstrative_decision(result, APPROVAL_THRESHOLD)
    (st.success if qualifies else st.warning)(f"{decision} (pérdida relativa ≤ {APPROVAL_THRESHOLD:.2f}).")
    st.caption(spec["copy"]["disclaimer"])


def render_scenarios(record: dict[str, Any], spec: dict[str, Any]) -> None:
    st.subheader("Sensibilidad")
    st.caption(spec["scenarios"]["label"])
    if not st.button("Explorar escenarios", type="secondary"):
        return
    qualifying: list[dict[str, Any]] = []
    # El warm-up se ejecuta una vez por acción, no por cada variante.
    if ENABLE_WARMUP:
        warm_api(f"{API_BASE_URL.rstrip('/')}{WARMUP_ENDPOINT}", WARMUP_TIMEOUT)
    for candidate in sensitivity_records(record, spec["scenarios"]["principal_factors"], spec["scenarios"]["terms"]):
        # Cada escenario es un POST nuevo: no se cachean predicciones ni decisiones.
        response = score_record(f"{API_BASE_URL.rstrip('/')}{SCORE_ENDPOINT}", candidate, DEFAULT_TIMEOUT)
        if response.ok and response.data:
            qualifies, _ = demonstrative_decision(response.data, APPROVAL_THRESHOLD)
            if qualifies:
                qualifying.append({"principal": candidate["principal"], "num_cuotas": candidate["num_cuotas"], "perdida_esperada_relativa": response.data["perdida_esperada_relativa"]})
        if len(qualifying) == spec["scenarios"]["max_qualifying"]:
            break
    if qualifying:
        st.dataframe(qualifying, hide_index=True, use_container_width=True)
        st.caption("Sólo cambian principal y num_cuotas; imp_cuota y el resto se mantienen fijos. No son ofertas ni recomendaciones.")
    else:
        st.info("No hubo escenarios calificables dentro del límite demostrativo. No se infiere una recomendación.")


def main() -> None:
    spec = load_design_spec()
    st.set_page_config(page_title=spec["copy"]["title"], page_icon="📊", layout="wide")
    st.markdown("""<style>
    .stApp { background: #FAF5FF; color: #0F172A; }
    [data-testid="stForm"] { background: #FFFFFF; border: 1px solid #DDD6FE; border-radius: 12px; padding: 1rem; }
    button:focus, input:focus, [role="combobox"]:focus { outline: 3px solid #EA580C !important; outline-offset: 2px; }
    @media (max-width: 768px) { [data-testid="stHorizontalBlock"] { flex-direction: column; } }
    </style>""", unsafe_allow_html=True)
    st.title(spec["copy"]["title"])
    st.caption(spec["copy"]["subtitle"])
    st.warning(spec["copy"]["drift_warning"], icon="⚠️")
    st.caption("La primera acción puede demorar mientras la API se activa. Editar campos no envía solicitudes.")

    values: dict[str, Any] = {}
    with st.form("credit-risk-form", clear_on_submit=False):
        left, right = st.columns(2)
        for index, (name, config) in enumerate(spec["fields"].items()):
            with (left if index % 2 == 0 else right):
                values[name] = render_field(name, config)
        submitted = st.form_submit_button("Calcular riesgo demostrativo", type="primary", use_container_width=True)

    state = get_state()
    if submitted:
        try:
            record = build_payload(values)
            set_state(transition(state, "submit"))
            response = score_action(API_BASE_URL, record, WARMUP_TIMEOUT, DEFAULT_TIMEOUT) if ENABLE_WARMUP else score_record(f"{API_BASE_URL.rstrip('/')}{SCORE_ENDPOINT}", record, DEFAULT_TIMEOUT)
            state = transition(state, "success" if response.ok else "error", response, record)
            set_state(state)
        except ValueError as error:
            state = transition(state, "error", HttpResult(False, error_message=str(error)))
            set_state(state)

    state = get_state()
    if state.status == "error":
        st.error(f"No pudimos actualizar el resultado. Revisá que la API esté activa y reintentá. Detalle: {state.error_message}")
    if state.last_valid_result:
        render_results(state.last_valid_result, spec, last_valid=state.status == "error")
        if state.submitted_payload:
            render_scenarios(state.submitted_payload, spec)
    if DEBUG_MODE and state.error_message:
        st.code(state.error_message, language="text")


if __name__ == "__main__":
    main()
