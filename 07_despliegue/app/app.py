"""Interfaz Streamlit demostrativa para la API canónica de scoring.

Composición de tablero analítico: panel de filtros angosto a la izquierda y
área de resultados (tarjeta principal, barra de impacto, indicador y métricas
secundarias) a la derecha. La app no contiene lógica de ML: sólo transporta
valores crudos hacia `POST /predict`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from app_core import (
    APPROVAL_THRESHOLD,
    AppState,
    HttpResult,
    build_payload,
    compose_payload_values,
    demonstrative_decision,
    risk_band,
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

BAND_COLORS = {"within": "#15803D", "near": "#B45309", "outside": "#B91C1C"}
BAND_FILLS = {"within": "#DCFCE7", "near": "#FEF9C3", "outside": "#FEE2E2"}
BAND_READING = {
    "within": "La pérdida estimada quedó por debajo del 5% de referencia.",
    "near": "La pérdida estimada quedó entre el 5% y el 10% de referencia.",
    "outside": "La pérdida estimada superó el 10% de referencia.",
}
BANDS_LEGEND = "hasta 5% riesgo bajo, entre 5% y 10% riesgo medio, más de 10% riesgo alto"
PRIMARY = "#1E40AF"


@st.cache_data(show_spinner=False)
def load_design_spec() -> dict[str, Any]:
    """Cachea sólo metadatos estáticos; nunca resultados ni respuestas HTTP."""
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def inject_styles() -> None:
    st.markdown(
        """<style>
    .stApp { background: #F8FAFC; color: #1E293B; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
    [data-testid="stColumn"] { min-width: 0; }
    [data-testid="stForm"] { background: #F1F5F9; border: 1px solid #E2E8F0; border-radius: 14px; padding: 1.1rem 1.15rem; }
    [data-testid="stForm"] label p, [data-testid="stWidgetLabel"] p { color: #1E293B !important; font-weight: 500; }
    [data-testid="stForm"] h4 { margin: 0 0 .4rem; color: #1E293B; }
    .result-card { background: #1E40AF; color: #FFFFFF; border-radius: 16px; padding: 1.6rem 1.8rem; box-shadow: 0 1px 3px rgba(15,23,42,.12); }
    .result-pill { display: inline-block; font-weight: 600; font-size: .82rem; padding: .28rem .7rem; border-radius: 999px; color: #FFFFFF; }
    .result-pill--within { background: #15803D; }
    .result-pill--near { background: #B45309; }
    .result-pill--outside { background: #B91C1C; }
    .result-figure { font-size: 2.9rem; font-weight: 700; line-height: 1.1; margin: .55rem 0 .1rem; }
    .result-label { font-size: 1rem; opacity: .92; }
    .result-reading { font-size: .9rem; opacity: .95; margin-top: .5rem; }
    .result-note { font-size: .82rem; opacity: .78; margin-top: .7rem; }
    .risk-empty { background: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 14px; padding: 2rem 1.5rem; color: #475569; text-align: center; }
    button:focus-visible, input:focus-visible, [role="combobox"]:focus-visible, [role="slider"]:focus-visible { outline: 3px solid #1E40AF !important; outline-offset: 2px; }
    @media (max-width: 900px) {
      [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex: 1 1 100% !important; width: 100% !important; min-width: 100% !important; }
    }
    </style>""",
        unsafe_allow_html=True,
    )


def render_filter_field(config: dict[str, Any]) -> Any:
    label = config["label"]
    help_text = config.get("help")
    widget = config["widget"]
    if widget == "radio":
        options = config["options"]
        return st.radio(label, options, index=options.index(config["default"]), help=help_text, horizontal=True)
    if widget == "selectbox":
        options = config["options"]
        return st.selectbox(label, options, index=options.index(config["default"]), help=help_text)
    numeric = [config["min"], config["max"], config["step"], config["default"]]
    if any(isinstance(value, float) for value in numeric):
        low, high, step, default = (float(value) for value in numeric)
    else:
        low, high, step, default = (int(value) for value in numeric)
    return st.slider(label, min_value=low, max_value=high, value=default, step=step, format=config.get("format"), help=help_text)


def get_state() -> AppState:
    raw = st.session_state.get("credit_risk_state")
    if isinstance(raw, AppState):
        return raw
    state = AppState()
    st.session_state.credit_risk_state = state
    return state


def set_state(state: AppState) -> None:
    st.session_state.credit_risk_state = state


def render_result_card(result: dict[str, float], spec: dict[str, Any]) -> float:
    scale = spec["scale"]
    relative = result["perdida_esperada_relativa"]
    pct = relative * 100
    key, label = risk_band(relative, scale["threshold"], scale["warn"])
    st.markdown(
        f"""<div class="result-card">
  <div class="result-pill result-pill--{key}">{label}</div>
  <div class="result-figure">{pct:.1f}%</div>
  <div class="result-label">{spec['copy']['result_title']}</div>
  <div class="result-reading">{BAND_READING[key]}</div>
  <div class="result-note">{spec['copy']['result_note']}</div>
</div>""",
        unsafe_allow_html=True,
    )
    return pct


def render_impact_bar(pct: float, spec: dict[str, Any]) -> None:
    scale = spec["scale"]
    axis_max = scale["max"] * 100
    threshold = scale["threshold"] * 100
    key, label = risk_band(pct / 100, scale["threshold"], scale["warn"])
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[min(pct, axis_max)],
            y=["Pérdida"],
            orientation="h",
            width=0.5,
            marker_color=BAND_COLORS[key],
            hovertemplate="%{x:.1f}%<extra></extra>",
        )
    )
    fig.add_shape(type="line", x0=threshold, x1=threshold, y0=-0.5, y1=0.5, line=dict(color=PRIMARY, width=2, dash="dot"))
    fig.add_annotation(x=threshold, y=0.62, text="5%", showarrow=False, font=dict(size=11, color=PRIMARY))
    fig.update_layout(
        height=120,
        margin=dict(l=8, r=8, t=8, b=8),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, axis_max], ticksuffix="%", fixedrange=True, gridcolor="#E2E8F0"),
        yaxis=dict(visible=False, fixedrange=True),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"Texto equivalente: la pérdida esperada relativa es {pct:.1f}% sobre una escala "
        f"de referencia de 0 a {axis_max:.0f}%. La línea punteada marca el 5%: {BANDS_LEGEND}."
    )


def render_gauge(pct: float, spec: dict[str, Any]) -> None:
    scale = spec["scale"]
    axis_max = scale["max"] * 100
    threshold = scale["threshold"] * 100
    warn = scale["warn"] * 100
    key, label = risk_band(pct / 100, scale["threshold"], scale["warn"])
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=min(pct, axis_max),
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 32, "color": "#1E293B"}},
            gauge={
                "axis": {"range": [0, axis_max], "ticksuffix": "%"},
                "bar": {"color": BAND_COLORS[key], "thickness": 0.28},
                "bgcolor": "#FFFFFF",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, threshold], "color": BAND_FILLS["within"]},
                    {"range": [threshold, warn], "color": BAND_FILLS["near"]},
                    {"range": [warn, axis_max], "color": BAND_FILLS["outside"]},
                ],
                "threshold": {"line": {"color": PRIMARY, "width": 3}, "value": threshold},
            },
        )
    )
    fig.update_layout(height=240, margin=dict(l=16, r=16, t=16, b=8), paper_bgcolor="rgba(0,0,0,0)", font={"color": "#1E293B"})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"Texto equivalente: indicador de pérdida esperada relativa en {pct:.1f}% — {label}. "
        f"Referencia demostrativa: {BANDS_LEGEND}."
    )


def render_secondary_metrics(result: dict[str, float], spec: dict[str, Any]) -> None:
    first, second, third = st.columns(3)
    first.metric("PD (probabilidad de default)", f"{result['score_pd']:.1%}")
    second.metric("EAD (exposición al default)", f"{result['score_ead']:.1%}")
    third.metric("LGD (pérdida dado el default)", f"{result['score_lgd']:.1%}")
    st.caption(
        f"Descomposición informativa: {spec['result_view']['formula']} = "
        f"{result['score_pd']:.3f} × {result['score_ead']:.3f} × {result['score_lgd']:.3f} = "
        f"{result['perdida_esperada_relativa']:.3f}."
    )


def render_scenarios(record: dict[str, Any], spec: dict[str, Any]) -> None:
    st.subheader(spec["copy"]["scenarios_heading"])
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
                qualifying.append(
                    {
                        "principal": candidate["principal"],
                        "num_cuotas": candidate["num_cuotas"],
                        "perdida_esperada_relativa": response.data["perdida_esperada_relativa"],
                    }
                )
        if len(qualifying) == spec["scenarios"]["max_qualifying"]:
            break
    if qualifying:
        st.dataframe(qualifying, hide_index=True, use_container_width=True)
        st.caption("Sólo cambian principal y num_cuotas; imp_cuota y el resto se mantienen fijos. No son ofertas ni recomendaciones.")
    else:
        st.info("No hubo escenarios calificables dentro del límite demostrativo. No se infiere una recomendación.")


def main() -> None:
    spec = load_design_spec()
    st.set_page_config(page_title=spec["copy"]["title"], layout="wide")
    inject_styles()
    st.title(spec["copy"]["title"])
    st.caption("La primera consulta puede demorar unos segundos mientras la API se activa. Editar filtros no envía solicitudes.")

    layout = spec["layout"]
    left, right = st.columns([layout["filter_ratio"], layout["content_ratio"]], gap="large")

    visible_values: dict[str, Any] = {}
    with left:
        with st.form("credit-risk-form", clear_on_submit=False):
            st.markdown(f"#### {spec['copy']['panel_title']}")
            for name, config in spec["visible_fields"].items():
                visible_values[name] = render_filter_field(config)
            submitted = st.form_submit_button(spec["copy"]["primary_button"], type="primary", use_container_width=True)

    state = get_state()
    if submitted:
        try:
            record = build_payload(compose_payload_values(visible_values, spec["hidden_fields"]))
            set_state(transition(state, "submit"))
            with st.spinner("Consultando la API…"):
                if ENABLE_WARMUP:
                    response = score_action(API_BASE_URL, record, WARMUP_TIMEOUT, DEFAULT_TIMEOUT)
                else:
                    response = score_record(f"{API_BASE_URL.rstrip('/')}{SCORE_ENDPOINT}", record, DEFAULT_TIMEOUT)
            state = transition(state, "success" if response.ok else "error", response, record)
            set_state(state)
        except ValueError as error:
            state = transition(state, "error", HttpResult(False, error_message=str(error)))
            set_state(state)

    state = get_state()
    with left:
        if state.status == "error":
            st.error(
                f"No pudimos actualizar el resultado. Revisá que la API esté activa y volvé a tocar "
                f"«{spec['copy']['primary_button']}». Detalle: {state.error_message}"
            )

    with right:
        if not state.last_valid_result:
            st.markdown(f'<div class="risk-empty">{spec["copy"]["empty_state"]}</div>', unsafe_allow_html=True)
        else:
            if state.status == "error":
                st.info("Mostramos el último resultado válido; el último envío no pudo completarse.")
            pct = render_result_card(state.last_valid_result, spec)
            st.subheader(spec["copy"]["impact_heading"])
            render_impact_bar(pct, spec)
            st.subheader(spec["copy"]["metrics_heading"])
            render_gauge(pct, spec)
            render_secondary_metrics(state.last_valid_result, spec)
            st.caption(spec["copy"]["disclaimer"])
            if state.submitted_payload:
                st.markdown("---")
                render_scenarios(state.submitted_payload, spec)

    if DEBUG_MODE and state.error_message:
        st.code(state.error_message, language="text")


if __name__ == "__main__":
    main()
