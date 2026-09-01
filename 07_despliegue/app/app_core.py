"""Núcleo puro para el cliente demostrativo de riesgo crediticio.

No carga artefactos de ML ni reproduce el scoring: únicamente transporta datos
sin normalizarlos hacia la API canónica.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from typing import Any, Callable, Iterable, Mapping, TypedDict

import requests

APPROVAL_THRESHOLD = 0.05
CANONICAL_METRICS = (
    "score_pd",
    "score_ead",
    "score_lgd",
    "perdida_esperada_relativa",
)
REQUIRED_FIELDS = (
    "ingresos_verificados", "vivienda", "finalidad", "num_cuotas",
    "antigüedad_empleo", "rating", "ingresos", "dti", "num_lineas_credito",
    "porc_uso_revolving", "principal", "tipo_interes", "imp_cuota",
    "num_derogatorios",
)


class InputRecord(TypedDict, total=False):
    id_cliente: int
    ingresos_verificados: str
    vivienda: str
    finalidad: str
    num_cuotas: str
    antigüedad_empleo: str
    rating: str
    ingresos: float
    dti: float
    num_lineas_credito: float
    porc_uso_revolving: float
    principal: float
    tipo_interes: float
    imp_cuota: float
    num_derogatorios: float


class ScoreResult(TypedDict):
    score_pd: float
    score_ead: float
    score_lgd: float
    perdida_esperada_relativa: float


@dataclass(frozen=True)
class HttpResult:
    ok: bool
    status_code: int | None = None
    data: ScoreResult | None = None
    error_message: str | None = None
    raw_text: str | None = None


@dataclass(frozen=True)
class AppState:
    status: str = "idle"
    last_valid_result: ScoreResult | None = None
    submitted_payload: InputRecord | None = None
    error_message: str | None = None


def build_payload(values: Mapping[str, Any]) -> InputRecord:
    """Construye el registro sin recortar, mapear ni escalar valores del usuario."""
    missing = [field for field in REQUIRED_FIELDS if field not in values]
    if missing:
        raise ValueError(f"Faltan campos requeridos: {', '.join(missing)}")
    record: InputRecord = {field: values[field] for field in REQUIRED_FIELDS}  # type: ignore[misc]
    if values.get("id_cliente") not in (None, ""):
        record["id_cliente"] = int(values["id_cliente"])
    return record


def parse_score_response(response: Any, status_code: int = 200) -> HttpResult:
    """Acepta sólo la primera respuesta válida de la lista definida por la API."""
    if not isinstance(response, list) or not response:
        return HttpResult(False, status_code, error_message="La API respondió una lista vacía o inválida.")
    first = response[0]
    if not isinstance(first, Mapping):
        return HttpResult(False, status_code, error_message="La primera respuesta de la API no es un objeto.")
    try:
        data: ScoreResult = {metric: float(first[metric]) for metric in CANONICAL_METRICS}
    except (KeyError, TypeError, ValueError):
        return HttpResult(False, status_code, error_message="La respuesta no contiene las cuatro métricas canónicas.")
    return HttpResult(True, status_code, data=data)


def _retryable_status(status_code: int) -> bool:
    return status_code in (502, 503, 504)


def warm_api(
    health_url: str,
    timeout: float,
    get: Callable[..., Any] = requests.get,
    pause: Callable[[float], None] = sleep,
) -> HttpResult:
    """Hace como máximo dos intentos de warm-up y no bloquea la carga de página."""
    last_error = "La API no respondió al warm-up."
    for attempt in range(2):
        try:
            response = get(health_url, timeout=timeout)
            if response.status_code in (200, 204):
                return HttpResult(True, response.status_code)
            last_error = f"Warm-up HTTP {response.status_code}."
            if not _retryable_status(response.status_code):
                break
        except (requests.Timeout, requests.ConnectionError) as error:
            last_error = f"No se pudo conectar durante el warm-up: {error}."
        if attempt == 0:
            pause(0.25)
    return HttpResult(False, error_message=last_error)


def score_record(
    score_url: str,
    record: InputRecord,
    timeout: float,
    post: Callable[..., Any] = requests.post,
    pause: Callable[[float], None] = sleep,
) -> HttpResult:
    """Envía exactamente ``[record]`` y reintenta sólo fallas transitorias una vez."""
    for attempt in range(2):
        try:
            response = post(score_url, json=[record], timeout=timeout)
            raw_text = getattr(response, "text", None)
            if response.status_code != 200:
                if _retryable_status(response.status_code) and attempt == 0:
                    pause(0.25)
                    continue
                return HttpResult(False, response.status_code, error_message=f"La API devolvió HTTP {response.status_code}.", raw_text=raw_text)
            try:
                parsed = response.json()
            except ValueError:
                return HttpResult(False, response.status_code, error_message="La API respondió contenido no JSON.", raw_text=raw_text)
            result = parse_score_response(parsed, response.status_code)
            return HttpResult(result.ok, result.status_code, result.data, result.error_message, raw_text)
        except (requests.Timeout, requests.ConnectionError) as error:
            if attempt == 0:
                pause(0.25)
                continue
            return HttpResult(False, error_message=f"No se pudo obtener el scoring: {error}.")
    return HttpResult(False, error_message="No se pudo obtener el scoring.")


def score_action(
    base_url: str,
    record: InputRecord,
    warmup_timeout: float,
    score_timeout: float,
    get: Callable[..., Any] = requests.get,
    post: Callable[..., Any] = requests.post,
    pause: Callable[[float], None] = sleep,
) -> HttpResult:
    """Ejecuta un warm-up acotado por acción; su fallo no impide el POST de scoring."""
    warm_api(f"{base_url.rstrip('/')}/health", warmup_timeout, get=get, pause=pause)
    return score_record(f"{base_url.rstrip('/')}/predict", record, score_timeout, post=post, pause=pause)


def transition(state: AppState, event: str, result: HttpResult | None = None, record: InputRecord | None = None) -> AppState:
    """Centraliza el estado para que un error nunca borre el último resultado válido."""
    if event == "submit":
        return AppState("submitting", state.last_valid_result, state.submitted_payload, None)
    if event == "success" and result and result.data and record:
        return AppState("success", result.data, record, None)
    if event == "error" and result:
        return AppState("error", state.last_valid_result, state.submitted_payload, result.error_message)
    return state


def demonstrative_decision(result: ScoreResult, threshold: float = APPROVAL_THRESHOLD) -> tuple[bool, str]:
    """Clasificación pedagógica, no una política de crédito ni una oferta."""
    qualifies = result["perdida_esperada_relativa"] <= threshold
    return qualifies, "Dentro del umbral demostrativo" if qualifies else "Fuera del umbral demostrativo"


def sensitivity_records(record: InputRecord, principal_factors: Iterable[float], terms: Iterable[str]) -> list[InputRecord]:
    """Genera variantes ceteris-paribus sin aumentar capital ni tocar otros campos."""
    original_principal = float(record["principal"])
    candidates: list[InputRecord] = []
    seen: set[tuple[float, str]] = set()
    for factor in principal_factors:
        principal = round(original_principal * float(factor), 2)
        if principal <= 0 or principal > original_principal:
            continue
        for term in terms:
            key = (principal, term)
            if key in seen or (principal == original_principal and term == record["num_cuotas"]):
                continue
            seen.add(key)
            candidate = dict(record)
            candidate["principal"] = principal
            candidate["num_cuotas"] = term
            candidates.append(candidate)
    return candidates
