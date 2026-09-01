from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core import (
    APPROVAL_THRESHOLD,
    AppState,
    build_payload,
    compose_payload_values,
    demonstrative_decision,
    parse_score_response,
    risk_band,
    sensitivity_records,
    transition,
)

SPEC_PATH = Path(__file__).resolve().parents[1] / "idea" / "design_spec.json"


def spec() -> dict[str, object]:
    import json

    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def record() -> dict[str, object]:
    return {"ingresos_verificados": "Source Verified", "vivienda": "RENT", "finalidad": "debt_consolidation", "num_cuotas": "36 months", "antigüedad_empleo": "5 years", "rating": "B", "ingresos": 50000, "dti": 15.0, "num_lineas_credito": 12, "porc_uso_revolving": 40.0, "principal": 10000, "tipo_interes": 0.15, "imp_cuota": 350, "num_derogatorios": 0}


class CoreTests(unittest.TestCase):
    def test_payload_preserves_raw_values(self) -> None:
        raw = record()
        payload = build_payload(raw)
        self.assertEqual(payload["num_cuotas"], "36 months")
        self.assertEqual(payload["tipo_interes"], 0.15)

    def test_metrics_require_exact_canonical_fields(self) -> None:
        result = parse_score_response([{"score_pd": 0.1, "score_ead": 0.2, "score_lgd": 0.3, "perdida_esperada_relativa": 0.05}])
        self.assertTrue(result.ok)
        self.assertFalse(parse_score_response([{"score_pd": 0.1}]).ok)

    def test_threshold_is_inclusive(self) -> None:
        qualifies, _ = demonstrative_decision({"score_pd": 0.1, "score_ead": 0.2, "score_lgd": 0.3, "perdida_esperada_relativa": APPROVAL_THRESHOLD})
        self.assertTrue(qualifies)

    def test_scenarios_only_change_allowed_fields_and_never_raise_principal(self) -> None:
        original = build_payload(record())
        scenarios = sensitivity_records(original, [0.8, 0.9, 1.0, 1.2], ["36 months", "60 months"])
        self.assertTrue(scenarios)
        for scenario in scenarios:
            self.assertLessEqual(scenario["principal"], original["principal"])
            self.assertTrue(set(key for key in scenario if scenario[key] != original.get(key)).issubset({"principal", "num_cuotas"}))
        self.assertEqual(original["principal"], 10000)

    def test_hidden_fields_reach_the_payload_with_fixture_values(self) -> None:
        design = spec()
        visible = {name: config["default"] for name, config in design["visible_fields"].items()}
        payload = build_payload(compose_payload_values(visible, design["hidden_fields"]))
        for field, value in design["hidden_fields"].items():
            self.assertEqual(payload[field], value)
        # Los seis ocultos coinciden con el fixture canónico y nunca se muestran como filtros.
        self.assertNotIn("vivienda", design["visible_fields"])
        self.assertEqual(payload["vivienda"], "RENT")
        self.assertEqual(payload["finalidad"], "debt_consolidation")

    def test_eight_visible_fields_are_sent_and_win_over_hidden(self) -> None:
        design = spec()
        self.assertEqual(
            set(design["visible_fields"]),
            {"principal", "num_cuotas", "tipo_interes", "imp_cuota", "ingresos", "dti", "porc_uso_revolving", "rating"},
        )
        visible = {"principal": 22000, "num_cuotas": "60 months", "tipo_interes": 0.21, "imp_cuota": 500,
                   "ingresos": 90000, "dti": 12.5, "porc_uso_revolving": 30, "rating": "C"}
        payload = build_payload(compose_payload_values(visible, design["hidden_fields"]))
        for field, value in visible.items():
            self.assertEqual(payload[field], value)

    def test_compose_payload_values_does_not_mutate_inputs(self) -> None:
        visible = {"principal": 15000}
        hidden = {"vivienda": "RENT"}
        compose_payload_values(visible, hidden)
        self.assertEqual(visible, {"principal": 15000})
        self.assertEqual(hidden, {"vivienda": "RENT"})

    def test_risk_band_splits_into_three_labelled_bands(self) -> None:
        self.assertEqual(risk_band(0.04, 0.05, 0.10)[0], "within")
        self.assertEqual(risk_band(0.05, 0.05, 0.10)[0], "within")
        self.assertEqual(risk_band(0.08, 0.05, 0.10)[0], "near")
        self.assertEqual(risk_band(0.20, 0.05, 0.10)[0], "outside")
        for _, label in (risk_band(v, 0.05, 0.10) for v in (0.01, 0.07, 0.3)):
            self.assertTrue(label)  # cada banda trae texto, no sólo color

    def test_error_preserves_last_valid_result(self) -> None:
        success = parse_score_response([{"score_pd": 0.1, "score_ead": 0.2, "score_lgd": 0.3, "perdida_esperada_relativa": 0.01}])
        current = transition(AppState(), "success", success, build_payload(record()))
        failed = transition(current, "error", type("Result", (), {"error_message": "timeout"})())
        self.assertEqual(failed.last_valid_result, current.last_valid_result)
        self.assertEqual(failed.status, "error")
