from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core import APPROVAL_THRESHOLD, AppState, build_payload, demonstrative_decision, parse_score_response, sensitivity_records, transition


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

    def test_error_preserves_last_valid_result(self) -> None:
        success = parse_score_response([{"score_pd": 0.1, "score_ead": 0.2, "score_lgd": 0.3, "perdida_esperada_relativa": 0.01}])
        current = transition(AppState(), "success", success, build_payload(record()))
        failed = transition(current, "error", type("Result", (), {"error_message": "timeout"})())
        self.assertEqual(failed.last_valid_result, current.last_valid_result)
        self.assertEqual(failed.status, "error")
