from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explainer import (  # type: ignore[import-not-found]
    DISCLAIMER,
    build_explainer_context,
    deterministic_answer,
    answer_question,
    evaluate_answer_contract,
    load_eval_cases,
)


SCORE = {
    "score_pd": 0.20,
    "score_ead": 0.80,
    "score_lgd": 0.90,
    "perdida_esperada_relativa": 0.144,
}

EVAL_CASES = Path(__file__).resolve().parents[1] / "evals" / "scoring_explainer_cases.json"


class ExplainerTests(unittest.TestCase):
    def test_context_contains_only_scores_and_the_decision_boundary(self) -> None:
        context = build_explainer_context(SCORE)
        self.assertEqual(context["metrics"], SCORE)
        self.assertNotIn("id_cliente", json.dumps(context))
        self.assertIn("no permite recomendar aprobar ni rechazar", context["decision_boundary"])

    def test_named_questions_have_simple_guarded_answers(self) -> None:
        questions = (
            "Explicame esto como si no conociera finanzas.",
            "¿Qué diferencia hay entre riesgo alto y pérdida alta?",
            "¿Qué debería verificar antes de usar este resultado?",
            "¿Cuál métrica está aumentando más la pérdida esperada?",
        )
        for question in questions:
            answer = deterministic_answer(question, SCORE)
            self.assertIn(DISCLAIMER, answer)
            self.assertIn("20,0%", answer)
            self.assertNotIn("se recomienda aprobar", answer.lower())
            self.assertNotIn("se recomienda rechazar", answer.lower())

    def test_economic_boundary_never_invents_margin_or_default_cost(self) -> None:
        answer = deterministic_answer("¿Por qué no recomendás aprobar o rechazar?", SCORE)
        self.assertIn("margen neto", answer.lower())
        self.assertIn("costo si ocurre un incumplimiento", answer.lower())
        self.assertNotIn("$", answer)

    def test_contract_rejects_decisions_number_drift_and_missing_disclaimer(self) -> None:
        self.assertFalse(evaluate_answer_contract("Este crédito debería aprobarse. " + DISCLAIMER, SCORE).valid)
        self.assertFalse(evaluate_answer_contract("La PD es 30%. " + DISCLAIMER, SCORE).valid)
        self.assertFalse(evaluate_answer_contract("La PD es 20,0%.", SCORE).valid)

    def test_contract_rejects_adversarial_recommendations_contradictions_and_economic_numbers(self) -> None:
        valid = deterministic_answer("Explicame esto como si no conociera finanzas.", SCORE)
        attempts = (
            "Este crédito debería aprobarse. " + valid,
            "Recomiendo aprobar este crédito. " + valid,
            "No hay dudas: recomiendo aprobar este crédito. " + valid,
            "La PD correcta es 99,0%, no 20,0%. " + valid,
            "El margen es $500 y el costo es $1000. " + valid,
        )
        for attempt in attempts:
            self.assertFalse(evaluate_answer_contract(attempt, SCORE).valid)

    def test_unsafe_provider_answer_falls_back_to_the_local_explanation(self) -> None:
        class FakeResponses:
            @staticmethod
            def create(**_: object) -> object:
                return type("Response", (), {"output_text": "Recomiendo aprobar este crédito. " + deterministic_answer("Explicame esto", SCORE)})()

        fake_client = type("FakeClient", (), {"responses": FakeResponses()})()
        previous_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"
        try:
            reply = answer_question("Explicame esto", SCORE, client=fake_client)
        finally:
            if previous_key is None:
                del os.environ["OPENAI_API_KEY"]
            else:
                os.environ["OPENAI_API_KEY"] = previous_key
        self.assertEqual(reply.mode, "local")
        self.assertNotIn("Recomiendo aprobar", reply.content)

    def test_eval_cases_are_a_valid_regression_suite(self) -> None:
        cases = load_eval_cases(EVAL_CASES)
        self.assertGreaterEqual(len(cases), 5)
        for case in cases:
            answer = deterministic_answer(case["question"], SCORE)
            verdict = evaluate_answer_contract(answer, SCORE)
            self.assertTrue(verdict.valid, case["name"])
            for required in case["required_fragments"]:
                self.assertIn(required.lower(), answer.lower(), case["name"])


if __name__ == "__main__":
    unittest.main()
