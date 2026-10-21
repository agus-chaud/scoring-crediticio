from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explainer import (  # type: ignore[import-not-found]
    DISCLAIMER,
    build_explainer_context,
    deterministic_answer,
    answer_question,
    evaluate_answer_contract,
    load_local_env,
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
    @patch("explainer.dotenv_values", return_value={"ANTHROPIC_API_KEY": "local-key", "ANTHROPIC_MODEL": "local-model"})
    def test_loads_nonempty_app_env_values_without_replacing_server_values(self, mock_dotenv_values: object) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "   ", "ANTHROPIC_MODEL": "   "}, clear=True):
            load_local_env()
            self.assertEqual(os.environ["ANTHROPIC_API_KEY"], "local-key")
            self.assertEqual(os.environ["ANTHROPIC_MODEL"], "local-model")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "server-key", "ANTHROPIC_MODEL": "server-model"}, clear=True):
            load_local_env()
            self.assertEqual(os.environ["ANTHROPIC_API_KEY"], "server-key")
            self.assertEqual(os.environ["ANTHROPIC_MODEL"], "server-model")

        mock_dotenv_values.assert_called_with(Path(__file__).resolve().parents[1] / ".env")

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

    def test_unsafe_anthropic_answer_falls_back_to_the_local_explanation(self) -> None:
        class FakeMessages:
            calls = 0

            @staticmethod
            def create(**_: object) -> object:
                FakeMessages.calls += 1
                text = "Recomiendo aprobar este crédito. " + deterministic_answer("Explicame esto", SCORE)
                return type("Response", (), {"content": [type("TextBlock", (), {"type": "text", "text": text})()]})()

        fake_client = type("FakeClient", (), {"messages": FakeMessages()})()
        previous_key = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        try:
            reply = answer_question("Explicame esto", SCORE, client=fake_client)
        finally:
            if previous_key is None:
                del os.environ["ANTHROPIC_API_KEY"]
            else:
                os.environ["ANTHROPIC_API_KEY"] = previous_key
        self.assertEqual(reply.mode, "local")
        self.assertNotIn("Recomiendo aprobar", reply.content)
        self.assertEqual(FakeMessages.calls, 1)

    def test_provider_answer_without_disclaimer_is_repaired_before_validation(self) -> None:
        class FakeMessages:
            @staticmethod
            def create(**_: object) -> object:
                text = "El resultado reúne tres medidas que conviene revisar juntas antes de sacar conclusiones."
                return type("Response", (), {"content": [type("TextBlock", (), {"type": "text", "text": text})()]})()

        fake_client = type("FakeClient", (), {"messages": FakeMessages()})()
        previous_key = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        try:
            reply = answer_question("Explicame esto", SCORE, client=fake_client)
        finally:
            if previous_key is None:
                del os.environ["ANTHROPIC_API_KEY"]
            else:
                os.environ["ANTHROPIC_API_KEY"] = previous_key
        self.assertEqual(reply.mode, "llm")
        self.assertTrue(reply.content.endswith(DISCLAIMER))
        self.assertIn("20,0%", reply.content)

    def test_provider_metric_drift_falls_back_before_metrics_are_appended(self) -> None:
        class FakeMessages:
            @staticmethod
            def create(**_: object) -> object:
                return type("Response", (), {"content": [type("TextBlock", (), {"type": "text", "text": "La PD es 80,0%."})()]})()

        fake_client = type("FakeClient", (), {"messages": FakeMessages()})()
        previous_key = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        try:
            reply = answer_question("Explicame esto", SCORE, client=fake_client)
        finally:
            if previous_key is None:
                del os.environ["ANTHROPIC_API_KEY"]
            else:
                os.environ["ANTHROPIC_API_KEY"] = previous_key
        self.assertEqual(reply.mode, "local")

    @patch("explainer.load_local_env")
    def test_provider_connection_failure_is_logged_and_distinguished_from_guardrails(self, _: object) -> None:
        class FakeMessages:
            @staticmethod
            def create(**_: object) -> object:
                raise ConnectionError("network disabled by sandbox")

        fake_client = type("FakeClient", (), {"messages": FakeMessages()})()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "secret-test-key", "ANTHROPIC_MODEL": "test-model"}):
            with self.assertLogs("explainer", level="WARNING") as logs:
                reply = answer_question("Explicame esto", SCORE, client=fake_client)
        self.assertEqual(reply.mode, "local")
        self.assertEqual(reply.fallback_reason, "provider_error")
        self.assertIn(DISCLAIMER, reply.content)
        log_text = "\n".join(logs.output)
        self.assertIn("ConnectionError", log_text)
        self.assertIn("test-model", log_text)
        self.assertNotIn("secret-test-key", log_text)

    @patch("explainer.load_local_env")
    def test_guardrail_rejection_is_logged_with_its_reason(self, _: object) -> None:
        class FakeMessages:
            @staticmethod
            def create(**_: object) -> object:
                text = "Recomiendo aprobar este crédito porque el riesgo es bajo."
                return type("Response", (), {"content": [type("TextBlock", (), {"type": "text", "text": text})()]})()

        fake_client = type("FakeClient", (), {"messages": FakeMessages()})()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "secret-test-key"}):
            with self.assertLogs("explainer", level="WARNING") as logs:
                reply = answer_question("Explicame esto", SCORE, client=fake_client)
        self.assertEqual(reply.mode, "local")
        self.assertEqual(reply.fallback_reason, "guardrail")
        self.assertNotIn("Recomiendo aprobar", reply.content)
        self.assertIn("recomendación crediticia prohibida", "\n".join(logs.output))

    @patch("explainer.load_local_env")
    def test_generated_percentage_is_reported_as_a_guardrail_fallback(self, _: object) -> None:
        class FakeMessages:
            @staticmethod
            def create(**_: object) -> object:
                return type("Response", (), {"content": [type("TextBlock", (), {"type": "text", "text": "La PD es 80,0%."})()]})()

        fake_client = type("FakeClient", (), {"messages": FakeMessages()})()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "secret-test-key"}):
            reply = answer_question("Explicame esto", SCORE, client=fake_client)
        self.assertEqual(reply.mode, "local")
        self.assertEqual(reply.fallback_reason, "guardrail")

    @patch("explainer.load_local_env")
    def test_truncated_provider_answer_is_never_shown(self, _: object) -> None:
        class FakeMessages:
            @staticmethod
            def create(**_: object) -> object:
                text = "El resultado reúne tres medidas que conviene"
                block = type("TextBlock", (), {"type": "text", "text": text})()
                return type("Response", (), {"content": [block], "stop_reason": "max_tokens"})()

        fake_client = type("FakeClient", (), {"messages": FakeMessages()})()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "secret-test-key"}):
            with self.assertLogs("explainer", level="WARNING") as logs:
                reply = answer_question("Explicame esto", SCORE, client=fake_client)
        self.assertEqual(reply.mode, "local")
        self.assertEqual(reply.fallback_reason, "incomplete")
        self.assertNotIn("conviene\n", reply.content)
        self.assertIn("max_tokens", "\n".join(logs.output))

    @patch("explainer.load_local_env")
    def test_request_leaves_room_for_thinking_and_answer(self, _: object) -> None:
        captured: dict[str, object] = {}

        class FakeMessages:
            @staticmethod
            def create(**kwargs: object) -> object:
                captured.update(kwargs)
                text = "El resultado reúne tres medidas que conviene revisar juntas."
                block = type("TextBlock", (), {"type": "text", "text": text})()
                return type("Response", (), {"content": [block], "stop_reason": "end_turn"})()

        fake_client = type("FakeClient", (), {"messages": FakeMessages()})()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "secret-test-key"}):
            reply = answer_question("Explicame esto", SCORE, client=fake_client)
        self.assertEqual(reply.mode, "llm")
        self.assertGreaterEqual(captured["max_tokens"], 4000)
        self.assertEqual(captured["output_config"], {"effort": "low"})

    @patch("explainer.load_local_env")
    def test_valid_provider_answer_has_no_fallback_reason(self, _: object) -> None:
        class FakeMessages:
            @staticmethod
            def create(**_: object) -> object:
                text = "El resultado reúne tres medidas que conviene revisar juntas."
                return type("Response", (), {"content": [type("TextBlock", (), {"type": "text", "text": text})()]})()

        fake_client = type("FakeClient", (), {"messages": FakeMessages()})()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "secret-test-key"}):
            reply = answer_question("Explicame esto", SCORE, client=fake_client)
        self.assertEqual(reply.mode, "llm")
        self.assertIsNone(reply.fallback_reason)

    @patch("explainer.load_local_env")
    def test_openai_key_does_not_enable_the_anthropic_provider(self, _: object) -> None:
        previous_anthropic_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        previous_openai_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "legacy-key"
        try:
            reply = answer_question("Explicame esto", SCORE)
        finally:
            if previous_anthropic_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = previous_anthropic_key
            if previous_openai_key is None:
                del os.environ["OPENAI_API_KEY"]
            else:
                os.environ["OPENAI_API_KEY"] = previous_openai_key
        self.assertEqual(reply.mode, "local")
        self.assertEqual(reply.fallback_reason, "not_configured")

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
