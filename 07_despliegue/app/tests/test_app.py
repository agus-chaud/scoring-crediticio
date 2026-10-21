from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core import AppState
from explainer import ExplainerReply, deterministic_answer
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
RESULT = {"score_pd": 0.062, "score_ead": 0.537, "score_lgd": 0.886, "perdida_esperada_relativa": 0.03}


class AppSmokeTests(unittest.TestCase):
    def test_layout_renders_without_making_a_request(self) -> None:
        app = AppTest.from_file(str(APP_PATH))
        app.run()
        self.assertFalse(app.exception)

    def test_filter_panel_exposes_the_nine_visible_controls(self) -> None:
        app = AppTest.from_file(str(APP_PATH))
        app.run()
        self.assertFalse(app.exception)
        # 6 sliders numéricos + 1 radio (plazo) + 2 selectbox (rating, sector) = 9 filtros visibles.
        self.assertEqual(len(app.slider), 6)
        self.assertEqual(len(app.radio), 1)
        self.assertEqual(len(app.selectbox), 2)
        labels = {button.label for button in app.button}
        self.assertIn("Calcular riesgo", labels)

    def test_sector_selectbox_defaults_to_administrativo_and_keeps_the_api_token(self) -> None:
        app = AppTest.from_file(str(APP_PATH))
        app.run()
        sector = next(box for box in app.selectbox if box.label == "Sector de empleo")
        self.assertEqual(sector.value, "administrativo")
        sector.select("salud").run()
        self.assertFalse(app.exception)
        sector = next(box for box in app.selectbox if box.label == "Sector de empleo")
        self.assertEqual(sector.value, "salud")

    def test_empty_state_is_shown_before_any_result(self) -> None:
        app = AppTest.from_file(str(APP_PATH))
        app.run()
        self.assertFalse(app.exception)
        rendered = " ".join(element.value for element in app.markdown)
        self.assertIn("Ajustá los filtros", rendered)


    def test_explainer_is_a_floating_popover_after_a_score(self) -> None:
        app = AppTest.from_file(str(APP_PATH))
        app.session_state["credit_risk_state"] = AppState(
            status="success",
            last_valid_result={
                "score_pd": 0.062,
                "score_ead": 0.537,
                "score_lgd": 0.886,
                "perdida_esperada_relativa": 0.03,
            },
        )
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("popover")), 1)
        self.assertEqual(len(app.chat_input), 1)

    def _ask_with_local_reply(self, fallback_reason: str) -> AppTest:
        reply = ExplainerReply(deterministic_answer("Explicame esto", RESULT), "local", fallback_reason)
        app = AppTest.from_file(str(APP_PATH))
        app.session_state["credit_risk_state"] = AppState(status="success", last_valid_result=RESULT)
        with patch("explainer.answer_question", return_value=reply):
            app.run()
            next(button for button in app.button if "conociera" in button.label).click().run()
        self.assertFalse(app.exception)
        return app

    def test_guardrail_fallback_shows_a_plain_language_notice(self) -> None:
        captions = " ".join(caption.value for caption in self._ask_with_local_reply("guardrail").caption)
        self.assertIn("no pudo usarse con seguridad", captions)
        self.assertIn("explicación local validada", captions)
        self.assertNotIn(".env", captions)
        self.assertNotIn("Anthropic", captions)

    def test_provider_failure_shows_a_plain_language_notice(self) -> None:
        captions = " ".join(caption.value for caption in self._ask_with_local_reply("provider_error").caption)
        self.assertIn("no está disponible", captions)
        self.assertIn("explicación local validada", captions)
        self.assertNotIn(".env", captions)
        self.assertNotIn("Anthropic", captions)


if __name__ == "__main__":
    unittest.main()
