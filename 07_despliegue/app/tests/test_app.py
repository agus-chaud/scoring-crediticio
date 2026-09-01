from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AppSmokeTests(unittest.TestCase):
    def test_layout_renders_without_making_a_request(self) -> None:
        app = AppTest.from_file(str(APP_PATH))
        app.run()
        self.assertFalse(app.exception)

    def test_filter_panel_exposes_the_eight_visible_controls(self) -> None:
        app = AppTest.from_file(str(APP_PATH))
        app.run()
        self.assertFalse(app.exception)
        # 6 sliders numéricos + 1 radio (plazo) + 1 selectbox (rating) = 8 filtros visibles.
        self.assertEqual(len(app.slider), 6)
        self.assertEqual(len(app.radio), 1)
        self.assertEqual(len(app.selectbox), 1)
        labels = {button.label for button in app.button}
        self.assertIn("Calcular riesgo", labels)

    def test_empty_state_is_shown_before_any_result(self) -> None:
        app = AppTest.from_file(str(APP_PATH))
        app.run()
        self.assertFalse(app.exception)
        rendered = " ".join(element.value for element in app.markdown)
        self.assertIn("Ajustá los filtros", rendered)


if __name__ == "__main__":
    unittest.main()
