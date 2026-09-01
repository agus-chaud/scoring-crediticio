from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AppSmokeTests(unittest.TestCase):
    def test_form_renders_without_making_a_request(self) -> None:
        app = AppTest.from_file(str(APP_PATH))
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.button), 1)
        self.assertEqual(app.button[0].label, "Calcular riesgo demostrativo")
