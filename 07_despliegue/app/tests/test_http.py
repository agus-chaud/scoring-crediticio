from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core import build_payload, score_record, warm_api
from test_core import record


def response(status: int, body: object = None, json_error: bool = False) -> Mock:
    item = Mock(status_code=status, text="body")
    item.json.side_effect = ValueError("invalid") if json_error else None
    if not json_error:
        item.json.return_value = body
    return item


class HttpTests(unittest.TestCase):
    def test_warmup_retries_once_for_connection_error(self) -> None:
        get = Mock(side_effect=[requests.ConnectionError("cold"), response(200)])
        result = warm_api("http://api/health", 1, get=get, pause=Mock())
        self.assertTrue(result.ok)
        self.assertEqual(get.call_count, 2)

    def test_score_retries_one_transient_status(self) -> None:
        post = Mock(side_effect=[response(503), response(200, [{"score_pd": 0.1, "score_ead": 0.2, "score_lgd": 0.3, "perdida_esperada_relativa": 0.01}])])
        result = score_record("http://api/predict", build_payload(record()), 1, post=post, pause=Mock())
        self.assertTrue(result.ok)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs["json"], [build_payload(record())])

    def test_non_json_and_client_error_are_actionable_without_retry(self) -> None:
        post = Mock(return_value=response(200, json_error=True))
        result = score_record("http://api/predict", build_payload(record()), 1, post=post)
        self.assertFalse(result.ok)
        self.assertIn("no JSON", result.error_message or "")
        post = Mock(return_value=response(422))
        result = score_record("http://api/predict", build_payload(record()), 1, post=post)
        self.assertFalse(result.ok)
        self.assertEqual(post.call_count, 1)
