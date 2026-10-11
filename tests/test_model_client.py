"""Consumer-visible boundaries of the model client: real HTTP round trips against a loopback endpoint.

The loopback server is an in-process stdlib HTTP endpoint (no fixtures, no
external network); parser, usage, security, and retry-count behavior are
exercised through ModelClient.complete and validate_model_config exactly as the
research controller calls them.
"""
from __future__ import annotations

import json
import io
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError

from autonomous_research_lab.model_client import MAX_RESPONSE_BYTES, ModelClient, ModelError, validate_model_config
from autonomous_research_lab.models import LabError

TEST_KEY = "sk-lab-test-key-123"
TEST_KEY_ENV = "ARL_TEST_MODEL_KEY"
GOOD_ENVELOPE = {
    "id": "chatcmpl-test-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "{\"answer\": 42}"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
}


class _Endpoint(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_POST(self):
        holder = self.server.holder
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        holder["requests"].append({
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "body": json.loads(raw.decode("utf-8")) if raw else None,
        })
        if holder["delay"]:
            time.sleep(holder["delay"])
        status, body = holder["response"]
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in holder["headers"].items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class ModelClientTests(unittest.TestCase):
    def setUp(self):
        self.holder = {"requests": [], "response": (200, json.dumps(GOOD_ENVELOPE).encode("utf-8")), "delay": 0, "headers": {}}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Endpoint)
        self.server.holder = self.holder
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        os.environ.pop(TEST_KEY_ENV, None)

    def client(self, **overrides) -> ModelClient:
        config = {"base_url": self.base_url, "model": "test-model"}
        config.update(overrides)
        return ModelClient(config)

    def respond(self, payload, status=200, headers=None, delay=0):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.holder["response"] = (status, body)
        self.holder["delay"] = delay
        self.holder["headers"] = headers or {}

    def complete(self, **overrides) -> dict:
        return self.client(**overrides).complete([{"role": "user", "content": "Return the JSON object."}], 512)

    def envelope(self, **changes) -> dict:
        envelope = json.loads(json.dumps(GOOD_ENVELOPE))
        envelope.update(changes)
        return envelope

    def test_decodes_structured_content_and_usage(self):
        result = self.complete()
        self.assertEqual(result["content"], {"answer": 42})
        self.assertEqual(result["usage"], {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18})

    def test_bearer_credentials_cannot_use_plaintext_transport(self):
        os.environ[TEST_KEY_ENV] = TEST_KEY
        with self.assertRaises(LabError):
            self.client(api_key_env=TEST_KEY_ENV)
        self.assertEqual(self.holder["requests"], [])

    def test_missing_env_variable_is_a_model_error_naming_only_the_variable(self):
        with self.assertRaises(ModelError) as raised:
            ModelClient({"base_url": "https://api.example.com/v1", "model": "test", "api_key_env": "ARL_TEST_ABSENT_KEY"}).complete([{"role": "user", "content": "hi"}], 16)
        self.assertNotIn("raw", raised.exception.record)

    def test_finish_reason_length_is_rejected(self):
        self.respond(self.envelope(choices=[{"message": {"role": "assistant", "content": "{\"ans"}, "finish_reason": "length"}]))
        with self.assertRaises(ModelError) as raised:
            self.complete()
        self.assertEqual(raised.exception.record["finish_reason"], "length")
        self.assertEqual(raised.exception.record["usage"]["completion_tokens"], 7)

    def test_unsupported_finish_reason_is_rejected(self):
        self.respond(self.envelope(choices=[{"message": {"role": "assistant", "content": "{}"}, "finish_reason": "content_filter"}]))
        with self.assertRaises(ModelError):
            self.complete()

    def test_missing_content_is_rejected(self):
        self.respond(self.envelope(choices=[{"message": {"role": "assistant"}, "finish_reason": "stop"}]))
        with self.assertRaises(ModelError) as raised:
            self.complete()

    def test_reasoning_only_response_is_an_explicit_failure(self):
        self.respond(self.envelope(choices=[{
            "message": {"role": "assistant", "content": None, "reasoning_content": "Let me think this through..."},
            "finish_reason": "stop",
        }]))
        with self.assertRaises(ModelError) as raised:
            self.complete()
        self.assertIn("Let me think", raised.exception.record["raw"])

    def test_content_must_be_a_json_object_without_markdown_extraction(self):
        for content in ("[1, 2, 3]", "\"just a string\"", "```json\n{\"a\": 1}\n```", "{\"a\": 1} trailing"):
            with self.subTest(content=content):
                self.respond(self.envelope(choices=[{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]))
                with self.assertRaises(ModelError) as raised:
                    self.complete()

    def test_non_finite_json_is_rejected_in_body_and_content(self):
        content_with_nan = json.dumps({"choices": [{"message": {"content": "{\"a\": NaN}"}}]})
        body_with_nan = json.dumps({"score": float("nan")})
        overflow = json.dumps({"choices": [{"message": {"content": "{\"a\": 1e999}"}}]})
        for body in (content_with_nan, body_with_nan, overflow):
            with self.subTest(body=body):
                self.respond(body.encode("utf-8"))
                with self.assertRaises(ModelError):
                    self.complete()

    def test_body_must_be_valid_json(self):
        self.respond(b"<html>Service Busy</html>")
        with self.assertRaises(ModelError) as raised:
            self.complete()
        self.assertIn("Service Busy", raised.exception.record["raw"])

    def test_malformed_usage_is_rejected_preserving_the_raw_usage(self):
        for usage in ("high", {"prompt_tokens": "10"}, {"total_tokens": -1}, {"completion_tokens": True}):
            with self.subTest(usage=usage):
                self.respond(self.envelope(usage=usage))
                with self.assertRaises(ModelError) as raised:
                    self.complete()
                self.assertEqual(raised.exception.record["usage"], usage)

    def envelope_no_usage(self) -> dict:
        envelope = json.loads(json.dumps(GOOD_ENVELOPE))
        envelope.pop("usage")
        return envelope

    def test_omitted_usage_stays_unknown(self):
        self.respond(self.envelope_no_usage())
        result = self.complete()
        self.assertEqual(result["usage"], {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None})

    def test_empty_usage_object_stays_unknown(self):
        self.respond(self.envelope(usage={}))
        result = self.complete()
        self.assertEqual(result["usage"], {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None})

    def test_secrets_are_redacted_when_endpoint_echoes_credentials(self):
        os.environ[TEST_KEY_ENV] = TEST_KEY
        client = ModelClient({"base_url": "https://api.example.com/v1", "model": "test", "api_key_env": TEST_KEY_ENV})
        body = json.dumps({"error": "rejected " + TEST_KEY}).encode()
        error = HTTPError(client.config["base_url"], 500, "failed", {}, io.BytesIO(body))
        with patch.object(client._opener, "open", side_effect=error):
            with self.assertRaises(ModelError) as raised:
                client.complete([{"role": "user", "content": "hi"}], 16)
        self.assertEqual(raised.exception.record["status"], 500)
        self.assertNotIn(TEST_KEY, json.dumps(raised.exception.record))
        self.assertNotIn(TEST_KEY, str(raised.exception))
        envelope = self.envelope(choices=[{"message": {"content": json.dumps({"echo": TEST_KEY})}, "finish_reason": "stop"}])
        with patch.object(client._opener, "open", return_value=io.BytesIO(json.dumps(envelope).encode())):
            result = client.complete([{"role": "user", "content": "hi"}], 16)
        self.assertNotIn(TEST_KEY, json.dumps(result))

    def test_invalid_utf8_cannot_become_valid_completion_content(self):
        self.respond(json.dumps(GOOD_ENVELOPE).encode().replace(b"answer", b"\xff"))
        with self.assertRaises(ModelError):
            self.complete()

    def test_redirects_are_not_followed_and_no_request_is_resent(self):
        self.respond(b"", status=302, headers={"Location": f"{self.base_url}/elsewhere"})
        with self.assertRaises(ModelError) as raised:
            self.complete()
        self.assertEqual(raised.exception.record["status"], 302)
        self.assertEqual(len(self.holder["requests"]), 1)

    def test_there_is_exactly_one_request_even_after_failure(self):
        self.respond(b"{\"error\": \"no\"}", status=503)
        with self.assertRaises(ModelError):
            self.complete()
        self.assertEqual(len(self.holder["requests"]), 1)

    def test_transport_failure_record_has_no_raw(self):
        client = ModelClient({"base_url": "http://127.0.0.1:1/v1", "model": "test-model", "request_timeout_seconds": 2})
        with self.assertRaises(ModelError) as raised:
            client.complete([{"role": "user", "content": "hi"}], 16)
        self.assertNotIn("raw", raised.exception.record)
        self.assertIn("elapsed_seconds", raised.exception.record)

    def test_oversized_response_is_rejected(self):
        self.respond(b"{\"choices\": [{\"message\": {\"content\": \"" + b"a" * MAX_RESPONSE_BYTES + b"\"}}]}")
        with self.assertRaises(ModelError) as raised:
            self.complete()
        self.assertEqual(len(self.holder["requests"]), 1)

    def test_timeout_is_a_model_error(self):
        self.respond(json.dumps(GOOD_ENVELOPE).encode("utf-8"), delay=1.5)
        with self.assertRaises(ModelError) as raised:
            self.complete(request_timeout_seconds=1)
        self.assertNotIn("raw", raised.exception.record)
        self.assertGreaterEqual(raised.exception.record["elapsed_seconds"], 0.5)



class ValidateModelConfigTests(unittest.TestCase):
    def test_saved_configuration_can_be_reloaded(self):
        snapshot = validate_model_config({"base_url": "https://api.example.com/v1/", "model": "cloud-model"})
        client = ModelClient(json.loads(json.dumps(snapshot)))
        self.assertEqual(client.config, snapshot)

    def test_invalid_configs_are_rejected(self):
        invalid = [
            ("missing fields", {}),
            ("missing model", {"base_url": "https://api.example.com/v1"}),
            ("malformed port", {"base_url": "https://api.example.com:bad/v1", "model": "m"}),
            ("unknown key", {"api_key": "sk-x", "base_url": "http://h/v1", "model": "m"}),
            ("no api key values", {"base_url": "http://h/v1", "model": "m", "api_key": "sk-x"}),
            ("scheme", {"base_url": "ftp://h/v1", "model": "m"}),
            ("scheme", {"base_url": "not a url", "model": "m"}),
            ("host", {"base_url": "http://", "model": "m"}),
            ("empty", {"base_url": "", "model": "m"}),
            ("credentials", {"base_url": "http://user:pass@host/v1", "model": "m"}),
            ("query", {"base_url": "http://host/v1?x=1", "model": "m"}),
            ("fragment", {"base_url": "http://host/v1#frag", "model": "m"}),
            ("whitespace", {"base_url": "  http://host/v1  ", "model": "m"}),
            ("model", {"base_url": "http://host/v1", "model": "   "}),
            ("model", {"base_url": "http://host/v1", "model": 7}),
            ("api_key_env", {"base_url": "http://host/v1", "model": "m", "api_key_env": "not-a-name"}),
            ("api_key_env", {"base_url": "http://host/v1", "model": "m", "api_key_env": ""}),
            ("api_key_env", {"base_url": "http://host/v1", "model": "m", "api_key_env": 5}),
            ("timeout", {"base_url": "http://host/v1", "model": "m", "request_timeout_seconds": 0}),
            ("timeout", {"base_url": "http://host/v1", "model": "m", "request_timeout_seconds": "600"}),
            ("timeout", {"base_url": "http://host/v1", "model": "m", "request_timeout_seconds": True}),
            ("timeout", {"base_url": "http://host/v1", "model": "m", "request_timeout_seconds": 1.5}),
            ("not an object", "http://host/v1"),
            ("not an object", None),
        ]
        for label, config in invalid:
            with self.subTest(label=label, config=config):
                with self.assertRaises(LabError):
                    validate_model_config(config)


class RequestValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = ModelClient({"base_url": "http://127.0.0.1:9/v1", "model": "m"})

    def test_messages_must_be_well_formed(self):
        for messages in ([], ["nope"], [{"role": "user"}], [{"role": "wizard", "content": "x"}], [{"role": "user", "content": 5}], [{"role": "user", "content": "x", "name": "y"}], [{"role": "user", "content": "x"}, {"nope": True}]):
            with self.subTest(messages=messages):
                with self.assertRaises(LabError):
                    self.client.complete(messages, 16)

    def test_max_tokens_must_be_a_positive_integer(self):
        for max_tokens in (0, -1, "16", 1.5, True, None):
            with self.subTest(max_tokens=max_tokens):
                with self.assertRaises(LabError):
                    self.client.complete([{"role": "user", "content": "hi"}], max_tokens)


if __name__ == "__main__":
    unittest.main()
