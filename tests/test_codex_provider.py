"""Codex completion boundaries, not successful fixtures advertised as live inference."""
import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from autonomous_research_lab import codex_provider
from autonomous_research_lab.model_client import ModelClient, ModelError, validate_model_config
from autonomous_research_lab.models import LabError
from autonomous_research_lab.research import validate_brief


class Auth:
    def credentials(self):
        return {"access_token": "private-access-sentinel", "account_id": "private-account-sentinel"}


class Stream(io.BytesIO):
    headers = {"Content-Type": "text/event-stream"}

    def read1(self, size):
        return super().read(min(size, 17))  # Split UTF-8 and SSE fields across reads.


def message(text, **changes):
    return {"type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": text}], **changes}


def event(kind, **fields):
    return ("event: " + kind + "\r\ndata: " + json.dumps({"type": kind, **fields}, ensure_ascii=False) + "\r\n\r\n").encode()


def terminal(output, status="completed", **fields):
    return event("response." + status, response={"id": "response-42", "status": status, "output": output, **fields})


class CodexProviderTests(unittest.TestCase):
    def complete(self, stream):
        with patch("autonomous_research_lab.codex_provider.urllib.request.build_opener") as factory:
            factory.return_value.open.return_value = Stream(stream)
            return ModelClient({"provider": "openai-codex", "model": "protocol-fixture"}, auth=Auth()).complete([{"role": "user", "content": "Respond with JSON"}], None)

    def test_ordered_completed_items_reconstruct_empty_terminal_output(self):
        stream = b": keepalive\r\n\r\n" + event("response.output_item.done", output_index=1, item=message('{"answer":"café"}'))
        stream += event("response.output_item.done", output_index=0, item={"type": "reasoning", "summary": [{"text": "private reasoning"}]})
        stream += terminal([], usage={"input_tokens": 12, "output_tokens": 9, "total_tokens": 21})
        result = self.complete(stream)
        self.assertEqual(result["content"], {"answer": "café"})
        self.assertEqual(result["usage"], {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21})
        self.assertNotIn("private reasoning", json.dumps(result))

    def test_missing_usage_remains_unknown(self):
        self.assertEqual(self.complete(terminal([message('{"answer":42}')]))["usage"], {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None})

    def test_incomplete_terminal_retains_public_answer_and_usage(self):
        with self.assertRaises(ModelError) as raised:
            self.complete(terminal([{"type": "reasoning", "summary": [{"text": "hidden"}]}, message('{"unfinished":')], "incomplete", usage={"input_tokens": 11, "output_tokens": 7}))
        self.assertEqual(raised.exception.record["raw"], '{"unfinished":')
        self.assertEqual(raised.exception.record["usage"]["completion_tokens"], 7)
        self.assertIsNone(raised.exception.record["usage"]["total_tokens"])
        self.assertEqual(raised.exception.record["response_id"], "response-42")
        self.assertNotIn("hidden", json.dumps(raised.exception.record))

    def test_eof_and_error_never_promote_partial_text_to_success(self):
        prefix = event("response.output_item.done", output_index=0, item=message('{"answer":42}'))
        for suffix in (b"", event("error", message="upstream failure"), b"data: [DONE]\n\n"):
            with self.subTest(suffix=suffix), self.assertRaises(ModelError) as raised:
                self.complete(prefix + suffix)
            self.assertEqual(raised.exception.record["raw"], '{"answer":42}')

    def test_malformed_item_sequences_and_unfinished_messages_fail(self):
        item = event("response.output_item.done", output_index=1, item=message('{"answer":42}'))
        for stream in (item + terminal([]), item + item + terminal([]), terminal([message('{"answer":42}', status="incomplete")]), terminal([{"type": "function_call", "name": "shell", "arguments": "{}"}])):
            with self.subTest(stream=stream), self.assertRaises(ModelError):
                self.complete(stream)

    def test_invalid_json_and_nonfinite_answers_fail_with_evidence(self):
        for text in ('{"x":NaN}', '[]', '{"cutoff":'):
            with self.subTest(text=text), self.assertRaises(ModelError) as raised:
                self.complete(terminal([message(text)]))
            self.assertEqual(raised.exception.record["raw"], text)

    def test_byte_limit_counts_ignored_keepalives(self):
        with patch.object(codex_provider, "MAX_RESPONSE_BYTES", 64), self.assertRaises(ModelError):
            self.complete(b": " + b"x" * 80 + b"\n\n" + terminal([message('{}')]))

    def test_partial_failure_evidence_redacts_account_credentials(self):
        text = json.dumps(Auth().credentials())
        with self.assertRaises(ModelError) as raised:
            self.complete(event("response.output_text.delta", delta=text))
        saved = json.dumps(raised.exception.record)
        for secret in Auth().credentials().values():
            self.assertNotIn(secret, saved)

    def test_http_failure_is_not_retried_or_routed_to_another_provider(self):
        with patch("autonomous_research_lab.codex_provider.urllib.request.build_opener") as factory:
            factory.return_value.open.side_effect = HTTPError("https://chatgpt.com/backend-api/codex/responses", 429, "limit", {}, io.BytesIO(b"private-access-sentinel"))
            with self.assertRaises(ModelError) as raised:
                ModelClient({"provider": "openai-codex", "model": "protocol-fixture"}, auth=Auth()).complete([{"role": "user", "content": "JSON"}], None)
            self.assertEqual(factory.return_value.open.call_count, 1)
            self.assertEqual(raised.exception.record["status"], 429)
            self.assertNotIn("private-access-sentinel", json.dumps(raised.exception.record))

    def test_codex_cannot_redirect_credentials_or_claim_a_numeric_token_cap(self):
        config = {"provider": "openai-codex", "model": "protocol-fixture"}
        for changes in ({"base_url": "https://attacker.example"}, {"api_key_env": "API_KEY"}):
            with self.subTest(changes=changes), self.assertRaises(LabError):
                validate_model_config({**config, **changes})
        brief = {"goal": "Investigate an approved question", "model": config}
        self.assertIsNone(validate_brief(brief)["max_output_tokens_per_call"])
        with self.assertRaises(LabError):
            validate_brief({**brief, "max_output_tokens_per_call": 8192})
        with self.assertRaises(LabError):
            validate_brief({**brief, "model": {"provider": "spark", "model": "protocol-fixture"}, "max_output_tokens_per_call": None})
