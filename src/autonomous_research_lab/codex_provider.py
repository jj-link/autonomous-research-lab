"""Tool-free Codex Responses transport for the user's ChatGPT account.

Wire references: https://github.com/openai/codex/tree/main/codex-rs/codex-api
and https://github.com/openai/codex/tree/main/codex-rs/model-provider-info
The account service, token destination, and catalog are fixed. No CLI agent or
host tools run here; only the coordinator can schedule an isolated experiment.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .model_client import MAX_RESPONSE_BYTES, ModelClient, ModelError, _NoRedirect, _parse_object, _redact, _strict_json_loads

BASE_URL = "https://chatgpt.com/backend-api/codex"
CLIENT_VERSION = "0.153.4"
MAX_STREAM_BYTES = 64 * 1024 * 1024


def _headers(credentials, stream=True):
    values = [credentials.get(key) for key in ("access_token", "account_id")]
    if any(not isinstance(v, str) or not v or not v.isascii() or any(ord(c) < 33 or ord(c) > 126 for c in v) for v in values):
        raise ModelError("Codex sign-in has invalid account metadata; sign in again")
    return {"Authorization": "Bearer " + values[0], "ChatGPT-Account-Id": values[1],
            "originator": "autonomous-research-lab", "User-Agent": "autonomous-research-lab/" + CLIENT_VERSION,
            "OpenAI-Beta": "responses=experimental", "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json"}


def _safe(value, credentials):
    for name in ("access_token", "account_id"):
        value = _redact(value, credentials.get(name))
    return value


def _body(config, messages):
    instructions, inputs = [], []
    for message in messages:
        if message["role"] == "system":
            instructions.append(message["content"])
        else:
            assistant = message["role"] == "assistant"
            item = {"role": message["role"], "content": [{"type": "output_text" if assistant else "input_text", "text": message["content"]}]}
            if assistant:
                item.update(type="message", status="completed")
            inputs.append(item)
    if not inputs:
        inputs.append({"role": "user", "content": [{"type": "input_text", "text": "Respond according to the instructions."}]})
    return {"model": config["model"], "store": False, "stream": True,
            "instructions": "\n\n".join(instructions), "input": inputs, "tools": [],
            "text": {"format": {"type": "json_schema", "name": "research_response", "strict": False,
                                "schema": {"type": "object", "additionalProperties": True}}}}


def _public_text(items):
    text = []
    size = 0
    for item in items:
        if item.get("type") == "reasoning":
            continue
        if item.get("type") != "message" or item.get("role") != "assistant" or not isinstance(item.get("content"), list):
            raise ModelError("Codex returned unsupported output; no tools were executed")
        if item.get("status") not in (None, "completed"):
            raise ModelError("Codex returned an unfinished message")
        for part in item["content"]:
            if not isinstance(part, dict):
                raise ModelError("Codex returned invalid message content")
            if part.get("type") == "refusal":
                raise ModelError("Codex refused the request")
            if part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                raise ModelError("Codex returned unsupported message content")
            size += len(part["text"].encode("utf-8"))
            if size > MAX_RESPONSE_BYTES:
                raise ModelError("Codex public answer exceeds the response-byte limit")
            text.append(part["text"])
    return "".join(text)


class _Events:
    def __init__(self, started):
        self.started = started
        self.data = []
        self.event = None
        self.parts = []
        self.public_bytes = 0
        self.completed_bytes = 0
        self.items = {}
        self.output = None
        self.response_id = None
        self.usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    def evidence(self):
        raw = "".join(self.parts)
        if not raw:
            items = self.output if self.output else [self.items[i] for i in sorted(self.items)]
            fragments, remaining = [], MAX_RESPONSE_BYTES
            for item in items:
                if not isinstance(item, dict) or item.get("type") != "message" or item.get("role") != "assistant" or not isinstance(item.get("content"), list):
                    continue
                for part in item["content"]:
                    if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        fragment = part["text"][:remaining]
                        fragments.append(fragment)
                        remaining -= len(fragment)
                    if not remaining:
                        break
                if not remaining:
                    break
            raw = "".join(fragments).encode("utf-8")[:MAX_RESPONSE_BYTES].decode("utf-8", errors="ignore")
        return {"raw": raw, "usage": self.usage, "response_id": self.response_id,
                "elapsed_seconds": time.monotonic() - self.started}

    def feed(self, line):
        if line:
            if line.startswith(":"):
                return None
            field, separator, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "data" and separator:
                self.data.append(value)
            elif field == "event" and separator:
                self.event = value
            elif field not in {"id", "retry"}:
                raise ModelError("Codex returned malformed stream framing")
            return None
        if not self.data:
            self.event = None
            return None
        raw, event = "\n".join(self.data), self.event
        self.data, self.event = [], None
        if raw == "[DONE]":
            raise ModelError("Codex stream ended without a terminal response")
        value = _strict_json_loads(raw)
        if not isinstance(value, dict) or not isinstance(value.get("type"), str) or (event and event != value["type"]):
            raise ModelError("Codex returned inconsistent stream events")
        kind = value["type"]
        response = value.get("response")
        if isinstance(response, dict):
            if isinstance(response.get("output"), list):
                self.output = response["output"]
            if isinstance(response.get("id"), str):
                self.response_id = response["id"]
            usage = response.get("usage")
            if usage is not None:
                if not isinstance(usage, dict):
                    raise ModelError("Codex returned malformed usage")
                self.usage = ModelClient._parse_usage({"usage": {
                    "prompt_tokens": usage.get("input_tokens"), "completion_tokens": usage.get("output_tokens"),
                    "total_tokens": usage.get("total_tokens")}}, self.evidence())
        if kind == "response.output_text.delta":
            if not isinstance(value.get("delta"), str):
                raise ModelError("Codex returned an invalid text delta")
            self.public_bytes += len(value["delta"].encode("utf-8"))
            if self.public_bytes > MAX_RESPONSE_BYTES:
                raise ModelError("Codex public answer exceeds the response-byte limit")
            self.parts.append(value["delta"])
        elif kind == "response.output_item.done":
            index, item = value.get("output_index"), value.get("item")
            if type(index) is not int or index < 0 or index in self.items or not isinstance(item, dict):
                raise ModelError("Codex returned duplicate or invalid completed output")
            self.completed_bytes += len(_public_text([item]).encode("utf-8"))
            if self.completed_bytes > MAX_RESPONSE_BYTES:
                raise ModelError("Codex completed public output exceeds the response-byte limit")
            self.items[index] = item
        elif kind in {"error", "response.failed"}:
            raise ModelError("Codex response failed; no provider was substituted")
        elif kind in {"response.completed", "response.done", "response.incomplete"}:
            if not isinstance(response, dict):
                raise ModelError("Codex terminal response is missing")
            status = response.get("status")
            if status != "completed" or kind == "response.incomplete" or response.get("error"):
                raise ModelError("Codex response was incomplete, failed, or cancelled")
            output = response.get("output")
            if output == [] and self.items:
                indexes = sorted(self.items)
                if indexes != list(range(len(indexes))):
                    raise ModelError("Codex stream is missing a completed output item")
                output = [self.items[i] for i in indexes]
            if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
                raise ModelError("Codex terminal response has invalid output")
            text = _public_text(output)
            self.parts = [text]
            if not text.strip():
                raise ModelError("Codex returned no public answer text")
            evidence = {**self.evidence(), "finish_reason": "stop"}
            return {**evidence, "content": _parse_object(text, evidence)}
        return None


def complete(config, messages, credentials):
    started = time.monotonic()
    parser = _Events(started)
    request = urllib.request.Request(BASE_URL + "/responses", data=json.dumps(_body(config, messages)).encode("utf-8"),
                                     headers=_headers(credentials), method="POST")
    opener = urllib.request.build_opener(_NoRedirect(), urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=config["request_timeout_seconds"]) as response:
            media = response.headers.get("Content-Type", "text/event-stream").split(";", 1)[0].strip()
            if media != "text/event-stream":
                raise ModelError("Codex did not return an event stream")
            received = 0
            pending = bytearray()
            while True:
                if time.monotonic() - started >= config["request_timeout_seconds"]:
                    raise ModelError("Codex request exceeded its timeout")
                chunk = response.read1(4096)
                if not chunk:
                    raise ModelError("Codex stream closed before its terminal response")
                received += len(chunk)
                if received > MAX_STREAM_BYTES:
                    raise ModelError("Codex event stream exceeded the transport-byte limit")
                pending.extend(chunk)
                while (end := pending.find(b"\n")) >= 0:
                    line = bytes(pending[:end]).rstrip(b"\r").decode("utf-8")
                    del pending[:end + 1]
                    result = parser.feed(line)
                    if result is not None:
                        return _safe(result, credentials)
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        raise ModelError(f"Codex request failed (HTTP {status}); check sign-in or account limits", _safe({**parser.evidence(), "status": status}, credentials)) from None
    except ModelError as exc:
        raise ModelError(_safe(str(exc), credentials), _safe({**parser.evidence(), **exc.record}, credentials)) from None
    except Exception:
        raise ModelError("Codex transport or stream decoding failed; partial evidence was retained", _safe(parser.evidence(), credentials)) from None


def list_models(config, credentials):
    request = urllib.request.Request(BASE_URL + "/models?client_version=" + CLIENT_VERSION, headers=_headers(credentials, stream=False))
    opener = urllib.request.build_opener(_NoRedirect(), urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=min(config["request_timeout_seconds"], 30)) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError()
        models = _strict_json_loads(raw.decode("utf-8"))["models"]
        if not isinstance(models, list) or any(not isinstance(m, dict) or not isinstance(m.get("slug"), str) or not m["slug"].strip() for m in models):
            raise ValueError()
        return list(dict.fromkeys(m["slug"] for m in models))
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        raise ModelError(f"Codex model discovery failed (HTTP {status}); check account sign-in") from None
    except Exception:
        raise ModelError("Codex model discovery failed; no static model list or alternate provider was substituted") from None
