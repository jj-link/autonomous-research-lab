"""Explicit model providers with strict JSON results and no inference retries.

API keys stay in environment variables; Codex account credentials stay in its
separate protected store. No provider is substituted when a request fails.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from typing import Any

from .models import LabError, positive_int, require_keys

__all__ = [
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "MAX_RESPONSE_BYTES",
    "ModelClient",
    "ModelError",
    "validate_model_config",
    "PROVIDERS",
    "SPARK_BASE_URL",
    "list_models",
]

DEFAULT_REQUEST_TIMEOUT_SECONDS = 600
PROVIDERS = ("openai-codex", "spark", "openai-compatible")
SPARK_BASE_URL = "http://100.92.139.82:8888/v1"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})
_ALLOWED_MESSAGE_KEYS = frozenset({"role", "content"})
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ModelError(LabError):
    """A model request or response failure.

    ``record`` carries the evidence available at failure time (``error``,
    ``raw``, ``usage``, ``status``, ``finish_reason``, ``elapsed_seconds``).
    It never contains credentials.
    """

    def __init__(self, message: str, record: dict | None = None):
        super().__init__(message)
        self.record: dict = dict(record or {})
        self.record.setdefault("error", message)


def validate_model_config(config: Any) -> dict:
    """Return a persistable provider/model assignment containing no credentials."""
    return _validate_config(config, require_model=True)


def _validate_config(config, *, require_model):
    require_keys(config, {"schema_version", "provider", "base_url", "model", "api_key_env", "request_timeout_seconds"}, "model config")
    if type(config.get("schema_version", 1)) is not int or config.get("schema_version", 1) != 1:
        raise LabError("Unsupported model configuration schema")
    provider = config.get("provider", "openai-compatible")
    if provider not in PROVIDERS:
        raise LabError("Unknown model provider")
    model = config.get("model")
    if require_model or model is not None:
        if not isinstance(model, str) or not model.strip():
            raise LabError("model config.model must be a nonempty string")
    timeout = config.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS)
    positive_int(timeout, "model config.request_timeout_seconds")
    result = {"schema_version": 1, "provider": provider, "model": model, "request_timeout_seconds": timeout}
    if provider == "openai-codex":
        if any(config.get(key) is not None for key in ("base_url", "api_key_env")):
            raise LabError("Codex uses ChatGPT sign-in and a fixed service, not endpoint or API-key overrides")
        return result
    base_url = config.get("base_url", SPARK_BASE_URL if provider == "spark" else None)
    if not isinstance(base_url, str) or not base_url or any(char.isspace() for char in base_url):
        raise LabError("model config.base_url must be a nonempty URL without whitespace")
    try:
        parts = urlsplit(base_url)
        parts.port
    except ValueError as exc:
        raise LabError(f"Invalid model endpoint: {exc}") from exc
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise LabError("model config.base_url must include an HTTP or HTTPS host")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise LabError("Model endpoints must not contain credentials, queries, or fragments")
    api_key_env = config.get("api_key_env")
    if api_key_env is not None and (not isinstance(api_key_env, str) or not _ENV_NAME.fullmatch(api_key_env)):
        raise LabError("model config.api_key_env must be None or a valid environment variable name")
    if api_key_env and parts.scheme != "https":
        raise LabError("Credentialed model endpoints must use HTTPS")
    return {**result, "base_url": base_url.rstrip("/"), "api_key_env": api_key_env}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Disable redirect following: one request per call, credentials never leave the configured host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 - urllib hook
        return None


def _strict_json_loads(text: str) -> Any:
    def reject_constant(name: str) -> Any:
        raise ValueError(f"non-finite JSON constant {name}")

    value = json.loads(text, parse_constant=reject_constant)
    json.dumps(value, allow_nan=False)
    return value


def _parse_object(content, evidence):
    try:
        parsed = _strict_json_loads(content)
    except (ValueError, RecursionError):
        raise ModelError("Model content is not valid finite JSON", evidence) from None
    if not isinstance(parsed, dict):
        raise ModelError("Model content must be a JSON object", evidence)
    return parsed


def _validate_messages(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        raise LabError("messages must be a nonempty list")
    for message in messages:
        require_keys(message, _ALLOWED_MESSAGE_KEYS, "message")
        if message.get("role") not in _ALLOWED_ROLES:
            raise LabError("message.role must be system, user, or assistant")
        if not isinstance(message.get("content"), str):
            raise LabError("message.content must be a string")


def _has_reasoning(message: dict) -> bool:
    return any(isinstance(message.get(name), str) and message[name].strip() for name in ("reasoning_content", "reasoning"))


def _redact(value, secret):
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]").replace(json.dumps(secret)[1:-1], "[REDACTED]")
    if isinstance(value, dict):
        return {_redact(key, secret): _redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    return value


class ModelClient:
    """Make one request to the selected provider; never execute model tools."""

    def __init__(self, config: dict, *, auth=None):
        self.config = validate_model_config(config)
        self.auth = auth
        self._opener = urllib.request.build_opener(_NoRedirect(), urllib.request.ProxyHandler({}))

    def check_auth(self):
        if self.config["provider"] != "openai-codex":
            return self._api_key()
        if self.auth is None:
            raise ModelError("Codex requires the research store's sign-in session")
        try:
            return self.auth.credentials()
        except LabError as exc:
            raise ModelError(str(exc)) from None

    def complete(self, messages: list[dict], max_tokens: int | None) -> dict:
        """Send one request and return {content, raw, usage, response_id, finish_reason, elapsed_seconds}.

        ``content`` is the parsed JSON object; ``raw`` is the completion text it
        was parsed from. ``usage`` values are None when the provider omits
        them. All failures raise :class:`ModelError`.
        """
        _validate_messages(messages)
        if self.config["provider"] == "openai-codex":
            if max_tokens is not None:
                raise LabError("Codex output-token limits are provider-managed; use null, not an unenforced token cap")
            from .codex_provider import complete
            return complete(self.config, messages, self.check_auth())
        max_tokens = positive_int(max_tokens, "max_tokens")
        request = urllib.request.Request(
            self.config["base_url"] + "/chat/completions",
            data=json.dumps({
                "model": self.config["model"],
                "messages": messages,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "autonomous-research-lab/0.1",
            },
            method="POST",
        )
        api_key = self._api_key()
        if api_key is not None:
            request.add_header("Authorization", f"Bearer {api_key}")
        started = time.monotonic()
        status = None
        try:
            try:
                response = self._opener.open(request, timeout=self.config["request_timeout_seconds"])
            except urllib.error.HTTPError as exc:
                response, status = exc, exc.code
            with response:
                payload = response.read(MAX_RESPONSE_BYTES + 1)
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            elapsed = time.monotonic() - started
            raise ModelError(_redact(f"Model request failed: {exc}", api_key), {"status": status, "elapsed_seconds": elapsed}) from None
        elapsed = time.monotonic() - started
        if len(payload) > MAX_RESPONSE_BYTES:
            raise ModelError(f"Model response exceeds the {MAX_RESPONSE_BYTES} byte limit", {"status": status, "elapsed_seconds": elapsed})
        if status is not None:
            detail = f"Model endpoint returned HTTP {status}"
            if 300 <= status < 400:
                detail += " (redirects are not followed)"
            raise ModelError(detail, {"status": status, "raw": _redact(payload.decode("utf-8", errors="backslashreplace"), api_key), "elapsed_seconds": elapsed})
        try:
            raw = payload.decode("utf-8")
        except UnicodeDecodeError:
            raise ModelError("Model response is not valid UTF-8", {"raw": _redact(payload.decode("utf-8", errors="backslashreplace"), api_key), "elapsed_seconds": elapsed}) from None
        try:
            return _redact(self._parse(raw, elapsed), api_key)
        except ModelError as exc:
            raise ModelError(_redact(str(exc), api_key), _redact(exc.record, api_key)) from None

    def _api_key(self) -> str | None:
        name = self.config["api_key_env"]
        if name is None:
            return None
        value = os.environ.get(name)
        if not value:
            raise ModelError(f"API key environment variable {name} is not set")
        if any(ord(char) < 33 or ord(char) > 126 for char in value):
            raise ModelError(f"API key environment variable {name} contains invalid header characters")
        return value


    def _parse(self, raw_body: str, elapsed: float) -> dict:
        evidence = {"raw": raw_body, "elapsed_seconds": elapsed}
        try:
            envelope = _strict_json_loads(raw_body)
        except ValueError as exc:
            raise ModelError(f"Model response body is not valid JSON: {exc}", evidence) from exc
        if not isinstance(envelope, dict):
            raise ModelError("Model response body must be a JSON object", evidence)
        usage = self._parse_usage(envelope, evidence)
        evidence["usage"] = usage
        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelError("Model response contains no choices", evidence)
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ModelError("Model response choice is not an object", evidence)
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ModelError("Model response finish_reason must be a string or null", evidence)
        if finish_reason == "length":
            raise ModelError("Model response was truncated: finish_reason is 'length'", evidence | {"finish_reason": finish_reason})
        if finish_reason not in (None, "stop"):
            raise ModelError(f"Model response has unsupported finish_reason {finish_reason!r}", evidence | {"finish_reason": finish_reason})
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ModelError("Model response choice has no message object", evidence)
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            detail = "Model returned a reasoning-only response without content" if _has_reasoning(message) else "Model response is missing content"
            raise ModelError(detail, evidence | {"finish_reason": finish_reason})
        evidence["raw"] = content
        parsed = _parse_object(content, evidence | {"usage": usage, "finish_reason": finish_reason})
        response_id = envelope.get("id")
        return {
            "content": parsed,
            "raw": content,
            "usage": usage,
            "response_id": response_id if isinstance(response_id, str) else None,
            "finish_reason": finish_reason,
            "elapsed_seconds": elapsed,
        }

    @staticmethod
    def _parse_usage(envelope: dict, evidence: dict) -> dict:
        raw_usage = envelope.get("usage")
        if raw_usage is None:
            return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        if not isinstance(raw_usage, dict):
            raise ModelError("Model response usage must be an object", evidence | {"usage": raw_usage})
        usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = raw_usage.get(name)
            if value is None:
                continue
            if type(value) is not int or value < 0:
                raise ModelError(f"Model response usage field {name} is malformed", evidence | {"usage": raw_usage})
            usage[name] = value
        return usage


def list_models(config, *, auth=None):
    """Discover the selected endpoint's actual catalog, without generating text."""
    config = _validate_config(config, require_model=False)
    if config["provider"] == "openai-codex":
        if auth is None:
            raise ModelError("Connect Codex before discovering its models")
        from .codex_provider import list_models as codex_models
        return codex_models(config, auth.credentials())
    name = config["api_key_env"]
    key = os.environ.get(name) if name else None
    if name and not key:
        raise ModelError(f"API key environment variable {name} is not set")
    if key and any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise ModelError("The configured API key contains invalid header characters")
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(config["base_url"] + "/models", headers=headers)
    opener = urllib.request.build_opener(_NoRedirect(), urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=min(config["request_timeout_seconds"], 30)) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError()
        rows = _strict_json_loads(raw.decode("utf-8"))["data"]
        if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip() for row in rows):
            raise ValueError()
        return list(dict.fromkeys(row["id"] for row in rows))
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        raise ModelError(f"Model discovery failed (HTTP {status}); no provider was substituted") from None
    except Exception:
        raise ModelError("Model discovery failed; check the selected endpoint and credentials") from None
