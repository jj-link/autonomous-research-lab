"""OpenAI-compatible chat completion client: one HTTP request per call, strict parsing.

The client is configured from a research brief's ``model`` object; see
:func:`validate_model_config` for the accepted schema and the persistable,
secret-free snapshot it returns. API credentials are only ever referenced by
environment-variable name and are read from the process environment at request
time; they never appear in snapshots, records, or error messages.

Each :meth:`ModelClient.complete` call sends exactly one ``chat/completions``
POST asking for a JSON object. Anything else — transport failure, HTTP error,
oversized body, truncated or unsupported finish reason, missing content,
reasoning-only responses, non-object or non-finite JSON, malformed usage — is
raised as :class:`ModelError` with a ``.record`` preserving the available raw
text, usage, status, and timing. There are no retries, no markdown extraction,
and no invented fields; redirects are not followed so a credential is never
resent to another host.
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
]

DEFAULT_REQUEST_TIMEOUT_SECONDS = 600
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
    """Validate a brief's ``model`` object and return a persistable snapshot without secrets.

    The snapshot contains ``schema_version``, ``base_url`` (without a trailing
    slash), ``model``, ``api_key_env`` (a variable *name*, or None), and
    ``request_timeout_seconds``. Any other key is rejected; a key value would
    have nowhere to go and is therefore never accepted.
    """
    require_keys(config, {"schema_version", "base_url", "model", "api_key_env", "request_timeout_seconds"}, "model config")
    if type(config.get("schema_version", 1)) is not int or config.get("schema_version", 1) != 1:
        raise LabError("Unsupported model configuration schema")
    base_url = config.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        raise LabError("model config.base_url must be a nonempty string")
    if any(char.isspace() for char in base_url):
        raise LabError("model config.base_url must not contain whitespace")
    try:
        parts = urlsplit(base_url)
        parts.port
    except ValueError as exc:
        raise LabError(f"Invalid model endpoint: {exc}") from exc
    if parts.scheme not in ("http", "https"):
        raise LabError("model config.base_url must use http or https")
    if not parts.hostname:
        raise LabError("model config.base_url must include a host")
    if parts.username or parts.password:
        raise LabError("model config.base_url must not contain credentials; name an environment variable via api_key_env instead")
    if parts.query or parts.fragment:
        raise LabError("model config.base_url must not contain a query or fragment")
    model = config.get("model")
    if not isinstance(model, str) or not model.strip():
        raise LabError("model config.model must be a nonempty string")
    api_key_env = config.get("api_key_env")
    if api_key_env is not None and (not isinstance(api_key_env, str) or not _ENV_NAME.fullmatch(api_key_env)):
        raise LabError("model config.api_key_env must be None or a valid environment variable name")
    if api_key_env and parts.scheme != "https":
        raise LabError("Credentialed model endpoints must use HTTPS")
    timeout = config.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS)
    positive_int(timeout, "model config.request_timeout_seconds")
    return {
        "schema_version": 1,
        "base_url": base_url.rstrip("/"),
        "model": model,
        "api_key_env": api_key_env,
        "request_timeout_seconds": timeout,
    }


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
    """Sends one chat/completions request per :meth:`complete` call and parses the response strictly."""

    def __init__(self, config: dict):
        self.config = validate_model_config(config)
        self._opener = urllib.request.build_opener(_NoRedirect(), urllib.request.ProxyHandler({}))

    def complete(self, messages: list[dict], max_tokens: int) -> dict:
        """Send one request and return {content, raw, usage, response_id, finish_reason, elapsed_seconds}.

        ``content`` is the parsed JSON object; ``raw`` is the completion text it
        was parsed from. ``usage`` values are None when the provider omits
        them. All failures raise :class:`ModelError`.
        """
        _validate_messages(messages)
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
        try:
            parsed = _strict_json_loads(content)
        except ValueError as exc:
            raise ModelError(f"Model content is not valid JSON: {exc}", evidence | {"usage": usage, "finish_reason": finish_reason}) from exc
        if not isinstance(parsed, dict):
            raise ModelError("Model content must be a JSON object", evidence | {"usage": usage, "finish_reason": finish_reason})
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
