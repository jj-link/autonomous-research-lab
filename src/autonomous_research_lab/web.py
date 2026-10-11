"""Loopback web interface: a stdlib HTTP server exposing the research engine plus the static dashboard.

The server binds to loopback addresses only, embeds a per-server token in the index page
(required for every mutation), and serves attempt artifacts as attachments. There is no
CORS: browser clients are same-origin by construction.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import socket
import stat
import sys
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from .execution import configuration, run as run_experiment, stop as stop_attempt
from .models import LabError, TERMINAL_STATES, relative_path
from .research import ResearchEngine
from .store import Store

MAX_BODY_BYTES = 1024 * 1024  # JSON bodies are capped at 1 MiB.
LOG_TAIL_BYTES = 64 * 1024
STATIC_DIR = Path(__file__).resolve().parent / "static"
TOKEN_PLACEHOLDER = "__ARL_SERVER_TOKEN__"
TOKEN_HEADER = "X-ARL-Token"
LOOPBACK_NAMES = frozenset({"127.0.0.1", "::1", "localhost"})
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; "
       "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")


def _require_loopback(host: str) -> str:
    """Reject any bind host that is not loopback; the dashboard is a local control surface."""
    if not isinstance(host, str) or not host:
        raise LabError("Bind host must be a nonempty loopback address")
    if host.lower() == "localhost":
        return "127.0.0.1"
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise LabError(f"Cannot resolve bind host {host!r}: {exc}") from exc
    if not addresses or any(not ipaddress.ip_address(address).is_loopback for address in addresses):
        raise LabError(
            f"Refusing to bind {host!r}: the dashboard exposes research records and controls "
            "and must stay on loopback"
        )
    return host


def _loopback_name(hostname: str | None) -> bool:
    return (hostname or "").lower() in LOOPBACK_NAMES


@dataclass
class _Response:
    status: int
    headers: list[tuple[str, str]] = field(default_factory=list)
    body: bytes = b""
    file_fd: int | None = None
    file_size: int = 0


def _json_response(status: int, value: Any) -> _Response:
    body = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
    return _Response(status, _headers("application/json; charset=utf-8"), body)


def _headers(content_type: str, extra: list[tuple[str, str]] | None = None) -> list[tuple[str, str]]:
    pairs = [
        ("Content-Type", content_type),
        ("Cache-Control", "no-store"),
        ("X-Content-Type-Options", "nosniff"),
        ("Referrer-Policy", "no-referrer"),
        ("X-Frame-Options", "DENY"),
    ]
    return pairs + (extra or [])


class _HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _open_regular(base: Path, relative: str) -> int:
    """Open a record through no-follow directory descriptors, including the leaf."""
    relative_path(relative, "Record path")
    if os.open not in os.supports_dir_fd:
        raise LabError("Safe record-file access requires Linux/WSL")
    directory = os.open(base, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise LabError("Record is not a regular file")
        return fd
    finally:
        os.close(directory)


def _preview(base: Path, relative: str, *, tail=False, limit=65536) -> dict:
    with os.fdopen(_open_regular(base, relative), "rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if tail:
            stream.seek(max(0, size - limit))
        return {"text": stream.read(limit).decode("utf-8", errors="replace"), "truncated": size > limit}


class _App:
    """Concurrent request routing; authoritative mutations use database transactions."""

    def __init__(self, store: Store):
        self.store = store
        self.engine = ResearchEngine(store)
        self.token = secrets.token_urlsafe(32)
        self.bound_port = 0

    # -- request entry --------------------------------------------------

    def handle(self, method: str, raw_path: str, headers: Any, body: bytes | None) -> _Response:
        try:
            if not self._host_ok(headers):
                raise _HttpError(403, "Requests must address this server by its loopback host")
            if method == "POST":
                if not self._origin_ok(headers):
                    raise _HttpError(403, "Cross-origin mutation rejected: Origin is not this dashboard")
                if not self._token_ok(headers):
                    raise _HttpError(403, "Missing or invalid session token for this mutation")
            return self._dispatch(method, raw_path, body)
        except _HttpError as exc:
            return _json_response(exc.status, {"error": exc.message})
        except KeyError as exc:
            # Stores and the engine signal unknown records with KeyError; anything else is an input error.
            missing = exc.args[0] if exc.args else exc
            return _json_response(404, {"error": f"No such record: {missing}"})
        except LabError as exc:
            return _json_response(400, {"error": str(exc)})
        except Exception:  # pragma: no cover - unexpected; details go to the server log.
            traceback.print_exc()
            return _json_response(500, {"error": "Internal server error; details are in the server log"})

    # -- origin/host/token gates ----------------------------------------

    def _host_ok(self, headers: Any) -> bool:
        host = headers.get("Host")
        if not host or any(char.isspace() for char in host):
            return False
        if hasattr(headers, "get_all") and len(headers.get_all("Host", [])) != 1:
            return False
        try:
            parts = urlsplit(f"http://{host}")
            return (_loopback_name(parts.hostname) and not any((parts.username, parts.password, parts.path, parts.query, parts.fragment))
                    and (parts.port or 80) == self.bound_port)
        except ValueError:
            return False

    def _origin_ok(self, headers: Any) -> bool:
        origin = headers.get("Origin")
        if origin is None:  # Non-browser clients (tests, curl) carry no Origin; the token still applies.
            return True
        if not origin or origin.lower() == "null":
            return False
        try:
            parts = urlsplit(origin)
            host = urlsplit("http://" + headers.get("Host", ""))
            return (parts.scheme == "http" and parts.hostname == host.hostname
                    and (parts.port or 80) == (host.port or 80) == self.bound_port
                    and not any((parts.username, parts.password, parts.path, parts.query, parts.fragment)))
        except ValueError:
            return False

    def _token_ok(self, headers: Any) -> bool:
        supplied = headers.get(TOKEN_HEADER)
        return isinstance(supplied, str) and hmac.compare_digest(supplied, self.token)

    # -- routing ---------------------------------------------------------

    def _dispatch(self, method: str, raw_path: str, body: bytes | None) -> _Response:
        path = urlsplit(raw_path).path
        segments = [segment for segment in path.split("/") if segment]
        if method == "GET":
            return self._get(path, segments)
        if method == "POST":
            return self._post(path, segments, body)
        raise _HttpError(405, "Method not allowed")

    def _get(self, path: str, segments: list[str]) -> _Response:
        if path == "/":
            return self._index()
        if path == "/app.js":
            return self._static("app.js", "text/javascript; charset=utf-8")
        if path == "/style.css":
            return self._static("style.css", "text/css; charset=utf-8")
        if segments == ["api", "overview"]:
            return self._overview()
        if len(segments) == 3 and segments[0] == "api":
            kind, record_id = segments[1], segments[2]
            if kind == "programs":
                return _json_response(200, self._record(self.engine.get, record_id))
            if kind == "experiments":
                return _json_response(200, self._record(self.store.get_experiment, record_id))
            if kind == "attempts":
                return _json_response(200, self._attempt_with_events(record_id))
        if len(segments) == 4 and segments[:2] == ["api", "programs"] and segments[3] == "report":
            self._record(self.engine.get, segments[2])  # Unknown programs 404 before reporting.
            return _json_response(200, {"markdown": self.engine.report(segments[2])})
        if len(segments) == 4 and segments[:2] == ["api", "attempts"] and segments[3] == "logs":
            return _json_response(200, self._attempt_logs(segments[2]))
        if len(segments) >= 4 and segments[:2] == ["api", "attempts"] and segments[3] == "artifacts":
            return self._artifact(segments[2], path)
        if len(segments) >= 4 and segments[:2] == ["api", "experiments"] and segments[3] == "code":
            self._record(self.store.get_experiment, segments[2])
            base = self.store.bundle_dir(segments[2]) / "code"
            if len(segments) == 4:
                files = []
                for parent, dirs, names in os.walk(base, followlinks=False):
                    dirs[:] = sorted(d for d in dirs if not (Path(parent) / d).is_symlink())
                    files.extend((Path(parent) / name).relative_to(base).as_posix() for name in sorted(names) if not (Path(parent) / name).is_symlink())
                return _json_response(200, {"files": files})
            try:
                return _json_response(200, _preview(base, unquote("/".join(segments[4:]))))
            except OSError:
                raise _HttpError(404, "Source file is not available") from None
        if len(segments) == 5 and segments[:2] == ["api", "programs"] and segments[3] == "calls":
            program = self._record(self.engine.get, segments[2])
            if not any(c["id"] == segments[4] for c in program["calls"]):
                raise _HttpError(404, "Model call does not exist")
            base = self.engine.folder(segments[2]) / "calls"
            records = {}
            for kind, suffix in (("request", ".request.json"), ("response", ".json")):
                try:
                    records[kind] = _preview(base, segments[4] + suffix, limit=262144)
                except FileNotFoundError:
                    records[kind] = None
            return _json_response(200, records)
        if self._is_post_route(segments):
            raise _HttpError(405, "This route accepts POST only")
        raise _HttpError(404, "Not found")

    def _post(self, path: str, segments: list[str], body: bytes | None) -> _Response:
        if segments == ["api", "programs"]:
            return _json_response(201, self.engine.create(self._json_body(body)))
        if len(segments) == 4 and segments[:2] == ["api", "programs"] and segments[3] in {"start", "pause", "resume", "stop"}:
            program_id, action = segments[2], segments[3]
            self._record(self.engine.get, program_id)  # Unknown programs 404 before any state change.
            if action == "start":
                return _json_response(200, self.engine.start(program_id, background=True))
            if action == "resume":
                return _json_response(200, self.engine.resume(program_id, background=True))
            if action == "pause":
                return _json_response(200, self.engine.pause(program_id))
            return _json_response(200, self.engine.stop(program_id))
        if len(segments) == 4 and segments[:2] == ["api", "experiments"] and segments[3] == "run":
            self._record(self.store.get_experiment, segments[2])
            return _json_response(200, run_experiment(self.store, segments[2], wait=False))
        if len(segments) == 4 and segments[:2] == ["api", "attempts"] and segments[3] == "stop":
            self._record(self.store.get_attempt, segments[2])
            return _json_response(200, stop_attempt(self.store, segments[2], wait=False))
        if self._is_get_route(path, segments):
            raise _HttpError(405, "This route accepts GET only")
        raise _HttpError(404, "Not found")

    @staticmethod
    def _is_post_route(segments: list[str]) -> bool:
        if segments == ["api", "programs"]:
            return True
        return len(segments) == 4 and segments[0] == "api" and (
            (segments[1] == "programs" and segments[3] in {"start", "pause", "resume", "stop"})
            or (segments[1] == "experiments" and segments[3] == "run")
            or (segments[1] == "attempts" and segments[3] == "stop")
        )

    @staticmethod
    def _is_get_route(path: str, segments: list[str]) -> bool:
        if path == "/" or path in {"/app.js", "/style.css"}:
            return True
        if segments == ["api", "overview"]:
            return True
        if len(segments) == 3 and segments[0] == "api" and segments[1] in {"programs", "experiments", "attempts"}:
            return True
        if len(segments) >= 4 and segments[0] == "api":
            if segments[1] == "programs" and segments[3] == "report":
                return True
            if segments[1] == "attempts" and segments[3] in {"logs", "artifacts"}:
                return True
        return False

    # -- data helpers -----------------------------------------------------

    @staticmethod
    def _record(loader: Any, *args: str) -> Any:
        """Pure record lookups: unknown or malformed IDs are 404 (the store raises LabError)."""
        try:
            return loader(*args)
        except KeyError as exc:  # Defense in depth if an implementation signals absence this way.
            missing = exc.args[0] if exc.args else exc
            raise _HttpError(404, f"No such record: {missing}") from None
        except LabError as exc:
            raise _HttpError(404, str(exc)) from None

    @staticmethod
    def _json_body(body: bytes | None) -> dict:
        if not body:
            raise _HttpError(400, "A JSON body is required")
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _HttpError(400, f"Invalid JSON body: {exc}") from exc
        if not isinstance(value, dict):
            raise _HttpError(400, "JSON body must be an object")
        return value

    def _overview(self) -> _Response:
        try:
            runner = dict(configuration(self.store))
            runner["configured"] = True
        except LabError as exc:
            runner = {"configured": False, "detail": str(exc)}
        return _json_response(200, {
            "experiments": self.store.list_experiments(),
            "attempts": self.store.list_attempts(active_only=False),
            "programs": self.engine.list(),
            "runner": runner,
        })

    def _attempt_with_events(self, attempt_id: str) -> dict:
        attempt = self._record(self.store.get_attempt, attempt_id)
        events = self.store.events(attempt_id)
        return {**attempt, "events": list(events)}

    def _attempt_logs(self, attempt_id: str) -> dict:
        self._record(self.store.get_attempt, attempt_id)  # Unknown attempts 404 before any filesystem access.
        log_path = self.store.attempt_dir(attempt_id) / "console.log"
        try:
            return _preview(log_path.parent, log_path.name, tail=True, limit=LOG_TAIL_BYTES)
        except FileNotFoundError:
            return {"text": "", "truncated": False}
        except OSError as exc:
            raise LabError(f"Cannot read console log {log_path}: {exc}") from exc

    def _artifact(self, attempt_id: str, raw_path: str) -> _Response:
        attempt = self._record(self.store.get_attempt, attempt_id)
        if attempt["state"] not in TERMINAL_STATES:
            raise _HttpError(409, "Artifacts are available after the attempt finishes")
        marker = f"/api/attempts/{attempt_id}/artifacts/"
        relative = unquote(raw_path[len(marker):])
        try:
            relative = relative_path(relative, "Artifact path")
        except LabError:
            raise _HttpError(404, "Artifact not found") from None
        record = next(
            (row for row in (attempt.get("result") or {}).get("artifacts", []) if row.get("path") == relative),
            None,
        )
        if record is None:
            raise _HttpError(404, "Artifact is not listed in this attempt's record")
        base = self.store.attempt_dir(attempt_id) / "output"
        try:
            fd = _open_regular(base, relative)
        except (OSError, LabError):
            raise _HttpError(404, "Artifact cannot be opened as a regular unlinked file") from None
        try:
            open_stat = os.fstat(fd)
            if open_stat.st_size != record.get("bytes"):
                raise _HttpError(409, "Artifact no longer matches its recorded size")
            digest = hashlib.sha256()
            while chunk := os.read(fd, 262144):
                digest.update(chunk)
            if digest.hexdigest() != record.get("sha256"):
                raise _HttpError(409, "Artifact no longer matches its recorded hash")
            os.lseek(fd, 0, os.SEEK_SET)
        except BaseException:
            os.close(fd)
            raise
        name = relative.rsplit("/", 1)[-1]
        safe = name if len(name) <= 128 and all(ch.isalnum() or ch in "._-" for ch in name) and name.isascii() else "artifact"
        disposition = f'attachment; filename="{safe}"'
        if name != safe:
            disposition += f"; filename*=UTF-8''{quote(name)}"
        headers = _headers("application/octet-stream", [("Content-Disposition", disposition)])
        return _Response(200, headers, file_fd=fd, file_size=open_stat.st_size)

    # -- static assets ------------------------------------------------------

    def _index(self) -> _Response:
        try:
            html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        except OSError as exc:
            raise LabError(f"Dashboard index.html is missing: {exc}") from exc
        if TOKEN_PLACEHOLDER not in html:
            raise LabError("Dashboard index.html does not contain the token placeholder")
        body = html.replace(TOKEN_PLACEHOLDER, self.token).encode("utf-8")
        return _Response(200, _headers("text/html; charset=utf-8", [("Content-Security-Policy", CSP)]), body)

    def _static(self, name: str, content_type: str) -> _Response:
        try:
            body = (STATIC_DIR / name).read_bytes()
        except OSError as exc:
            raise LabError(f"Dashboard asset {name} is missing: {exc}") from exc
        return _Response(200, _headers(content_type, [("Content-Security-Policy", CSP)]), body)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "arl-web"
    app: _App  # assigned as server.app by create_server

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    @property
    def _app(self) -> _App:
        return self.server.app  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._dispatch("GET", None)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            body = self._read_body()
        except _HttpError as exc:
            self.close_connection = True
            self._respond(_json_response(exc.status, {"error": exc.message}))
            return
        self._dispatch("POST", body)

    def _read_body(self) -> bytes:
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            raise _HttpError(400, "Chunked request bodies are not supported")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return b""
        try:
            length = int(raw_length)
        except ValueError:
            raise _HttpError(400, "Invalid Content-Length") from None
        if length < 0:
            raise _HttpError(400, "Invalid Content-Length")
        if length > MAX_BODY_BYTES:
            raise _HttpError(413, "Request body exceeds the 1 MiB JSON limit")
        body = self.rfile.read(length)
        if len(body) != length:
            raise _HttpError(400, "Request body was shorter than its Content-Length")
        return body

    def _dispatch(self, method: str, body: bytes | None) -> None:
        try:
            response = self._app.handle(method, self.path, self.headers, body)
        except Exception:  # pragma: no cover - belt and braces; handle() catches everything already.
            traceback.print_exc()
            response = _json_response(500, {"error": "Internal server error; details are in the server log"})
        self._respond(response)

    def _respond(self, response: _Response) -> None:
        try:
            self.send_response(response.status)
            if self.close_connection:
                self.send_header("Connection", "close")
            for key, value in response.headers:
                self.send_header(key, value)
            if response.file_fd is None:
                self.send_header("Content-Length", str(len(response.body)))
                self.end_headers()
                self.wfile.write(response.body)
            else:
                self.send_header("Content-Length", str(response.file_size))
                self.end_headers()
                remaining = response.file_size
                while remaining > 0:
                    chunk = os.read(response.file_fd, min(262144, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # Client hung up; nothing useful can be sent anymore.
        finally:
            if response.file_fd is not None:
                os.close(response.file_fd)

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        if code == "-" or (isinstance(code, int) and code >= 400):
            super().log_request(code, size)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _IPv6Server(_Server):
    address_family = socket.AF_INET6


def create_server(store: Store, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Create (but do not start) the dashboard server; useful for tests. serve() is the CLI path."""
    bound_host = _require_loopback(host)
    app = _App(store)
    server_class = _IPv6Server if ":" in bound_host else _Server
    server = server_class((bound_host, port), _Handler)
    server.app = app  # type: ignore[attr-defined]
    app.bound_port = server.server_address[1]
    return server


def serve(store: Store, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the research dashboard until interrupted. Loopback only."""
    server = create_server(store, host, port)
    address = server.server_address
    display = f"[{address[0]}]" if ":" in str(address[0]) else str(address[0])
    print(f"Research dashboard: http://{display}:{address[1]}/ (loopback only)", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
