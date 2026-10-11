"""Independent ChatGPT OAuth session; never import another application's grant.

Protocol: https://github.com/openai/codex/tree/main/codex-rs/login
Windows/WSL stores DPAPI-encrypted bytes, bound to the Windows user. Other hosts
require a supported native keyring; there is no plaintext credential fallback.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import platform
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlencode, urlsplit

from .execution import atomic_json
from .model_client import _NoRedirect, _strict_json_loads
from .models import LabError

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
TOKEN_BYTES = 128 * 1024


class CodexAuthError(LabError):
    """Only credential-safe messages may cross the authentication boundary."""


class _InvalidGrant(CodexAuthError):
    pass


def _safe_header(value):
    return isinstance(value, str) and 0 < len(value) <= 65536 and all(32 < ord(c) < 127 for c in value)


class _Vault:
    def __init__(self, root):
        self.folder = Path(root).resolve() / "auth"
        self.dpapi = sys.platform == "win32" or "microsoft" in platform.release().lower()
        self.name = "codex-" + sys.platform
        self.path = self.folder / (self.name + ".encrypted.json")
        self.service = "autonomous-research-lab.codex." + hashlib.sha256(str(self.folder).encode()).hexdigest() + "." + sys.platform

    def _keyring(self):
        try:
            import keyring
            backend = keyring.get_keyring()
            if (type(backend).__module__, type(backend).__name__) not in {
                ("keyring.backends.SecretService", "Keyring"),
                ("keyring.backends.libsecret", "Keyring"),
                ("keyring.backends.kwallet", "DBusKeyring"),
                ("keyring.backends.macOS", "Keyring"),
            }:
                raise ValueError()
            return backend
        except Exception:
            raise CodexAuthError("Codex requires a supported native keyring. Install the secure-keyring extra and unlock your OS vault; plaintext storage is not supported.") from None

    def _crypt(self, data, decrypt=False):
        executable = (Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
                      if sys.platform == "win32" else Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"))
        if not executable.is_file() or len(data) > TOKEN_BYTES * 2:
            raise CodexAuthError("Windows user-protected credential storage is unavailable in this process.")
        operation = "Unprotect" if decrypt else "Protect"
        command = ("$ErrorActionPreference='Stop'; Add-Type -AssemblyName System.Security; "
                   "$e=[Convert]::FromBase64String([Console]::In.ReadLine()); "
                   "$b=[Convert]::FromBase64String([Console]::In.ReadToEnd()); "
                   "$r=[Security.Cryptography.ProtectedData]::" + operation +
                   "($b,$e,[Security.Cryptography.DataProtectionScope]::CurrentUser); "
                   "[Console]::Out.Write([Convert]::ToBase64String($r))")
        try:
            # Secret bytes travel only through captured pipes, never arguments or logs.
            entropy = hashlib.sha256(self.service.encode("utf-8")).digest()
            process = subprocess.run([str(executable), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
                                     input=base64.b64encode(entropy) + b"\n" + base64.b64encode(data),
                                     capture_output=True, timeout=20, check=False)
            if process.returncode or len(process.stdout) > TOKEN_BYTES * 4:
                raise ValueError()
            return base64.b64decode(process.stdout.strip(), validate=True)
        except Exception:
            raise CodexAuthError("Windows credential protection failed. Use the same Windows account that signed in; no plaintext fallback was used.") from None

    def probe(self):
        if self.dpapi:
            if self._crypt(self._crypt(b"arl-credential-store-check"), decrypt=True) != b"arl-credential-store-check":
                raise CodexAuthError("Windows credential protection is unavailable.")
        else:
            self._keyring()

    def exists(self):
        if self.dpapi:
            return self.path.is_file()
        return self._keyring().get_password(self.service, "session") is not None

    def load(self):
        try:
            if self.dpapi:
                if not self.path.exists():
                    return None
                if self.path.stat().st_size > TOKEN_BYTES * 4:
                    raise ValueError()
                envelope = _strict_json_loads(self.path.read_text(encoding="utf-8"))
                if envelope.get("version") != 1:
                    raise ValueError()
                raw = self._crypt(base64.b64decode(envelope["ciphertext"], validate=True), decrypt=True).decode("utf-8")
            else:
                raw = self._keyring().get_password(self.service, "session")
                if raw is None:
                    return None
            if len(raw.encode("utf-8")) > TOKEN_BYTES * 2:
                raise ValueError()
            value = _strict_json_loads(raw)
            if not isinstance(value, dict) or any(not _safe_header(value.get(k)) for k in ("access_token", "refresh_token", "account_id")):
                raise ValueError()
            expiry = value.get("expires_at")
            if type(expiry) not in (int, float) or not math.isfinite(expiry) or expiry <= 0:
                raise ValueError()
            return value
        except CodexAuthError:
            raise
        except Exception:
            raise CodexAuthError("The saved Codex session cannot be read. Disconnect it and sign in again.") from None

    def save(self, value):
        try:
            raw = json.dumps(value, allow_nan=False, separators=(",", ":"))
            if self.dpapi:
                encrypted = self._crypt(raw.encode("utf-8"))
                atomic_json(self.path, {"version": 1, "ciphertext": base64.b64encode(encrypted).decode("ascii")})
            else:
                self._keyring().set_password(self.service, "session", raw)
        except CodexAuthError:
            raise
        except Exception:
            raise CodexAuthError("Could not save the Codex session in protected storage. Sign in again after fixing the vault.") from None

    def delete(self):
        try:
            if self.dpapi:
                self.path.unlink(missing_ok=True)
            else:
                backend = self._keyring()
                if backend.get_password(self.service, "session") is not None:
                    backend.delete_password(self.service, "session")
        except Exception:
            raise CodexAuthError("Could not remove the saved Codex session from protected storage.") from None


class CodexAuth:
    def __init__(self, root, *, vault=None, opener=None, login_seconds=300):
        self.vault = vault if vault is not None else _Vault(root)
        self.lock_path = Path(root).resolve() / "auth" / ("codex-" + sys.platform + ".lock")
        if type(login_seconds) not in (int, float) or not math.isfinite(login_seconds) or not 0 < login_seconds <= 600:
            raise ValueError("OAuth login timeout must be positive and at most 600 seconds")
        self.opener = opener or urllib.request.build_opener(_NoRedirect(), urllib.request.ProxyHandler({}))
        self.login_seconds = login_seconds
        self.mutex = threading.RLock()
        self.pending = None
        self.error = None
        self.generation = 0
        self.closed = False

    def _epoch(self):
        path = self.lock_path.with_suffix(".epoch")
        try:
            value = path.read_text(encoding="ascii")
        except FileNotFoundError:
            return None
        if len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
            raise CodexAuthError("Codex sign-in state is invalid; inspect the protected-store directory")
        return value

    def _invalidate_login(self):
        # Called under the process lock. A logout in another process must also
        # invalidate a web callback whose token exchange is already in flight.
        path = self.lock_path.with_suffix(".epoch")
        pending = path.with_name(path.name + "." + secrets.token_hex(8))
        try:
            pending.write_text(secrets.token_hex(16), encoding="ascii")
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)

    @contextmanager
    def _locked(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.lock_path.open("a+b") as lock:
            if os.name == "nt":
                import msvcrt
                if lock.tell() == 0:
                    lock.write(b"0")
                    lock.flush()
                def acquire():
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                def release():
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                def acquire():
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                def release():
                    fcntl.flock(lock, fcntl.LOCK_UN)
            deadline = time.monotonic() + 35
            while True:
                try:
                    acquire()
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise CodexAuthError("Codex authentication is busy in another process. Retry shortly.") from None
                    time.sleep(0.05)
            try:
                yield
            finally:
                release()

    def status(self):
        with self.mutex:
            attempt = self.pending
        if attempt and self._epoch() != attempt["epoch"]:
            self._finish(attempt, "Codex sign-in was cancelled in another process.")
        with self.mutex:
            pending, error = self.pending is not None, self.error
        try:
            connected = self.vault.exists()
        except Exception:
            return {"connected": False, "pending": pending, "error": "Protected credential storage is unavailable."}
        return {"connected": connected, "pending": pending, "error": error}

    def _exchange(self, fields):
        request = urllib.request.Request(TOKEN_URL, data=urlencode({"client_id": CLIENT_ID, **fields}).encode("ascii"),
                                         headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}, method="POST")
        try:
            with self.opener.open(request, timeout=30) as response:
                raw = response.read(TOKEN_BYTES + 1)
            if len(raw) > TOKEN_BYTES:
                raise ValueError()
            value = _strict_json_loads(raw.decode("utf-8"))
            access, refresh, expiry = value.get("access_token"), value.get("refresh_token"), value.get("expires_in")
            if not _safe_header(access) or not _safe_header(refresh) or type(expiry) not in (int, float) or not math.isfinite(expiry) or not 0 < expiry <= 366 * 86400:
                raise ValueError()
            payload = access.split(".")[1]
            claims = _strict_json_loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode())
            account = claims["https://api.openai.com/auth"]["chatgpt_account_id"]
            if not _safe_header(account):
                raise ValueError()
            # JWT decoding extracts metadata only; trust comes from the fixed HTTPS exchange.
            return {"access_token": access, "refresh_token": refresh, "account_id": account, "expires_at": time.time() + expiry}
        except urllib.error.HTTPError as exc:
            invalid = False
            try:
                payload = _strict_json_loads(exc.read(TOKEN_BYTES + 1).decode("utf-8"))
                detail = payload.get("error") if isinstance(payload, dict) else None
                code = detail.get("code") if isinstance(detail, dict) else detail
                invalid = fields.get("grant_type") == "refresh_token" and code in {
                    "invalid_grant", "refresh_token_expired", "refresh_token_reused", "refresh_token_invalid"}
            except Exception:
                pass
            finally:
                exc.close()
            if invalid:
                raise _InvalidGrant("Codex authorization expired or was revoked. Sign in again.") from None
            raise CodexAuthError("Codex authorization was rejected. Sign in again; no model request was retried.") from None
        except Exception:
            raise CodexAuthError("Codex authorization could not be completed. Check the connection and sign in again.") from None

    def credentials(self):
        with self.mutex:
            if self.closed:
                raise CodexAuthError("Codex authentication is closed")
            generation = self.generation
        with self._locked():
            value = self.vault.load()
            if value is None:
                raise CodexAuthError("Codex is not connected. Sign in with ChatGPT in the dashboard or run arl provider login.")
            if value["expires_at"] <= time.time() + 60:
                try:
                    value = self._exchange({"grant_type": "refresh_token", "refresh_token": value["refresh_token"]})
                except _InvalidGrant:
                    self.vault.delete()
                    raise
                with self.mutex:
                    if generation != self.generation:
                        raise CodexAuthError("Codex authentication was disconnected during refresh.")
                    self.vault.save(value)
            with self.mutex:
                if generation != self.generation:
                    raise CodexAuthError("Codex authentication was disconnected.")
            return {"access_token": value["access_token"], "account_id": value["account_id"]}

    @staticmethod
    def _stop(attempt):
        if attempt:
            attempt["timer"].cancel()
            server = attempt["server"]
            server.shutdown()
            server.server_close()

    def _finish(self, attempt, error=None):
        with self.mutex:
            if self.pending is not attempt:
                return
            self.pending = None
            self.error = error
        self._stop(attempt)

    def begin_login(self):
        self.vault.probe()
        with self._locked():
            epoch = self._epoch()
        with self.mutex:
            if self.closed:
                raise CodexAuthError("Codex authentication is closed")
            if self.pending is not None:
                raise CodexAuthError("A Codex sign-in is already pending. Finish or cancel it first.")
            attempt = {"state": secrets.token_urlsafe(32), "verifier": secrets.token_urlsafe(64),
                       "deadline": time.monotonic() + self.login_seconds, "consumed": False, "epoch": epoch}
            owner = self
            class Callback(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass  # OAuth codes and state never enter request logs.
                def do_GET(self):
                    status, message = owner._callback(attempt, self.path, self.headers.get_all("Host", []), self.headers.get_all("Origin", []))
                    body = message.encode()
                    self.send_response(status)
                    for key, value in (("Content-Type", "text/plain; charset=utf-8"), ("Cache-Control", "no-store"),
                                       ("Referrer-Policy", "no-referrer"), ("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'"),
                                       ("Content-Length", str(len(body))), ("Connection", "close")):
                        self.send_header(key, value)
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except OSError:
                        pass
            class Server(ThreadingHTTPServer):
                daemon_threads = True
                allow_reuse_address = os.name != "nt"
                def server_bind(self):
                    if os.name == "nt":
                        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    super().server_bind()
                def get_request(self):
                    connection, address = super().get_request()
                    connection.settimeout(5)
                    return connection, address
                def handle_error(self, *args):
                    pass
            try:
                attempt["server"] = Server(("127.0.0.1", 1455), Callback)
            except OSError:
                raise CodexAuthError("Cannot open Codex sign-in on localhost:1455. Finish another sign-in before retrying.") from None
            attempt["timer"] = threading.Timer(self.login_seconds, self._finish, (attempt, "Codex sign-in expired. Start a new sign-in."))
            attempt["timer"].daemon = True
            self.pending, self.error = attempt, None
            threading.Thread(target=attempt["server"].serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
            attempt["timer"].start()
        challenge = base64.urlsafe_b64encode(hashlib.sha256(attempt["verifier"].encode()).digest()).rstrip(b"=").decode()
        params = {"response_type": "code", "client_id": CLIENT_ID, "redirect_uri": REDIRECT_URI,
                  "scope": "openid profile email offline_access", "code_challenge": challenge, "code_challenge_method": "S256",
                  "state": attempt["state"], "id_token_add_organizations": "true", "codex_cli_simplified_flow": "true", "originator": "autonomous-research-lab"}
        return {"authorization_url": AUTHORIZE_URL + "?" + urlencode(params), "expires_in": self.login_seconds}

    def _callback(self, attempt, path, hosts, origins):
        if len(hosts) != 1 or hosts[0] not in ("localhost:1455", "127.0.0.1:1455") or (origins and origins != ["https://auth.openai.com"]):
            return 403, "Request rejected."
        try:
            url = urlsplit(path)
            if url.scheme or url.netloc or url.fragment or url.path != "/auth/callback":
                return 404, "Request rejected."
            query = parse_qs(url.query, keep_blank_values=True, strict_parsing=True, max_num_fields=16)
            state, code, error = query.get("state", []), query.get("code", []), query.get("error", [])
            if len(state) != 1 or not state[0].isascii() or not secrets.compare_digest(state[0], attempt["state"]):
                return 400, "Invalid OAuth state."
            if not ((len(code) == 1 and _safe_header(code[0]) and not error) or (len(error) == 1 and not code)):
                return 400, "Invalid OAuth response."
        except (ValueError, UnicodeError):
            return 400, "Invalid OAuth response."
        with self.mutex:
            if self.pending is not attempt or attempt["consumed"] or time.monotonic() >= attempt["deadline"]:
                return 409, "Sign-in is no longer active."
            attempt["consumed"] = True
        if error:
            self._finish(attempt, "Codex sign-in was declined. Start a new sign-in if needed.")
            return 400, "Sign-in was not completed."
        try:
            value = self._exchange({"grant_type": "authorization_code", "code": code[0], "code_verifier": attempt["verifier"], "redirect_uri": REDIRECT_URI})
            with self._locked():
                with self.mutex:
                    if self.pending is not attempt or time.monotonic() >= attempt["deadline"] or self._epoch() != attempt["epoch"]:
                        return 409, "Sign-in is no longer active."
                    self.vault.save(value)
                    self.pending, self.error = None, None
            self._stop(attempt)
            return 200, "Codex connected. Return to Autonomous Research Lab."
        except CodexAuthError as exc:
            self._finish(attempt, str(exc))
            return 503, "Sign-in failed. Return to the dashboard for details."

    def cancel_login(self):
        with self.mutex:
            attempt, self.pending = self.pending, None
            self.error = None
        self._stop(attempt)
        with self._locked():
            self._invalidate_login()
        return self.status()

    def logout(self):
        with self.mutex:
            self.generation += 1
        self.cancel_login()
        with self._locked():
            self._invalidate_login()
            self.vault.delete()
        return self.status()

    def close(self):
        with self.mutex:
            self.closed = True
            pending = self.pending is not None
        if pending:
            self.cancel_login()
