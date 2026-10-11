"""Single-use OAuth, late callbacks, refresh serialization, and protected storage."""
import base64
from concurrent.futures import ThreadPoolExecutor
import http.client
import io
import json
import multiprocessing
from multiprocessing.managers import SyncManager
from pathlib import Path
import platform
import shutil
import sys
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit, urlencode
import uuid

from autonomous_research_lab.codex_auth import CodexAuth, CodexAuthError, _Vault
from autonomous_research_lab.research import ResearchEngine
from autonomous_research_lab.store import Store


def token_response():
    claims = base64.urlsafe_b64encode(json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "fixture-account"}}).encode()).rstrip(b"=").decode()
    return {"access_token": "fixture." + claims + ".signature", "refresh_token": "fixture-rotated-grant", "expires_in": 3600}


def expired_session():
    return {"access_token": "fixture-expired", "refresh_token": "fixture-old-grant", "account_id": "fixture-account", "expires_at": 1}


class MemoryVault:
    def __init__(self, values=None):
        self.values = values if values is not None else {}

    def probe(self):
        pass

    def exists(self):
        return self.values.get("session") is not None

    def load(self):
        value = self.values.get("session")
        return dict(value) if value is not None else None

    def save(self, value):
        self.values["session"] = dict(value)

    def delete(self):
        self.values.pop("session", None)


class TokenEndpoint:
    def __init__(self, requests=None):
        self.requests = requests if requests is not None else []
        self.entered = threading.Event()
        self.release = None
        self.failure = None

    def open(self, request, timeout):
        self.requests.append(parse_qs(request.data.decode()))
        self.entered.set()
        if self.release is not None and not self.release.wait(5):
            raise AssertionError("exchange barrier was not released")
        if self.failure is not None:
            raise self.failure
        return io.BytesIO(json.dumps(token_response()).encode())


def refresh_worker(root, values, requests, barrier, output):
    auth = CodexAuth(root, vault=MemoryVault(values), opener=TokenEndpoint(requests))
    try:
        barrier.wait(timeout=10)
        output.put(auth.credentials())
    except Exception as error:
        output.put({"error": type(error).__name__})
    finally:
        auth.close()


class CodexAuthTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / ".arl-cache" / "tests" / uuid.uuid4().hex
        self.vault, self.endpoint = MemoryVault(), TokenEndpoint()
        self.auth = CodexAuth(self.root, vault=self.vault, opener=self.endpoint)

    def tearDown(self):
        self.auth.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def login(self):
        url = self.auth.begin_login()["authorization_url"]
        state = parse_qs(urlsplit(url).query)["state"][0]
        return self.auth.pending, "/auth/callback?" + urlencode({"state": state, "code": "fixture-authorization-code"})

    def http_get(self, path, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", 1455, timeout=5)
        try:
            connection.request("GET", path, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read().decode()
        finally:
            connection.close()

    def test_bad_callbacks_do_not_consume_the_valid_login(self):
        attempt, path = self.login()
        invalid = [
            ("/auth/callback?code=code", {}),
            (path + "&state=duplicate", {}),
            (path.replace(attempt["state"], "wrong"), {}),
            (path + "&code=duplicate", {}),
            (path + "&error=denied", {}),
            (path, {"Host": "attacker.example"}),
            (path, {"Origin": "https://attacker.example"}),
            (path.replace("/auth/callback", "/wrong"), {}),
        ]
        for query, headers in invalid:
            with self.subTest(query=query, headers=headers):
                self.assertIn(self.http_get(query, headers)[0], (400, 403, 404))
                self.assertTrue(self.auth.status()["pending"])
        self.assertEqual(self.endpoint.requests, [])
        status, body = self.http_get(path)
        self.assertEqual(status, 200)
        self.assertTrue(self.auth.status()["connected"])
        for secret in (attempt["state"], attempt["verifier"], "fixture-authorization-code", token_response()["access_token"]):
            self.assertNotIn(secret, body + json.dumps(self.auth.status()))
        self.assertEqual(self.auth._callback(attempt, path, ["localhost:1455"], [])[0], 409)
        self.assertEqual(len(self.endpoint.requests), 1)

    def test_duplicate_callback_cannot_exchange_twice(self):
        attempt, path = self.login()
        self.endpoint.release = threading.Event()
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(self.auth._callback, attempt, path, ["localhost:1455"], [])
            try:
                self.assertTrue(self.endpoint.entered.wait(5))
                self.assertEqual(self.auth._callback(attempt, path, ["localhost:1455"], [])[0], 409)
            finally:
                self.endpoint.release.set()
            self.assertEqual(pending.result(timeout=5)[0], 200)
        self.assertEqual(len(self.endpoint.requests), 1)

    def test_cancel_and_server_close_defeat_late_exchange(self):
        for action in ("cancel_login", "close"):
            with self.subTest(action=action):
                self.auth = CodexAuth(self.root, vault=self.vault, opener=TokenEndpoint())
                self.endpoint = self.auth.opener
                self.endpoint.release = threading.Event()
                attempt, path = self.login()
                with ThreadPoolExecutor(1) as pool:
                    pending = pool.submit(self.auth._callback, attempt, path, ["localhost:1455"], [])
                    try:
                        self.assertTrue(self.endpoint.entered.wait(5))
                        getattr(self.auth, action)()
                    finally:
                        self.endpoint.release.set()
                    self.assertEqual(pending.result(timeout=5)[0], 409)
                self.assertFalse(self.vault.exists())
                self.assertFalse(self.auth.status()["pending"])
                self.auth.close()

    def test_other_controller_logout_defeats_inflight_login(self):
        attempt, path = self.login()
        self.endpoint.release = threading.Event()
        other = CodexAuth(self.root, vault=self.vault, opener=TokenEndpoint())
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(self.auth._callback, attempt, path, ["localhost:1455"], [])
            try:
                self.assertTrue(self.endpoint.entered.wait(5))
                other.logout()
            finally:
                self.endpoint.release.set()
                other.close()
            self.assertEqual(pending.result(timeout=5)[0], 409)
        self.assertFalse(self.auth.status()["connected"])
        self.assertFalse(self.auth.status()["pending"])

    def test_expired_attempt_cannot_commit_a_completed_exchange(self):
        attempt, path = self.login()
        self.endpoint.release = threading.Event()
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(self.auth._callback, attempt, path, ["localhost:1455"], [])
            try:
                self.assertTrue(self.endpoint.entered.wait(5))
                attempt["deadline"] = time.monotonic() - 1
            finally:
                self.endpoint.release.set()
            self.assertEqual(pending.result(timeout=5)[0], 409)
        self.assertFalse(self.vault.exists())

    def test_cancelled_state_cannot_authorize_the_next_login(self):
        old_attempt, old_path = self.login()
        self.auth.cancel_login()
        attempt, path = self.login()
        self.assertEqual(self.auth._callback(attempt, old_path, ["localhost:1455"], [])[0], 400)
        self.assertEqual(self.auth._callback(old_attempt, old_path, ["localhost:1455"], [])[0], 409)
        self.assertEqual(self.auth._callback(attempt, path, ["localhost:1455"], [])[0], 200)

    def test_transient_refresh_failure_preserves_grant_but_revocation_deletes_it(self):
        for code, retained in (("server_error", True), ("invalid_grant", False)):
            with self.subTest(code=code):
                self.vault.save(expired_session())
                self.endpoint.failure = HTTPError("https://auth.openai.com/oauth/token", 400, "rejected", {}, io.BytesIO(json.dumps({"error": code, "detail": "fixture-old-grant"}).encode()))
                with self.assertRaises(CodexAuthError) as raised:
                    self.auth.credentials()
                self.assertEqual(self.vault.exists(), retained)
                self.assertNotIn("fixture-old-grant", str(raised.exception))

    def test_separate_processes_rotate_the_shared_grant_once(self):
        context = multiprocessing.get_context("spawn")
        with SyncManager(address=("127.0.0.1", 0), ctx=context) as manager:
            values = manager.dict({"session": expired_session()})
            requests, output = manager.list(), context.Queue()
            barrier = context.Barrier(2)
            processes = [context.Process(target=refresh_worker, args=(str(self.root), values, requests, barrier, output)) for _ in range(2)]
            try:
                for process in processes:
                    process.start()
                results = [output.get(timeout=20) for _ in processes]
                for process in processes:
                    process.join(timeout=10)
                    self.assertEqual(process.exitcode, 0)
                self.assertEqual(len(requests), 1)
                self.assertEqual(results, [{"access_token": token_response()["access_token"], "account_id": "fixture-account"}] * 2)
                self.assertEqual(values["session"]["refresh_token"], "fixture-rotated-grant")
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                output.close()
                output.join_thread()

    def test_missing_signin_blocks_before_model_budget_or_execution(self):
        engine = ResearchEngine(Store(self.root))
        engine.auth = self.auth
        program = engine.create({"goal": "Investigate an approved question", "model": {"provider": "openai-codex", "model": "protocol-fixture"}})
        result = engine.start(program["id"], background=False)
        self.assertEqual(result["state"], "blocked")
        self.assertEqual(result["calls"], [])
        self.assertEqual(result["implementations"], [])

    @unittest.skipUnless(sys.platform == "win32" or "microsoft" in platform.release().lower(), "DPAPI is Windows/WSL-specific")
    def test_real_dpapi_is_ciphertext_only_and_bound_to_store_root(self):
        vault = _Vault(self.root)
        vault.folder.mkdir(parents=True)
        value = {**expired_session(), "expires_at": time.time() + 3600}
        vault.save(value)
        encrypted = vault.path.read_text()
        for key in ("access_token", "refresh_token", "account_id"):
            self.assertNotIn(value[key], encrypted)
        self.assertEqual(vault.load(), value)
        other = _Vault(self.root / "other-store")
        other.folder.mkdir(parents=True)
        shutil.copyfile(vault.path, other.path)
        with self.assertRaises(CodexAuthError):
            other.load()
        vault.delete()
        self.assertFalse(vault.exists())
