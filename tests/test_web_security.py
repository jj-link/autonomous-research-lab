import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import unittest
from unittest.mock import Mock
import uuid

from autonomous_research_lab.models import ExperimentSpec
from autonomous_research_lab.store import Store
from autonomous_research_lab.web import _Handler, _Response, create_server


@unittest.skipUnless(sys.platform == "linux", "No-follow artifact access uses Linux directory descriptors")
class WebSecurityTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(__file__).resolve().parents[1] / ".arl-cache" / "tests" / uuid.uuid4().hex
        source = self.folder / "source"
        source.mkdir(parents=True)
        (source / "main.py").write_text("print('fixture')\n")
        self.store = Store(self.folder / "state")
        experiment = self.store.submit(ExperimentSpec.from_dict({"name": "fixture", "source": str(source), "entrypoint": "main.py"}, self.folder))
        attempt = self.store.create_attempt(experiment["id"])
        self.id = attempt["id"]
        self.output = self.store.attempt_dir(self.id) / "output"
        (self.output / "nested").mkdir()
        (self.output / "nested" / "data.txt").write_bytes(b"visible")
        result = {"state": "completed", "artifacts": [{"path": "nested/data.txt", "bytes": 7, "sha256": hashlib.sha256(b"visible").hexdigest()}]}
        self.store.update_attempt(self.id, state="starting")
        self.store.update_attempt(self.id, state="running")
        self.store.update_attempt(self.id, state="completed", result=result)
        self.server = create_server(self.store, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.origin = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        shutil.rmtree(self.folder)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_mutations_require_token_and_exact_origin(self):
        brief = {"goal": "Security check", "model": {"base_url": "https://api.example.com/v1", "model": "fixture"}}
        token = self.server.app.token
        for headers in ({}, {"X-ARL-Token": "wrong"}, {"X-ARL-Token": token, "Origin": "https://127.0.0.1:" + str(self.server.server_port)},
                        {"X-ARL-Token": token, "Origin": "http://localhost:" + str(self.server.server_port)}, {"X-ARL-Token": token, "Origin": "http://127.0.0.1"}):
            with self.subTest(headers=headers):
                status, _, _ = self.request("POST", "/api/programs", brief, headers)
                self.assertEqual(status, 403)
        self.assertEqual(self.server.app.engine.list(), [])
        status, _, body = self.request("POST", "/api/programs", brief, {"X-ARL-Token": token, "Origin": self.origin})
        self.assertEqual(status, 201)
        program_id = json.loads(body)["id"]
        for action, expected in (("pause", "paused"), ("stop", "stopped")):
            status, _, body = self.request("POST", f"/api/programs/{program_id}/{action}", {}, {"X-ARL-Token": token, "Origin": self.origin})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["state"], expected)

    def test_malformed_host_authorities_are_rejected(self):
        for host in ("evil.example", "127.0.0.1:1", f"attacker@127.0.0.1:{self.server.server_port}", f"127.0.0.1:{self.server.server_port}/path"):
            with self.subTest(host=host):
                self.assertEqual(self.request("GET", "/api/overview", headers={"Host": host})[0], 403)

    def test_artifact_download_checks_inventory_hash_and_every_path_component(self):
        path = f"/api/attempts/{self.id}/artifacts/nested/data.txt"
        status, headers, body = self.request("GET", path)
        self.assertEqual((status, body), (200, b"visible"))
        self.assertTrue(headers["Content-Disposition"].startswith("attachment;"))
        (self.output / "nested" / "data.txt").write_bytes(b"changed")
        self.assertEqual(self.request("GET", path)[0], 409)
        (self.output / "nested" / "data.txt").unlink()
        (self.output / "secret").write_bytes(b"private")
        (self.output / "nested" / "data.txt").symlink_to(self.output / "secret")
        self.assertEqual(self.request("GET", path)[0], 404)
        (self.output / "nested" / "data.txt").unlink()
        (self.output / "nested").rmdir()
        (self.output / "elsewhere").mkdir()
        (self.output / "elsewhere" / "data.txt").write_bytes(b"visible")
        (self.output / "nested").symlink_to(self.output / "elsewhere", target_is_directory=True)
        self.assertEqual(self.request("GET", path)[0], 404)
        self.assertEqual(self.request("GET", f"/api/attempts/{self.id}/artifacts/secret")[0], 404)

    def test_console_log_symlinks_cannot_disclose_host_files(self):
        private = self.folder / "private"
        private.write_bytes(b"not-for-the-browser")
        (self.store.attempt_dir(self.id) / "console.log").symlink_to(private)
        status, _, body = self.request("GET", f"/api/attempts/{self.id}/logs")
        self.assertIn(status, {400, 404})
        self.assertNotIn(b"not-for-the-browser", body)

    def test_oversized_body_is_rejected_without_reading_it(self):
        status, headers, _ = self.request("POST", "/api/programs", headers={"Content-Length": str(1024**2 + 1)})
        self.assertEqual(status, 413)
        self.assertEqual(headers["Connection"], "close")

    def test_artifact_descriptor_closes_if_client_disconnects_during_headers(self):
        descriptor = os.open(self.output / "nested" / "data.txt", os.O_RDONLY)
        handler = object.__new__(_Handler)
        handler.send_response = Mock(side_effect=BrokenPipeError)
        handler._respond(_Response(200, file_fd=descriptor, file_size=7))
        with self.assertRaises(OSError):
            os.fstat(descriptor)


if __name__ == "__main__":
    unittest.main()
