"""Opt-in real rootless Podman tests: ARL_TEST_IMAGE=localhost/arl-runtime:0.1."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import time
import unittest
from unittest.mock import patch
import uuid

from autonomous_research_lab import execution
from autonomous_research_lab.execution import (
    attempt_lock, configure, container_name, environment, inspect_owned,
    podman, recover, run, stop, supervise,
)
from autonomous_research_lab.models import ExperimentSpec, TERMINAL_STATES
from autonomous_research_lab.store import Store


@unittest.skipUnless(os.environ.get("ARL_TEST_IMAGE") and os.name == "posix", "set ARL_TEST_IMAGE on Linux for real container tests")
class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(__file__).resolve().parents[1] / ".arl-cache" / "tests" / uuid.uuid4().hex
        self.source = self.folder / "source"
        self.source.mkdir(parents=True)
        self.store = Store(self.folder / "state")
        configure(self.store, os.environ["ARL_TEST_IMAGE"])

    def tearDown(self):
        for attempt in self.store.list_attempts():
            if attempt["state"] not in TERMINAL_STATES:
                stop(self.store, attempt["id"], wait=True)
            container = inspect_owned(self.store, attempt["id"])
            if container:
                podman("rm", container_name(attempt["id"]))
        shutil.rmtree(self.folder)

    def submit(self, code, *, inputs=None, **limits):
        (self.source / "main.py").write_text(code, encoding="utf-8")
        budget = {"wall_seconds": 8, "memory_bytes": 64 * 1024**2, "output_bytes": 1024**2, "log_bytes": 65536, "pids": 32}
        budget.update(limits)
        return self.store.submit(ExperimentSpec.from_dict({"name": "integration", "source": str(self.source), "entrypoint": "main.py", "limits": budget, "inputs": inputs or {}}, self.folder))

    def await_running(self, attempt_id):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            attempt = self.store.get_attempt(attempt_id)
            if attempt["state"] == "running":
                return attempt
            if attempt["state"] in TERMINAL_STATES:
                self.fail(f"Did not reach running: {attempt}")
            time.sleep(0.05)
        log = self.store.attempt_dir(attempt_id) / "supervisor.log"
        self.fail(f"Supervisor startup did not finish: {self.store.get_attempt(attempt_id)}; {log.read_text() if log.exists() else 'no supervisor log'}")

    def test_snapshot_and_container_boundaries(self):
        experiment = self.submit('''import json, os, pathlib, socket
assert os.environ.get("ARL_BOUNDARY_TEST_SECRET") is None
assert not pathlib.Path("/mnt/c/Users").exists()
try:
    pathlib.Path("/experiment/code/main.py").write_text("changed")
except OSError:
    pass
else:
    raise AssertionError("bundle is writable")
sock = socket.socket()
sock.settimeout(0.2)
try:
    sock.connect(("1.1.1.1", 80))
except OSError:
    pass
else:
    raise AssertionError("outbound network is available")
finally:
    sock.close()
assert pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip() == "67108864"
pathlib.Path("/workspace/results.json").write_text(json.dumps({"metrics": {"answer": 42}}))
pathlib.Path("/workspace/escape").symlink_to("/etc/passwd")
print("boundary checks passed")
''')
        (self.source / "main.py").write_text("raise RuntimeError('mutable original used')", encoding="utf-8")
        previous = os.environ.get("ARL_BOUNDARY_TEST_SECRET")
        os.environ["ARL_BOUNDARY_TEST_SECRET"] = "must-not-enter-container"
        try:
            result = run(self.store, experiment["id"], wait=True)
        finally:
            if previous is None:
                os.environ.pop("ARL_BOUNDARY_TEST_SECRET", None)
            else:
                os.environ["ARL_BOUNDARY_TEST_SECRET"] = previous
        self.assertEqual(result["state"], "completed", result)
        self.assertEqual(result["result"]["observations"]["metrics"]["answer"], 42)
        self.assertNotIn("escape", {row["path"] for row in result["result"]["artifacts"]})
        self.assertTrue(any("escape" in warning for warning in result["result"]["artifact_warnings"]))

    def test_spawned_processes_share_memory_without_writable_root(self):
        experiment = self.submit('''import json, multiprocessing
from pathlib import Path

def increment(counter):
    for _ in range(1000):
        with counter.get_lock():
            counter.value += 1

if __name__ == "__main__":
    for folder in ("/", "/tmp", "/run"):
        try:
            (Path(folder) / "must-not-write").write_text("unsafe")
        except OSError:
            pass
        else:
            raise AssertionError(f"{folder} became writable")
    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    children = [context.Process(target=increment, args=(counter,)) for _ in range(2)]
    for child in children:
        child.start()
    for child in children:
        child.join(15)
        assert child.exitcode == 0, child.exitcode
    Path("/workspace/results.json").write_text(json.dumps({"metrics": {"increments": counter.value}}))
''', wall_seconds=30, memory_bytes=128 * 1024**2)
        result = run(self.store, experiment["id"], wait=True)
        self.assertEqual(result["state"], "completed", result)
        self.assertEqual(result["result"]["observations"]["metrics"]["increments"], 2000)

    def test_only_approved_input_is_readable_and_cannot_be_modified(self):
        data = self.folder / "approved"
        data.mkdir()
        (data / "value.txt").write_text("verified checkpoint contents")
        (self.folder / "private.txt").write_text("must not be mounted")
        approved = self.store.register_input(data, "checkpoint")
        experiment = self.submit('''import json, os
from pathlib import Path
path = Path(json.loads(os.environ["ARL_INPUTS"])["model"])
assert (path / "value.txt").read_text() == "verified checkpoint contents"
assert not (path / "../private.txt").exists()
try:
    (path / "value.txt").write_text("tampered")
except OSError:
    pass
else:
    raise AssertionError("approved input is writable")
Path("/workspace/results.json").write_text(json.dumps({"metrics": {"input_read_only": 1}}))
''', inputs={"model": approved["id"]})
        result = run(self.store, experiment["id"], wait=True)
        self.assertEqual(result["state"], "completed", result)
        self.assertEqual(result["result"]["observations"]["metrics"]["input_read_only"], 1)
        self.assertEqual((data / "value.txt").read_text(), "verified checkpoint contents")

    def test_changed_input_prevents_container_execution(self):
        data = self.folder / "approved"
        data.mkdir()
        file = data / "weights.bin"
        file.write_bytes(b"original")
        approved = self.store.register_input(data, "checkpoint")
        experiment = self.submit("raise AssertionError('must not execute')\n", inputs={"model": approved["id"]})
        file.write_bytes(b"modified")
        result = run(self.store, experiment["id"], wait=True)
        self.assertEqual(result["state"], "failed", result)
        self.assertIsNone(inspect_owned(self.store, result["id"]))

    def test_different_rootless_mount_view_prevents_execution(self):
        data = self.folder / "approved"
        data.mkdir()
        (data / "data.txt").write_text("approved")
        approved = self.store.register_input(data, "checkpoint")
        foreign = self.folder / "foreign"
        foreign.mkdir()
        meta = foreign.stat()
        experiment = self.submit("raise AssertionError('must not execute')\n", inputs={"model": approved["id"]})
        attempt = self.store.create_attempt(experiment["id"])

        def stale_namespace(*args, **kwargs):
            if args[:2] == ("unshare", "stat"):
                return f"{meta.st_dev} {meta.st_ino}\n"
            return podman(*args, **kwargs)

        with patch("autonomous_research_lab.execution.podman", side_effect=stale_namespace):
            result = supervise(self.store, attempt["id"])
        self.assertEqual(result["state"], "failed", result)
        self.assertIsNone(inspect_owned(self.store, result["id"]))

    def test_failure_retains_output_and_logs(self):
        experiment = self.submit("from pathlib import Path\nPath('/workspace/partial.txt').write_text('retained')\nprint('before failure', flush=True)\nraise RuntimeError('deliberate failure')\n")
        result = run(self.store, experiment["id"], wait=True)
        self.assertEqual(result["state"], "failed", result)
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual((self.store.attempt_dir(result["id"]) / "output/partial.txt").read_text(), "retained")
        self.assertIn("deliberate failure", (self.store.attempt_dir(result["id"]) / "console.log").read_text())
        restored = Store(self.store.root).get_attempt(result["id"])
        self.assertEqual(restored["result"], result["result"])

    def test_database_failure_cannot_leave_the_owned_container_running(self):
        experiment = self.submit("import time\ntime.sleep(60)\n", wall_seconds=30)
        attempt = self.store.create_attempt(experiment["id"])
        get_attempt = self.store.get_attempt
        update_attempt = self.store.update_attempt
        unavailable = False

        def read_record(*args, **kwargs):
            if unavailable:
                raise sqlite3.OperationalError("injected database failure")
            return get_attempt(*args, **kwargs)

        def write_record(*args, **kwargs):
            nonlocal unavailable
            if unavailable:
                raise sqlite3.OperationalError("injected database failure")
            result = update_attempt(*args, **kwargs)
            if kwargs.get("state") == "running":
                unavailable = True
                raise sqlite3.OperationalError("injected database failure")
            return result

        with patch.object(self.store, "get_attempt", side_effect=read_record), patch.object(
            self.store, "update_attempt", side_effect=write_record
        ):
            with self.assertRaises(sqlite3.OperationalError):
                supervise(self.store, attempt["id"])
        self.assertFalse(inspect_owned(self.store, attempt["id"])["State"]["Running"])
        recovered = recover(self.store, wait=True)
        self.assertEqual(recovered[0]["id"], attempt["id"])
        self.assertEqual(recovered[0]["state"], "failed")

    def test_inspection_timeout_preserves_already_finished_outcomes_without_reexecution(self):
        cases = (
            ("starting", 0, True, "inspect", "completed"),
            ("running", 0, True, "inspect", "completed"),
            ("running", 7, True, "inspect", "failed"),
            ("running", 0, False, "inspect", "failed"),
            ("running", 0, True, "unrelated", "failed"),
        )
        for phase, exit_code, writes_results, failure, expected in cases:
            with self.subTest(phase=phase, exit_code=exit_code, writes_results=writes_results, failure=failure):
                code = (
                    "import json, time\nfrom pathlib import Path\n"
                    "with Path('/workspace/executions').open('a') as stream: stream.write('once\\n')\n"
                    "while not Path('/workspace/release').exists(): time.sleep(0.02)\n"
                )
                if writes_results:
                    code += "Path('/workspace/results.json').write_text(json.dumps({'metrics': {'value': 7}}))\n"
                code += f"raise SystemExit({exit_code})\n"
                experiment = self.submit(code, wall_seconds=30)
                attempt = self.store.create_attempt(experiment["id"])
                process = execution._podman_process
                injected = False

                def observe(*args, **kwargs):
                    nonlocal injected
                    result = process(*args, **kwargs)
                    if args == ("container", "inspect", container_name(attempt["id"])) and not result.returncode and not injected:
                        container = json.loads(result.stdout)[0]
                        if container["State"].get("Running") and self.store.get_attempt(attempt["id"])["state"] == phase:
                            (self.store.attempt_dir(attempt["id"]) / "output/release").write_text("finish\n")
                            podman("wait", container["Id"], timeout=15)
                            injected = True
                            if failure == "inspect":
                                raise subprocess.TimeoutExpired(["podman", *args], 30)
                            raise OSError("unrelated supervisor failure")
                    return result

                with patch.object(execution, "_podman_process", side_effect=observe):
                    result = supervise(self.store, attempt["id"])
                self.assertTrue(injected)
                self.assertEqual(result["state"], expected, result)
                self.assertEqual(result["exit_code"], exit_code)
                self.assertEqual((self.store.attempt_dir(attempt["id"]) / "output/executions").read_text(), "once\n")
                diagnostics = [event for event in self.store.events(attempt["id"]) if event["kind"] == "container_observation_timeout"]
                self.assertEqual(bool(diagnostics), failure == "inspect")
                if diagnostics:
                    self.assertIn("TimeoutExpired", diagnostics[0]["details"]["error"])

    def test_inspection_timeout_cannot_promote_work_terminated_during_cleanup(self):
        experiment = self.submit("import time\ntime.sleep(60)\n", wall_seconds=30)
        attempt = self.store.create_attempt(experiment["id"])
        process = execution._podman_process
        injected = False

        def observe(*args, **kwargs):
            nonlocal injected
            if args == ("container", "inspect", container_name(attempt["id"])) and not injected:
                if self.store.get_attempt(attempt["id"])["state"] == "running":
                    injected = True
                    raise subprocess.TimeoutExpired(["podman", *args], 30)
            return process(*args, **kwargs)

        with patch.object(execution, "_podman_process", side_effect=observe):
            result = supervise(self.store, attempt["id"])
        self.assertTrue(injected)
        self.assertEqual(result["state"], "failed", result)
        self.assertNotEqual(result["exit_code"], 0)
        self.assertFalse(inspect_owned(self.store, attempt["id"])["State"]["Running"])

    def test_wall_timeout_terminates_container(self):
        experiment = self.submit("import time\ntime.sleep(60)\n", wall_seconds=1)
        result = run(self.store, experiment["id"], wait=True)
        self.assertEqual(result["state"], "timed_out", result)
        self.assertFalse(inspect_owned(self.store, result["id"])["State"]["Running"])

    def test_stop_retains_attempt_and_terminates_owned_work(self):
        experiment = self.submit("from pathlib import Path\nimport time\nPath('/workspace/partial').write_text('started')\ntime.sleep(60)\n", wall_seconds=30)
        attempt = run(self.store, experiment["id"])
        self.await_running(attempt["id"])
        result = stop(self.store, attempt["id"], wait=True)
        self.assertEqual(result["state"], "stopped", result)
        self.assertFalse(inspect_owned(self.store, result["id"])["State"]["Running"])
        self.assertEqual(Store(self.store.root).get_attempt(result["id"])["state"], "stopped")

    def test_stop_during_snapshot_verification_prevents_launch(self):
        experiment = self.submit("raise AssertionError('must not execute')\n")
        attempt = self.store.create_attempt(experiment["id"])
        original_verify = self.store.verify_experiment

        def verify_then_stop(experiment_id):
            original_verify(experiment_id)
            self.store.request_stop(attempt["id"])

        self.store.verify_experiment = verify_then_stop
        result = supervise(self.store, attempt["id"])
        self.assertEqual(result["state"], "stopped", result)
        self.assertIsNone(inspect_owned(self.store, attempt["id"]))

    def test_recovery_adopts_existing_container_without_repeating_code(self):
        experiment = self.submit("import json, time\nfrom pathlib import Path\nwith Path('/workspace/executions').open('a') as f: f.write('once\\n')\ntime.sleep(4)\nPath('/workspace/results.json').write_text(json.dumps({'metrics': {'value': 7}}))\n")
        attempt = run(self.store, experiment["id"])
        active = self.await_running(attempt["id"])
        container_id = active["container_id"]
        os.kill(active["supervisor_pid"], signal.SIGKILL)
        deadline = time.monotonic() + 5
        while True:
            with attempt_lock(self.store, attempt["id"]) as acquired:
                if acquired:
                    break
            if time.monotonic() >= deadline:
                self.fail("Killed supervisor did not release its lock")
            time.sleep(0.02)
        recover(self.store, wait=True)
        result = self.store.get_attempt(attempt["id"])
        self.assertEqual(result["state"], "completed", result)
        self.assertEqual(result["container_id"], container_id)
        self.assertEqual((self.store.attempt_dir(attempt["id"]) / "output/executions").read_text(), "once\n")
        self.assertEqual(len(self.store.list_attempts()), 1)

    def test_memory_limit_is_enforced(self):
        experiment = self.submit("x = bytearray(256 * 1024**2)\nprint(len(x))\n")
        result = run(self.store, experiment["id"], wait=True)
        self.assertEqual(result["state"], "limit_exceeded", result)
        self.assertTrue(inspect_owned(self.store, result["id"])["State"]["OOMKilled"])

    def test_total_output_limit_stops_a_multi_file_writer(self):
        experiment = self.submit("from pathlib import Path\nimport time\nfor i in range(5): Path(f'/workspace/{i}.bin').write_bytes(b'x' * 65536)\ntime.sleep(60)\n", output_bytes=262144)
        result = run(self.store, experiment["id"], wait=True)
        self.assertEqual(result["state"], "limit_exceeded", result)
        self.assertGreater(result["result"]["output_bytes"], 262144)
        self.assertFalse(inspect_owned(self.store, result["id"])["State"]["Running"])

    def test_unavailable_cpu_quota_fails_without_running_code(self):
        if "cpu" in environment()["host"].get("cgroupControllers", []):
            self.skipTest("CPU controller is delegated on this host")
        experiment = self.submit("raise AssertionError('must not execute')\n", cpus=1)
        result = run(self.store, experiment["id"], wait=True)
        self.assertEqual(result["state"], "failed", result)
        self.assertIsNone(inspect_owned(self.store, result["id"]))

    def test_success_exit_with_missing_results_is_not_success(self):
        experiment = self.submit("print('no results')\n")
        result = run(self.store, experiment["id"], wait=True)
        self.assertEqual(result["state"], "failed", result)
        self.assertEqual(result["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
