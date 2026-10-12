import concurrent.futures
import os
from pathlib import Path
import shutil
import threading
import unittest
import uuid

from autonomous_research_lab.models import ExperimentSpec, LabError
from autonomous_research_lab.store import Store

GPU = "GPU-6ac06ab6-ac30-f3ad-b246-064a8e6f1309"


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(__file__).resolve().parents[1] / ".arl-cache" / "tests" / uuid.uuid4().hex
        self.source = self.folder / "source"
        self.source.mkdir(parents=True)
        (self.source / "main.py").write_text("print('original')\n", encoding="utf-8")
        self.store = Store(self.folder / "state")

    def tearDown(self):
        shutil.rmtree(self.folder)

    def submit(self, gpu=False):
        return self.store.submit(ExperimentSpec.from_dict({"name": "record-test", "source": str(self.source), "entrypoint": "main.py", "gpu": gpu, "settings": {"seed": 42}}, self.folder))

    def test_snapshot_survives_original_edits_and_detects_tampering(self):
        experiment = self.submit()
        (self.source / "main.py").write_text("changed source", encoding="utf-8")
        self.store.verify_experiment(experiment["id"])
        saved = self.store.bundle_dir(experiment["id"]) / "code/main.py"
        self.assertEqual(saved.read_text(), "print('original')\n")
        saved.write_text("tampered snapshot", encoding="utf-8")
        with self.assertRaises(LabError):
            self.store.verify_experiment(experiment["id"])

    def test_failed_history_and_events_survive_reopening(self):
        experiment = self.submit()
        attempt = self.store.create_attempt(experiment["id"])
        self.store.update_attempt(attempt["id"], state="failed", error="retained failure", result={"observations": None})
        reopened = Store(self.store.root)
        result = reopened.get_attempt(attempt["id"])
        self.assertEqual((result["state"], result["error"]), ("failed", "retained failure"))
        self.assertEqual(reopened.events(attempt["id"])[-1]["details"], {"from": "queued", "to": "failed"})
        with self.assertRaises(LabError):
            reopened.update_attempt(attempt["id"], error="overwrite")
        self.assertEqual(reopened.request_stop(attempt["id"]), result)
        retry = reopened.create_attempt(experiment["id"])
        self.assertNotEqual(retry["id"], attempt["id"])
        self.assertEqual(len(reopened.list_attempts(experiment["id"])), 2)

    @unittest.skipUnless(os.name == "posix", "POSIX link fixture")
    def test_external_database_preserves_history_and_gpu_lease_in_same_workspace(self):
        experiment = self.submit(gpu=True)
        attempt = self.store.create_attempt(experiment["id"], gpu_uuid=GPU)
        external = self.folder / "external.sqlite3"
        original = self.store.db_path
        original.replace(external)
        original.symlink_to(external)
        reopened = Store(self.store.root)
        self.assertEqual(reopened.root, self.store.root)
        self.assertEqual(reopened.get_attempt(attempt["id"]), attempt)
        with self.assertRaises(LabError):
            reopened.create_attempt(experiment["id"], gpu_uuid=GPU)
        reopened.update_attempt(attempt["id"], state="failed", error="preserved after relocation")
        self.assertEqual(self.store.get_attempt(attempt["id"])["error"], "preserved after relocation")
        self.assertEqual(reopened.db_path, external.resolve())

    @unittest.skipUnless(os.name == "posix", "POSIX link fixture")
    def test_missing_linked_database_never_creates_an_empty_history(self):
        original = self.store.db_path
        retained = self.folder / "retained.sqlite3"
        original.replace(retained)
        missing = self.folder / "missing.sqlite3"
        original.symlink_to(missing)
        with self.assertRaisesRegex(LabError, "no empty replacement"):
            Store(self.store.root)
        self.assertFalse(missing.exists())
        self.assertTrue(retained.is_file())

    def test_gpu_reservation_is_atomic_and_case_insensitive(self):
        experiment = self.submit(gpu=True)
        barrier = threading.Barrier(2)

        def reserve(gpu):
            barrier.wait(timeout=10)
            try:
                return self.store.create_attempt(experiment["id"], gpu_uuid=gpu)
            except LabError:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(reserve, (GPU, GPU.upper())))
        successes = [result for result in results if result is not None]
        self.assertEqual(len(successes), 1)
        active = successes[0]
        self.assertEqual(active["gpu_uuid"], GPU)
        self.store.update_attempt(active["id"], state="failed")
        replacement = self.store.create_attempt(experiment["id"], gpu_uuid=GPU.upper())
        self.assertEqual(replacement["gpu_uuid"], GPU)

    def test_stop_wins_a_compare_and_set_race(self):
        experiment = self.submit()
        attempt = self.store.create_attempt(experiment["id"])
        self.store.request_stop(attempt["id"])
        with self.assertRaises(LabError):
            self.store.update_attempt(attempt["id"], state="starting", expected_states={"queued"})
        self.store.update_attempt(attempt["id"], supervisor_pid=123)
        actual = self.store.get_attempt(attempt["id"])
        self.assertEqual(actual["state"], "stopping")
        self.assertTrue(actual["stop_requested"])
        with self.assertRaises(LabError):
            self.store.update_attempt(attempt["id"], stop_requested=False)

    def test_planned_attempt_identity_cannot_be_reused(self):
        experiment = self.submit()
        planned = "run_" + uuid.uuid4().hex
        self.store.create_attempt(experiment["id"], attempt_id=planned)
        with self.assertRaises(LabError):
            self.store.create_attempt(experiment["id"], attempt_id=planned)
        self.assertTrue((self.store.attempt_dir(planned) / "output").is_dir())
        self.assertEqual(len(self.store.list_attempts()), 1)

    def test_missing_entrypoint_does_not_publish_a_partial_record(self):
        (self.source / "main.py").unlink()
        with self.assertRaises(LabError):
            self.submit()
        self.assertEqual(self.store.list_experiments(), [])

    @unittest.skipUnless(os.name == "posix", "POSIX link fixture")
    def test_snapshot_rejects_symbolic_links(self):
        (self.source / "outside").symlink_to(self.folder)
        with self.assertRaises(LabError):
            self.submit()
        self.assertEqual(self.store.list_experiments(), [])

    def test_state_directory_is_not_recursively_snapshotted(self):
        spec = ExperimentSpec.from_dict({"name": "unsafe", "source": str(self.folder), "entrypoint": "source/main.py"}, self.folder)
        with self.assertRaises(LabError):
            self.store.submit(spec)

    def test_record_identifiers_cannot_escape_storage(self):
        with self.assertRaises(LabError):
            self.store.attempt_dir("../../outside")
        with self.assertRaises(LabError):
            self.store.bundle_dir("../outside")

    def test_approved_input_hashes_survive_reopening_and_detect_same_size_tampering(self):
        data = self.folder / "approved"
        data.mkdir()
        file = data / "weights.bin"
        file.write_bytes(b"original")
        approved = self.store.register_input(data, "checkpoint")
        reopened = Store(self.store.root)
        self.assertEqual(reopened.verify_input(approved["id"]), approved)
        before = file.stat()
        file.write_bytes(b"modified")
        os.utime(file, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaises(LabError):
            reopened.verify_input(approved["id"])
        self.assertEqual(reopened.get_input(approved["id"]), approved)

    def test_publisher_hash_mismatch_is_not_registered(self):
        with self.assertRaises(LabError):
            self.store.register_input(self.source, "checkpoint", expected_hashes={"main.py": "0" * 64})
        self.assertEqual(self.store.list_inputs(), [])

    def test_large_input_is_not_copied_into_the_source_snapshot(self):
        data = self.folder / "large-input"
        data.mkdir()
        with (data / "weights.bin").open("wb") as stream:
            stream.truncate(256 * 1024**2 + 1)
        approved = self.store.register_input(data, "large-checkpoint")
        spec = ExperimentSpec.from_dict({"name": "large-input", "source": str(self.source),
                                        "entrypoint": "main.py", "inputs": {"model": approved["id"]}}, self.folder)
        experiment = self.store.submit(spec)
        self.store.verify_experiment(experiment["id"])
        self.assertFalse((self.store.bundle_dir(experiment["id"]) / "code/weights.bin").exists())
        self.assertEqual(self.store.input_source(approved["id"]), data.resolve())

    def test_input_approval_cannot_include_credentials_or_cross_stores(self):
        for path in (self.store.root, self.folder):
            with self.subTest(path=path), self.assertRaises(LabError):
                self.store.register_input(path, "unsafe")
        foreign = Store(self.folder / "foreign")
        approved = foreign.register_input(self.source, "foreign-input")
        spec = ExperimentSpec.from_dict({"name": "foreign", "source": str(self.source),
                                        "entrypoint": "main.py", "inputs": {"model": approved["id"]}}, self.folder)
        with self.assertRaises(LabError):
            self.store.submit(spec)
        self.assertEqual(self.store.list_experiments(), [])

    @unittest.skipUnless(os.name == "posix", "POSIX link fixtures")
    def test_input_links_are_rejected_at_registration_and_before_reuse(self):
        linked = self.folder / "linked"
        linked.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(LabError):
            self.store.register_input(linked, "linked")
        approved = self.store.register_input(self.source, "original")
        original = self.folder / "moved"
        self.source.rename(original)
        self.source.symlink_to(original, target_is_directory=True)
        with self.assertRaises(LabError):
            self.store.verify_input(approved["id"])
        nested = self.folder / "nested"
        nested.mkdir()
        (nested / "secret").symlink_to(original / "main.py")
        with self.assertRaises(LabError):
            self.store.register_input(nested, "nested-link")


if __name__ == "__main__":
    unittest.main()
