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


if __name__ == "__main__":
    unittest.main()
