import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import threading
import unittest
from unittest.mock import patch
import uuid

from autonomous_research_lab.models import ExperimentSpec, LabError
from autonomous_research_lab.store import Store

if sys.platform != "linux":
    raise unittest.SkipTest("The study controller runs inside Linux")

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "research/qwen38-nvfp4"
sys.path.insert(0, str(SOURCE))
try:
    SPEC = importlib.util.spec_from_file_location("qwen38_study", SOURCE / "study.py")
    study = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(study)
finally:
    sys.path.remove(str(SOURCE))


class StudyExecutionTests(unittest.TestCase):
    def setUp(self):
        self.root = PROJECT / ".arl-cache" / "study-tests" / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.store = Store(self.root / "store")
        self.code = self.root / "code"
        self.code.mkdir()
        (self.code / "generate.py").write_text("pass\n")

    def tearDown(self):
        shutil.rmtree(self.root)

    def retained_attempt(self, *, legacy=False):
        settings = {"checkpoint": "nvidia", "split": "primary", "series": "test",
                    "engine_options": study.ENGINE_OPTIONS}
        settings.update({"seed": 17} if legacy else {"seeds": [29, 43]})
        experiment = self.store.submit(ExperimentSpec(name="retained-generation", source=self.code,
                                                       entrypoint="generate.py", settings=settings))
        attempt = self.store.create_attempt(experiment["id"])
        self.store.update_attempt(attempt["id"], state="starting")
        self.store.update_attempt(attempt["id"], state="running")
        folder = self.store.attempt_dir(attempt["id"]) / "output"
        folder.mkdir(parents=True, exist_ok=True)
        metrics = {"requested": 256, "recorded": 256, "generation_errors": 0,
                   "generated_tokens": 2048, "generation_wall_seconds": 5, "load_seconds": 2}
        artifacts = []

        def artifact(name, value):
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = value.encode()
            path.write_bytes(data)
            artifacts.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})

        if legacy:
            artifact("generations.jsonl", "{}\n")
        else:
            for seed, status in ((29, "completed"), (43, "failed")):
                artifact(f"seed-{seed}/results.json", json.dumps({
                    "status": status, "metrics": metrics,
                    "details": {"checkpoint": "nvidia", "split": "primary", "seed": seed}}))
                artifact(f"seed-{seed}/generations.jsonl", "{}\n")
        self.store.update_attempt(attempt["id"], state="completed" if legacy else "failed",
                                  container_id=f"container-{attempt['id']}",
                                  result={"artifacts": artifacts, "observations": {"metrics": metrics}})
        record = self.root / "generation.json"
        study.atomic_json(record, {"experiment_id": experiment["id"], "attempt_id": attempt["id"]})
        return record, folder

    def test_completed_seed_remains_available_after_a_later_seed_fails(self):
        record, _ = self.retained_attempt()
        attempt, outcome, generations = study.generation_block(self.store, record, "nvidia", "primary", 29, "test")
        self.assertEqual(attempt["state"], "failed")
        self.assertEqual(outcome["status"], "completed")
        self.assertEqual(generations.read_text(), "{}\n")
        with self.assertRaises(LabError):
            study.generation_block(self.store, record, "nvidia", "primary", 43, "test")

    def test_changed_receipt_cannot_reclassify_a_failed_seed_as_complete(self):
        record, folder = self.retained_attempt()
        path = folder / "seed-43/results.json"
        receipt = json.loads(path.read_text())
        receipt["status"] = "completed"
        path.write_text(json.dumps(receipt))
        with self.assertRaises(LabError):
            study.generation_block(self.store, record, "nvidia", "primary", 43, "test")

    def test_historical_completed_attempt_does_not_require_new_artifact_paths(self):
        record, _ = self.retained_attempt(legacy=True)
        attempt, _, generations = study.generation_block(self.store, record, "nvidia", "primary", 17, "test")
        self.assertEqual(attempt["state"], "completed")
        self.assertEqual(generations.read_text(), "{}\n")
        with self.assertRaises(LabError):
            study.generation_identity(self.store, record, "nvidia", "primary", 29, "test")

    def test_retiring_completed_container_preserves_logs_and_result_evidence(self):
        record, output = self.retained_attempt(legacy=True)
        attempt = self.store.get_attempt(study.read_json(record)["attempt_id"])
        log = output.parent / "console.log"
        log.write_text("retained execution log\n")
        container = {"Id": attempt["container_id"], "State": {"Running": False, "Paused": False, "ExitCode": 0}}

        def remove_container(*args):
            # A runtime may remove a custom log path with the container.
            log.unlink(missing_ok=True)

        with patch.object(study, "inspect_owned", return_value=container), patch.object(study, "podman", side_effect=remove_container):
            study.retire_completed_grader(self.store, attempt)
        self.assertEqual(log.read_text(), "retained execution log\n")
        self.assertEqual((output / "generations.jsonl").read_text(), "{}\n")
        self.assertEqual(self.store.get_attempt(attempt["id"]), attempt)
        archive = next(event for event in self.store.events(attempt["id"]) if event["kind"] == "completed_container_archived")
        self.assertEqual(archive["details"]["console_sha256"], hashlib.sha256(log.read_bytes()).hexdigest())

    def test_retirement_cannot_delete_a_container_that_is_still_running(self):
        record, _ = self.retained_attempt(legacy=True)
        attempt = self.store.get_attempt(study.read_json(record)["attempt_id"])
        container = {"Id": attempt["container_id"], "State": {"Running": True, "ExitCode": 0}}
        with patch.object(study, "inspect_owned", return_value=container), patch.object(study, "podman", side_effect=AssertionError("live container removed")):
            with self.assertRaises(LabError):
                study.retire_completed_grader(self.store, attempt)

    def test_amendment_excludes_already_submitted_work_without_losing_seeds(self):
        inventory = {"checkpoints": [{"label": "bf16"}, {"label": "nvidia"}]}
        plan = {"blocks": [{"checkpoint": label, "seed": seed}
                           for seed in study.SEEDS for label in ("bf16", "nvidia")], "wall_seconds": 21600}
        folder = self.store.root / study.STATE / "primary/test"
        study.atomic_json(folder / "cohort.json", plan)
        study.atomic_json(folder / "seed-17/bf16/generation.json",
                          {"experiment_id": "original", "attempt_id": "still-running"})
        amendment = study.execution_amendment(self.store, inventory, "test", plan, self.root / "cache")
        grouped = {group["checkpoint"]: group["seeds"] for group in amendment["remaining_groups"]}
        self.assertEqual(grouped, {"bf16": [29, 43], "nvidia": [17, 29, 43]})
        self.assertEqual(amendment["retained_blocks"][0]["attempt_id"], "still-running")
        self.assertEqual(json.loads((folder / "cohort.json").read_text()), plan)

    def test_next_gpu_group_runs_while_prior_cpu_grading_is_in_progress(self):
        inventory = {"checkpoints": [{"label": "bf16"}, {"label": "nvidia"}]}
        for label in ("bf16", "nvidia"):
            study.atomic_json(self.store.root / study.STATE / f"compatibility/test/seed-17/{label}/summary.json", {})
        study.atomic_json(self.store.root / study.STATE / "primary/test/seed-17/bf16/generation.json",
                          {"experiment_id": "original", "attempt_id": "original-attempt"})
        amendment = {"retained_blocks": [{"checkpoint": "bf16", "seed": 17,
                                          "experiment_id": "original", "attempt_id": "original-attempt"}],
                     "remaining_groups": [{"checkpoint": "nvidia", "seeds": [29, 43]}]}
        grading_started = threading.Event()
        next_generation_started = threading.Event()
        graded = []

        def grade(store, label, **options):
            if label == "bf16":
                grading_started.set()
                if not next_generation_started.wait(5):
                    raise AssertionError("GPU generation incorrectly waited for grading")
            graded.append((label, options["seed"]))

        def generate_group(*args):
            self.assertTrue(grading_started.wait(5))
            next_generation_started.set()

        with patch.object(study, "primary_plan", return_value={"wall_seconds": 21600}), \
             patch.object(study, "execution_amendment", return_value=amendment), \
             patch.object(study, "generate"), patch.object(study, "grade", side_effect=grade), \
             patch.object(study, "generate_resident_group", side_effect=generate_group), \
             patch.object(study, "analyze_cohort", side_effect=lambda *_: list(graded)):
            result = study.run_cohort(self.store, inventory, "test", self.root / "cache")
        self.assertEqual(result, [("bf16", 17), ("nvidia", 29), ("nvidia", 43)])


if __name__ == "__main__":
    unittest.main()
