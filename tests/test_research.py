"""Recovery/control invariants; cached protocol records are fixtures, not model calls."""
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from autonomous_research_lab.execution import atomic_json
from autonomous_research_lab.models import LabError
from autonomous_research_lab.research import ResearchEngine, new_id, resource_spec, validate_brief, within
from autonomous_research_lab.store import Store


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(__file__).resolve().parents[1] / ".arl-cache" / "tests" / uuid.uuid4().hex
        self.engine = ResearchEngine(Store(self.folder / "state"))
        self.program = self.engine.create({"goal": "Investigate a bounded question", "model": {"base_url": "https://api.example.com/v1", "model": "protocol-fixture"}})
        self.id = self.program["id"]

    def tearDown(self):
        shutil.rmtree(self.folder)

    def pending(self, response=None, role="director"):
        call_id = new_id("call")
        def seed(p):
            p.update(state="running", phase="decide", pending_call_id=call_id)
            p["calls"].append({"id": call_id, "role": role, "status": "pending", "usage": None})
            p["usage"]["model_calls"] = 1
        self.engine._change(self.id, seed)
        if response is not None:
            atomic_json(self.engine.folder(self.id) / "calls" / (call_id + ".json"), {"status": "completed", "response": {
                "content": response, "raw": json.dumps(response), "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}})
        return call_id

    def test_interrupted_request_is_charged_and_not_replayed(self):
        self.pending()
        with patch("autonomous_research_lab.research.ModelClient.complete", side_effect=AssertionError("must not replay")):
            program = self.engine.work(self.id)
        self.assertEqual(program["state"], "blocked")
        self.assertEqual(program["calls"][0]["status"], "interrupted")
        self.assertEqual(program["usage"]["model_calls"], 1)
        self.assertEqual(program["usage"]["unknown_token_calls"], 1)
        self.assertEqual(program["implementations"], [])

    def test_saved_response_recovers_without_another_request(self):
        call_id = self.pending({"action": "finish", "rationale": "The recorded scope cannot support an experiment", "evidence_refs": []})
        with patch("autonomous_research_lab.research.ModelClient.complete", side_effect=AssertionError("must not replay")):
            result = ResearchEngine(self.engine.store).work(self.id)
        self.assertEqual(result["state"], "completed")
        self.assertEqual([call["id"] for call in result["calls"]], [call_id])
        self.assertEqual(result["usage"]["completion_tokens_known"], 7)
        self.assertIsNone(result["pending_call_id"])

    def test_pause_wins_over_late_request_completion(self):
        self.pending({"action": "finish", "rationale": "No further work", "evidence_refs": []})
        self.engine.pause(self.id)
        with patch("autonomous_research_lab.research.ModelClient.complete", side_effect=AssertionError("must not schedule")):
            result = ResearchEngine(self.engine.store).work(self.id)
        self.assertEqual(result["state"], "paused")
        self.assertEqual(result["calls"][0]["status"], "completed")
        self.assertEqual(result["decisions"], [])
        self.assertEqual(result["usage"]["completion_tokens_known"], 7)

    def test_terminal_stop_cannot_restart_or_lose_history(self):
        self.engine.pause(self.id)
        stopped = self.engine.stop(self.id)
        self.assertEqual(stopped["state"], "stopped")
        with self.assertRaises(LabError):
            self.engine.start(self.id)
        self.assertEqual(self.engine.get(self.id)["questions"], self.program["questions"])

    def test_unknown_evidence_is_not_promoted_to_a_decision(self):
        self.pending({"action": "finish", "rationale": "Claims unsupported evidence", "evidence_refs": ["invented_record"]})
        program = self.engine.work(self.id)
        self.assertEqual(program["state"], "blocked")
        self.assertEqual(program["calls"][0]["status"], "invalid")
        self.assertEqual(program["decisions"], [])
        self.assertEqual(program["usage"]["model_calls"], 1)

    def test_missing_credential_blocks_before_requests_or_execution(self):
        brief = self.program["brief"]
        brief["model"]["api_key_env"] = "ARL_TEST_DELIBERATELY_ABSENT"
        program = self.engine.create(brief)
        with patch.dict("os.environ", {}, clear=True):
            result = self.engine.start(program["id"], background=False)
        self.assertEqual(result["state"], "blocked")
        self.assertEqual(result["calls"], [])
        self.assertEqual(result["implementations"], [])

    def test_malformed_usage_cannot_corrupt_accounting(self):
        call_id = self.pending()
        self.engine._record_call(self.id, call_id, {"status": "failed", "response": {"usage": {"prompt_tokens": "bad", "completion_tokens": 7}}})
        result = self.engine.get(self.id)
        self.assertEqual(result["usage"]["unknown_token_calls"], 1)
        self.assertEqual(result["usage"]["completion_tokens_known"], 7)
        self.assertEqual(result["usage"]["prompt_tokens_known"], 0)

    def test_allocations_cannot_exceed_permissions_or_remove_a_quota(self):
        allowed = resource_spec({"wall_seconds": 20, "memory_bytes": 64 * 1024**2, "cpus": 2})
        for field, value in (("wall_seconds", 21), ("memory_bytes", 128 * 1024**2), ("gpu", True), ("cpus", None)):
            with self.subTest(field=field), self.assertRaises(LabError):
                within({**allowed, field: value}, allowed)

    def test_data_paths_and_unbounded_budgets_are_rejected(self):
        for extra in ({"data_files": {"../secret": "x"}}, {"max_model_calls": 101}, {"max_repairs": True}, {"max_experiments": 0}):
            with self.subTest(extra=extra), self.assertRaises(LabError):
                validate_brief({**self.program["brief"], **extra})

    def test_builder_cannot_drop_replace_rename_or_expand_approved_inputs(self):
        ids = []
        for name in ("approved", "unapproved"):
            directory = self.folder / name
            directory.mkdir()
            (directory / "data.txt").write_text(name)
            ids.append(self.engine.store.register_input(directory, name)["id"])
        permitted = {"model": ids[0]}
        original_brief = self.program["brief"]
        for bindings in ({}, {"model": ids[1]}, {"other": ids[0]}, {**permitted, "private": ids[1]}):
            with self.subTest(bindings=bindings):
                self.program = self.engine.create({**original_brief, "inputs": permitted})
                self.id = self.program["id"]
                design = {"id": new_id("design"), "question": "Read the approved input",
                          "procedure": "Measure its contents", "resources": self.program["brief"]["resources"]}
                self.pending({"files": {"main.py": "raise AssertionError('must not execute')"},
                              "manifest": {"name": "input-boundary", "source": "code", "entrypoint": "main.py",
                                           "question": design["question"], "procedure": design["procedure"],
                                           "gpu": design["resources"]["gpu"],
                                           "limits": {k: v for k, v in design["resources"].items() if k != "gpu"},
                                           "inputs": bindings}}, role="builder")
                self.engine._change(self.id, lambda p: p.update(
                    phase="build", designs=[design], selected_design_id=design["id"]))
                self.engine._build(self.id)
                implementation = self.engine.get(self.id)["implementations"][-1]
                self.assertEqual(implementation["status"], "invalid")
                self.assertIsNone(implementation["experiment_id"])
                self.assertEqual(self.engine.store.list_experiments(), [])


if __name__ == "__main__":
    unittest.main()
