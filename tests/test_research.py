"""Recovery/control invariants; cached protocol records are fixtures, not model calls."""
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from autonomous_research_lab.execution import atomic_json
from autonomous_research_lab.models import ExperimentSpec, LabError
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

    def test_registered_input_citations_require_program_approval(self):
        source = self.folder / "citation-source"
        source.mkdir()
        (source / "data.json").write_text('{"value":7}')
        input_id = self.engine.store.register_input(source, "citation-source")["id"]
        self.engine._change(self.id, lambda p: p["brief"].update(inputs={"approved": input_id}))
        self.pending({"action": "finish", "rationale": "The approved input cannot answer this question", "evidence_refs": [input_id]})
        self.assertEqual(self.engine.work(self.id)["state"], "completed")
        other = self.engine.create({"goal": "Another program without that input",
                                    "model": {"base_url": "https://api.example.com/v1", "model": "protocol-fixture"}})
        self.id = other["id"]
        self.pending({"action": "finish", "rationale": "Citing another program's input", "evidence_refs": [input_id]})
        rejected = self.engine.work(self.id)
        self.assertEqual(rejected["state"], "blocked")
        self.assertEqual(rejected["calls"][-1]["status"], "invalid")

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

    def recorded_attempt(self, *, inputs=None, metrics=None, outputs=None, scratch="not a declared handoff"):
        code = self.folder / uuid.uuid4().hex
        code.mkdir()
        (code / "main.py").write_text("pass\n")
        spec = ExperimentSpec.from_dict({"name": "artifact-boundary", "source": str(code), "entrypoint": "main.py",
                                         "inputs": inputs or {}, "outputs": outputs or ["results.json"]}, code)
        experiment = self.engine.store.submit(spec)
        attempt = self.engine.store.create_attempt(experiment["id"])
        self.engine.store.update_attempt(attempt["id"], state="starting")
        self.engine.store.update_attempt(attempt["id"], state="running")
        output = self.engine.store.attempt_dir(attempt["id"]) / "output"
        output.mkdir(exist_ok=True)
        contents = json.dumps({"metrics": metrics or {"observed": 7}})
        (output / "results.json").write_text(contents)
        (output / "scratch.txt").write_text(scratch)
        artifacts = [{"path": name, "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
                      "bytes": (output / name).stat().st_size} for name in ("results.json", "scratch.txt")]
        attempt = self.engine.store.update_attempt(attempt["id"], state="completed", result={
            "artifacts": artifacts, "observations": json.loads(contents)})
        implementation = {"id": new_id("implementation"), "design_id": new_id("design"), "status": "collected",
                          "attempt_id": attempt["id"], "experiment_id": experiment["id"]}
        self.engine._change(self.id, lambda p: p["implementations"].append(implementation))
        return implementation, attempt

    def test_model_evidence_excludes_scratch_without_losing_recorded_artifacts(self):
        _, attempt = self.recorded_attempt()
        evidence = self.engine._attempt_record(attempt["id"])
        self.assertEqual([item["path"] for item in evidence["result"]["artifacts"]], ["results.json"])
        self.assertEqual(evidence["result"]["omitted_undeclared_artifacts"], 1)
        self.assertEqual([item["path"] for item in evidence["artifact_previews"]], ["results.json"])
        self.assertEqual(evidence["result"]["observations"], {"metrics": {"observed": 7}})
        retained = self.engine.store.get_attempt(attempt["id"])
        self.assertEqual(retained["result"], attempt["result"])
        scratch = self.engine.store.read_artifact(attempt["id"], "scratch.txt")
        self.assertEqual(scratch["text"], "not a declared handoff")

    def test_large_declared_diagnostics_receive_a_bounded_preview(self):
        contents = "runtime diagnostics\n" + "x" * 70000
        _, attempt = self.recorded_attempt(outputs=["results.json", "scratch.txt"], scratch=contents)
        previews = self.engine._attempt_record(attempt["id"])["artifact_previews"]
        diagnostic = next(item for item in previews if item["path"] == "scratch.txt")
        self.assertEqual(diagnostic["text"], contents[:65536])
        self.assertTrue(diagnostic["truncated"])
        self.assertEqual(diagnostic["next_offset_bytes"], 65536)

    def test_recorded_artifact_can_be_read_in_bounded_pages(self):
        _, attempt = self.recorded_attempt()
        first = self.engine.store.read_artifact(attempt["id"], "results.json", max_bytes=5)
        self.assertTrue(first["truncated"])
        self.assertEqual(first["next_offset_bytes"], 5)
        rest = self.engine.store.read_artifact(attempt["id"], "results.json", offset_bytes=5)
        self.assertFalse(rest["truncated"])
        self.assertEqual(json.loads(first["text"] + rest["text"]), {"metrics": {"observed": 7}})
        for options in ({"offset_bytes": -1}, {"offset_bytes": first["bytes"] + 1}, {"max_bytes": 65537}):
            with self.subTest(options=options), self.assertRaises(LabError):
                self.engine.store.read_artifact(attempt["id"], "results.json", **options)

    def test_artifact_reads_reject_changed_unlisted_or_linked_evidence(self):
        _, attempt = self.recorded_attempt()
        path = self.engine.store.attempt_dir(attempt["id"]) / "output" / "results.json"
        original = path.read_text()
        path.write_text(original.replace("7", "8"))
        with self.assertRaises(LabError):
            self.engine.store.read_artifact(attempt["id"], "results.json")
        outside = self.folder / "outside.json"
        outside.write_text(original)
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaises((OSError, LabError)):
            self.engine.store.read_artifact(attempt["id"], "results.json")
        with self.assertRaises(LabError):
            self.engine.store.read_artifact(attempt["id"], "../outside.json")
        with self.assertRaises(LabError):
            self.engine.store.read_artifact(attempt["id"], "unlisted.json")

    def test_director_can_read_evidence_but_not_another_programs_artifacts(self):
        _, attempt = self.recorded_attempt()
        request = {"action": "read_artifacts", "rationale": "Inspect the measured result", "evidence_refs": [],
                   "artifacts": [{"attempt_id": attempt["id"], "path": "results.json"}]}
        self.pending(request)
        self.engine._decision(self.id)
        recorded = self.engine.get(self.id)
        self.assertEqual(json.loads(recorded["requested_records"][0]["text"]), {"metrics": {"observed": 7}})
        self.assertEqual(recorded["decisions"][-1]["artifacts"], request["artifacts"])
        other = self.engine.create({"goal": "An unrelated program",
                                    "model": {"base_url": "https://api.example.com/v1", "model": "protocol-fixture"}})
        self.id = other["id"]
        self.pending(request)
        with self.assertRaises(LabError):
            self.engine._decision(self.id)
        self.assertEqual(self.engine.get(self.id)["requested_records"], [])

    def test_followup_can_cite_an_individual_batch_execution(self):
        implementation, attempt = self.recorded_attempt()
        def nest(p):
            item = p["implementations"][0]
            item.update(attempt_id=None, experiment_id=None,
                        jobs=[{"attempt_id": attempt["id"], "experiment_id": implementation["experiment_id"]}],
                        batch_record={"id": implementation["id"], "state": "completed"})
        self.engine._change(self.id, nest)
        candidate = {key: "Use the observed child result" for key in (
            "question", "motivation", "procedure", "measurements", "interpretation", "requirements", "relationship")}
        candidate.update(parent_question_id=self.program["questions"][0]["id"], evidence_refs=[attempt["id"]],
                         resources=self.program["brief"]["resources"])
        self.pending({"action": "experiment", "rationale": "Follow the recorded child execution",
                      "evidence_refs": [attempt["id"]], "candidates": [candidate]})
        self.engine._decision(self.id)
        self.assertEqual(self.engine.get(self.id)["designs"][0]["evidence_refs"], [attempt["id"]])

    def test_handoff_excludes_scratch_and_preserves_private_lineage(self):
        directory = self.folder / "private"
        directory.mkdir()
        (directory / "tests.json").write_text('{"secret":7}')
        input_id = self.engine.store.register_input(directory, "private")["id"]
        self.engine._change(self.id, lambda p: p["brief"].update(inputs={"private": input_id}, private_inputs=["private"]))
        implementation, attempt = self.recorded_attempt(inputs={"private": input_id})
        self.engine._publish_outputs(self.id, implementation, attempt)
        saved = self.engine.get(self.id)["implementations"][-1]
        published = saved["published_input"]
        record = self.engine.store.verify_input(published["id"])
        self.assertEqual([item["path"] for item in record["files"]], ["results.json"])
        self.assertEqual(json.loads((self.engine.store.input_source(record["id"]) / "results.json").read_text())["metrics"], {"observed": 7})
        bindings, private_ids = self.engine._available_inputs(self.engine.get(self.id))
        self.assertEqual(bindings[published["alias"]], record["id"])
        self.assertIn(record["id"], private_ids)
        self.engine._publish_outputs(self.id, saved, attempt)
        self.assertEqual(self.engine.get(self.id)["implementations"][-1]["published_input"], published)
        candidate = {key: "Bounded design" for key in ("question", "motivation", "procedure", "measurements", "interpretation", "requirements", "relationship")}
        candidate.update(parent_question_id=self.program["questions"][0]["id"], inputs=[published["alias"]],
                         resources={"gpu": True}, evidence_refs=[attempt["id"]])
        self.pending({"action": "experiment", "rationale": "Read derived private data", "evidence_refs": [attempt["id"]], "candidates": [candidate]})
        result = self.engine.work(self.id)
        self.assertEqual(result["state"], "blocked")
        self.assertEqual(result["designs"], [])

    def test_changed_artifact_cannot_be_registered_as_recorded_evidence(self):
        implementation, attempt = self.recorded_attempt()
        (self.engine.store.attempt_dir(attempt["id"]) / "output" / "results.json").write_text('{"metrics":{"observed":8}}')
        with self.assertRaises(LabError):
            self.engine._publish_outputs(self.id, implementation, attempt)
        self.assertNotIn("published_input", self.engine.get(self.id)["implementations"][-1])
        self.assertEqual(self.engine.store.list_inputs(), [])

    def test_completion_requires_all_metrics_in_one_completed_execution(self):
        self.engine._change(self.id, lambda p: p["brief"].update(completion_metrics={"responses": 12, "graded": 12}))
        self.recorded_attempt(metrics={"responses": 12, "graded": 0})
        self.recorded_attempt(metrics={"responses": 0, "graded": 12})
        with self.assertRaises(LabError):
            self.engine._check_completion(self.engine.get(self.id))
        self.recorded_attempt(metrics={"responses": 12, "graded": 12})
        self.engine._check_completion(self.engine.get(self.id))

    def test_budget_exhaustion_is_blocked_not_completed(self):
        self.pending()
        self.engine._change(self.id, lambda p: (p.update(pending_call_id=None), p["brief"].update(max_model_calls=1)))
        result = self.engine.work(self.id)
        self.assertEqual(result["state"], "blocked")
        self.assertEqual(result["implementations"], [])

    def test_finish_cannot_bypass_missing_completion_evidence(self):
        self.engine._change(self.id, lambda p: p["brief"].update(completion_metrics={"graded": 12}))
        self.pending({"action": "finish", "rationale": "Claiming completion without execution", "evidence_refs": []})
        result = self.engine.work(self.id)
        self.assertEqual(result["state"], "blocked")
        self.assertNotIn("report", result)


if __name__ == "__main__":
    unittest.main()
