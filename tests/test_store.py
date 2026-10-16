from contextlib import redirect_stderr, redirect_stdout
import copy
import concurrent.futures
import io
import json
import os
from pathlib import Path
import shutil
import threading
import unittest
import uuid

from autonomous_research_lab.cli import main
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

    def catalog(self, experiment_ids):
        return {
            "studies": [{
                "id": "attention-study", "title": "Attention comparison", "question": "Does the intervention help?",
                "parent_study_id": None, "rationale": None,
            }],
            "experiment_groups": [{
                "id": "comparison-main", "study_id": "attention-study", "purpose": "comparison",
                "label": "Main comparison", "model": "example/model", "seeds": [42, 7],
                "experiment_ids": experiment_ids,
            }],
        }

    def program(self, experiment_ids=()):
        return {
            "id": "program_" + uuid.uuid4().hex, "brief": {"goal": "  Original goal verbatim  "},
            "designs": [{"id": "design-original", "question": "Recorded design question"}],
            "implementations": [
                {"design_id": "design-original", "experiment_id": experiment_id,
                 "status": "failed" if index == 0 else "ready"}
                for index, experiment_id in enumerate(experiment_ids)
            ],
        }

    def test_coordinator_registration_preserves_authored_lineage_on_reopen(self):
        program = self.program()
        original = copy.deepcopy(program)
        links = self.store.reconcile_programs([program])
        study_id = links[program["id"]]
        study = self.store.list_catalog()["studies"][0]
        self.assertEqual(study, {"id": study_id, "title": program["brief"]["goal"],
                                "question": program["brief"]["goal"], "parent_study_id": None, "rationale": None})
        parent = self.catalog([])["studies"][0]
        study.update(title="Authored title", question="Authored question",
                     parent_study_id=parent["id"], rationale="Explicit follow-up")
        expected = self.store.import_catalog({"studies": [study, parent], "experiment_groups": []})
        reopened = Store(self.store.root)
        before = reopened.db_path.read_bytes()
        self.assertEqual(reopened.reconcile_programs([program]), links)
        self.assertEqual(reopened.reconcile_programs([program]), links)
        self.assertEqual(reopened.db_path.read_bytes(), before)
        self.assertEqual(reopened.list_catalog(), expected)
        self.assertEqual(program, original)

    def test_coordinator_groups_repaired_definitions_without_changing_execution(self):
        first, repair = self.submit(), self.submit()
        attempt = self.store.create_attempt(first["id"])
        self.store.update_attempt(attempt["id"], state="failed", error="Retained failure")
        program = self.program([first["id"], repair["id"]])
        program["implementations"].append({"design_id": "design-original", "experiment_id": None, "status": "invalid"})
        original = copy.deepcopy(program)
        experiments, attempts = self.store.list_experiments(), self.store.list_attempts()
        events = self.store.events(attempt["id"])
        links = self.store.reconcile_programs([program])
        group = self.store.list_catalog()["experiment_groups"][0]
        self.assertEqual(group["study_id"], links[program["id"]])
        self.assertEqual(group["experiment_ids"], [first["id"], repair["id"]])
        self.assertEqual((group["purpose"], group["label"], group["model"], group["seeds"]),
                         ("comparison", "Recorded design question", None, []))
        later = self.submit()
        program["implementations"].append({"design_id": "design-original", "experiment_id": later["id"]})
        self.store.reconcile_programs([program])
        groups = self.store.list_catalog()["experiment_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["id"], group["id"])
        self.assertEqual(groups[0]["experiment_ids"], [first["id"], repair["id"], later["id"]])
        self.assertEqual(self.store.list_attempts(), attempts)
        self.assertEqual(self.store.events(attempt["id"]), events)
        self.assertEqual(self.store.get_experiment(first["id"]), experiments[0])
        self.assertEqual(program["implementations"][:-1], original["implementations"])

    def test_coordinator_does_not_create_empty_groups_for_manual_memberships(self):
        experiment = self.submit()
        expected = self.store.import_catalog(self.catalog([experiment["id"]]))["experiment_groups"]
        program = self.program([experiment["id"]])
        self.store.reconcile_programs([program])
        self.assertEqual(self.store.list_catalog()["experiment_groups"], expected)

    def test_coordinator_preserves_manual_classification_and_group_metadata(self):
        first, second = self.submit(), self.submit()
        program = self.program([first["id"], second["id"]])
        self.store.import_catalog(self.catalog([first["id"]]))
        self.store.reconcile_programs([program])
        groups = self.store.list_catalog()["experiment_groups"]
        auto = next(group for group in groups if group["id"] != "comparison-main")
        auto.update(label="Authored group", purpose="setup", model="explicit/model", seeds=[123])
        self.store.import_catalog({"studies": [], "experiment_groups": [auto]})
        self.store.import_catalog(self.catalog([second["id"]]))
        expected = self.store.list_catalog()
        self.store.reconcile_programs([program])
        self.assertEqual(self.store.list_catalog(), expected)
        self.assertEqual(next(group for group in expected["experiment_groups"] if group["id"] == "comparison-main")["experiment_ids"],
                         [first["id"], second["id"]])

    def test_invalid_coordinator_references_roll_back_all_registration(self):
        first = self.submit()
        good = self.program([first["id"]])
        for bad in (self.program(["exp_" + uuid.uuid4().hex]), self.program([first["id"]])):
            if bad["implementations"][0]["experiment_id"] == first["id"]:
                bad["implementations"][0]["design_id"] = "missing-design"
            with self.assertRaises(LabError):
                self.store.reconcile_programs([good, bad])
            self.assertEqual(self.store.list_catalog(), {"studies": [], "experiment_groups": []})
            with self.store._db() as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM study_coordinators").fetchone()[0], 0)

    def test_catalog_is_idempotent_chronological_and_survives_reopening(self):
        first, second = self.submit(), self.submit()
        catalog = self.catalog([second["id"], first["id"]])
        original = copy.deepcopy(catalog)
        expected = self.store.import_catalog(catalog)
        self.assertEqual(catalog, original)
        self.assertEqual(expected["studies"], catalog["studies"])
        self.assertEqual(expected["experiment_groups"][0]["experiment_ids"], [first["id"], second["id"]])
        self.assertEqual(expected["experiment_groups"][0]["seeds"], [7, 42])
        self.assertEqual(self.store.import_catalog(catalog), expected)
        reopened = Store(self.store.root)
        self.assertEqual(reopened.list_catalog(), expected)
        self.assertEqual(reopened.import_catalog({"studies": [], "experiment_groups": []}), expected)
        with reopened._db() as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)

    def test_child_first_nested_lineage_persists(self):
        root = self.catalog([])["studies"][0]
        child = {
            "id": "attention-follow-up", "title": "Longer comparison",
            "question": "Does the parent finding persist with a longer schedule?",
            "parent_study_id": root["id"], "rationale": "The parent comparison found sensitivity to schedule length.",
        }
        grandchild = {
            "id": "attention-follow-up-replication", "title": "Longer comparison replication",
            "question": "Does the longer-schedule finding replicate across seeds?",
            "parent_study_id": child["id"], "rationale": "The follow-up found a difference on one seed.",
        }
        catalog = {"studies": [grandchild, child, root], "experiment_groups": []}
        expected = {"studies": sorted(catalog["studies"], key=lambda study: study["id"]), "experiment_groups": []}
        self.assertEqual(self.store.import_catalog(catalog), expected)
        reopened = Store(self.store.root)
        self.assertEqual(reopened.list_catalog(), expected)
        self.assertEqual(reopened.import_catalog({"studies": [], "experiment_groups": []}), expected)

    def test_invalid_lineage_graph_rolls_back_entire_catalog(self):
        experiment = self.submit()
        catalog = self.catalog([experiment["id"]])
        root = catalog["studies"][0]
        child = {
            **root, "id": "follow-up", "parent_study_id": root["id"],
            "rationale": "The parent finding requires a longer comparison.",
        }
        grandchild = {
            **child, "id": "replication", "parent_study_id": child["id"],
            "rationale": "The follow-up finding requires replication.",
        }
        catalog["studies"] = [grandchild, child, root]
        original = self.store.import_catalog(catalog)
        changed_root = {**root, "title": "Must roll back"}
        reassignment = {**catalog["experiment_groups"][0], "id": "must-not-exist"}
        for studies, error in (
            ([changed_root, {**child, "parent_study_id": "missing-study"}], "Unknown parent study"),
            ([{**changed_root, "parent_study_id": root["id"], "rationale": "Self-link"}], "own parent"),
            ([{**changed_root, "parent_study_id": grandchild["id"], "rationale": "Indirect cycle"}], "cycles"),
            ([changed_root, {**child, "id": "new-a", "parent_study_id": "new-b"},
              {**child, "id": "new-b", "parent_study_id": "new-a"}], "cycles"),
        ):
            with self.subTest(studies=studies):
                with self.assertRaisesRegex(LabError, error):
                    self.store.import_catalog({"studies": studies, "experiment_groups": [reassignment]})
                self.assertEqual(self.store.list_catalog(), original)
                self.assertEqual(self.store.get_experiment(experiment["id"]), experiment)

    def test_reparenting_preserves_separate_comparison_and_preparation_history(self):
        comparison, calibration = self.submit(), self.submit()
        comparison_attempt = self.store.create_attempt(comparison["id"])
        failed_calibration = self.store.create_attempt(calibration["id"])
        self.store.update_attempt(failed_calibration["id"], state="failed", error="Preparation failed")
        replacement = self.store.create_attempt(calibration["id"])
        for attempt in (comparison_attempt, replacement):
            for state in ("starting", "running", "completed"):
                self.store.update_attempt(attempt["id"], state=state)
        attempts = self.store.list_attempts()
        events = {attempt["id"]: self.store.events(attempt["id"]) for attempt in attempts}
        catalog = self.catalog([comparison["id"]])
        root = catalog["studies"][0]
        other_root = {**root, "id": "other-study", "title": "Another comparison"}
        child = {
            "id": "follow-up", "title": "Longer comparison", "question": "Does a longer schedule change the result?",
            "parent_study_id": root["id"], "rationale": "The comparison suggests schedule sensitivity.",
        }
        catalog["studies"].extend([child, other_root])
        catalog["experiment_groups"].append({
            "id": "follow-up-calibration", "study_id": child["id"], "purpose": "calibration",
            "label": "Schedule calibration", "model": "example/model", "seeds": [42],
            "experiment_ids": [calibration["id"]],
        })
        original = self.store.import_catalog(catalog)
        expected_studies = {study["id"]: study for study in original["studies"]}
        for changes in (
            {"title": "Revised follow-up", "question": "Does a longer schedule replicate the other finding?",
             "parent_study_id": other_root["id"], "rationale": "The other study isolates the relevant schedule effect."},
            {"parent_study_id": None, "rationale": None},
            {"parent_study_id": root["id"], "rationale": "The original finding remains the motivation."},
        ):
            child = {**child, **changes}
            expected_studies[child["id"]] = child
            result = self.store.import_catalog({"studies": [child], "experiment_groups": []})
            self.assertEqual({study["id"]: study for study in result["studies"]}, expected_studies)
            self.assertEqual(result["experiment_groups"], original["experiment_groups"])
        reopened = Store(self.store.root)
        self.assertEqual(reopened.list_catalog(), result)
        self.assertEqual(reopened.get_experiment(comparison["id"]), comparison)
        self.assertEqual(reopened.get_experiment(calibration["id"]), calibration)
        self.assertEqual(reopened.list_attempts(), attempts)
        for attempt_id, history in events.items():
            self.assertEqual(reopened.events(attempt_id), history)

    def test_pre_lineage_catalog_migrates_as_roots(self):
        experiment = self.submit()
        attempt = self.store.create_attempt(experiment["id"])
        events = self.store.events(attempt["id"])
        catalog = self.catalog([experiment["id"]])
        catalog["studies"].append({
            **catalog["studies"][0], "id": "attention-study-follow-up", "title": "Named follow-up without explicit lineage",
        })
        original = self.store.import_catalog(catalog)
        with self.store._db(write=True) as connection:
            connection.execute("DROP TABLE study_lineage")
        reopened = Store(self.store.root)
        self.assertEqual(reopened.list_catalog(), original)
        self.assertTrue(all(study["parent_study_id"] is None and study["rationale"] is None
                            for study in reopened.list_catalog()["studies"]))
        self.assertEqual(reopened.get_experiment(experiment["id"]), experiment)
        self.assertEqual(reopened.get_attempt(attempt["id"]), attempt)
        self.assertEqual(reopened.events(attempt["id"]), events)
        with reopened._db() as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
        self.store.update_attempt(attempt["id"], state="starting")
        self.assertEqual(reopened.get_attempt(attempt["id"])["state"], "starting")

    def test_invalid_catalog_reference_rolls_back_metadata_and_reassignment(self):
        experiment = self.submit()
        original_catalog = self.catalog([experiment["id"]])
        original_catalog["studies"].append({
            **original_catalog["studies"][0], "id": "other-study",
        })
        original = self.store.import_catalog(original_catalog)
        for bad_group in (
            {"experiment_ids": ["exp_" + "0" * 32]},
            {"study_id": "missing-study", "experiment_ids": []},
        ):
            with self.subTest(bad_group=bad_group):
                catalog = self.catalog([])
                catalog["studies"][0]["title"] = "Must roll back"
                catalog["studies"][0].update({
                    "parent_study_id": "other-study", "rationale": "A finding motivates this follow-up.",
                })
                group = catalog["experiment_groups"][0]
                group["label"] = "Must also roll back"
                catalog["experiment_groups"].extend([
                    {**group, "id": "new-group", "experiment_ids": [experiment["id"]]},
                    {**group, "id": "invalid-group", **bad_group},
                ])
                with self.assertRaises(LabError):
                    self.store.import_catalog(catalog)
                self.assertEqual(self.store.list_catalog(), original)

    def test_catalog_rejects_malformed_and_conflicting_assignments(self):
        experiment = self.submit()
        original = self.store.import_catalog(self.catalog([experiment["id"]]))
        invalid = [None, [], {}, {"studies": {}, "experiment_groups": []}]
        for changes in (
            {"id": "../outside"}, {"study_id": None}, {"purpose": []}, {"purpose": "unknown"},
            {"label": " "}, {"model": False}, {"seeds": [True]}, {"seeds": [1.5]},
            {"seeds": [7, 7]}, {"seeds": "7"}, {"experiment_ids": "exp"},
            {"experiment_ids": [None]}, {"experiment_ids": [experiment["id"], experiment["id"]]},
            {"unexpected": "field"},
        ):
            catalog = self.catalog([])
            catalog["experiment_groups"][0].update(changes)
            invalid.append(catalog)
        for changes in (
            {"parent_study_id": None, "rationale": "Root rationale is invalid"},
            {"parent_study_id": None, "rationale": ""},
            {"parent_study_id": "other-study", "rationale": None},
            {"parent_study_id": "other-study", "rationale": ""},
            {"parent_study_id": "other-study", "rationale": " \t\n"},
            {"parent_study_id": "other-study", "rationale": False},
            {"parent_study_id": "../outside", "rationale": "Follow up"},
            {"parent_study_id": [], "rationale": "Follow up"},
        ):
            catalog = self.catalog([])
            catalog["studies"][0].update(changes)
            invalid.append(catalog)
        legacy_shape = self.catalog([])
        del legacy_shape["studies"][0]["parent_study_id"]
        del legacy_shape["studies"][0]["rationale"]
        invalid.append(legacy_shape)
        duplicate_study = self.catalog([])
        duplicate_study["studies"] *= 2
        invalid.append(duplicate_study)
        duplicate_group = self.catalog([])
        duplicate_group["experiment_groups"] *= 2
        invalid.append(duplicate_group)
        duplicate_membership = self.catalog([experiment["id"]])
        duplicate_membership["experiment_groups"].append({
            **duplicate_membership["experiment_groups"][0], "id": "other-group",
        })
        invalid.append(duplicate_membership)
        missing_field = self.catalog([])
        del missing_field["studies"][0]["question"]
        invalid.append(missing_field)
        for catalog in invalid:
            with self.subTest(catalog=catalog), self.assertRaises(LabError):
                self.store.import_catalog(catalog)
            self.assertEqual(self.store.list_catalog(), original)

    def test_partial_catalog_import_reassigns_only_explicit_memberships(self):
        first, second, third = self.submit(), self.submit(), self.submit()
        original = self.store.import_catalog(self.catalog([first["id"], second["id"]]))
        existing = self.catalog([])["experiment_groups"][0]
        existing["label"] = "Renamed comparison"
        diagnostic = {
            "id": "diagnostic", "study_id": None, "purpose": "diagnostic",
            "label": "Standalone check", "model": None, "seeds": [], "experiment_ids": [second["id"]],
        }
        result = self.store.import_catalog({"studies": [], "experiment_groups": [diagnostic, existing]})
        self.assertEqual(result["studies"], original["studies"])
        self.assertEqual([group["id"] for group in result["experiment_groups"]], ["comparison-main", "diagnostic"])
        self.assertEqual(result["experiment_groups"][0]["label"], "Renamed comparison")
        self.assertEqual(result["experiment_groups"][0]["experiment_ids"], [first["id"]])
        diagnostic["experiment_ids"] = [third["id"]]
        result = self.store.import_catalog({"studies": [], "experiment_groups": [diagnostic]})
        self.assertEqual(result["experiment_groups"][0]["experiment_ids"], [first["id"]])
        self.assertEqual(result["experiment_groups"][1]["experiment_ids"], [second["id"], third["id"]])
        self.assertEqual(self.store.list_experiments(), [first, second, third])

    def test_catalog_upgrade_preserves_history_bundles_and_active_workers(self):
        experiment = self.submit(gpu=True)
        failed = self.store.create_attempt(experiment["id"], gpu_uuid=GPU)
        self.store.update_attempt(failed["id"], state="failed", error="Historical failure")
        active = self.store.create_attempt(experiment["id"], gpu_uuid=GPU)
        attempts = self.store.list_attempts()
        events = {attempt["id"]: self.store.events(attempt["id"]) for attempt in attempts}
        bundle = self.store.bundle_dir(experiment["id"])
        contents = {str(path.relative_to(bundle)): path.read_bytes() for path in bundle.rglob("*") if path.is_file()}
        # Model an existing version-1 store before the additive catalog tables existed.
        with self.store._db(write=True) as connection:
            connection.execute("DROP TABLE experiment_group_memberships")
            connection.execute("DROP TABLE study_lineage")
            connection.execute("DROP TABLE experiment_groups")
            connection.execute("DROP TABLE studies")
        reopened = Store(self.store.root)
        reopened.import_catalog(self.catalog([experiment["id"]]))
        self.assertEqual(reopened.get_experiment(experiment["id"]), experiment)
        self.assertEqual(reopened.list_attempts(), attempts)
        for attempt_id, history in events.items():
            self.assertEqual(reopened.events(attempt_id), history)
        self.assertEqual({str(path.relative_to(bundle)): path.read_bytes() for path in bundle.rglob("*") if path.is_file()}, contents)
        reopened.verify_experiment(experiment["id"])
        with self.assertRaises(LabError):
            self.store.create_attempt(experiment["id"], gpu_uuid=GPU)
        self.store.update_attempt(active["id"], state="starting")
        self.assertEqual(reopened.get_attempt(active["id"])["state"], "starting")

    def test_study_cli_import_list_and_invalid_import(self):
        experiment = self.submit()
        path = self.folder / "catalog.json"
        path.write_text(json.dumps(self.catalog([experiment["id"]])), encoding="utf-8")
        for command in (["import", str(path)], ["list"]):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["--root", str(self.store.root), "study", *command]), 0)
            self.assertEqual(json.loads(output.getvalue()), self.store.list_catalog())
        expected = self.store.list_catalog()
        path.write_text(json.dumps(self.catalog(["exp_" + "0" * 32])), encoding="utf-8")
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(main(["--root", str(self.store.root), "study", "import", str(path)]), 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("Unknown experiment", errors.getvalue())
        self.assertEqual(self.store.list_catalog(), expected)

    def test_overview_adds_catalog_without_hiding_unclassified_records(self):
        from autonomous_research_lab.web import create_server

        classified, unclassified = self.submit(), self.submit()
        attempt = self.store.create_attempt(classified["id"])
        self.store.update_attempt(attempt["id"], state="failed", error="Preserved submission")
        attempt = self.store.get_attempt(attempt["id"])
        catalog = self.store.import_catalog(self.catalog([classified["id"]]))
        with create_server(self.store, port=0) as server:
            headers = {"Host": f"127.0.0.1:{server.server_port}"}
            response = server.app.handle("GET", "/api/overview", headers, None)
            self.assertEqual(response.status, 200)
            overview = json.loads(response.body)
        self.assertEqual(overview["studies"], catalog["studies"])
        self.assertEqual(overview["experiment_groups"], catalog["experiment_groups"])
        self.assertEqual(overview["experiments"], [classified, unclassified])
        self.assertEqual(overview["attempts"], [attempt])
        self.assertEqual(overview["programs"], [])
        self.assertIn("runner", overview)
        self.assertIn("codex", overview["providers"])

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
