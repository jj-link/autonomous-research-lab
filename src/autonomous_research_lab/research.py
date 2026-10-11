"""Durable research state machine. Models propose; this coordinator owns authority."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import uuid

from . import execution, prompts
from .model_client import ModelClient, ModelError, validate_model_config
from .models import LabError, Limits, ExperimentSpec, TERMINAL_STATES, canonical_json, positive_int, read_json, relative_path, require_keys
from .sources import SourceTools
from .store import Store, identifier, utc_now

FINISHED = {"completed", "failed", "stopped"}
POLICIES = {"fixed_plan", "direct", "candidate_comparison"}


def new_id(prefix):
    return prefix + "_" + uuid.uuid4().hex


def text(value, label, limit=12000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise LabError(f"{label} must be nonempty text of at most {limit} characters")
    return value


def strings(value, label, maximum=20):
    if not isinstance(value, list) or len(value) > maximum:
        raise LabError(f"{label} must be an array of at most {maximum} strings")
    return [text(item, label) for item in value]


def resource_spec(value):
    require_keys(value, set(Limits.__dataclass_fields__) | {"gpu"}, "resources")
    gpu = value.get("gpu", False)
    if type(gpu) is not bool:
        raise LabError("resources.gpu must be boolean")
    return {"gpu": gpu, **Limits.from_dict({k: v for k, v in value.items() if k != "gpu"}).to_dict()}


def within(requested, maximum):
    if requested["gpu"] and not maximum["gpu"]:
        raise LabError("GPU execution is outside the brief")
    for key in Limits.__dataclass_fields__:
        cap, needed = maximum[key], requested[key]
        if cap is not None and (needed is None or needed > cap):
            raise LabError(f"Requested {key} exceeds the permitted allocation")


def validate_brief(value):
    allowed = {"goal", "scope", "sources", "model", "max_experiments", "max_model_calls", "max_output_tokens_per_call", "max_wall_seconds", "max_repairs", "resources", "policy", "data_files"}
    require_keys(value, allowed, "brief")
    brief = {"goal": text(value.get("goal"), "goal"), "scope": value.get("scope", ""),
             "model": validate_model_config(value.get("model")), "sources": strings(value.get("sources", []), "sources"),
             "resources": resource_spec(value.get("resources", {"wall_seconds": 120})),
             "policy": value.get("policy", "candidate_comparison")}
    if not isinstance(brief["scope"], str) or len(brief["scope"]) > 12000:
        raise LabError("scope must be text of at most 12000 characters")
    if brief["policy"] not in POLICIES:
        raise LabError("Unknown research policy")
    for key, default, low, high in (("max_experiments", 2, 1, 20), ("max_model_calls", 12, 1, 100),
                                    ("max_output_tokens_per_call", 8192, 128, 32768), ("max_wall_seconds", 600, 1, 86400), ("max_repairs", 1, 0, 3)):
        count = positive_int(value.get(key, default), key, low)
        if count > high:
            raise LabError(f"{key} must be <= {high}")
        brief[key] = count
    data = value.get("data_files", {})
    if not isinstance(data, dict) or len(data) > 16:
        raise LabError("data_files must map at most 16 relative names to UTF-8 text")
    for name, content in data.items():
        relative_path(name, "data file")
        if not isinstance(content, str):
            raise LabError("data_files contents must be text, not host paths")
    if sum(len(v.encode("utf-8")) for v in data.values()) > 256 * 1024:
        raise LabError("Approved data exceeds 256 KiB")
    brief["data_files"] = data
    canonical_json(brief)
    return brief


class BudgetReached(LabError):
    pass


class SchedulingInterrupted(LabError):
    pass


class ResearchEngine:
    def __init__(self, store: Store):
        self.store = store
        self.path = store.root / "research.sqlite"
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS programs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    @contextmanager
    def _db(self, write=False):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    def folder(self, program_id):
        return self.store.root / "programs" / identifier(program_id, "program")

    @staticmethod
    def _load(db, program_id):
        row = db.execute("SELECT payload FROM programs WHERE id=?", (program_id,)).fetchone()
        if row is None:
            raise LabError("Research program does not exist")
        return json.loads(row[0])

    def get(self, program_id):
        identifier(program_id, "program")
        with self._db() as db:
            return self._load(db, program_id)

    def list(self):
        with self._db() as db:
            return sorted((json.loads(row[0]) for row in db.execute("SELECT payload FROM programs")), key=lambda p: p["created_at"], reverse=True)

    def _change(self, program_id, operation):
        with self._db(True) as db:
            program = self._load(db, program_id)
            operation(program)
            program["updated_at"] = utc_now()
            db.execute("UPDATE programs SET payload=? WHERE id=?", (canonical_json(program), program_id))
        return program

    def _patch(self, program_id, **fields):
        return self._change(program_id, lambda p: p.update(fields))

    def create(self, brief):
        brief = validate_brief(brief)
        program_id, question_id = new_id("program"), new_id("question")
        stamp = utc_now()
        program = {"id": program_id, "state": "draft", "phase": "sources", "brief": brief, "created_at": stamp, "updated_at": stamp,
                   "active_attempt_id": None, "pending_call_id": None, "selected_design_id": None,
                   "questions": [{"id": question_id, "text": brief["goal"], "parent_id": None, "status": "open", "history": []}],
                   "sources": [], "source_cursor": 0, "designs": [], "decisions": [], "analyses": [], "calls": [], "implementations": [],
                   "fixed_queue": [], "requested_records": [], "stop_reason": None, "environment": None,
                   "usage": {"model_calls": 0, "attempts": 0, "implementation_attempts": 0, "wall_seconds_charged": 0.0,
                             "wall_seconds_estimated": False, "prompt_tokens_known": 0, "completion_tokens_known": 0, "unknown_token_calls": 0}}
        self.folder(program_id).mkdir(parents=True)
        with self._db(True) as db:
            db.execute("INSERT INTO programs VALUES (?,?)", (program_id, canonical_json(program)))
        return program

    def _preflight(self, program):
        config = program["brief"]["model"]
        if not program["active_attempt_id"] and config.get("api_key_env") and not os.environ.get(config["api_key_env"]):
            raise LabError(f"Cloud API credential variable {config['api_key_env']} is not set in the coordinator environment")
        resources = program["brief"]["resources"]
        spec = ExperimentSpec.from_dict({"name": "preflight", "source": "code", "entrypoint": "main.py", "gpu": resources["gpu"],
                                        "limits": {k: resources[k] for k in Limits.__dataclass_fields__}}, self.folder(program["id"]))
        runner = execution.configuration(self.store)
        return execution.prepare(self.store, {"gpu_uuid": runner.get("gpu_uuid") if spec.gpu else None}, spec)

    def start(self, program_id, background=True):
        program = self.get(program_id)
        if program["state"] not in {"draft", "paused", "blocked", "running", "pausing", "stopping"}:
            raise LabError("A terminal research program cannot be restarted; create a new brief")
        try:
            environment = self._preflight(program)
        except LabError as exc:
            def block(p):
                if p["state"] not in FINISHED | {"stopping"}:
                    p.update(state="blocked", stop_reason=str(exc))
            return self._change(program_id, block)
        def activate(p):
            if p["state"] in {"draft", "paused", "blocked"}:
                p.update(state="running", stop_reason=None)
            p["environment"] = environment
        self._change(program_id, activate)
        if background:
            self._launch(program_id)
            return self.get(program_id)
        return self.work(program_id)

    def _launch(self, program_id):
        with (self.folder(program_id) / "coordinator.log").open("ab") as stream:
            process = subprocess.Popen([sys.executable, "-m", "autonomous_research_lab.research_worker", "--root", str(self.store.root), "--program", program_id],
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, start_new_session=True)
        threading.Thread(target=process.wait, daemon=True, name=f"arl-program-reap-{process.pid}").start()

    def pause(self, program_id):
        def update(p):
            if p["state"] == "running":
                p["state"] = "pausing"
            elif p["state"] in {"draft", "blocked"}:
                p["state"] = "paused"
        return self._change(program_id, update)

    def stop(self, program_id):
        def update(p):
            if p["state"] not in FINISHED:
                p.update(state="stopping", stop_reason="Stopped by user")
        program = self._change(program_id, update)
        attempt_id = program["active_attempt_id"]
        if attempt_id:
            attempts = {a["id"]: a for a in self.store.list_attempts()}
            if attempt_id in attempts and attempts[attempt_id]["state"] not in TERMINAL_STATES:
                self.store.request_stop(attempt_id)
                execution.launch_supervisor(self.store, attempt_id, recovery=True)
        if program["state"] == "stopping":
            if attempt_id or program["pending_call_id"]:
                self._launch(program_id)
            else:
                self._patch(program_id, state="stopped")
        return self.get(program_id)

    def resume(self, program_id, background=True):
        program = self.get(program_id)
        if program["state"] not in {"paused", "blocked"}:
            raise LabError("Only a paused or blocked program can resume")
        # A failed/uncertain request is never invisibly replayed by recovery. Explicit resume
        # permits another request, which receives a new ID and spends another call allowance.
        pending = next((c for c in program["calls"] if c["id"] == program["pending_call_id"]), None)
        if pending and pending["status"] in {"failed", "interrupted", "invalid"}:
            self._patch(program_id, pending_call_id=None)
        return self.start(program_id, background)

    def recover(self, background=True):
        return [self.start(p["id"], background) for p in self.list() if p["state"] in {"running", "pausing", "stopping"}]

    def _finish(self, program_id, state, reason):
        def update(p):
            if p["state"] == "stopping":
                p.update(state="stopped", stop_reason="Stopped by user")
            elif p["state"] not in FINISHED:
                p.update(state=state, stop_reason=reason)
        return self._change(program_id, update)

    def _known(self, program):
        records = {}
        for category in ("questions", "sources", "designs", "decisions", "analyses", "implementations", "calls"):
            records.update((r["id"], r) for r in program[category])
        for implementation in program["implementations"]:
            if implementation.get("attempt_id"):
                records[implementation["attempt_id"]] = {"id": implementation["attempt_id"], "kind": "attempt"}
        return records

    def _attempt_record(self, attempt_id):
        record = self.store.get_attempt(attempt_id)
        path = self.store.attempt_dir(attempt_id) / "console.log"
        try:
            with path.open("rb") as stream:
                size = stream.seek(0, os.SEEK_END)
                stream.seek(max(0, size - 8192))
                record["console"] = {"text": stream.read(8192).decode("utf-8", errors="replace"), "truncated": size > 8192}
        except FileNotFoundError:
            record["console"] = {"text": "", "truncated": False}
        return record

    def _refs(self, program, value):
        refs = strings(value, "evidence_refs", 30)
        if set(refs) - self._known(program).keys():
            raise LabError("Model cited unknown evidence IDs")
        return refs

    def _context(self, program):
        brief = {k: v for k, v in program["brief"].items() if k not in {"model", "data_files"}}
        return {"brief": brief, "approved_data": list(program["brief"]["data_files"]), "usage": program["usage"],
                "environment": program["environment"], "questions": program["questions"], "sources": program["sources"][-5:],
                "designs": program["designs"][-6:], "analyses": program["analyses"][-4:], "decisions": program["decisions"][-4:],
                "record_index": [{"id": r["id"], "kind": key} for key in ("sources", "designs", "analyses", "decisions", "implementations", "calls") for r in program[key]]
                    + [{"id": i["attempt_id"], "kind": "attempt"} for i in program["implementations"] if i.get("attempt_id")],
                "requested_records": program["requested_records"]}

    def _call(self, program_id, role, payload):
        program = self.get(program_id)
        pending = program["pending_call_id"]
        if pending:
            call = next(c for c in program["calls"] if c["id"] == pending)
            path = self.folder(program_id) / "calls" / (pending + ".json")
            if call["role"] != role:
                raise LabError("Pending model call does not match the durable phase")
            if call["status"] == "pending" and path.exists():
                self._record_call(program_id, pending, read_json(path, 16 * 1024**2))
                return self._call(program_id, role, payload)
            if call["status"] == "completed":
                return read_json(path, 16 * 1024**2)["response"]["content"]
            if call["status"] == "pending":
                record = {"error": "Coordinator interrupted during request; provider usage is unknown. Explicit resume is required.", "status": "interrupted", "response": {}}
                self._record_call(program_id, pending, record)
            raise LabError(call.get("error") or "Model request was interrupted; inspect the recorded call before resuming")
        if len(program["calls"]) >= program["brief"]["max_model_calls"]:
            raise BudgetReached("Model-call budget exhausted")
        messages = [{"role": "system", "content": getattr(prompts, role.upper())}, {"role": "user", "content": canonical_json(payload)}]
        if len(canonical_json(messages).encode()) > 1024 * 1024:
            raise LabError("Model context exceeds the 1 MiB request bound; inspect the retained records")
        call_id = new_id("call")
        def reserve(p):
            if p["state"] != "running":
                raise SchedulingInterrupted("Scheduling was interrupted by pause or stop")
            p["calls"].append({"id": call_id, "role": role, "status": "pending", "started_at": utc_now(), "usage": None})
            p["pending_call_id"] = call_id
            p["usage"]["model_calls"] += 1
        self._change(program_id, reserve)
        path = self.folder(program_id) / "calls" / (call_id + ".json")
        execution.atomic_json(path.with_suffix(".request.json"), {"messages": messages, "model": program["brief"]["model"], "max_tokens": program["brief"]["max_output_tokens_per_call"]})
        try:
            response = ModelClient(program["brief"]["model"]).complete(messages, program["brief"]["max_output_tokens_per_call"])
            record = {"status": "completed", "response": response}
        except ModelError as exc:
            record = {"status": "failed", "error": str(exc), "response": exc.record}
        execution.atomic_json(path, record)
        self._record_call(program_id, call_id, record)
        if record["status"] != "completed":
            raise LabError(record["error"])
        return record["response"]["content"]

    def _record_call(self, program_id, call_id, record):
        def update(p):
            call = next(c for c in p["calls"] if c["id"] == call_id)
            if call["status"] != "pending":
                return
            usage = record.get("response", {}).get("usage") or {}
            call.update(status=record["status"], finished_at=utc_now(), usage=usage, error=record.get("error"))
            unknown = False
            for field in ("prompt_tokens", "completion_tokens"):
                value = usage.get(field) if isinstance(usage, dict) else None
                if type(value) is not int or value < 0:
                    unknown = True
                else:
                    p["usage"][field + "_known"] += value
            p["usage"]["unknown_token_calls"] += int(unknown)
        self._change(program_id, update)

    def _source(self, program_id, url=None, query=None, decision=None):
        if self.get(program_id)["state"] != "running":
            raise SchedulingInterrupted("Source retrieval interrupted by pause or stop")
        try:
            tools = SourceTools()
            records = tools.search(query, limit=5) if query is not None else [tools.fetch(url)]
        except LabError as exc:
            records = [{"id": new_id("source"), "url": url, "query": query, "error": str(exc), "retrieved_at": utc_now()}]
        def update(p):
            for record in records:
                if not any(s["id"] == record["id"] for s in p["sources"]):
                    p["sources"].append(record)
            if decision is not None:
                p["decisions"].append(decision)
            p["pending_call_id"] = None
        self._change(program_id, update)

    def _decision(self, program_id):
        program = self.get(program_id)
        if program["usage"]["attempts"] >= program["brief"]["max_experiments"]:
            raise BudgetReached("Experiment-attempt budget exhausted")
        if program["usage"]["wall_seconds_charged"] >= program["brief"]["max_wall_seconds"]:
            raise BudgetReached("Execution-time budget exhausted")
        response = self._call(program_id, "director", self._context(program))
        action = response.get("action")
        rationale = text(response.get("rationale"), "decision rationale")
        refs = self._refs(program, response.get("evidence_refs", []))
        decision = {"id": new_id("decision"), "action": action, "rationale": rationale, "evidence_refs": refs, "call_id": self.get(program_id)["pending_call_id"]}
        if action in {"search", "fetch"}:
            self._source(program_id, url=text(response.get("url"), "url", 2048) if action == "fetch" else None,
                         query=text(response.get("query"), "query", 500) if action == "search" else None, decision=decision)
            return
        if action == "finish":
            self._change(program_id, lambda p: (p["decisions"].append(decision), p.update(pending_call_id=None)))
            self._finish(program_id, "completed", rationale)
            return
        if action == "records":
            ids = strings(response.get("ids"), "record ids", 3)
            known = self._known(program)
            if set(ids) - known.keys():
                raise LabError("Requested record IDs do not exist")
            records = [self._attempt_record(i) if known[i].get("kind") == "attempt" else known[i] for i in ids]
            self._change(program_id, lambda p: (p["decisions"].append(decision), p.update(requested_records=records, pending_call_id=None)))
            return
        if action == "questions":
            updates = response.get("updates")
            if not isinstance(updates, list) or not 1 <= len(updates) <= 10:
                raise LabError("Question updates must contain 1–10 items")
            by_id = {q["id"]: q for q in program["questions"]}
            for item in updates:
                if not isinstance(item, dict) or item.get("id") not in by_id or item.get("status") not in {"open", "answered", "unresolved", "closed"}:
                    raise LabError("Invalid question update")
                text(item.get("rationale"), "question rationale")
                self._refs(program, item.get("evidence_refs", []))
            def apply(p):
                for item in updates:
                    question = next(q for q in p["questions"] if q["id"] == item["id"])
                    question["history"].append({**item, "decision_id": decision["id"]})
                    question["status"] = item["status"]
                p["decisions"].append(decision)
                p["pending_call_id"] = None
            self._change(program_id, apply)
            return
        if action != "experiment":
            raise LabError("Director returned an unsupported action")
        candidates = response.get("candidates")
        policy = program["brief"]["policy"]
        maximum = program["brief"]["max_experiments"] if policy == "fixed_plan" else (1 if policy == "direct" else 3)
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= maximum:
            raise LabError("Director returned an invalid candidate count")
        designs = []
        prior_evidence = {a["id"] for a in program["analyses"]} | {i["attempt_id"] for i in program["implementations"] if i.get("attempt_id")}
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise LabError("Candidate must be an object")
            design = {key: text(candidate.get(key), key) for key in ("question", "parent_question_id", "motivation", "procedure", "measurements", "interpretation", "requirements", "relationship")}
            if design["parent_question_id"] not in {q["id"] for q in program["questions"]}:
                raise LabError("Design parent question does not exist")
            design.update(id=new_id("design"), decision_id=decision["id"], evidence_refs=self._refs(program, candidate.get("evidence_refs", [])), resources=resource_spec(candidate.get("resources")), feasible=True)
            if prior_evidence and policy != "fixed_plan" and not prior_evidence.intersection(design["evidence_refs"]):
                raise LabError("Follow-up design does not cite any earlier analysis or attempt")
            try:
                within(design["resources"], program["brief"]["resources"])
                if design["resources"]["wall_seconds"] + program["usage"]["wall_seconds_charged"] > program["brief"]["max_wall_seconds"]:
                    raise LabError("Design exceeds the remaining execution-time budget")
            except LabError as exc:
                design.update(feasible=False, rejection=str(exc))
            designs.append(design)
        feasible = [d["id"] for d in designs if d["feasible"]]
        decision["candidate_ids"] = [d["id"] for d in designs]
        def save(p):
            p["designs"].extend(designs)
            p["decisions"].append(decision)
            p.update(pending_call_id=None, requested_records=[], candidate_ids=feasible, phase="select")
            if policy == "fixed_plan":
                p["fixed_queue"] = feasible.copy()
        self._change(program_id, save)
        if not feasible:
            raise BudgetReached("No proposed experiment fits the permitted resources")
        if len(feasible) == 1 or policy == "fixed_plan":
            self._select(program_id, feasible[0])

    def _select(self, program_id, design_id, decision=None):
        def select(p):
            if decision is not None:
                p["decisions"].append(decision)
            design = next(d for d in p["designs"] if d["id"] == design_id)
            question = {"id": new_id("question"), "text": design["question"], "parent_id": design["parent_question_id"], "status": "open", "history": []}
            p["questions"].append(question)
            design["question_id"] = question["id"]
            p.update(selected_design_id=design_id, phase="build", pending_call_id=None)
            if design_id in p["fixed_queue"]:
                p["fixed_queue"].remove(design_id)
        self._change(program_id, select)

    def _selection(self, program_id):
        program = self.get(program_id)
        candidates = [d for d in program["designs"] if d["id"] in program["candidate_ids"]]
        if not candidates:
            raise BudgetReached("No feasible candidate remains")
        if len(candidates) == 1 or program["brief"]["policy"] == "fixed_plan":
            self._select(program_id, candidates[0]["id"])
            return
        response = self._call(program_id, "selector", {"context": self._context(program), "feasible_candidates": candidates})
        selected = response.get("selected_id")
        ids = {d["id"] for d in candidates}
        alternatives = response.get("alternatives")
        if selected not in ids or not isinstance(alternatives, list) or {a.get("id") for a in alternatives if isinstance(a, dict)} != ids - {selected}:
            raise LabError("Selection must name one feasible design and explain every alternative")
        for alternative in alternatives:
            text(alternative.get("reason"), "alternative reason")
        decision = {"id": new_id("decision"), "action": "select", "selected_id": selected, "rationale": text(response.get("rationale"), "selection rationale"), "alternatives": alternatives}
        self._select(program_id, selected, decision)

    def _build(self, program_id):
        program = self.get(program_id)
        design = next(d for d in program["designs"] if d["id"] == program["selected_design_id"])
        prior = [i for i in program["implementations"] if i["design_id"] == design["id"]]
        if program["pending_call_id"] is None and len(prior) > program["brief"]["max_repairs"]:
            raise LabError("Implementation repair allowance exhausted")
        previous = []
        for item in prior:
            proposal = read_json(self.folder(program_id) / "calls" / (item["call_id"] + ".json"), 16 * 1024**2)["response"]["content"]
            previous.append({**item, "proposal": proposal, "attempt": self._attempt_record(item["attempt_id"]) if item.get("attempt_id") else None})
        response = self._call(program_id, "builder", {"design": design, "environment": program["environment"], "approved_data": program["brief"]["data_files"], "prior_implementations": previous})
        call_id = self.get(program_id)["pending_call_id"]
        implementation = next((i for i in prior if i["call_id"] == call_id), None)
        if implementation is None:
            implementation = {"id": new_id("implementation"), "call_id": call_id, "design_id": design["id"], "status": "building", "attempt_id": None, "experiment_id": None}
            self._change(program_id, lambda p: (p["implementations"].append(implementation), p["usage"].update(implementation_attempts=p["usage"]["implementation_attempts"] + 1)))
        folder = self.folder(program_id) / "implementations" / implementation["id"]
        try:
            files, manifest = response.get("files"), response.get("manifest")
            if not isinstance(files, dict) or not 1 <= len(files) <= 24 or not all(isinstance(v, str) for v in files.values()):
                raise LabError("Builder files must contain 1–24 UTF-8 text files")
            if sum(len(v.encode()) for v in files.values()) > 256 * 1024:
                raise LabError("Generated code exceeds 256 KiB")
            if not isinstance(manifest, dict) or manifest.get("source") != "code":
                raise LabError("Builder manifest.source must be exactly code")
            spec = ExperimentSpec.from_dict(manifest, folder)
            within({"gpu": spec.gpu, **spec.limits.to_dict()}, design["resources"])
            if spec.question != design["question"] or spec.procedure != design["procedure"]:
                raise LabError("Builder changed the selected question or procedure")
            if not spec.entrypoint.endswith(".py") or "results.json" not in spec.outputs:
                raise LabError("Builder must supply a Python entrypoint and results.json output")
            approved = {"data/" + name: value for name, value in program["brief"]["data_files"].items()}
            for name, content in files.items():
                relative_path(name, "generated file")
                if name == "data" or name.startswith("data/"):
                    raise LabError("Builder cannot replace approved data")
                if name.endswith(".py"):
                    ast.parse(content, filename=name)
            for name, content in {**files, **approved}.items():
                path = spec.source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    if path.is_symlink() or path.read_text(encoding="utf-8") != content:
                        raise LabError("Previously recorded implementation was changed")
                else:
                    with path.open("x", encoding="utf-8", newline="\n") as stream:
                        stream.write(content)
            matches = [e for e in self.store.list_experiments() if e["manifest"] == spec.to_dict()]
            experiment = matches[0] if matches else self.store.submit(spec)
            self.store.verify_experiment(experiment["id"])
            def saved(p):
                item = next(i for i in p["implementations"] if i["id"] == implementation["id"])
                item.update(status="ready", experiment_id=experiment["id"])
                p.update(phase="execute", pending_call_id=None)
            self._change(program_id, saved)
        except (LabError, SyntaxError, ValueError, RecursionError) as exc:
            def failed(p):
                item = next(i for i in p["implementations"] if i["id"] == implementation["id"])
                item.update(status="invalid", error=str(exc))
                p["pending_call_id"] = None
            self._change(program_id, failed)
            if len(prior) >= program["brief"]["max_repairs"]:
                raise LabError(f"Implementation repair allowance exhausted: {exc}") from exc

    def _execute(self, program_id):
        program = self.get(program_id)
        implementation = program["implementations"][-1]
        experiment = self.store.get_experiment(implementation["experiment_id"])
        reservation = experiment["manifest"]["limits"]["wall_seconds"]
        if implementation["attempt_id"] is None:
            def reserve(p):
                if p["state"] != "running":
                    raise SchedulingInterrupted("Scheduling was interrupted by pause or stop")
                if p["usage"]["attempts"] >= p["brief"]["max_experiments"] or p["usage"]["wall_seconds_charged"] + reservation > p["brief"]["max_wall_seconds"]:
                    raise BudgetReached("Insufficient remaining experiment or execution-time budget")
                item = p["implementations"][-1]
                item.update(attempt_id=new_id("run"), status="reserved", reserved_seconds=reservation)
                p["active_attempt_id"] = item["attempt_id"]
                p["usage"]["attempts"] += 1
            program = self._change(program_id, reserve)
            implementation = program["implementations"][-1]
        attempt_id = implementation["attempt_id"]
        known = {a["id"]: a for a in self.store.list_attempts(experiment["id"])}
        if attempt_id not in known:
            if self.get(program_id)["state"] == "stopping":
                def cancel(p):
                    p["implementations"][-1].update(status="cancelled_before_launch", wall_seconds_charged=0, wall_seconds_estimated=False)
                    p["active_attempt_id"] = None
                self._change(program_id, cancel)
                return
            config = execution.configuration(self.store)
            attempt = self.store.create_attempt(experiment["id"], gpu_uuid=config.get("gpu_uuid") if experiment["manifest"]["gpu"] else None, attempt_id=attempt_id)
        else:
            attempt = known[attempt_id]
        if attempt["state"] not in TERMINAL_STATES:
            if self.get(program_id)["state"] == "stopping":
                self.store.request_stop(attempt_id)
            attempt = execution.supervise(self.store, attempt_id, recovery=attempt["state"] != "queued")
        if attempt["state"] not in TERMINAL_STATES:
            if attempt.get("error") and "termination unconfirmed" in attempt["error"]:
                raise LabError("Worker termination is unconfirmed; retain allocation and run recovery")
            attempt = execution.wait_for_attempt(self.store, attempt_id)
        result = attempt.get("result") or {}
        measured = result.get("wall_seconds")
        estimated = measured is None
        charged = reservation if estimated else measured
        def collected(p):
            item = next(i for i in p["implementations"] if i["attempt_id"] == attempt_id)
            if item["status"] == "collected":
                return
            item.update(status="collected", execution_state=attempt["state"], error=attempt.get("error"), wall_seconds_charged=charged, wall_seconds_estimated=estimated)
            p["usage"]["wall_seconds_charged"] += charged
            p["usage"]["wall_seconds_estimated"] |= estimated
            p.update(active_attempt_id=None, phase="analyze")
        self._change(program_id, collected)

    def _analyze(self, program_id):
        program = self.get(program_id)
        implementation = program["implementations"][-1]
        attempt = self._attempt_record(implementation["attempt_id"])
        design = next(d for d in program["designs"] if d["id"] == implementation["design_id"])
        response = self._call(program_id, "analyst", {"design": design, "attempt": attempt, "questions": program["questions"], "sources": program["sources"][-5:]})
        outcome = response.get("outcome")
        if outcome not in {"supported", "contradicted", "inconclusive", "execution_failed"} or (attempt["state"] != "completed" and outcome != "execution_failed"):
            raise LabError("Analysis outcome conflicts with execution status")
        refs = self._refs(program, response.get("evidence_refs", []))
        if attempt["id"] not in refs:
            raise LabError("Analysis must cite the actual attempt")
        interpretations = response.get("interpretations")
        questions = response.get("next_questions")
        if not isinstance(interpretations, list) or len(interpretations) > 20 or not isinstance(questions, list) or len(questions) > 3:
            raise LabError("Invalid analysis interpretations or next questions")
        for item in interpretations:
            if not isinstance(item, dict):
                raise LabError("Interpretation must be an object")
            text(item.get("claim"), "claim")
            if not self._refs(program, item.get("evidence_refs", [])):
                raise LabError("Interpretations need evidence references")
        for question in questions:
            if not isinstance(question, dict) or question.get("parent_question_id") not in {q["id"] for q in program["questions"]}:
                raise LabError("Follow-up question parent does not exist")
            text(question.get("text"), "question")
            text(question.get("rationale"), "question rationale")
        analysis = {"id": new_id("analysis"), "attempt_id": attempt["id"], "design_id": design["id"], "outcome": outcome,
                    "observations": strings(response.get("observations"), "observations"), "interpretations": interpretations,
                    "limitations": strings(response.get("limitations"), "limitations"), "next_questions": questions, "evidence_refs": refs}
        def save(p):
            p["analyses"].append(analysis)
            for question in questions:
                p["questions"].append({"id": new_id("question"), "text": question["text"], "parent_id": question["parent_question_id"], "status": "open", "history": [{"analysis_id": analysis["id"], "rationale": question["rationale"]}]})
            p.update(pending_call_id=None, phase="advance")
        self._change(program_id, save)

    def _advance(self, program_id):
        program = self.get(program_id)
        implementation = program["implementations"][-1]
        attempt = self.store.get_attempt(implementation["attempt_id"])
        design = next(d for d in program["designs"] if d["id"] == implementation["design_id"])
        repairs = sum(i["design_id"] == design["id"] for i in program["implementations"])
        if attempt["state"] == "failed" and repairs <= program["brief"]["max_repairs"]:
            self._patch(program_id, phase="build")
        elif program["brief"]["policy"] == "fixed_plan":
            if program["fixed_queue"]:
                self._select(program_id, program["fixed_queue"][0])
            else:
                self._finish(program_id, "completed", "Fixed experimental plan completed")
        else:
            self._patch(program_id, phase="decide")

    def work(self, program_id):
        import fcntl
        with (self.folder(program_id) / "coordinator.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return self.get(program_id)
            try:
                initial = self.get(program_id)
                pending = next((c for c in initial["calls"] if c["id"] == initial["pending_call_id"]), None)
                if pending and pending["status"] == "pending":
                    path = self.folder(program_id) / "calls" / (pending["id"] + ".json")
                    record = read_json(path, 16 * 1024**2) if path.exists() else {
                        "status": "interrupted", "error": "Coordinator interrupted during request; provider usage is unknown. Explicit resume is required.", "response": {}}
                    self._record_call(program_id, pending["id"], record)
                while True:
                    program = self.get(program_id)
                    if program["state"] in FINISHED | {"draft", "paused", "blocked"}:
                        return program
                    if program["state"] == "stopping" and not program["active_attempt_id"]:
                        return self._finish(program_id, "stopped", "Stopped by user")
                    if program["state"] == "pausing" and not program["active_attempt_id"]:
                        return self._patch(program_id, state="paused")
                    phase = program["phase"]
                    if phase == "sources":
                        cursor = program["source_cursor"]
                        if cursor < len(program["brief"]["sources"]):
                            self._source(program_id, url=program["brief"]["sources"][cursor])
                            self._patch(program_id, source_cursor=cursor + 1)
                        else:
                            self._patch(program_id, phase="decide")
                    elif phase == "decide":
                        self._decision(program_id)
                    elif phase == "select":
                        self._selection(program_id)
                    elif phase == "build":
                        self._build(program_id)
                    elif phase == "execute":
                        self._execute(program_id)
                    elif phase == "analyze":
                        self._analyze(program_id)
                    elif phase == "advance":
                        self._advance(program_id)
                    else:
                        raise LabError("Unknown durable research phase")
            except BudgetReached as exc:
                return self._finish(program_id, "completed", str(exc))
            except SchedulingInterrupted:
                current = self.get(program_id)
                if current["state"] == "stopping":
                    return self._finish(program_id, "stopped", "Stopped by user")
                return self._patch(program_id, state="paused")
            except Exception as exc:
                current = self.get(program_id)
                pending = next((c for c in current["calls"] if c["id"] == current["pending_call_id"]), None)
                if pending and pending["status"] == "completed":
                    def invalidate(p):
                        call = next(c for c in p["calls"] if c["id"] == pending["id"])
                        call.update(status="invalid", error=str(exc))
                    self._change(program_id, invalidate)
                if current["state"] == "pausing" and not current["active_attempt_id"]:
                    return self._patch(program_id, state="paused")
                # Retain an unconfirmed worker rather than claiming stop or releasing its lease.
                if current["state"] == "stopping" and current["active_attempt_id"]:
                    return self._patch(program_id, stop_reason=str(exc))
                return self._finish(program_id, "blocked", f"{type(exc).__name__}: {exc}")

    def report(self, program_id):
        program = self.get(program_id)
        lines = ["# Research report", "", program["brief"]["goal"], "", f"Program: `{program_id}`", f"State: **{program['state']}**", f"Reason: {program['stop_reason'] or 'Research remains active or unstarted.'}", "", "## Resource accounting", "", "```json", json.dumps(program["usage"], indent=2), "```", "", "Known tokens exclude calls whose provider usage was unavailable. Reserved runtime is charged when actual runtime is unknown. No monetary or energy total is inferred."]
        for item in program["implementations"]:
            lines.extend(["", f"## Implementation `{item['id']}`", f"Design: `{item['design_id']}`; status: {item['status']}"])
            if item.get("error"):
                lines.append("Recorded error: " + item["error"])
            if item.get("attempt_id"):
                attempt = next((a for a in self.store.list_attempts(item["experiment_id"]) if a["id"] == item["attempt_id"]), None)
                if attempt:
                    result = attempt.get("result") or {}
                    lines.extend([f"Attempt: `{attempt['id']}`; execution: **{attempt['state']}**", "", "### Recorded observations (not model interpretation)", "```json", json.dumps(result.get("observations"), indent=2), "```", "Artifacts:", *[f"- `{a['path']}` — SHA-256 `{a['sha256']}`" for a in result.get("artifacts", [])]])
        for analysis in program["analyses"]:
            lines.extend(["", f"## Model-authored analysis `{analysis['id']}`", f"Attempt: `{analysis['attempt_id']}`; assessment: {analysis['outcome']}", *["- " + claim["claim"] + " [" + ", ".join(claim["evidence_refs"]) + "]" for claim in analysis["interpretations"]], "Limitations:", *["- " + item for item in analysis["limitations"]]])
        lines.extend(["", "## Decisions", *[f"- `{d['id']}` ({d['action']}): {d['rationale']}" for d in program["decisions"]], "", "## Questions", *[f"- `{q['id']}` [{q['status']}] {q['text']} (parent: {q['parent_id']})" for q in program["questions"]], "", "## Sources", *[f"- `{s['id']}`: {s.get('title') or s.get('url') or s.get('query')} — {s.get('url') or ''}; retrieved {s.get('retrieved_at')}; {s.get('error') or 'retained source record'}" for s in program["sources"]], "", "This report exposes recorded evidence and model-authored interpretations. It is not independent confirmation of novelty or scientific validity.", ""])
        return "\n".join(lines)
