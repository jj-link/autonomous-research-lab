"""Bounded CPU job batches executed by the research coordinator, not worker code."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import shutil
import uuid

from . import execution
from .models import ExperimentSpec, LabError, TERMINAL_STATES, canonical_json, positive_int, read_json, relative_path, require_keys


def namespace(store):
    return store.root.parent / ".arl-cache" / "research-inputs" / hashlib.sha256(str(store.root).encode()).hexdigest()[:16]


def snapshot_files(store, folder, files, name, registered=None):
    """Copy only explicitly selected, hash-pinned files outside credential storage."""
    folder.mkdir(parents=True, exist_ok=True)
    if folder.resolve() != folder.absolute():
        raise LabError("Artifact snapshot must not traverse links")
    expected = {}
    for destination, (source_root, source_name, digest) in files.items():
        destination = relative_path(destination, "snapshot destination")
        source_name = relative_path(source_name, "snapshot source")
        source, target = source_root / source_name, folder / destination
        if source.is_symlink() or not source.resolve().is_relative_to(source_root.resolve()):
            raise LabError("Snapshot source escaped its approved directory")
        expected[destination] = digest
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.parent.resolve() != target.parent.absolute():
            raise LabError("Snapshot destination must not traverse links")
        if not target.exists():
            pending = target.with_name("." + target.name + ".pending")
            with source.open("rb") as incoming, pending.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            pending.replace(target)
    existing = (registered or {}).get(str(folder))
    if registered is None:
        existing = next((item for item in store.list_inputs() if item["source"] == str(folder)), None)
    record = store.verify_input(existing["id"]) if existing else store.register_input(folder, name, expected_hashes=expected)
    if {item["path"]: item["sha256"] for item in record["files"]} != expected:
        raise LabError("Snapshot bytes differ from the recorded source inventory")
    if registered is not None:
        registered[str(folder)] = record
    return record


def prepare(engine, program, implementation, spec, descriptor):
    require_keys(descriptor, {"input", "path", "parallelism", "keys"}, "batch")
    if spec.gpu:
        raise LabError("Isolated job batches are CPU-only; GPU stages execute separately")
    parallelism = positive_int(descriptor.get("parallelism", 1), "batch parallelism")
    if parallelism > program["brief"].get("max_parallel_workers", 1):
        raise LabError("Batch concurrency exceeds the approved worker allowance")
    if any(getattr(spec.limits, field) * parallelism > program["brief"]["resources"][field] for field in ("memory_bytes", "pids")):
        raise LabError("Concurrent batch workers exceed the program's memory or PID allocation")
    alias = descriptor.get("input")
    if alias not in spec.inputs:
        raise LabError("Batch index must be an input of the selected design")
    index_name = relative_path(descriptor.get("path"), "batch index")
    source_records = {key: engine.store.get_input(value) for key, value in spec.inputs.items()}
    index_record = engine.store.verify_input(spec.inputs[alias])
    if index_name not in {item["path"] for item in index_record["files"]}:
        raise LabError("Batch index is absent from its approved inventory")
    index = read_json(engine.store.input_source(index_record["id"]) / index_name, 16 * 1024**2)
    require_keys(index, {"jobs"}, "batch index")
    rows = index.get("jobs")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 2000:
        raise LabError("Batch index must contain 1–2000 jobs")
    selected = descriptor.get("keys")
    if selected is not None and (not isinstance(selected, list) or not selected or any(not isinstance(key, str) for key in selected) or len(set(selected)) != len(selected)):
        raise LabError("Batch repair keys must be a nonempty array of distinct job keys")
    keys = []
    for row in rows:
        require_keys(row, {"key", "settings", "inputs", "wall_seconds"}, "batch job")
        key = row.get("key")
        if not isinstance(key, str) or not key or len(key) > 200:
            raise LabError("Batch jobs require bounded nonempty keys")
        keys.append(key)
    if len(set(keys)) != len(keys) or (selected is not None and set(selected) - set(keys)):
        raise LabError("Batch keys must be unique and any repair selection must exist")
    if selected is not None:
        rows = [row for row in rows if row["key"] in set(selected)]
    if len(rows) + program["usage"]["attempts"] > program["brief"]["max_experiments"]:
        raise LabError("Batch exceeds the remaining execution-attempt budget")
    existing_inputs = {item["source"]: item for item in engine.store.list_inputs()}
    existing_experiments = {canonical_json(item["manifest"]): item for item in engine.store.list_experiments()}
    approved_files = {key: {item["path"]: item for item in record["files"]} for key, record in source_records.items()}
    jobs, copied, copied_bytes = [], set(), 0
    for row in rows:
        settings, projections = row.get("settings", {}), row.get("inputs")
        if not isinstance(settings, dict) or len(canonical_json(settings).encode()) > 64 * 1024:
            raise LabError("Job settings must be a bounded JSON object; mount large response files instead")
        if not isinstance(projections, dict) or not 1 <= len(projections) <= 16:
            raise LabError("Each job needs 1–16 explicitly projected input bindings")
        bindings = {}
        for mount, projection in projections.items():
            require_keys(projection, {"input", "files"}, "job input projection")
            origin = alias if projection.get("input") == "$batch" else projection.get("input")
            if origin not in spec.inputs:
                raise LabError("Job input projection is outside the selected design")
            mapping = projection.get("files")
            if not isinstance(mapping, dict) or not 1 <= len(mapping) <= 32:
                raise LabError("Job input projection must select 1–32 files")
            files = {}
            record = source_records[origin]
            source_root = engine.store.input_source(record["id"])
            for destination, source_name in mapping.items():
                relative_path(destination, "projected filename")
                relative_path(source_name, "approved projected source")
                if source_name not in approved_files[origin]:
                    raise LabError("Job selected a file absent from its approved input inventory")
                metadata = approved_files[origin][source_name]
                files[destination] = (source_root, source_name, metadata["sha256"])
            identity = hashlib.sha256(canonical_json({"input": record["id"], "files": mapping}).encode()).hexdigest()
            if identity not in copied:
                copied.add(identity)
                copied_bytes += sum(approved_files[origin][source_name]["bytes"] for source_name in mapping.values())
                if copied_bytes > program["brief"]["resources"]["output_bytes"]:
                    raise LabError("Projected batch inputs exceed the approved snapshot byte allowance")
            projected = snapshot_files(engine.store, namespace(engine.store) / program["id"] / "projections" / identity,
                                       files, "projection-" + identity[:32], existing_inputs)
            bindings[mount] = projected["id"]
        wall = positive_int(row.get("wall_seconds", spec.limits.wall_seconds), "job wall_seconds")
        if wall > spec.limits.wall_seconds:
            raise LabError("Job runtime exceeds the selected design's allowance")
        job_spec = replace(spec, name=spec.name[:40] + "-" + hashlib.sha256(row["key"].encode()).hexdigest()[:16],
                           settings={**spec.settings, **settings}, inputs=bindings, limits=replace(spec.limits, wall_seconds=wall))
        # Validate generated aliases through the same public manifest boundary.
        job_spec = ExperimentSpec.from_dict(job_spec.to_dict(), spec.source)
        signature = canonical_json(job_spec.to_dict())
        experiment = existing_experiments.get(signature)
        if experiment is None:
            experiment = engine.store.submit(job_spec)
            existing_experiments[signature] = experiment
        jobs.append({"id": "job_" + uuid.uuid4().hex, "key": row["key"], "experiment_id": experiment["id"],
                     "attempt_id": None, "status": "ready", "reserved_seconds": wall})
    _, private_ids = engine._available_inputs(program)
    return {"jobs": jobs, "parallelism": parallelism, "private": bool(private_ids.intersection(spec.inputs.values()))}


def _run_job(engine, program_id, job):
    known = {attempt["id"]: attempt for attempt in engine.store.list_attempts(job["experiment_id"])}
    attempt = known.get(job["attempt_id"])
    if attempt is None:
        attempt = engine.store.create_attempt(job["experiment_id"], attempt_id=job["attempt_id"])
    if attempt["state"] not in TERMINAL_STATES:
        if engine.get(program_id)["state"] == "stopping":
            engine.store.request_stop(attempt["id"])
        attempt = execution.supervise(engine.store, attempt["id"], recovery=attempt["state"] != "queued")
    if attempt["state"] not in TERMINAL_STATES:
        raise LabError("Batch worker termination is unconfirmed; its attempt and reservation were retained")
    measured = (attempt.get("result") or {}).get("wall_seconds")
    def collected(program):
        implementation = program["implementations"][-1]
        item = next(item for item in implementation["jobs"] if item["id"] == job["id"])
        if item["status"] == "collected":
            return
        item.update(status="collected", execution_state=attempt["state"], error=attempt.get("error"))
        program["usage"]["wall_seconds_charged"] += item["reserved_seconds"] if measured is None else measured
        program["usage"]["wall_seconds_estimated"] |= measured is None
        active = [identity for identity in program.get("active_attempt_ids", []) if identity != attempt["id"]]
        program.update(active_attempt_ids=active, active_attempt_id=active[0] if active else None)
    engine._change(program_id, collected)


def execute(engine, program_id):
    implementation = engine.get(program_id)["implementations"][-1]
    parallelism = implementation["batch"]["parallelism"]
    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures = {}
        while True:
            program = engine.get(program_id)
            implementation = program["implementations"][-1]
            candidates = sorted((job for job in implementation["jobs"] if job["status"] != "collected" and job["id"] not in futures.values()),
                                key=lambda job: job["attempt_id"] is None)
            for candidate in candidates:
                if len(futures) >= parallelism:
                    break
                if program["state"] != "running" and candidate["attempt_id"] is None:
                    break
                def reserve(current):
                    item = next(job for job in current["implementations"][-1]["jobs"] if job["id"] == candidate["id"])
                    if item["attempt_id"] is not None:
                        return
                    if current["state"] != "running":
                        return
                    outstanding = sum(job["reserved_seconds"] for job in current["implementations"][-1]["jobs"] if job["attempt_id"] and job["status"] != "collected")
                    if current["usage"]["attempts"] >= current["brief"]["max_experiments"] or current["usage"]["wall_seconds_charged"] + outstanding + item["reserved_seconds"] > current["brief"]["max_wall_seconds"]:
                        return
                    item.update(attempt_id="run_" + uuid.uuid4().hex, status="reserved")
                    current["usage"]["attempts"] += 1
                    active = [*current.get("active_attempt_ids", []), item["attempt_id"]]
                    current.update(active_attempt_ids=active, active_attempt_id=active[0])
                program = engine._change(program_id, reserve)
                job = next(item for item in program["implementations"][-1]["jobs"] if item["id"] == candidate["id"])
                if job["attempt_id"] is None:
                    break
                futures[pool.submit(_run_job, engine, program_id, job)] = job["id"]
            if not futures:
                break
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                del futures[future]
                future.result()
    program = engine.get(program_id)
    implementation = program["implementations"][-1]
    if any(job["status"] != "collected" for job in implementation["jobs"]):
        if program["state"] == "running":
            raise LabError("Remaining batch jobs do not fit the execution-attempt or runtime budget")
        return
    publish(engine, program, implementation)


def publish(engine, program, implementation):
    folder = namespace(engine.store) / program["id"] / implementation["id"]
    files, index = {}, []
    sums, completed = {}, 0
    errors = []
    for job in implementation["jobs"]:
        attempt = engine.store.get_attempt(job["attempt_id"])
        manifest = engine.store.get_experiment(job["experiment_id"])["manifest"]
        spec = ExperimentSpec.from_dict(manifest, engine.store.root)
        result = attempt.get("result") or {}
        prefix = "jobs/" + job["id"] + "/"
        artifacts = {}
        for artifact in result.get("artifacts", []):
            if spec.includes_output(artifact["path"]):
                path = prefix + artifact["path"]
                files[path] = (engine.store.attempt_dir(attempt["id"]) / "output", artifact["path"], artifact["sha256"])
                artifacts[artifact["path"]] = path
        observations = result.get("observations")
        index.append({"key": job["key"], "attempt_id": attempt["id"], "experiment_id": job["experiment_id"],
                      "state": attempt["state"], "settings": manifest["settings"], "observations": observations, "artifacts": artifacts})
        if attempt["state"] == "completed":
            completed += 1
            for key, value in (observations or {}).get("metrics", {}).items():
                if type(value) in (int, float):
                    sums[key] = sums.get(key, 0) + value
        elif len(errors) < 5:
            errors.append({"key": job["key"], "attempt_id": attempt["id"], "error": attempt.get("error"), "state": attempt["state"]})
    index_folder = engine.folder(program["id"]) / "implementations" / implementation["id"]
    execution.atomic_json(index_folder / "batch.json", {"jobs": index})
    data = (index_folder / "batch.json").read_bytes()
    files["batch.json"] = (index_folder, "batch.json", hashlib.sha256(data).hexdigest())
    record = snapshot_files(engine.store, folder, files, "batch-" + implementation["id"].split("_", 1)[1])
    summary = {"id": implementation["id"], "kind": "isolated_worker_batch", "state": "completed" if completed == len(index) else "failed",
               "result": {"observations": {"metrics": {"jobs_requested": len(index), "jobs_completed": completed,
                                                           "jobs_failed": len(index) - completed},
                                              "details": {"summed_worker_metrics": sums, "failed_examples": errors,
                                                          "complete_index": "batch.json", "parallelism": implementation["batch"]["parallelism"]}}}}
    published = {"alias": "output_" + implementation["id"].split("_", 1)[1], "id": record["id"], "private": implementation["batch"]["private"]}
    def saved(current):
        item = current["implementations"][-1]
        item.update(status="collected", execution_state=summary["state"], batch_record=summary, published_input=published)
        current.update(phase="analyze", active_attempt_id=None, active_attempt_ids=[])
    engine._change(program["id"], saved)
