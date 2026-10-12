"""Immutable input bundles and transactional execution history."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import uuid

from .models import ACTIVE_STATES, ALL_STATES, TERMINAL_STATES, ExperimentSpec, LabError, canonical_json, input_bindings, normalize_gpu_uuid

MAX_BUNDLE_BYTES = 256 * 1024**2
TRANSITIONS = {
    "queued": {"starting", "stopping", "failed", "stopped", "interrupted", "timed_out", "limit_exceeded"},
    "starting": {"running", "stopping", "failed", "stopped", "interrupted", "timed_out", "limit_exceeded"},
    "running": {"stopping"} | TERMINAL_STATES,
    "stopping": TERMINAL_STATES,
}
MUTABLE = {"started_at", "finished_at", "container_id", "supervisor_pid", "stop_requested", "error", "exit_code", "execution", "result"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def identifier(value, prefix):
    if not isinstance(value, str) or not re.fullmatch(prefix + r"_[0-9a-f]{32}", value):
        raise LabError(f"Invalid {prefix} identifier")
    return value


def _regular(meta):
    return stat.S_ISREG(meta.st_mode) and not (getattr(meta, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _tree(folder):
    entries = 0
    for directory, dirs, files in os.walk(folder, followlinks=False):
        dirs.sort()
        files.sort()
        for name in dirs + files:
            path = Path(directory) / name
            meta = path.lstat()
            entries += 1
            if entries > 10000:
                raise LabError("Bundle exceeds 10,000 filesystem entries")
            if stat.S_ISLNK(meta.st_mode) or (getattr(meta, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
                raise LabError(f"Bundle must not contain links: {path}")
            if stat.S_ISDIR(meta.st_mode):
                continue
            if not _regular(meta):
                raise LabError(f"Bundle must contain only regular files: {path}")
            yield path, meta


def _digest(folder):
    digest = hashlib.sha256()
    total = 0
    for path, meta in _tree(folder):
        relative = path.relative_to(folder).as_posix().encode("utf-8")
        total += meta.st_size
        if total > MAX_BUNDLE_BYTES:
            raise LabError("Bundle exceeds the 256 MiB snapshot limit")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(meta.st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _input_directory(path):
    path = Path(path).absolute()
    try:
        for part in (path, *path.parents):
            meta = part.lstat()
            if not stat.S_ISDIR(meta.st_mode) or (getattr(meta, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
                raise LabError(f"Input directory and its parents must be real directories, not links: {part}")
    except OSError as exc:
        raise LabError(f"Input directory is unavailable: {path}: {exc}") from exc
    return path.resolve()


def _input_inventory(path):
    files = []
    try:
        for file, original in _tree(_input_directory(path)):
            digest = hashlib.sha256()
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            with os.fdopen(os.open(file, flags), "rb") as stream:
                before = os.fstat(stream.fileno())
                if not _regular(before) or (before.st_dev, before.st_ino) != (original.st_dev, original.st_ino):
                    raise LabError(f"Input changed while hashing: {file}")
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                after = os.fstat(stream.fileno())
            current = file.lstat()
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, field) != getattr(after, field) or getattr(after, field) != getattr(current, field) for field in fields):
                raise LabError(f"Input changed while hashing: {file}")
            files.append({"path": file.relative_to(path).as_posix(), "bytes": before.st_size, "sha256": digest.hexdigest()})
    except OSError as exc:
        raise LabError(f"Cannot inventory approved input {path}: {exc}") from exc
    if not files:
        raise LabError("An approved input must contain at least one regular file")
    return {"files": files, "bytes": sum(item["bytes"] for item in files),
            "sha256": hashlib.sha256(canonical_json(files).encode("utf-8")).hexdigest()}


class Store:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "research.sqlite3"
        with self._db() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise LabError(f"Unsupported record schema version: {version}")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    manifest TEXT NOT NULL, bundle_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS approved_inputs (
                    id TEXT PRIMARY KEY, record TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id),
                    state TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
                    container_id TEXT, supervisor_pid INTEGER, gpu_uuid TEXT,
                    stop_requested INTEGER NOT NULL DEFAULT 0, error TEXT, exit_code INTEGER,
                    execution TEXT, result TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS active_gpu_lease ON attempts(gpu_uuid)
                    WHERE gpu_uuid IS NOT NULL AND state IN ('queued','starting','running','stopping');
                CREATE INDEX IF NOT EXISTS attempt_experiment ON attempts(experiment_id,created_at);
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    attempt_id TEXT NOT NULL REFERENCES attempts(id),
                    timestamp TEXT NOT NULL, kind TEXT NOT NULL, details TEXT NOT NULL
                );
                PRAGMA user_version=1;
            """)

    @contextmanager
    def _db(self, write=False):
        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _experiment(row):
        if row is None:
            raise LabError("Experiment does not exist")
        result = dict(row)
        result["manifest"] = json.loads(result["manifest"])
        return result

    @staticmethod
    def _attempt(row):
        if row is None:
            raise LabError("Execution attempt does not exist")
        result = dict(row)
        result["stop_requested"] = bool(result["stop_requested"])
        for key in ("execution", "result"):
            result[key] = json.loads(result[key]) if result[key] is not None else None
        return result

    @staticmethod
    def _event(connection, attempt_id, kind, details):
        connection.execute("INSERT INTO events(attempt_id,timestamp,kind,details) VALUES (?,?,?,?)",
                           (attempt_id, utc_now(), kind, canonical_json(details)))

    def bundle_dir(self, experiment_id):
        return self.root / "experiments" / identifier(experiment_id, "exp") / "bundle"

    def attempt_dir(self, attempt_id):
        return self.root / "runs" / identifier(attempt_id, "run")

    def register_input(self, path, name, *, expected_hashes=None):
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", name):
            raise LabError("Input name must be a 1–80 character slug")
        path = _input_directory(path)
        if self.root.is_relative_to(path) or path.is_relative_to(self.root):
            raise LabError("Approved inputs must be separate from the research store and its credentials")
        record = {"id": "input_" + uuid.uuid4().hex, "name": name, "source": str(path),
                  "registered_at": utc_now(), **_input_inventory(path)}
        observed = {item["path"]: item["sha256"] for item in record["files"]}
        if expected_hashes is not None and any(observed.get(name) != digest for name, digest in expected_hashes.items()):
            raise LabError("Input files do not match their expected SHA-256 hashes; approval was not recorded")
        with self._db(write=True) as connection:
            connection.execute("INSERT INTO approved_inputs VALUES (?,?)", (record["id"], canonical_json(record)))
        return record

    def get_input(self, input_id):
        identifier(input_id, "input")
        with self._db() as connection:
            row = connection.execute("SELECT record FROM approved_inputs WHERE id=?", (input_id,)).fetchone()
        if row is None:
            raise LabError(f"Input is not registered in this store: {input_id}")
        return json.loads(row[0])

    def list_inputs(self):
        with self._db() as connection:
            return [json.loads(row[0]) for row in connection.execute("SELECT record FROM approved_inputs ORDER BY id")]

    def verify_input(self, input_id):
        record = self.get_input(input_id)
        actual = _input_inventory(Path(record["source"]))
        if any(actual[key] != record[key] for key in ("files", "bytes", "sha256")):
            raise LabError(f"Approved input changed: {input_id}; register a new input rather than reusing this record")
        return record

    def input_source(self, input_id):
        return _input_directory(self.get_input(input_id)["source"])

    def resolve_inputs(self, bindings, *, verify=False):
        result = {}
        for alias, input_id in input_bindings(bindings).items():
            record = self.verify_input(input_id) if verify else self.get_input(input_id)
            result[alias] = {key: record[key] for key in ("id", "name", "sha256", "bytes")}
            result[alias].update(path=f"/inputs/{alias}", file_count=len(record["files"]))
        return result

    def submit(self, spec: ExperimentSpec):
        self.resolve_inputs(spec.inputs)
        if spec.source.is_symlink() or not spec.source.is_dir():
            raise LabError("source must be a real directory, not a symbolic link")
        source = spec.source.resolve()
        if self.root.is_relative_to(source):
            raise LabError("Source directory must not contain the research store; use a dedicated code directory")
        experiment_id = "exp_" + uuid.uuid4().hex
        bundle = self.bundle_dir(experiment_id)
        code = bundle / "code"
        code.mkdir(parents=True)
        try:
            total = 0
            for path, original in _tree(source):
                relative = path.relative_to(source)
                target = code / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                fd = os.open(path, flags)
                with os.fdopen(fd, "rb") as incoming, target.open("xb") as outgoing:
                    before = os.fstat(incoming.fileno())
                    if not _regular(before) or (before.st_dev, before.st_ino) != (original.st_dev, original.st_ino):
                        raise LabError(f"Source changed while snapshotting: {path}")
                    for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                        total += len(chunk)
                        if total > MAX_BUNDLE_BYTES:
                            raise LabError("Source exceeds the 256 MiB snapshot limit")
                        outgoing.write(chunk)
                    after = os.fstat(incoming.fileno())
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise LabError(f"Source changed while snapshotting: {path}")
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
            if not (code / spec.entrypoint).is_file():
                raise LabError(f"Entrypoint is missing from source: {spec.entrypoint}")
            manifest = spec.to_dict()
            for name, value in (("settings.json", spec.settings), ("manifest.json", manifest)):
                with (bundle / name).open("x", encoding="utf-8") as stream:
                    stream.write(canonical_json(value) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            digest = _digest(bundle)
            with self._db(write=True) as connection:
                connection.execute("INSERT INTO experiments VALUES (?,?,?,?)",
                                   (experiment_id, utc_now(), canonical_json(manifest), digest))
        except BaseException:
            shutil.rmtree(bundle.parent)
            raise
        return self.get_experiment(experiment_id)

    def verify_experiment(self, experiment_id):
        experiment = self.get_experiment(experiment_id)
        if _digest(self.bundle_dir(experiment_id)) != experiment["bundle_hash"]:
            raise LabError(f"Experiment snapshot was modified: {experiment_id}")
        return experiment

    def get_experiment(self, experiment_id):
        identifier(experiment_id, "exp")
        with self._db() as connection:
            return self._experiment(connection.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone())

    def list_experiments(self):
        with self._db() as connection:
            return [self._experiment(row) for row in connection.execute("SELECT * FROM experiments ORDER BY created_at,id")]

    def create_attempt(self, experiment_id, gpu_uuid=None, *, attempt_id=None):
        experiment = self.get_experiment(experiment_id)
        if experiment["manifest"]["gpu"]:
            if gpu_uuid is None:
                raise LabError("This experiment requires an allocated GPU UUID")
            gpu_uuid = normalize_gpu_uuid(gpu_uuid)
        elif gpu_uuid is not None:
            raise LabError("A CPU-only experiment cannot reserve a GPU")
        attempt_id = identifier(attempt_id or "run_" + uuid.uuid4().hex, "run")
        folder = self.attempt_dir(attempt_id)
        created_folder = False
        try:
            with self._db(write=True) as connection:
                if connection.execute("SELECT 1 FROM attempts WHERE id=?", (attempt_id,)).fetchone():
                    raise LabError("Attempt identifier already exists; inspect or recover it instead of repeating it")
                connection.execute("INSERT INTO attempts(id,experiment_id,state,created_at,gpu_uuid) VALUES (?,?,'queued',?,?)",
                                   (attempt_id, experiment_id, utc_now(), gpu_uuid))
                if folder.exists():
                    # Only an empty pre-commit directory can be adopted after a crash.
                    entries = list(folder.iterdir())
                    if entries and not (len(entries) == 1 and entries[0].name == "output" and entries[0].is_dir() and not entries[0].is_symlink() and not any(entries[0].iterdir())):
                        raise LabError("Unrecorded attempt files already exist; refusing to overwrite them")
                else:
                    folder.mkdir(parents=True)
                    created_folder = True
                (folder / "output").mkdir(exist_ok=True)
                self._event(connection, attempt_id, "attempt_created", {"experiment_id": experiment_id, "gpu_uuid": gpu_uuid})
        except sqlite3.IntegrityError as exc:
            if created_folder:
                shutil.rmtree(folder)
            raise LabError("Allocated GPU already has an active attempt in this store") from exc
        except BaseException:
            if created_folder:
                shutil.rmtree(folder)
            raise
        return self.get_attempt(attempt_id)

    def get_attempt(self, attempt_id):
        identifier(attempt_id, "run")
        with self._db() as connection:
            return self._attempt(connection.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone())

    def list_attempts(self, experiment_id=None, active_only=False):
        clauses, values = [], []
        if experiment_id is not None:
            identifier(experiment_id, "exp")
            clauses.append("experiment_id=?")
            values.append(experiment_id)
        if active_only:
            clauses.append("state IN ('queued','starting','running','stopping')")
        query = "SELECT * FROM attempts" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY created_at,id"
        with self._db() as connection:
            return [self._attempt(row) for row in connection.execute(query, values)]

    def update_attempt(self, attempt_id, *, state=None, expected_states=None, **fields):
        identifier(attempt_id, "run")
        if set(fields) - MUTABLE:
            raise LabError(f"Immutable or unknown attempt fields: {sorted(set(fields) - MUTABLE)}")
        if state is not None and state not in ALL_STATES:
            raise LabError("Unknown execution state")
        with self._db(write=True) as connection:
            current = self._attempt(connection.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone())
            if expected_states is not None and current["state"] not in expected_states:
                raise LabError(f"Attempt state changed concurrently: {current['state']}")
            if current["state"] in TERMINAL_STATES and (state is not None or fields):
                raise LabError("Completed attempt records are immutable; create a new attempt explicitly")
            if state is not None and state != current["state"] and state not in TRANSITIONS[current["state"]]:
                raise LabError(f"Illegal attempt transition: {current['state']} -> {state}")
            if current["stop_requested"] and fields.get("stop_requested") is False:
                raise LabError("A stop request cannot be withdrawn")
            changes = dict(fields)
            if state is not None:
                changes["state"] = state
                if state in TERMINAL_STATES:
                    changes.setdefault("finished_at", utc_now())
            for key in ("execution", "result"):
                if key in changes and changes[key] is not None:
                    if not isinstance(changes[key], dict):
                        raise LabError(f"{key} must be an object or null")
                    changes[key] = canonical_json(changes[key])
            if "stop_requested" in changes:
                if type(changes["stop_requested"]) is not bool:
                    raise LabError("stop_requested must be boolean")
                changes["stop_requested"] = int(changes["stop_requested"])
            if changes:
                setters = ",".join(f"{key}=?" for key in changes)
                connection.execute(f"UPDATE attempts SET {setters} WHERE id=?", (*changes.values(), attempt_id))
            if state is not None and state != current["state"]:
                self._event(connection, attempt_id, "state_changed", {"from": current["state"], "to": state})
            return self._attempt(connection.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone())

    def request_stop(self, attempt_id):
        identifier(attempt_id, "run")
        with self._db(write=True) as connection:
            current = self._attempt(connection.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone())
            if current["state"] in TERMINAL_STATES or current["stop_requested"]:
                return current
            connection.execute("UPDATE attempts SET stop_requested=1,state='stopping' WHERE id=?", (attempt_id,))
            self._event(connection, attempt_id, "stop_requested", {"from": current["state"]})
            return self._attempt(connection.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone())

    def add_event(self, attempt_id, kind, details):
        identifier(attempt_id, "run")
        with self._db(write=True) as connection:
            if not connection.execute("SELECT 1 FROM attempts WHERE id=?", (attempt_id,)).fetchone():
                raise LabError("Execution attempt does not exist")
            self._event(connection, attempt_id, kind, details)

    def events(self, attempt_id):
        self.get_attempt(attempt_id)
        with self._db() as connection:
            return [{**dict(row), "details": json.loads(row["details"])} for row in connection.execute("SELECT * FROM events WHERE attempt_id=? ORDER BY sequence", (attempt_id,))]
