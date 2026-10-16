"""Host-owned native checkpoint cache; immutable HDD approvals remain history.

Only the controller may release a cache, after its inference attempt is terminal.
A failed copy remains explicitly owned but cannot be admitted; release it using
its source input ID. Released state is retained for idempotent release calls.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys

from autonomous_research_lab.execution import atomic_json
from autonomous_research_lab.models import LabError, canonical_json, read_json, relative_path
from autonomous_research_lab.store import Store

RESERVE_BYTES = 8 * 1024**3
STATE_NAME = "cache-state.json"
DATA_NAME = "checkpoint"
NATIVE_FILESYSTEMS = {"ext2", "ext3", "ext4", "xfs", "btrfs"}


def _identity(path):
    meta = path.lstat()
    return [meta.st_dev, meta.st_ino]


def _directories(path):
    for part in (path, *path.parents):
        if not part.exists() and not part.is_symlink():
            raise LabError(f"Cache path parent is unavailable: {part}")
        meta = part.lstat()
        if not stat.S_ISDIR(meta.st_mode) or getattr(meta, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise LabError(f"Cache paths must be real directories without links: {part}")


def _native(path):
    if sys.platform != "linux":
        raise LabError("Checkpoint caches require native Linux storage")
    try:
        mounted = json.loads(subprocess.check_output(
            ["findmnt", "--target", str(path), "--json", "--output", "FSTYPE,TARGET,SOURCE"], text=True
        ))["filesystems"]
        if len(mounted) != 1 or mounted[0]["fstype"] not in NATIVE_FILESYSTEMS:
            raise LabError("Checkpoint cache is not on an approved native Linux filesystem")
        target = Path(mounted[0]["target"])
        if not target.is_absolute() or not path.is_relative_to(target):
            raise LabError("Cannot establish the checkpoint cache mount identity")
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        raise LabError(f"Cannot establish native checkpoint cache storage: {exc}") from exc


def _root(store, cache_root, source):
    root = Path(cache_root)
    if not root.is_absolute() or ".." in root.parts or root == root.parent:
        raise LabError("Cache root must be an absolute dedicated directory")
    _directories(root if root.exists() or root.is_symlink() else root.parent)
    root = root.absolute()
    protected = (source, store.root, store.db_path)
    if any(root.is_relative_to(path) or path.is_relative_to(root) for path in protected):
        raise LabError("Cache root must not overlap the original input or research store")
    _native(root if root.exists() else root.parent)
    return root


def _approved(store, input_id):
    record = store.get_input(input_id)  # Never rehash the slow HDD source.
    files = record["files"]
    seen = set()
    for item in files:
        name = relative_path(item["path"], "approved checkpoint file")
        if name in seen or type(item["bytes"]) is not int or item["bytes"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise LabError("Invalid pinned checkpoint inventory")
        seen.add(name)
    if not files or record["bytes"] != sum(item["bytes"] for item in files) or record["sha256"] != hashlib.sha256(canonical_json(files).encode("utf-8")).hexdigest():
        raise LabError("Invalid pinned checkpoint inventory digest")
    return record


def _inventory_matches(left, right):
    return all(left[key] == right[key] for key in ("files", "bytes", "sha256"))


def _load(store, root):
    marker = root / STATE_NAME
    if not marker.is_file() or marker.is_symlink() or marker.stat().st_nlink != 1:
        raise LabError("Existing cache has no regular ownership state; it will not be reused or deleted")
    state = read_json(marker, max_bytes=16 * 1024**2)
    if not isinstance(state, dict) or state.get("schema_version") != 1 or state.get("root") != str(root) or state.get("root_identity") != _identity(root) or state.get("store") != str(store.root) or state.get("database") != str(store.db_path):
        raise LabError("Cache ownership does not match this root and research store")
    if state.get("status") not in {"copying", "ready", "releasing", "released"} or not isinstance(state.get("owned"), dict):
        raise LabError("Invalid checkpoint cache state")
    if not isinstance(state.get("checkpoint"), dict) or set(state["checkpoint"]) != {"label", "repo_id", "revision"}:
        raise LabError("Cache checkpoint identity is unavailable")
    original = _approved(store, state.get("source_input_id"))
    if state.get("source") != original["source"] or state.get("inventory_sha256") != original["sha256"] or state.get("bytes") != original["bytes"]:
        raise LabError("Cache source identity does not match its pinned approval")
    _root(store, root, Path(original["source"]))
    allowed = {DATA_NAME: "directory"}
    for item in original["files"]:
        path = Path(DATA_NAME) / item["path"]
        allowed[path.as_posix()] = "file"
        for parent in path.parents:
            if parent != Path("."):
                allowed[parent.as_posix()] = "directory"
    for name, owner in state["owned"].items():
        relative_path(name, "owned cache path")
        if not isinstance(owner, dict) or owner.get("kind") != allowed.get(name) or not isinstance(owner.get("identity"), list) or len(owner["identity"]) != 2:
            raise LabError("Cache contains an invalid ownership inventory")
    actual = {}
    def walk(directory):
        for path in directory.iterdir():
            if directory == root and path.name == STATE_NAME:
                continue
            meta = path.lstat()
            if meta.st_dev != root.stat().st_dev or stat.S_ISLNK(meta.st_mode) or getattr(meta, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                raise LabError(f"Cache contains a link or foreign mount: {path}")
            kind = "directory" if stat.S_ISDIR(meta.st_mode) else "file"
            if kind == "file" and (not stat.S_ISREG(meta.st_mode) or meta.st_nlink != 1):
                raise LabError(f"Cache contains a non-owned regular file: {path}")
            name = path.relative_to(root).as_posix()
            owner = {"kind": kind, "identity": [meta.st_dev, meta.st_ino]}
            if state["owned"].get(name) != owner:
                raise LabError(f"Cache contains unowned or replaced content: {path}")
            actual[name] = owner
            if kind == "directory":
                walk(path)
    walk(root)
    if actual != state["owned"]:
        raise LabError("Cache ownership inventory is incomplete")
    if state["status"] == "released" and actual:
        raise LabError("Released cache still contains checkpoint content")
    if state["status"] == "ready":
        if set(actual) != set(allowed):
            raise LabError("Completed checkpoint cache is incomplete")
        cached = store.get_input(state.get("cached_input_id"))
        if cached["source"] != str(root / DATA_NAME) or not _inventory_matches(cached, original):
            raise LabError("Cached input approval does not match its pinned source")
    return state, original


def _required_space(root, original):
    # Include block rounding and metadata, not just the logical weights size.
    unit = os.statvfs(root).f_frsize if hasattr(os, "statvfs") else 4096
    unit = max(unit, 4096)
    directories = {str(parent) for item in original["files"] for parent in Path(item["path"]).parents}
    return sum(((item["bytes"] + unit - 1) // unit) * unit for item in original["files"]) + (len(directories) + 2) * unit + 1024**2


def _capacity(path, required):
    if shutil.disk_usage(path).free < required + RESERVE_BYTES:
        raise LabError("Insufficient native cache capacity: the checkpoint must leave at least 8 GiB free")


def cache_checkpoint(store, checkpoint, source_input_id, cache_root):
    """Copy pinned inventory once and return a normal, execution-verified Store input.

    A ready cache is reverified on admission. A copying cache is never resumed;
    explicit release by the controller is required before staging another input.
    """
    original = _approved(store, source_input_id)
    source = store.input_source(source_input_id)
    root = _root(store, cache_root, source)
    checkpoint_identity = {key: checkpoint[key] for key in ("label", "repo_id", "revision")}
    if root.exists():
        state, prior = _load(store, root)
        if state["status"] != "released":
            if state["status"] != "ready" or state["source_input_id"] != source_input_id or state["checkpoint"] != checkpoint_identity or not _inventory_matches(prior, original):
                raise LabError("A different, stale or partial checkpoint cache requires explicit release")
            _capacity(root, 0)
            return store.verify_input(state["cached_input_id"])
    _capacity(root if root.exists() else root.parent, _required_space(root if root.exists() else root.parent, original))
    if not root.exists():
        root.mkdir(mode=0o700)
    state = {"schema_version": 1, "root": str(root), "root_identity": _identity(root),
             "store": str(store.root), "database": str(store.db_path),
             "source_input_id": source_input_id, "source": original["source"],
             "inventory_sha256": original["sha256"], "bytes": original["bytes"],
             "checkpoint": checkpoint_identity, "status": "copying", "cached_input_id": None, "owned": {}}
    atomic_json(root / STATE_NAME, state)
    def own(path, kind):
        state["owned"][path.relative_to(root).as_posix()] = {"kind": kind, "identity": _identity(path)}
        atomic_json(root / STATE_NAME, state)
    data = root / DATA_NAME
    data.mkdir()
    own(data, "directory")
    for item in original["files"]:
        incoming_path = source / item["path"]
        _directories(incoming_path.parent)
        before = incoming_path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size != item["bytes"]:
            raise LabError(f"Approved checkpoint file is missing, linked or changed: {item['path']}")
        outgoing_path = data / item["path"]
        for directory in reversed(outgoing_path.parent.parents):
            if directory.is_relative_to(data) and not directory.exists():
                directory.mkdir()
                own(directory, "directory")
        if not outgoing_path.parent.exists():
            outgoing_path.parent.mkdir()
            own(outgoing_path.parent, "directory")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        with os.fdopen(os.open(incoming_path, flags), "rb") as incoming, outgoing_path.open("xb") as outgoing:
            own(outgoing_path, "file")
            opened = os.fstat(incoming.fileno())
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(opened, field) != getattr(before, field) for field in fields):
                raise LabError(f"Approved checkpoint changed before copying: {item['path']}")
            remaining = item["bytes"]
            while remaining:
                chunk = incoming.read(min(8 * 1024**2, remaining))
                if not chunk:
                    raise LabError(f"Approved checkpoint was truncated while copying: {item['path']}")
                _capacity(root, ((len(chunk) + 4095) // 4096) * 4096)
                outgoing.write(chunk)
                remaining -= len(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
            after = os.fstat(incoming.fileno())
        current = incoming_path.lstat()
        if any(getattr(opened, field) != getattr(after, field) or getattr(after, field) != getattr(current, field) for field in fields):
            raise LabError(f"Approved checkpoint changed while copying: {item['path']}")
    _load(store, root)  # Reject unexpected content before granting an input approval.
    cached = store.register_input(data, original["name"][:76] + "-ssd",
                                  expected_hashes={item["path"]: item["sha256"] for item in original["files"]})
    if not _inventory_matches(cached, original):
        raise LabError("Cached checkpoint inventory differs from its pinned approval")
    state.update(status="ready", cached_input_id=cached["id"])
    atomic_json(root / STATE_NAME, state)
    return cached


def release_checkpoint_cache(store, input_id, cache_root):
    """Delete only inode-identified owned cache content; retain all Store history.

    Accept the cached input ID, or the original input ID after a failed copy.
    Releasing the same retained released receipt is idempotent. The caller must
    establish that no inference is using this cache before invoking this function.
    """
    supplied = store.get_input(input_id)
    root = Path(cache_root)
    if not root.is_absolute() or ".." in root.parts or not root.exists():
        raise LabError("Cache release requires an existing absolute owned root")
    _directories(root)
    state, original = _load(store, root)
    if input_id not in {state["source_input_id"], state["cached_input_id"]}:
        raise LabError("Release input does not own the active checkpoint cache")
    if input_id == state["cached_input_id"] and (supplied["source"] != str(root / DATA_NAME) or not _inventory_matches(supplied, original)):
        raise LabError("Release input approval does not match this checkpoint cache")
    if state["status"] == "released":
        return
    state["status"] = "releasing"
    atomic_json(root / STATE_NAME, state)
    # No recursive deletion: each child was checked against its recorded inode.
    for name in sorted(state["owned"], key=lambda value: (value.count("/"), value), reverse=True):
        owner = state["owned"][name]
        path = root / name
        if _identity(path) != owner["identity"]:
            raise LabError(f"Owned cache changed before release: {path}")
        if owner["kind"] == "directory":
            path.rmdir()
        else:
            path.unlink()
        del state["owned"][name]
        atomic_json(root / STATE_NAME, state)
    state["status"] = "released"
    atomic_json(root / STATE_NAME, state)


def cached_input_id(cache_root):
    """Return the active registered cache ID without admitting it for execution.

    Missing roots and owned copying/releasing/released roots have no active ID.
    Existing unrelated or inconsistent roots fail closed. Full byte verification
    remains mandatory in cache_checkpoint and the ordinary execution admission.
    """
    root = Path(cache_root)
    if not root.is_absolute() or ".." in root.parts or root == root.parent:
        raise LabError("Cache inspection requires an absolute dedicated root")
    _directories(root if root.exists() or root.is_symlink() else root.parent)
    _native(root if root.exists() else root.parent)
    if not root.exists():
        return None
    marker = root / STATE_NAME
    if not marker.is_file() or marker.is_symlink() or marker.stat().st_nlink != 1:
        raise LabError("Existing cache has no regular ownership state")
    state = read_json(marker, max_bytes=16 * 1024**2)
    if not isinstance(state, dict) or state.get("schema_version") != 1 or state.get("root") != str(root) or state.get("root_identity") != _identity(root):
        raise LabError("Cache inspection ownership does not match this root")
    if not isinstance(state.get("store"), str) or not isinstance(state.get("database"), str):
        raise LabError("Cache inspection requires its original research store")
    store_root = Path(state["store"])
    database = Path(state["database"])
    if not store_root.is_absolute() or not database.is_absolute():
        raise LabError("Cache inspection requires absolute research store paths")
    _directories(store_root)
    _directories(database.parent)
    if database.is_symlink() or not database.is_file() or (store_root / "research.sqlite3").resolve() != database:
        raise LabError("Cache inspection research history is unavailable or changed")
    store = Store(store_root)
    state, _ = _load(store, root)
    return state["cached_input_id"] if state["status"] == "ready" else None
