"""Versioned, JSON-serializable experiment inputs. No execution occurs here."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any


class LabError(Exception):
    """An actionable input, storage, or execution failure."""


ACTIVE_STATES = frozenset({"queued", "starting", "running", "stopping"})
TERMINAL_STATES = frozenset({"completed", "failed", "stopped", "timed_out", "interrupted", "limit_exceeded"})
ALL_STATES = ACTIVE_STATES | TERMINAL_STATES


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise LabError(f"Value is not finite JSON: {exc}") from exc


def normalize_gpu_uuid(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
        raise LabError("GPU allocation must be a full GPU-UUID reported by nvidia-smi")
    return "GPU-" + value[4:].lower()


def read_json(path: Path, max_bytes: int = 1024 * 1024) -> Any:
    try:
        with path.open("rb") as stream:
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise LabError(f"JSON file exceeds {max_bytes} bytes: {path}")
        value = json.loads(data)
        canonical_json(value)
        return value
    except (OSError, UnicodeError, ValueError) as exc:
        raise LabError(f"Cannot read JSON {path}: {exc}") from exc


def relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise LabError(f"{label} must be a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")) or ":" in value:
        raise LabError(f"{label} must remain inside its bundle: {value!r}")
    return str(path)


def positive_int(value: Any, label: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise LabError(f"{label} must be an integer >= {minimum}")
    return value


def require_keys(value: Any, allowed: set[str], label: str) -> dict:
    if not isinstance(value, dict):
        raise LabError(f"{label} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise LabError(f"Unknown {label} fields: {', '.join(sorted(unknown))}")
    return value


@dataclass(frozen=True)
class Limits:
    wall_seconds: int = 300
    memory_bytes: int = 2 * 1024**3
    output_bytes: int = 64 * 1024**2
    log_bytes: int = 8 * 1024**2
    pids: int = 64
    cpus: float | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "Limits":
        require_keys(value, set(cls.__dataclass_fields__), "limits")
        result = cls(**value)
        for name in ("wall_seconds", "output_bytes", "log_bytes", "pids"):
            positive_int(getattr(result, name), f"limits.{name}")
        positive_int(result.memory_bytes, "limits.memory_bytes", 16 * 1024**2)
        if result.cpus is not None and (
            type(result.cpus) not in (int, float)
            or not math.isfinite(result.cpus)
            or result.cpus <= 0
        ):
            raise LabError("limits.cpus must be a finite positive number or null (no CPU quota)")
        return result

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    source: Path
    entrypoint: str
    question: str = ""
    procedure: str = ""
    args: tuple[str, ...] = ()
    settings: dict = field(default_factory=dict)
    outputs: tuple[str, ...] = ("results.json",)
    gpu: bool = False
    limits: Limits = field(default_factory=Limits)

    @classmethod
    def from_dict(cls, value: Any, base: Path) -> "ExperimentSpec":
        require_keys(value, {"schema_version", "name", "source", "entrypoint", "question", "procedure", "args", "settings", "outputs", "gpu", "limits"}, "experiment")
        if value.get("schema_version", 1) != 1:
            raise LabError("Unsupported experiment schema_version; expected 1")
        name = value.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", name):
            raise LabError("name must be 1–80 letters, digits, dots, underscores, or hyphens")
        source = value.get("source")
        if not isinstance(source, str) or not source or "\x00" in source:
            raise LabError("source must name a directory relative to the manifest, or an absolute directory")
        source_path = (base / source).absolute()
        entrypoint = relative_path(value.get("entrypoint"), "entrypoint")
        args = value.get("args", [])
        if not isinstance(args, list) or any(not isinstance(arg, str) or "\x00" in arg for arg in args):
            raise LabError("args must be an array of strings without NUL characters")
        settings = value.get("settings", {})
        if not isinstance(settings, dict):
            raise LabError("settings must be an object")
        canonical_json(settings)
        outputs = value.get("outputs", ["results.json"])
        if not isinstance(outputs, list):
            raise LabError("outputs must be an array of relative file paths")
        outputs = tuple(relative_path(path, "output") for path in outputs)
        if len(set(outputs)) != len(outputs):
            raise LabError("outputs must not contain duplicates")
        gpu = value.get("gpu", False)
        if type(gpu) is not bool:
            raise LabError("gpu must be true or false")
        for key in ("question", "procedure"):
            if not isinstance(value.get(key, ""), str):
                raise LabError(f"{key} must be a string")
        return cls(name, source_path, entrypoint, value.get("question", ""), value.get("procedure", ""), tuple(args), settings, outputs, gpu, Limits.from_dict(value.get("limits", {})))

    def to_dict(self) -> dict:
        return {"schema_version": 1, "name": self.name, "source": str(self.source), "entrypoint": self.entrypoint,
                "question": self.question, "procedure": self.procedure, "args": list(self.args),
                "settings": self.settings, "outputs": list(self.outputs), "gpu": self.gpu, "limits": self.limits.to_dict()}


def load_manifest(path: Path) -> ExperimentSpec:
    path = path.absolute()
    return ExperimentSpec.from_dict(read_json(path), path.parent)
