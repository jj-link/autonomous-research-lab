"""Local rootless Podman execution; container lifetime is independent of the CLI."""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
import threading
import uuid

from .models import TERMINAL_STATES, ExperimentSpec, LabError, canonical_json, normalize_gpu_uuid, read_json
from .store import Store

LABEL = "org.autonomous-research-lab.attempt"
ROOT_LABEL = "org.autonomous-research-lab.store"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        with pending.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def configure(store: Store, image: str, gpu_uuid: str | None = None, gpu_mode: str = "cdi") -> dict:
    if not image or image.startswith("-") or any(ch.isspace() for ch in image):
        raise LabError("image must be a local image name, digest, or ID")
    if gpu_uuid is not None:
        gpu_uuid = normalize_gpu_uuid(gpu_uuid)
    if gpu_mode not in {"cdi", "wsl"}:
        raise LabError("gpu_mode must be cdi or wsl")
    config = {"schema_version": 1, "image": image, "gpu_uuid": gpu_uuid, "gpu_mode": gpu_mode}
    atomic_json(store.root / "runner.json", config)
    return config


def configuration(store: Store) -> dict:
    path = store.root / "runner.json"
    if not path.is_file():
        raise LabError("Runner is not configured. Build Containerfile, then use arl configure --image IMAGE [--gpu UUID --gpu-mode wsl].")
    config = read_json(path)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise LabError("Unsupported runner configuration")
    return config


def podman(*args: str, timeout: int = 30) -> str:
    try:
        result = subprocess.run(["podman", *args], capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LabError(f"Podman {args[0] if args else ''} failed: {exc}") from exc
    if result.returncode:
        raise LabError(f"Podman {args[0]} failed ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout


def environment() -> dict:
    if platform.system() != "Linux":
        raise LabError("Execution requires Linux with local rootless Podman. On Windows, run this CLI inside WSL; record inspection also works on Windows.")
    if os.environ.get("CONTAINER_HOST") or os.environ.get("CONTAINER_CONNECTION"):
        raise LabError("Remote Podman connections are not supported: bundle paths must refer to this host")
    if not shutil.which("podman"):
        raise LabError("podman is not installed in this Linux environment")
    info = json.loads(podman("info", "--format=json"))
    host = info["host"]
    if not host.get("security", {}).get("rootless"):
        raise LabError("The runner requires rootless Podman, not a privileged/root daemon")
    if host.get("cgroupVersion") != "v2":
        raise LabError("The runner requires cgroup v2 resource accounting")
    return info


def gpu_inventory() -> list[dict]:
    executable = shutil.which("nvidia-smi")
    if executable is None and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    if executable is None:
        raise LabError("nvidia-smi is unavailable; allocated GPU capacity cannot be recorded")
    try:
        result = subprocess.run(
            [executable, "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        )
        return [{"uuid": normalize_gpu_uuid(row[0].strip()), "name": row[1].strip(),
                 "memory_bytes": int(row[2].strip()) * 1024**2}
                for row in csv.reader(result.stdout.splitlines())]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise LabError(f"Cannot read GPU inventory: {exc}") from exc


def doctor(store: Store) -> dict:
    info = environment()
    config = configuration(store) if (store.root / "runner.json").exists() else None
    try:
        inventory, inventory_error = gpu_inventory(), None
    except LabError as exc:
        inventory, inventory_error = [], str(exc)
    return {"root": str(store.root), "runner": config, "rootless": True,
            "controllers": info["host"].get("cgroupControllers", []),
            "logical_cpus": os.cpu_count(), "kernel": platform.release(),
            "host_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "gpus": inventory, "gpu_inventory_error": inventory_error,
            "gpu_note": "WSL GPU selection is not a security boundary against other adapters." if "microsoft" in platform.release().lower() else None}


def container_name(attempt_id: str) -> str:
    return "arl-" + attempt_id


def root_label(store: Store) -> str:
    return hashlib.sha256(str(store.root.resolve()).encode()).hexdigest()


def inspect_owned(store: Store, attempt_id: str) -> dict | None:
    name = store.get_attempt(attempt_id).get("container_id") or container_name(attempt_id)
    result = subprocess.run(["podman", "container", "inspect", name], capture_output=True, text=True, timeout=30)
    if result.returncode:
        if "no such" in result.stderr.lower() or "does not exist" in result.stderr.lower():
            return None
        raise LabError(f"Cannot inspect container {name}: {result.stderr.strip()}")
    record = json.loads(result.stdout)[0]
    labels = record.get("Config", {}).get("Labels") or {}
    if labels.get(LABEL) != attempt_id or labels.get(ROOT_LABEL) != root_label(store):
        raise LabError(f"Refusing to operate on container {name}: ownership labels do not match this store")
    return record


def terminate_owned(store: Store, attempt_id: str) -> dict | None:
    container = inspect_owned(store, attempt_id)
    if container and container["State"].get("Running"):
        try:
            podman("kill", "--signal=KILL", container["Id"])
        except LabError:
            # Exiting naturally between inspect and kill is not a termination failure.
            current = inspect_owned(store, attempt_id)
            if current and current["State"].get("Running"):
                raise
            return current
        return inspect_owned(store, attempt_id)
    return container


def prepare(store: Store, attempt: dict, spec: ExperimentSpec) -> dict:
    info = environment()
    controllers = set(info["host"].get("cgroupControllers", []))
    needed = {"memory", "pids"} | ({"cpu"} if spec.limits.cpus is not None else set())
    if not needed <= controllers:
        raise LabError(f"Requested resource limits cannot be enforced: missing delegated cgroup controllers {sorted(needed - controllers)}. No host settings were changed.")
    config = configuration(store)
    image = json.loads(podman("image", "inspect", config["image"]))[0]
    gpu_device = None
    isolation = None
    gpu_hardware = None
    if spec.gpu:
        gpu_uuid = attempt["gpu_uuid"]
        if not gpu_uuid:
            raise LabError("GPU experiment requires an allocated GPU UUID")
        gpu_hardware = next((gpu for gpu in gpu_inventory() if gpu["uuid"] == gpu_uuid), None)
        if gpu_hardware is None:
            raise LabError(f"Allocated GPU {gpu_uuid} is not present; no other GPU will be substituted")
        if config["gpu_mode"] == "wsl":
            if "microsoft" not in platform.release().lower():
                raise LabError("gpu_mode=wsl is only valid on WSL")
            if not Path("/dev/dxg").exists() or not all(Path(path).is_dir() for path in ("/usr/lib/wsl/lib", "/usr/lib/wsl/drivers")):
                raise LabError("WSL GPU device and driver library directories are not available")
            gpu_device = "/dev/dxg"
            isolation = "shared WSL /dev/dxg; CUDA device selection, NOT adapter security isolation"
        else:
            gpu_device = f"nvidia.com/gpu={gpu_uuid}"
            isolation = "device-specific NVIDIA CDI"
            try:
                cdi = subprocess.run(["nvidia-ctk", "cdi", "list"], capture_output=True, text=True, timeout=15, check=True)
            except (OSError, subprocess.SubprocessError) as exc:
                raise LabError(f"NVIDIA CDI inventory is unavailable: {exc}") from exc
            devices = set(re.findall(r"nvidia\.com/gpu=[^\s]+", cdi.stdout + "\n" + cdi.stderr))
            if gpu_device not in devices:
                raise LabError(f"Allocated CDI device {gpu_device} is not available; no other GPU will be substituted")
    return {"backend": "podman-rootless", "image_reference": config["image"], "image_id": image["Id"],
            "image_digests": image.get("RepoDigests", []), "gpu_uuid": attempt["gpu_uuid"],
            "gpu_device": gpu_device, "gpu_isolation": isolation, "limits": spec.limits.to_dict(),
            "driver_mounts": ["/usr/lib/wsl/lib", "/usr/lib/wsl/drivers"] if gpu_device == "/dev/dxg" else [],
            "hardware": {"logical_cpus": os.cpu_count(),
                         "host_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
                         "gpu": gpu_hardware},
            "cpu_quota_enforced": spec.limits.cpus is not None, "network": "none",
            "output_limit": "monitored total apparent file bytes; per-file RLIMIT_FSIZE; not a hard filesystem quota",
            "log_limit": "Podman bounded rotating log", "podman_version": info.get("version", {}),
            "termination_reason": None, "cpu_seconds_observed": None, "peak_memory_bytes_observed": None}


def create_container(store: Store, attempt: dict, spec: ExperimentSpec, execution: dict) -> str:
    bundle = store.bundle_dir(attempt["experiment_id"]).resolve()
    folder = store.attempt_dir(attempt["id"]).resolve()
    for path in (bundle, folder):
        if any(char in str(path) for char in ":,\n\r"):
            raise LabError("Execution paths cannot contain colon, comma, or newline characters")
    args = ["create", "--name", container_name(attempt["id"]), "--pull=never",
            "--label", f"{LABEL}={attempt['id']}", "--label", f"{ROOT_LABEL}={root_label(store)}",
            "--network=none", "--http-proxy=false", "--read-only", "--read-only-tmpfs=false",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--userns=keep-id",
            "--user", f"{os.getuid()}:{os.getgid()}", "--pid=private", "--ipc=private",
            "--init", "--no-healthcheck", "--systemd=false", "--restart=no", "--image-volume=ignore",
            "--memory", str(spec.limits.memory_bytes), "--memory-swap", str(spec.limits.memory_bytes),
            "--pids-limit", str(spec.limits.pids), "--timeout", str(spec.limits.wall_seconds),
            "--stop-timeout=1", "--ulimit", f"fsize={spec.limits.output_bytes}:{spec.limits.output_bytes}",
            "--log-driver=k8s-file", "--log-opt", f"path={folder / 'console.log'}",
            "--log-opt", f"max-size={spec.limits.log_bytes}", "--workdir=/workspace",
            "--volume", f"{bundle}:/experiment:ro", "--volume", f"{folder / 'output'}:/workspace:rw",
            "--env=ARL_OUTPUT_DIR=/workspace", "--env=ARL_SETTINGS=/experiment/settings.json",
            "--env=PYTHONDONTWRITEBYTECODE=1", "--env=PYTHONUNBUFFERED=1", "--env=HOME=/workspace",
            "--env=CUDA_CACHE_DISABLE=1", "--env=PYTHONPATH=/experiment/code", "--entrypoint=python3"]
    if spec.limits.cpus is not None:
        args += ["--cpus", str(spec.limits.cpus)]
    if execution["gpu_device"]:
        args += ["--device", execution["gpu_device"], "--env", f"CUDA_VISIBLE_DEVICES={attempt['gpu_uuid']}"]
        if execution["gpu_device"] == "/dev/dxg":
            # WSL's CDI hooks write linker caches/symlinks and fail on a read-only root.
            # Its standard driver layout can instead be mounted read-only without hooks.
            for path in execution["driver_mounts"]:
                args += ["--mount", f"type=bind,src={path},dst={path},ro"]
            args += ["--env=LD_LIBRARY_PATH=/usr/lib/wsl/lib"]
    args += [execution["image_id"], f"/experiment/code/{spec.entrypoint}", *spec.args]
    return podman(*args, timeout=60).strip()


@contextmanager
def attempt_lock(store: Store, attempt_id: str):
    if platform.system() != "Linux":
        raise LabError("Execution supervision requires Linux")
    import fcntl
    path = store.attempt_dir(attempt_id) / "supervisor.lock"
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def workspace_files(folder: Path, *, hash_contents: bool = False) -> tuple[int, list[dict], list[str]]:
    total, records, warnings = 0, [], []
    entries = 0
    # fwalk keeps directory descriptors, avoiding traversal through replaced symlinks.
    for directory, dirs, files, dirfd in os.fwalk(folder, follow_symlinks=False):
        for name in dirs + files:
            entries += 1
            if entries > 10000:
                raise LabError("Workspace exceeds the 10,000-entry metadata limit")
            try:
                meta = os.stat(name, dir_fd=dirfd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            relative = (Path(directory) / name).relative_to(folder).as_posix()
            if stat.S_ISDIR(meta.st_mode):
                continue
            if not stat.S_ISREG(meta.st_mode):
                warnings.append(f"Skipped non-regular output: {relative}")
                continue
            total += meta.st_size
            if hash_contents:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dirfd)
                with os.fdopen(fd, "rb") as stream:
                    current = os.fstat(stream.fileno())
                    if not stat.S_ISREG(current.st_mode):
                        warnings.append(f"Skipped changed output: {relative}")
                        continue
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                records.append({"path": relative, "bytes": current.st_size, "sha256": digest.hexdigest()})
    return total, sorted(records, key=lambda item: item["path"]), warnings


def sample_usage(container: dict, execution: dict) -> None:
    pid = container["State"].get("Pid", 0)
    if not pid:
        return
    try:
        line = next(line for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines() if line.startswith("0::"))
        cgroup = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
        cpu = dict(line.split() for line in (cgroup / "cpu.stat").read_text().splitlines())
        usage = int(cpu["usage_usec"]) / 1_000_000
        previous = execution.get("cpu_seconds_observed")
        execution["cpu_seconds_observed"] = max(usage, previous or 0)
        peak = int((cgroup / "memory.peak").read_text())
        execution["peak_memory_bytes_observed"] = max(peak, execution.get("peak_memory_bytes_observed") or 0)
    except (OSError, ValueError, KeyError, StopIteration):
        pass  # Missing cgroups are unknown usage, never a fabricated zero.


def timestamp(value: str | None) -> float | None:
    if not value or value.startswith("0001"):
        return None
    # Podman uses nanoseconds; Python 3.10's ISO parser accepts at most microseconds.
    normalized = re.sub(r"(\.\d{6})\d+", r"\1", value.replace("Z", "+00:00"))
    return datetime.fromisoformat(normalized).timestamp()


def finish(store: Store, attempt: dict, spec: ExperimentSpec, container: dict | None, execution: dict, forced: str | None = None, error: str | None = None) -> dict:
    state_info = container.get("State", {}) if container else {}
    started = timestamp(state_info.get("StartedAt"))
    ended = timestamp(state_info.get("FinishedAt"))
    elapsed = max(0.0, ended - started) if started is not None and ended is not None else None
    code = state_info.get("ExitCode") if container else None
    state = forced or execution.get("termination_reason")
    if state is None:
        if state_info.get("OOMKilled"):
            state, error = "limit_exceeded", "Container exceeded its memory allocation"
        elif elapsed is not None and elapsed >= spec.limits.wall_seconds - 0.05 and code != 0:
            state, error = "timed_out", "Container reached its wall-time allocation"
        elif attempt["stop_requested"]:
            state = "stopped"
        elif code == 0:
            state = "completed"
        else:
            state, error = "failed", error or f"Experiment exited with status {code}"
    output = store.attempt_dir(attempt["id"]) / "output"
    try:
        output_bytes, artifacts, warnings = workspace_files(output, hash_contents=True)
    except (OSError, LabError) as exc:
        output_bytes, artifacts, warnings = None, [], [str(exc)]
        if state == "completed":
            state, error = "failed", f"Cannot inventory experiment output: {exc}"
    if output_bytes is not None and output_bytes > spec.limits.output_bytes:
        state, error = "limit_exceeded", "Experiment output exceeded its monitored storage allocation"
    paths = {item["path"] for item in artifacts}
    missing = set(spec.outputs) - paths
    if state == "completed" and missing:
        state, error = "failed", f"Missing required outputs: {', '.join(sorted(missing))}"
    observations = None
    if "results.json" in paths:
        try:
            observations = read_json(output / "results.json")
            if not isinstance(observations, dict):
                raise LabError("results.json must be a JSON object")
            metrics = observations.get("metrics", {})
            if not isinstance(metrics, dict) or any(type(value) not in (int, float) for value in metrics.values()):
                raise LabError("results.json metrics must be finite JSON numbers")
        except LabError as exc:
            if state == "completed":
                state, error = "failed", str(exc)
            warnings.append(str(exc))
            observations = None
    result = {"schema_version": 1, "attempt_id": attempt["id"], "experiment_id": attempt["experiment_id"],
              "state": state, "exit_code": code, "error": error, "scientific_outcome": "not_assessed",
              "wall_seconds": elapsed, "output_bytes": output_bytes, "artifacts": artifacts,
              "artifact_warnings": warnings, "observations": observations, "execution": execution}
    atomic_json(store.attempt_dir(attempt["id"]) / "result.json", result)
    return store.update_attempt(attempt["id"], state=state, finished_at=now(), exit_code=code,
                                error=error, execution=execution, result=result)


def supervise(store: Store, attempt_id: str, *, recovery: bool = False) -> dict:
    with attempt_lock(store, attempt_id) as acquired:
        if not acquired:
            return store.get_attempt(attempt_id)
        attempt = store.get_attempt(attempt_id)
        if attempt["state"] in TERMINAL_STATES:
            return attempt
        experiment = store.get_experiment(attempt["experiment_id"])
        spec = ExperimentSpec.from_dict(experiment["manifest"], Path("/"))
        execution = attempt.get("execution") or {}
        container = None
        try:
            store.update_attempt(attempt_id, supervisor_pid=os.getpid())
            container = inspect_owned(store, attempt_id)
            if container is None:
                if attempt["stop_requested"]:
                    return finish(store, attempt, spec, None, execution, "stopped")
                if recovery or attempt["state"] != "queued":
                    return finish(store, attempt, spec, None, execution, "interrupted", "No owned container exists; recovery did not repeat the experiment")
                store.verify_experiment(attempt["experiment_id"])
                try:
                    store.update_attempt(attempt_id, state="starting", expected_states={"queued"})
                except LabError:
                    current = store.get_attempt(attempt_id)
                    if current["stop_requested"]:
                        return finish(store, current, spec, None, execution, "stopped")
                    raise
                execution = prepare(store, attempt, spec)
                store.update_attempt(attempt_id, execution=execution)
                container_id = create_container(store, attempt, spec, execution)
                store.update_attempt(attempt_id, container_id=container_id)
                container = inspect_owned(store, attempt_id)
            status = container["State"]["Status"]
            if status in {"created", "configured"}:
                attempt = store.get_attempt(attempt_id)
                if attempt["stop_requested"]:
                    return finish(store, attempt, spec, container, execution, "stopped")
                # A configured container has never executed. Starting it is safe on recovery.
                podman("start", container["Id"])
                container = inspect_owned(store, attempt_id)
            attempt = store.get_attempt(attempt_id)
            if attempt["state"] in {"queued", "starting"}:
                try:
                    if attempt["state"] == "queued":
                        store.update_attempt(attempt_id, state="starting", expected_states={"queued"})
                    # Short commands may already be finished when their start is observed.
                    store.update_attempt(attempt_id, state="running", expected_states={"starting"}, started_at=container["State"]["StartedAt"])
                except LabError:
                    if store.get_attempt(attempt_id)["state"] != "stopping":
                        raise
            if container["State"].get("Running"):
                store.add_event(attempt_id, "supervision_resumed" if recovery else "execution_started", {"container_id": container["Id"]})
            last_saved = 0.0
            while container["State"].get("Running"):
                attempt = store.get_attempt(attempt_id)
                reason = execution.get("termination_reason")
                if attempt["stop_requested"] and reason is None:
                    reason = "stopped"
                start = timestamp(container["State"].get("StartedAt"))
                if start is not None and time.time() - start >= spec.limits.wall_seconds and reason is None:
                    reason = "timed_out"
                try:
                    size, _, _ = workspace_files(store.attempt_dir(attempt_id) / "output")
                    if size > spec.limits.output_bytes:
                        reason = "limit_exceeded"
                except LabError:
                    reason = "limit_exceeded"
                sample_usage(container, execution)
                if reason:
                    execution["termination_reason"] = reason
                    store.update_attempt(attempt_id, execution=execution)
                    terminate_owned(store, attempt_id)
                if time.monotonic() - last_saved >= 1.0:
                    store.update_attempt(attempt_id, execution=execution)
                    last_saved = time.monotonic()
                time.sleep(0.2)
                container = inspect_owned(store, attempt_id)
                if container is None:
                    return finish(store, store.get_attempt(attempt_id), spec, None, execution, "interrupted", "Owned container disappeared during execution; not repeated")
            return finish(store, store.get_attempt(attempt_id), spec, container, execution)
        except Exception as exc:
            # Never release the GPU lease while a possibly-running container is unaccounted for.
            error = f"{type(exc).__name__}: {exc}"
            try:
                container = terminate_owned(store, attempt_id)
                if container and container["State"].get("Running"):
                    raise LabError("Container is still running after termination request")
            except Exception as stop_error:
                execution["termination_reason"] = "failed"
                store.update_attempt(attempt_id, state="stopping", error=f"{error}; termination unconfirmed: {stop_error}", execution=execution)
                store.add_event(attempt_id, "recovery_required", {"error": str(stop_error)})
                return store.get_attempt(attempt_id)
            current = store.get_attempt(attempt_id)
            if current["state"] in TERMINAL_STATES:
                return current
            return finish(store, current, spec, container, execution, "failed", error)


def launch_supervisor(store: Store, attempt_id: str, recovery: bool = False) -> int:
    command = [sys.executable, "-m", "autonomous_research_lab.supervisor", "--root", str(store.root), "--attempt", attempt_id]
    if recovery:
        command.append("--recover")
    with (store.attempt_dir(attempt_id) / "supervisor.log").open("ab") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    # A long-lived CLI/API process owns and reaps its children; CLI exit still leaves
    # the separate session running, at which point the OS becomes its reaper.
    threading.Thread(target=process.wait, name=f"arl-reap-{process.pid}", daemon=True).start()
    return process.pid


def run(store: Store, experiment_id: str, *, wait: bool = False) -> dict:
    environment()  # Don't create a queued lease when the CLI is running on an unsupported host.
    config = configuration(store)
    experiment = store.get_experiment(experiment_id)
    gpu_uuid = config.get("gpu_uuid") if experiment["manifest"]["gpu"] else None
    attempt = store.create_attempt(experiment_id, gpu_uuid=gpu_uuid)
    try:
        launch_supervisor(store, attempt["id"])
    except (OSError, LabError) as exc:
        store.update_attempt(attempt["id"], state="failed", error=str(exc), finished_at=now())
        raise LabError(f"Could not start supervisor: {exc}") from exc
    return wait_for_attempt(store, attempt["id"]) if wait else store.get_attempt(attempt["id"])


def wait_for_attempt(store: Store, attempt_id: str, timeout: float | None = None) -> dict:
    if timeout is None:
        attempt = store.get_attempt(attempt_id)
        experiment = store.get_experiment(attempt["experiment_id"])
        timeout = experiment["manifest"]["limits"]["wall_seconds"] + 90
    deadline = time.monotonic() + timeout
    while True:
        attempt = store.get_attempt(attempt_id)
        if attempt["state"] in TERMINAL_STATES:
            return attempt
        if time.monotonic() >= deadline:
            raise LabError(f"Wait timed out; attempt {attempt_id} remains recorded. Run arl recover to reconcile its owned container; do not resubmit it blindly.")
        time.sleep(0.2)


def stop(store: Store, attempt_id: str, *, wait: bool = False) -> dict:
    environment()
    attempt = store.request_stop(attempt_id)
    if attempt["state"] not in TERMINAL_STATES:
        launch_supervisor(store, attempt_id, recovery=True)
    return wait_for_attempt(store, attempt_id) if wait else attempt


def recover(store: Store, *, wait: bool = False) -> list[dict]:
    environment()
    results = []
    for attempt in store.list_attempts(active_only=True):
        launch_supervisor(store, attempt["id"], recovery=True)
        results.append(wait_for_attempt(store, attempt["id"]) if wait else store.get_attempt(attempt["id"]))
    return results
