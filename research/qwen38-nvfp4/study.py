"""Execute frozen compatibility checks in the existing lab; never run primary items."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from autonomous_research_lab.execution import atomic_json, configuration, configure, run, wait_for_attempt
from autonomous_research_lab.models import ExperimentSpec, LabError, Limits, read_json
from autonomous_research_lab.store import Store
from stage_tasks import DATA, file_hash, task_filename
from worker.grade import lcb_test_count, lcb_wall_seconds

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
STATE = "studies/qwen38-nvfp4"


def execute_once(store, record_path, spec):
    """Resume the recorded immutable attempt; never silently resubmit a failure."""
    if record_path.exists():
        record = read_json(record_path)
        experiment = store.get_experiment(record["experiment_id"])
        if experiment["manifest"] != spec.to_dict():
            raise LabError("Recorded experiment differs from the requested conditions; preserve it and record a protocol amendment")
    else:
        experiment = store.submit(spec)
        record = {"experiment_id": experiment["id"], "bundle_hash": experiment["bundle_hash"]}
        atomic_json(record_path, record)
    attempts = store.list_attempts(experiment["id"])
    if len(attempts) > 1:
        raise LabError("Multiple attempts exist for this study job; an explicit reconciliation is required")
    attempt = attempts[0] if attempts else run(store, experiment["id"])
    record["attempt_id"] = attempt["id"]
    atomic_json(record_path, record)
    print(json.dumps({"record": str(record_path), "attempt_id": attempt["id"], "state": attempt["state"]}), flush=True)
    attempt = wait_for_attempt(store, attempt["id"], timeout=spec.limits.wall_seconds + 7200)
    record.update(state=attempt["state"], error=attempt["error"], exit_code=attempt["exit_code"])
    atomic_json(record_path, record)
    return attempt


def require_completed(attempt):
    if attempt["state"] != "completed":
        raise LabError(f"Study attempt {attempt['id']} {attempt['state']}: {attempt['error']}. Its outputs and failure were retained; no automatic retry was made.")
    return attempt["result"]["observations"]


def inference_store(root):
    store = Store(root)
    inventory = read_json(HERE / "inputs.lock.json")
    config = configuration(store)
    expected = inventory["inference_runtime"]
    if config["image"] != expected["local_image_id"] or config["gpu_uuid"] != expected["gpu_uuid"] or config["gpu_mode"] != "wsl":
        raise LabError("Existing lab runner does not match the pinned inference image and allocated GPU; it was not changed")
    return store, inventory


def grading_store(store):
    receipt = read_json(PROJECT / ".arl-cache/qwen38-grader-build/receipt.json")
    grader = Store(store.root / STATE / "grading-runner")
    if (grader.root / "runner.json").exists():
        if configuration(grader)["image"] != receipt["image_id"]:
            raise LabError("The grading runtime changed; preserve the old grading records and declare an amendment")
    else:
        configure(grader, receipt["image_id"])
    return grader


def freeze(store, inventory):
    state = store.root / STATE
    frozen_path = state / "frozen.json"
    if frozen_path.exists():
        frozen = read_json(frozen_path)
        for item in frozen["inputs"].values():
            store.verify_input(item["id"])
        return frozen
    tasks = read_json(state / "tasks.json")
    tokenizers = read_json(state / "tokenizers.json")
    labels = [item["label"] for item in inventory["checkpoints"]]
    if set(tokenizers) != set(labels):
        raise LabError("All five pinned tokenizer subsets must be staged before freezing")
    for checkpoint in inventory["checkpoints"]:
        staged = tokenizers[checkpoint["label"]]
        if (staged["repo_id"], staged["revision"]) != (checkpoint["repo_id"], checkpoint["revision"]):
            raise LabError("Staged tokenizer differs from the pinned checkpoint")
    bindings = {f"tokenizer_{label}": tokenizers[label]["input_id"] for label in labels}
    bindings.update({f"public_{split}": values["public"]["id"] for split, values in tasks["inputs"].items()})
    spec = ExperimentSpec(name="qwen38-freeze", source=HERE / "worker", entrypoint="freeze.py",
                         question="Do all tokenizer identities match, and do the frozen prompts fit the common context?",
                         settings={"checkpoints": labels}, inputs=bindings,
                         outputs=("results.json", "rendered_compatibility.json", "rendered_primary.json"),
                         limits=Limits(wall_seconds=900, memory_bytes=4 * 1024**3, output_bytes=64 * 1024**2, pids=128))
    attempt = execute_once(store, state / "freeze-attempt.json", spec)
    result = require_completed(attempt)
    inputs = {}
    for split in ("compatibility", "primary"):
        source = store.attempt_dir(attempt["id"]) / "output" / f"rendered_{split}.json"
        destination = DATA / "rendered" / attempt["id"] / split / "tasks.json"
        expected = result["details"]["splits"][split]["sha256"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if file_hash(destination) != expected:
                raise LabError("Existing frozen prompt file differs; it was not replaced")
        else:
            shutil.copyfile(source, destination)
        approved = store.register_input(destination.parent, f"qwen38-{split}-rendered", expected_hashes={"tasks.json": expected})
        inputs[split] = {key: approved[key] for key in ("id", "sha256", "bytes")}
    frozen = {"attempt_id": attempt["id"], "inputs": inputs, "selection": tasks["selection"],
              "details": result["details"], "scorers": tasks["scorers"],
              "grading_image_id": configuration(grading_store(store))["image"]}
    atomic_json(frozen_path, frozen)
    return frozen


def generate(store, inventory, label):
    state = store.root / STATE
    frozen = read_json(state / "frozen.json")
    staged = read_json(state / "staged.json")
    checkpoint = next(item for item in inventory["checkpoints"] if item["label"] == label)
    if label not in staged or (staged[label]["repo_id"], staged[label]["revision"]) != (checkpoint["repo_id"], checkpoint["revision"]):
        raise LabError(f"Pinned {label} checkpoint is not staged and verified")
    tokenizers = read_json(state / "tokenizers.json")
    spec = ExperimentSpec(name=f"qwen38-compatibility-{label}", source=HERE / "worker", entrypoint="generate.py",
                         question="Can this checkpoint execute the frozen compatibility items under the common inference conditions?",
                         procedure="Compatibility only: eight items per family, seed 17, no primary estimates or hidden retries.",
                         settings={"checkpoint": label, "repo_id": checkpoint["repo_id"], "revision": checkpoint["revision"], "seed": 17},
                         gpu=True, inputs={"model": staged[label]["input_id"], "tokenizer": tokenizers["bf16"]["input_id"],
                                           "prompts": frozen["inputs"]["compatibility"]["id"]},
                         outputs=("results.json", "generations.jsonl", "runtime.json", "warmup.json"),
                         limits=Limits(wall_seconds=21600, memory_bytes=26 * 1024**3, output_bytes=8 * 1024**3,
                                       log_bytes=32 * 1024**2, pids=512))
    attempt = execute_once(store, state / "compatibility" / label / "generation.json", spec)
    require_completed(attempt)
    return {"checkpoint": label, "attempt_id": attempt["id"], "state": attempt["state"]}


def private_item(store, grader, tasks, task_id):
    source_id = tasks["inputs"]["compatibility"]["private"]["id"]
    approved = store.get_input(source_id)
    filename = task_filename(task_id)
    expected = next(item["sha256"] for item in approved["files"] if item["path"] == filename)
    record_path = store.root / STATE / "grading-inputs" / filename
    if record_path.exists():
        record = read_json(record_path)
        if record["source_input_id"] != source_id:
            raise LabError("Private task inventory changed after grading input registration")
        return record["input_id"]
    source = store.input_source(source_id) / filename
    destination = PROJECT / ".arl-cache/qwen38-grading-inputs" / approved["sha256"] / Path(filename).stem / "grading.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(destination) != expected:
            raise LabError("Existing private grading file differs; it was not replaced")
    else:
        if source.is_symlink() or not source.is_file():
            raise LabError("Private test file is not a regular file")
        shutil.copyfile(source, destination)
    copied = grader.register_input(destination.parent, f"qwen38-tests-{Path(filename).stem[:20]}", expected_hashes={"grading.json": expected})
    atomic_json(record_path, {"source_input_id": source_id, "input_id": copied["id"], "file_sha256": expected})
    return copied["id"]


def grade(store, label):
    state = store.root / STATE
    frozen = read_json(state / "frozen.json")
    tasks = read_json(state / "tasks.json")
    grader = grading_store(store)
    if configuration(grader)["image"] != frozen["grading_image_id"]:
        raise LabError("Grading image differs from the frozen protocol")
    record = read_json(state / "compatibility" / label / "generation.json")
    attempt = store.get_attempt(record["attempt_id"])
    require_completed(attempt)
    public = read_json(store.input_source(frozen["inputs"]["compatibility"]["id"]) / "tasks.json", max_bytes=64 * 1024**2)
    expected = {row["id"]: row for row in public["tasks"]}
    generations = {}
    with (store.attempt_dir(attempt["id"]) / "output/generations.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["id"] not in expected or row["id"] in generations:
                raise LabError("Generation inventory has duplicate or unexpected task IDs")
            task = expected[row["id"]]
            if row["rendered_prompt_sha256"] != task["rendered_prompt_sha256"] or row["checkpoint"] != label or row["seed"] != 17:
                raise LabError("Generation identity differs from the frozen compatibility request")
            generations[row["id"]] = row
    if set(generations) != set(expected):
        raise LabError("Generation inventory is incomplete; preserve the failed block rather than silently grading a subset")
    results = []
    for task_id, task in expected.items():
        input_id = private_item(store, grader, tasks, task_id)
        private = json.loads((grader.input_source(input_id) / "grading.json").read_text(encoding="utf-8"))
        seconds = lcb_wall_seconds(lcb_test_count(private)) + 60 if task["family"] == "coding" else 120
        spec = ExperimentSpec(name=f"qwen38-grade-{label}-{task_filename(task_id)[:16]}", source=HERE / "worker", entrypoint="grade.py",
                             question="Does this compatibility response pass the pinned official task evaluator?",
                             settings={"family": task["family"], "id": task_id, "generation": generations[task_id], "grading_file": "grading.json"},
                             inputs={"grading": input_id}, outputs=("results.json", "grading.json"),
                             limits=Limits(wall_seconds=seconds, memory_bytes=4 * 1024**3, pids=128))
        scored = execute_once(grader, state / "compatibility" / label / "grades" / task_filename(task_id), spec)
        outcome = require_completed(scored)
        results.append({"id": task_id, "family": task["family"], "attempt_id": scored["id"],
                        "metrics": outcome["metrics"], "correctness": outcome["details"]["correctness"]})
    summary = {"checkpoint": label, "split": "compatibility", "seed": 17, "generation_attempt": attempt["id"],
               "grader_root": str(grader.root), "results": results, "quality_comparison_performed": False}
    atomic_json(state / "compatibility" / label / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "generate", "grade"))
    parser.add_argument("checkpoint", nargs="?", choices=("bf16", "unsloth", "nvidia", "radixark", "quasar"))
    parser.add_argument("--root", type=Path, default=Path(".arl"))
    args = parser.parse_args()
    store, inventory = inference_store(args.root)
    if args.action == "freeze":
        result = freeze(store, inventory)
    else:
        if args.checkpoint is None:
            parser.error("generate and grade require a checkpoint label")
        result = generate(store, inventory, args.checkpoint) if args.action == "generate" else grade(store, args.checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
