"""Execute complete frozen study blocks with explicit series, split, and seed."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import re
import shutil

from autonomous_research_lab.execution import atomic_json, configuration, configure, run, wait_for_attempt
from autonomous_research_lab.models import ExperimentSpec, LabError, Limits, read_json
from autonomous_research_lab.store import Store
from analysis import SEEDS, analyze, render_report
from stage_tasks import DATA, file_hash, task_filename
from worker.grade import lcb_test_count, lcb_wall_seconds
from worker.serving import ENGINE_OPTIONS

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


def artifact_file(store, attempt, name):
    """Consume only the bytes retained by the completed attempt's inventory."""
    require_completed(attempt)
    entry = next((item for item in attempt["result"]["artifacts"] if item["path"] == name), None)
    path = store.attempt_dir(attempt["id"]) / "output" / name
    if entry is None or path.is_symlink() or not path.is_file() or file_hash(path) != entry["sha256"]:
        raise LabError(f"Retained artifact differs or is absent: {attempt['id']}/{name}")
    return path


def generate(store, inventory, label, *, split="compatibility", seed=17, series="baseline", wall_seconds=21600):
    state = store.root / STATE
    frozen = read_json(state / "frozen.json")
    staged = read_json(state / "staged.json")
    if split == "primary":
        plan = primary_plan(store, inventory, series)
        if wall_seconds != plan["wall_seconds"]:
            raise LabError("Primary generation must use the common measured reservation from its cohort plan")
    checkpoint = next(item for item in inventory["checkpoints"] if item["label"] == label)
    if label not in staged or (staged[label]["repo_id"], staged[label]["revision"]) != (checkpoint["repo_id"], checkpoint["revision"]):
        raise LabError(f"Pinned {label} checkpoint is not staged and verified")
    approved = store.get_input(staged[label]["input_id"])
    if approved["sha256"] != checkpoint["staged_input"]["inventory_sha256"]:
        raise LabError("Approved checkpoint inventory differs from the pinned release")
    if store.get_input(frozen["inputs"][split]["id"])["sha256"] != inventory["task_inventory"][split]["approved_inventory_sha256"]:
        raise LabError("Approved prompts differ from the frozen study inventory")
    tokenizers = read_json(state / "tokenizers.json")
    spec = ExperimentSpec(name=f"qwen38-{split}-{series}-{label}-seed-{seed}", source=HERE / "worker", entrypoint="generate.py",
                         question="How does this pinned checkpoint perform on the frozen tasks under the common inference conditions?",
                         procedure="One complete frozen split and shared seed; preserve every requested response and never retry invisibly.",
                         settings={"checkpoint": label, "repo_id": checkpoint["repo_id"], "revision": checkpoint["revision"],
                                   "split": split, "seed": seed, "series": series, "engine_options": ENGINE_OPTIONS},
                         gpu=True, inputs={"model": staged[label]["input_id"], "tokenizer": tokenizers["bf16"]["input_id"],
                                           "prompts": frozen["inputs"][split]["id"]},
                         outputs=("results.json", "generations.jsonl", "runtime.json", "warmup.json"),
                         limits=Limits(wall_seconds=wall_seconds, memory_bytes=26 * 1024**3, output_bytes=8 * 1024**3,
                                       log_bytes=32 * 1024**2, pids=512))
    attempt = execute_once(store, state / split / series / f"seed-{seed}" / label / "generation.json", spec)
    require_completed(attempt)
    return {"checkpoint": label, "attempt_id": attempt["id"], "state": attempt["state"]}


def private_item(store, grader, tasks, task_id, split):
    source_id = tasks["inputs"][split]["private"]["id"]
    approved = store.get_input(source_id)
    filename = task_filename(task_id)
    expected = next(item["sha256"] for item in approved["files"] if item["path"] == filename)
    record_path = store.root / STATE / "grading-inputs" / split / filename
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


def grade(store, label, *, split="compatibility", seed=17, series="baseline"):
    state = store.root / STATE
    frozen = read_json(state / "frozen.json")
    tasks = read_json(state / "tasks.json")
    grader = grading_store(store)
    if configuration(grader)["image"] != frozen["grading_image_id"]:
        raise LabError("Grading image differs from the frozen protocol")
    block = state / split / series / f"seed-{seed}" / label
    record = read_json(block / "generation.json")
    attempt = store.get_attempt(record["attempt_id"])
    require_completed(attempt)
    public = read_json(store.input_source(frozen["inputs"][split]["id"]) / "tasks.json", max_bytes=64 * 1024**2)
    expected = {row["id"]: row for row in public["tasks"]}
    generations = {}
    with artifact_file(store, attempt, "generations.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["id"] not in expected or row["id"] in generations:
                raise LabError("Generation inventory has duplicate or unexpected task IDs")
            task = expected[row["id"]]
            if row["rendered_prompt_sha256"] != task["rendered_prompt_sha256"] or row["checkpoint"] != label or row["seed"] != seed:
                raise LabError("Generation identity differs from the frozen request")
            generations[row["id"]] = row
    if set(generations) != set(expected):
        raise LabError("Generation inventory is incomplete; preserve the failed block rather than silently grading a subset")
    results = []
    for task_id, task in expected.items():
        input_id = private_item(store, grader, tasks, task_id, split)
        private = json.loads((grader.input_source(input_id) / "grading.json").read_text(encoding="utf-8"))
        test_count = lcb_test_count(private) if task["family"] == "coding" else None
        seconds = lcb_wall_seconds(test_count) + 60 if test_count is not None else 120
        spec = ExperimentSpec(name=f"qwen38-grade-{label}-{task_filename(task_id)[:16]}", source=HERE / "worker", entrypoint="grade.py",
                             question="Does this response pass the pinned official task evaluator?",
                             settings={"family": task["family"], "id": task_id, "generation": generations[task_id], "grading_file": "grading.json"},
                             inputs={"grading": input_id}, outputs=("results.json", "grading.json"),
                             limits=Limits(wall_seconds=seconds, memory_bytes=4 * 1024**3, pids=128))
        scored = execute_once(grader, block / "grades" / task_filename(task_id), spec)
        outcome = require_completed(scored)
        generation = generations[task_id]
        results.append({"id": task_id, "family": task["family"], "checkpoint": label, "seed": seed,
                        "instruction_count": len(private["instruction_id_list"]) if task["family"] == "ifeval" else None,
                        "prompt_tokens": generation["prompt_tokens"], "test_count": test_count,
                        "attempt_id": scored["id"], "metrics": outcome["metrics"], "correctness": outcome["details"]["correctness"],
                        "generated_tokens": len(generation["token_ids"]), "generation_seconds": generation["generation_seconds"],
                        "repeated_128_token_block": generation["repeated_128_token_block"], "finish_reason": generation["finish_reason"]})
    summary = {"checkpoint": label, "split": split, "seed": seed, "series": series, "generation_attempt": attempt["id"],
               "grader_root": str(grader.root), "results": results}
    atomic_json(block / "summary.json", summary)
    return summary


def primary_plan(store, inventory, series):
    """Gate the full cohort on all five complete, separately graded preflights."""
    state = store.root / STATE
    labels = [item["label"] for item in inventory["checkpoints"]]
    compatibility = {}
    slowest_seconds_per_token = 0.0
    longest_load = 0.0
    for label in labels:
        block = state / "compatibility" / series / "seed-17" / label
        if not (block / "summary.json").exists():
            raise LabError(f"Primary comparison requires the complete {label} compatibility block")
        summary = read_json(block / "summary.json")
        if (summary["checkpoint"], summary["split"], summary["seed"], summary["series"]) != (label, "compatibility", 17, series):
            raise LabError("Compatibility summary identity changed")
        attempt = store.get_attempt(summary["generation_attempt"])
        observed = require_completed(attempt)
        if observed["metrics"]["requested"] != 16 or observed["metrics"]["recorded"] != 16 or observed["metrics"]["generation_errors"]:
            raise LabError(f"{label} did not execute all sixteen compatibility requests without an engine error")
        rows = summary["results"]
        if len(rows) != 16 or len({row["id"] for row in rows}) != 16 or any(row["checkpoint"] != label or row["seed"] != 17 for row in rows):
            raise LabError("Compatibility grading inventory differs")
        if {family: sum(row["family"] == family for row in rows) for family in ("coding", "ifeval")} != {"coding": 8, "ifeval": 8}:
            raise LabError("Compatibility task-family inventory differs")
        grader = grading_store(store)
        for row in rows:
            scored = grader.get_attempt(row["attempt_id"])
            evidence = require_completed(scored)
            if evidence["details"]["id"] != row["id"] or evidence["metrics"] != row["metrics"]:
                raise LabError("Compatibility summary differs from retained scorer evidence")
        metadata = read_json(artifact_file(store, attempt, "runtime.json"), max_bytes=4 * 1024**2)
        if metadata["visible_gpu_uuid"] != inventory["inference_runtime"]["gpu_uuid"] or metadata["cuda_device_count"] != 1:
            raise LabError("Compatibility did not use the allocated GPU")
        if metadata["kv_cache_dtype"] != "bfloat16" or not metadata["model_implementation_paths"]:
            raise LabError("Compatibility runtime evidence is incomplete")
        if any(key not in metadata["engine_arguments"] or metadata["engine_arguments"][key] != value
               for key, value in ENGINE_OPTIONS.items()):
            raise LabError("Compatibility serving conditions differ from the common optimized configuration")
        total_tokens = sum(row["generated_tokens"] for row in rows)
        total_seconds = observed["metrics"]["generation_wall_seconds"]
        if total_tokens != observed["metrics"]["generated_tokens"]:
            raise LabError("Compatibility token accounting differs from the retained generation result")
        if total_tokens <= 0 or not math.isfinite(total_seconds) or total_seconds <= 0:
            raise LabError("Compatibility supplies no usable throughput measurement")
        slowest_seconds_per_token = max(slowest_seconds_per_token, total_seconds / total_tokens)
        longest_load = max(longest_load, metadata["load_seconds"])
        compatibility[label] = {"attempt_id": attempt["id"], "generated_tokens": total_tokens,
                                "generation_wall_seconds": total_seconds, "load_seconds": metadata["load_seconds"]}
    # Cover every requested response at the full output allowance, with 50% headroom.
    # This is a common reservation, not a claim that all checkpoints take this long.
    seconds = 1.5 * (longest_load + 256 * 16384 * slowest_seconds_per_token)
    wall_seconds = max(21600, math.ceil(seconds / 3600) * 3600)
    blocks = [{"checkpoint": label, "seed": seed} for offset, seed in enumerate(SEEDS)
              for label in labels[offset:] + labels[:offset]]
    plan = {"series": series, "split": "primary", "items": 256, "seeds": list(SEEDS),
            "wall_seconds": wall_seconds, "engine_options": ENGINE_OPTIONS, "compatibility": compatibility, "blocks": blocks,
            "reservation_rule": "1.5 * (longest measured load + 256 * 16384 * slowest aggregate generation wall-seconds per generated token), rounded up to an hour; at least six hours"}
    destination = state / "primary" / series / "cohort.json"
    if destination.exists():
        if read_json(destination) != plan:
            raise LabError("The recorded cohort differs; preserve it and declare an amendment")
    else:
        atomic_json(destination, plan)
    return plan


def analyze_cohort(store, inventory, series):
    state = store.root / STATE
    plan = primary_plan(store, inventory, series)
    frozen = read_json(state / "frozen.json")
    approved = frozen["inputs"]["primary"]
    store.verify_input(approved["id"])
    public = read_json(store.input_source(approved["id"]) / "tasks.json", max_bytes=64 * 1024**2)
    expected_ids = {family: [row["id"] for row in public["tasks"] if row["family"] == family] for family in ("coding", "ifeval")}
    if any(len(ids) != 128 for ids in expected_ids.values()):
        raise LabError("Primary frozen inventory does not contain 128 items per family")
    rows, provenance = [], []
    grader = grading_store(store)
    for block in plan["blocks"]:
        folder = state / "primary" / series / f"seed-{block['seed']}" / block["checkpoint"]
        summary = read_json(folder / "summary.json", max_bytes=8 * 1024**2)
        if (summary["checkpoint"], summary["seed"], summary["split"], summary["series"]) != (block["checkpoint"], block["seed"], "primary", series):
            raise LabError("Primary block identity differs")
        attempt = store.get_attempt(summary["generation_attempt"])
        artifact_file(store, attempt, "generations.jsonl")
        artifact_file(store, attempt, "runtime.json")
        for row in summary["results"]:
            scored = grader.get_attempt(row["attempt_id"])
            evidence = require_completed(scored)
            if (evidence["details"]["id"], evidence["details"]["correctness"], evidence["metrics"]) != (row["id"], row["correctness"], row["metrics"]):
                raise LabError("Primary summary differs from retained scorer evidence")
        rows.extend(summary["results"])
        experiment = store.get_experiment(attempt["experiment_id"])
        provenance.append({"checkpoint": block["checkpoint"], "seed": block["seed"], "attempt_id": attempt["id"],
                           "bundle_hash": experiment["bundle_hash"], "wall_seconds": attempt["result"]["wall_seconds"],
                           "generation_metrics": attempt["result"]["observations"]["metrics"],
                           "artifacts": attempt["result"]["artifacts"]})
    labels = [item["label"] for item in inventory["checkpoints"]]
    result = analyze(rows, expected_ids, labels)
    result["provenance"] = {"series": series, "inputs_lock_sha256": file_hash(HERE / "inputs.lock.json"),
                            "analysis_code_sha256": file_hash(HERE / "analysis.py"), "frozen_inputs": frozen["inputs"],
                            "runs": provenance, "checkpoints": inventory["checkpoints"],
                            "inference_runtime": inventory["inference_runtime"], "grading_runtime": inventory["grading_runtime"]}
    destination = state / "primary" / series
    atomic_json(destination / "analysis.json", result)
    (destination / "report.md").write_text(render_report(result), encoding="utf-8")
    return {"analysis": str(destination / "analysis.json"), "report": str(destination / "report.md"),
            "requested_responses": result["requested_responses"]}


def run_cohort(store, inventory, series):
    for checkpoint in inventory["checkpoints"]:
        label = checkpoint["label"]
        generate(store, inventory, label, series=series)
        grade(store, label, series=series)
    plan = primary_plan(store, inventory, series)
    for block in plan["blocks"]:
        options = {"split": "primary", "seed": block["seed"], "series": series}
        generate(store, inventory, block["checkpoint"], wall_seconds=plan["wall_seconds"], **options)
        grade(store, block["checkpoint"], **options)
    return analyze_cohort(store, inventory, series)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "generate", "grade", "run-all", "analyze"))
    parser.add_argument("checkpoint", nargs="?", choices=("bf16", "unsloth", "nvidia", "radixark", "quasar"))
    parser.add_argument("--root", type=Path, default=Path(".arl"))
    parser.add_argument("--split", choices=("compatibility", "primary"), default="compatibility")
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), default=17)
    parser.add_argument("--series", default="baseline")
    parser.add_argument("--wall-seconds", type=int, default=21600)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.series):
        parser.error("series must be a short lowercase identifier")
    if args.split == "compatibility" and args.seed != 17:
        parser.error("compatibility uses seed 17")
    store, inventory = inference_store(args.root)
    if args.action == "freeze":
        result = freeze(store, inventory)
    else:
        lock_path = store.root / STATE / f"{args.series}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise LabError("Another controller owns this study series; it was not duplicated") from exc
            if args.action == "run-all":
                result = run_cohort(store, inventory, args.series)
            elif args.action == "analyze":
                result = analyze_cohort(store, inventory, args.series)
            else:
                if args.checkpoint is None:
                    parser.error("generate and grade require a checkpoint label")
                options = {"split": args.split, "seed": args.seed, "series": args.series}
                result = generate(store, inventory, args.checkpoint, wall_seconds=args.wall_seconds, **options) if args.action == "generate" else grade(store, args.checkpoint, **options)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
