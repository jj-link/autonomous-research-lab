"""Run the approved 100k-output coding follow-up with a measured FP8 engine choice."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

from autonomous_research_lab.execution import atomic_json, configuration
from autonomous_research_lab.models import ExperimentSpec, LabError, Limits, read_json
from autonomous_research_lab.store import Store
from analysis import SEEDS, analyze, render_report, select_throughput
from stage_tasks import DATA, file_hash, rows, task_filename
from study import artifact_file, execute_once, require_completed, retire_completed_grader
from task_data import prepare_coding_followup
from worker.grade import lcb_test_count, lcb_wall_seconds
from worker.serving import ENGINE_OPTIONS

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
STUDY = PROJECT / ".arl/studies/qwen38-nvfp4"
STATE = STUDY / "followup-fp8"
CACHE = PROJECT / ".arl-cache/qwen38-fp8-followup"
CALIBRATION = STUDY / "dflash-calibration-20261015"
LABELS = ("bf16", "radixark", "quasar")
MODES = ("plain", "dflash")
IMAGE = "27e2c317d7b13ac6d7afe0642b350e5ef98c7206109bf7f59dfba20e41c20199"
GPU = "GPU-6ac06ab6-ac30-f3ad-b246-064a8e6f1309"
DRAFT_REVISION = "015e795645c74b1a0eeef3b570031fb62e769bc5"


def stores():
    store = Store(CALIBRATION / "runner")
    if configuration(store) != {"schema_version": 1, "image": IMAGE, "gpu_uuid": GPU, "gpu_mode": "wsl"}:
        raise LabError("The retained FP8/DFlash runner configuration changed")
    grader = Store(STUDY / "grading-runner")
    if configuration(grader)["image"] != read_json(HERE / "inputs.lock.json")["grading_runtime"]["local_image_id"]:
        raise LabError("The pinned official grading runtime changed")
    return store, grader


def immutable_json(path, value):
    if path.exists():
        if read_json(path, max_bytes=256 * 1024**2) != value:
            raise LabError(f"Retained follow-up data differs; it was not overwritten: {path}")
    else:
        atomic_json(path, value)


def prepare(store):
    receipt_path = STATE / "protocol.json"
    if receipt_path.exists():
        return read_json(receipt_path)
    original = read_json(STUDY / "tasks.json")
    frozen = read_json(STUDY / "frozen.json")
    inventory = read_json(HERE / "inputs.lock.json")
    dataset = next(item for item in inventory["datasets"] if item["label"] == "livecodebench")
    evidence = next(item for item in original["datasets"]
                    if item["repo_id"] == dataset["repo_id"] and item["revision"] == dataset["revision"])
    directory = DATA / "raw/livecodebench" / dataset["revision"]
    for item in evidence["files"]:
        path = directory / item["path"]
        if path.is_symlink() or not path.is_file() or file_hash(path) != item["sha256"]:
            raise LabError(f"Pinned source dataset file differs: {path}")
    excluded = original["selection"]["coding"]["compatibility"] + original["selection"]["coding"]["primary"]
    if len(excluded) != 136 or len(set(excluded)) != 136:
        raise LabError("The original 8+128 coding selection differs")
    selected = prepare_coding_followup(rows(directory, dataset["files"]), excluded)
    public = CACHE / "public"
    immutable_json(public / "tasks.json", {"split": "followup", "tasks": selected["public"]})
    for task in selected["private"]:
        immutable_json(CACHE / "private" / task_filename(task["id"])[:-5] / "grading.json", task["grading"])
    approved = store.register_input(public, "qwen38-followup-public")
    old_inputs = read_json(CALIBRATION / "inputs.json")
    tokenizer_id = old_inputs["tokenizer"]["input_id"]
    spec = ExperimentSpec(
        name="qwen38-followup-freeze", source=HERE / "worker", entrypoint="followup_freeze.py",
        question="Freeze 128 previously unused coding tasks under the original tokenizer and template.",
        settings={"items": 128, "max_output_tokens": 100000, "max_model_len": 131072,
                  "tokenizer_identity": frozen["details"]["tokenizers"]["bf16"],
                  "chat_template_sha256": frozen["details"]["chat_template_sha256"],
                  "stop_token_ids": frozen["details"]["stop_token_ids"]},
        inputs={"public": approved["id"], "tokenizer": tokenizer_id},
        outputs=("results.json", "rendered.json"),
        limits=Limits(wall_seconds=1800, memory_bytes=8 * 1024**3, pids=256))
    attempt = execute_once(store, STATE / "freeze.json", spec)
    outcome = require_completed(attempt)
    rendered = artifact_file(store, attempt, "rendered.json")
    destination = CACHE / "rendered/tasks.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(destination) != outcome["details"]["sha256"]:
            raise LabError("Existing rendered follow-up prompts differ")
    else:
        shutil.copyfile(rendered, destination)
    prompts = store.register_input(destination.parent, "qwen38-followup-rendered",
                                  expected_hashes={"tasks.json": outcome["details"]["sha256"]})
    record = {
        "schema_version": 1, "selection": selected["selection"], "dataset": evidence,
        "checkpoints": [item for item in inventory["checkpoints"] if item["label"] in LABELS],
        "seeds": list(SEEDS), "items": 128, "requested_responses": 1152,
        "max_output_tokens": 100000, "max_model_len": 131072,
        "kv_cache_dtype": "fp8", "calculate_kv_scales": False,
        "kv_scale_policy": "publisher scales when supplied; otherwise vLLM defaults, without separate calibration",
        "thinking_enabled": True, "reasoning_effort": "xhigh",
        "image_id": IMAGE, "gpu_uuid": GPU, "drafter_revision": DRAFT_REVISION,
        "inputs": {"prompts": prompts["id"], "tokenizer": tokenizer_id, "draft": old_inputs["draft"]["input_id"]},
        "prompt_inventory_sha256": prompts["sha256"], "freeze_attempt_id": attempt["id"],
        "comparison": {"seed": 17, "modes": list(MODES), "checkpoints": list(LABELS),
                       "responses_per_mode": 384, "max_num_seqs": 128,
                       "concurrency_policy": "queue the full cohort; the cache scheduler determines residency, with preemption costs included",
                       "selection": "highest total generated tokens / total generation wall seconds across checkpoints; exact ties choose plain",
                       "reuse": "retain the winning mode's complete seed-17 responses; choose before any grading",
                       "timing": "first submission through last durable response, including prefill; load and warmup recorded separately"},
        "interpretation": "New held-out tasks, FP8 KV cache, and vLLM 0.28 differ from the original study; this is not an isolated causal test of the larger output allowance.",
    }
    atomic_json(receipt_path, record)
    print(json.dumps({"prepared": str(receipt_path), "items": 128, "excluded_items": len(excluded),
                      "max_prompt_tokens": outcome["metrics"]["max_prompt_tokens"]}), flush=True)
    return record


def model_input(store, label):
    receipt = STATE / "models" / f"{label}.json"
    if receipt.exists():
        return read_json(receipt)["input_id"]
    if label == "bf16":
        input_id = read_json(CALIBRATION / "inputs.json")["model"]["input_id"]
        approved = store.get_input(input_id)
    else:
        main = Store(PROJECT / ".arl")
        previous = read_json(STUDY / "staged.json")[label]
        approved_original = main.get_input(previous["input_id"])
        approved = store.register_input(main.input_source(previous["input_id"]), f"qwen38-followup-{label}",
                                        expected_hashes={item["path"]: item["sha256"] for item in approved_original["files"]})
    atomic_json(receipt, {"input_id": approved["id"], "sha256": approved["sha256"], "bytes": approved["bytes"]})
    return approved["id"]


def generation_path(label, mode, stage):
    return STATE / "runs" / label / mode / f"{stage}.json"


def generate(store, protocol, label, mode, seeds, stage):
    # Stores have separate lease databases; do not overlap a main-lab GPU attempt.
    main = Store(PROJECT / ".arl")
    conflicts = [item["id"] for item in main.list_attempts(active_only=True) if item["gpu_uuid"] == GPU]
    if conflicts:
        raise LabError(f"The allocated GPU has active main-lab attempts: {conflicts}")
    checkpoint = next(item for item in protocol["checkpoints"] if item["label"] == label)
    speculative = {"method": "dflash", "model": "/inputs/draft", "num_speculative_tokens": 7} if mode == "dflash" else None
    settings = {"checkpoint": label, "repo_id": checkpoint["repo_id"], "revision": checkpoint["revision"],
                "split": "followup", "seeds": list(seeds), "max_output_tokens": protocol["max_output_tokens"],
                "dflash": mode == "dflash", "drafter_revision": DRAFT_REVISION if mode == "dflash" else None,
                "runtime_env": {"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_WSL2_ENABLE_PIN_MEMORY": "1"},
                "engine_options": {**ENGINE_OPTIONS, "max_num_seqs": protocol["items"], "speculative_config": speculative}}
    bindings = {"model": model_input(store, label), "tokenizer": protocol["inputs"]["tokenizer"],
                "prompts": protocol["inputs"]["prompts"]}
    if mode == "dflash":
        bindings["draft"] = protocol["inputs"]["draft"]
    outputs = ["results.json", "runtime.json", "warmup.json", "metrics.json"]
    for seed in seeds:
        outputs.extend((f"seed-{seed}/results.json", f"seed-{seed}/generations.jsonl"))
    spec = ExperimentSpec(
        name=f"qwen38-followup-{label}-{mode}-{stage}", source=HERE / "worker", entrypoint="long_generate.py",
        question="Which released checkpoint gives better coding answers with a 100,000-token output allowance?",
        procedure="Full natural-stop coding responses; all requests queued, FP8 KV cache, no artificial length window.",
        settings=settings, inputs=bindings, outputs=tuple(outputs), gpu=True,
        limits=Limits(wall_seconds=216000 * len(seeds), memory_bytes=26 * 1024**3,
                      output_bytes=8 * 1024**3, log_bytes=32 * 1024**2, pids=512))
    attempt = execute_once(store, generation_path(label, mode, stage), spec)
    require_completed(attempt)
    return attempt


def generation_block(store, protocol, label, mode, stage, seed):
    pointer = read_json(generation_path(label, mode, stage))
    attempt = store.get_attempt(pointer["attempt_id"])
    require_completed(attempt)
    summary = read_json(artifact_file(store, attempt, f"seed-{seed}/results.json"))
    metrics = summary["metrics"]
    if (summary["status"] != "completed" or metrics["requested"] != 128 or metrics["recorded"] != 128
            or metrics["generation_errors"] != 0 or summary["details"]["seed"] != seed):
        raise LabError("The complete requested seed block is not available")
    public = read_json(store.input_source(protocol["inputs"]["prompts"]) / "tasks.json", max_bytes=64 * 1024**2)
    expected = {task["id"]: task for task in public["tasks"]}
    generations = {}
    with artifact_file(store, attempt, f"seed-{seed}/generations.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["id"] not in expected or row["id"] in generations:
                raise LabError("The generation inventory contains unexpected or duplicate identities")
            task = expected[row["id"]]
            if (row["checkpoint"] != label or row["seed"] != seed or row["error"] is not None
                    or any(row[key] != task[key] for key in ("family", "dataset_id", "prompt_sha256", "rendered_prompt_sha256", "prompt_tokens_sha256"))
                    or row["prompt_tokens"] != len(task["prompt_token_ids"])):
                raise LabError("The retained response differs from the frozen request")
            generations[row["id"]] = row
    if set(generations) != set(expected) or sum(len(row["token_ids"]) for row in generations.values()) != metrics["generated_tokens"]:
        raise LabError("The retained generation inventory or token numerator is incomplete")
    return attempt, summary, generations


def compare(store, protocol):
    cases = []
    for index, label in enumerate(LABELS):
        for mode in MODES if index % 2 == 0 else tuple(reversed(MODES)):
            generate(store, protocol, label, mode, (17,), "comparison")
            attempt, summary, _ = generation_block(store, protocol, label, mode, "comparison", 17)
            runtime = read_json(artifact_file(store, attempt, "runtime.json"))
            measurements = read_json(artifact_file(store, attempt, "metrics.json"), max_bytes=16 * 1024**2)
            case = {"mode": mode, "checkpoint": label, "attempt_id": attempt["id"],
                    **{key: summary["metrics"][key] for key in ("generated_tokens", "generation_wall_seconds", "load_seconds", "warmup_seconds")},
                    "execution": runtime["worker_execution_state"], "measured": measurements["measured"]}
            case["aggregate_tokens_per_second"] = case["generated_tokens"] / case["generation_wall_seconds"]
            cases.append(case)
            print(json.dumps({"comparison_cell": case}), flush=True)
    result = select_throughput(cases, LABELS)
    result["protocol_sha256"] = file_hash(STATE / "protocol.json")
    immutable_json(STATE / "selection.json", result)
    print(json.dumps({"selected": result["winner"], "modes": result["modes"]}), flush=True)
    return result


def grade(store, grader, protocol, label, mode, stage, seed):
    attempt, _, generations = generation_block(store, protocol, label, mode, stage, seed)
    destination = STATE / "grades" / label / f"seed-{seed}"
    results = []
    for task_id in protocol["selection"]["ids"]:
        generation = generations[task_id]
        key = task_filename(task_id)[:-5]
        private_source = CACHE / "private" / key
        private = read_json(private_source / "grading.json", max_bytes=256 * 1024**2)
        count = lcb_test_count(private)
        binding_path = destination / key / "inputs.json"
        if binding_path.exists():
            bindings = read_json(binding_path)
        else:
            response_source = CACHE / "responses" / attempt["id"] / f"seed-{seed}" / key
            immutable_json(response_source / "generation.json", generation)
            bindings = {"grading": grader.register_input(private_source, f"qwen38-followup-tests-{key[:16]}")["id"],
                        "response": grader.register_input(response_source, f"qwen38-followup-answer-{key[:16]}")["id"]}
            atomic_json(binding_path, bindings)
        spec = ExperimentSpec(
            name=f"qwen38-followup-grade-{label}-{key[:16]}", source=HERE / "worker", entrypoint="followup_grade.py",
            question="Does the complete retained response pass every pinned official coding test?",
            settings={"id": task_id, "family": "coding", "grading_file": "grading.json"},
            inputs=bindings, outputs=("results.json", "grading.json"),
            limits=Limits(wall_seconds=lcb_wall_seconds(count) + 60, memory_bytes=4 * 1024**3, pids=128))
        scored = execute_once(grader, destination / key / "attempt.json", spec)
        outcome = require_completed(scored)
        results.append({"id": task_id, "family": "coding", "checkpoint": label, "seed": seed,
                        "instruction_count": None, "test_count": count, "prompt_tokens": generation["prompt_tokens"],
                        "attempt_id": scored["id"], "metrics": outcome["metrics"], "correctness": outcome["details"]["correctness"],
                        "generated_tokens": len(generation["token_ids"]), "generation_seconds": generation["generation_seconds"],
                        "repeated_128_token_block": generation["repeated_128_token_block"], "finish_reason": generation["finish_reason"]})
        retire_completed_grader(grader, scored)
    summary = {"checkpoint": label, "seed": seed, "mode": mode, "generation_attempt_id": attempt["id"], "results": results}
    immutable_json(destination / "summary.json", summary)
    return summary


def run_followup(store, grader, protocol, selection):
    if selection["protocol_sha256"] != file_hash(STATE / "protocol.json"):
        raise LabError("Engine selection belongs to a different frozen protocol")
    mode = selection["winner"]
    futures = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for label in LABELS:
            futures.append(pool.submit(grade, store, grader, protocol, label, mode, "comparison", 17))
        for label in LABELS:
            for future in futures:
                if future.done():
                    future.result()
            generate(store, protocol, label, mode, (29, 43), "continuation")
            for seed in (29, 43):
                futures.append(pool.submit(grade, store, grader, protocol, label, mode, "continuation", seed))
        summaries = [future.result() for future in futures]
    results = [row for summary in summaries for row in summary["results"]]
    result = analyze(results, {"coding": protocol["selection"]["ids"]}, list(LABELS))
    result["provenance"] = {"protocol_sha256": file_hash(STATE / "protocol.json"),
                            "selection_sha256": file_hash(STATE / "selection.json"),
                            "selected_mode": mode, "interpretation": protocol["interpretation"],
                            "generation_attempts": sorted({summary["generation_attempt_id"] for summary in summaries})}
    atomic_json(STATE / "analysis.json", result)
    print(render_report(result), flush=True)
    print(json.dumps({"completed_responses": result["requested_responses"], "analysis": str(STATE / "analysis.json")}), flush=True)


def main():
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "prepare", "compare", "run-all"))
    args = parser.parse_args()
    store, grader = stores()
    if args.action == "status":
        retained = read_json(CALIBRATION / "inputs.json")
        staged = read_json(STUDY / "staged.json")
        print(json.dumps({"runner": configuration(store), "active_runner": store.list_attempts(active_only=True),
                          "active_main": Store(PROJECT / ".arl").list_attempts(active_only=True),
                          "retained_inputs_present": {name: store.input_source(retained[name]["input_id"]).is_dir()
                                                      for name in ("model", "tokenizer", "draft")},
                          "checkpoint_sources_present": {label: Path(staged[label]["source"]).is_dir() for label in LABELS}}, indent=2))
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LabError("A follow-up controller is already active") from exc
        protocol = prepare(store)
        if args.action == "prepare":
            return
        selection = compare(store, protocol)
        if args.action == "run-all":
            run_followup(store, grader, protocol, selection)


if __name__ == "__main__":
    main()
