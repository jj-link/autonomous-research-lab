"""Resident checkpoint, one warm-up, and durable seed blocks over a frozen split."""
from __future__ import annotations

from enum import Enum
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from contextlib import suppress


def repeated_block(tokens):
    """Four adjacent identical 128-token blocks, at any output offset."""
    width = 128
    for start in range(len(tokens) - 4 * width + 1):
        if not tokens[start] == tokens[start + width] == tokens[start + 2 * width] == tokens[start + 3 * width]:
            continue
        block = tokens[start:start + width]
        if all(tokens[start + offset * width:start + (offset + 1) * width] == block for offset in (1, 2, 3)):
            return True
    return False


def model_paths(model):
    """Inspect selected implementations without copying any tensor contents."""
    def describe(value):
        if value is None or type(value) in (bool, int, float, str):
            return value
        if isinstance(value, Enum):
            return f"{type(value).__module__}.{type(value).__qualname__}.{value.name}"
        if inspect.isroutine(value):
            return f"{value.__module__}.{value.__qualname__}"
        kind = value if isinstance(value, type) else type(value)
        return f"{kind.__module__}.{kind.__qualname__}"
    groups = {}
    for name, module in model.named_modules():
        for attribute in ("quant_method", "impl", "attn_backend", "backend"):
            value = getattr(module, attribute, None)
            if value is None:
                continue
            selections = {key: describe(item) for key, item in getattr(value, "__dict__", {}).items()
                          if "kernel" in key or "backend" in key}
            descriptor = {"attribute": attribute, "implementation": describe(value), "selections": selections}
            key = json.dumps(descriptor, sort_keys=True)
            if key not in groups:
                groups[key] = {**descriptor, "module_count": 0, "example_module": name}
            groups[key]["module_count"] += 1
    return list(groups.values())


class WorkerMetadata:
    """Named worker RPC: no callable or pickle payload crosses the engine boundary."""

    def arl_implementation_paths(self):
        return model_paths(self.get_model())

    def arl_execution_state(self):
        config = self.vllm_config
        runner = self.model_runner
        entries = getattr(runner.model, "concrete_cudagraph_entries", {})
        return {
            "compilation_mode": str(config.compilation_config.mode),
            "cudagraph_mode": config.compilation_config.cudagraph_mode.name,
            "capture_sizes": config.compilation_config.cudagraph_capture_sizes,
            "captured_full_graphs": sum(entry.cudagraph is not None for entry in entries.values()),
            "dispatcher_mode": runner.cudagraph_dispatcher.cudagraph_mode.name,
            "async_scheduling": config.scheduler_config.async_scheduling,
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "compute_dtype": str(config.model_config.dtype),
            "kv_cache_dtype": config.cache_config.cache_dtype,
            "num_gpu_blocks": config.cache_config.num_gpu_blocks,
            "available_kv_cache_memory_bytes": getattr(self, "available_kv_cache_memory_bytes", None),
        }


def save(name, value):
    with Path("/workspace", name).open("w", encoding="utf-8") as destination:
        destination.write(json.dumps(value, ensure_ascii=False, allow_nan=False))
        destination.flush()
        os.fsync(destination.fileno())


def engine_arguments(settings):
    options = settings["engine_options"]
    return {"model": "/inputs/model", "tokenizer": "/inputs/tokenizer", "trust_remote_code": False,
            "dtype": "bfloat16", "kv_cache_dtype": "bfloat16", "max_model_len": 32768,
            "tensor_parallel_size": 1, "max_num_seqs": options["max_num_seqs"],
            "max_num_batched_tokens": options["max_num_batched_tokens"],
            "enable_chunked_prefill": options["enable_chunked_prefill"],
            "enable_prefix_caching": options["enable_prefix_caching"],
            "speculative_config": options["speculative_config"],
            "enforce_eager": options["enforce_eager"], "cpu_offload_gb": 0, "offload_group_size": 0,
            "gpu_memory_utilization": 0.90, "seed": settings["seeds"][0], "generation_config": "vllm",
            "load_format": "safetensors", "safetensors_load_strategy": "lazy",
            "limit_mm_per_prompt": {"image": 0, "video": 0}, "hf_token": False,
            "cudagraph_metrics": True, "disable_log_stats": False,
            "worker_extension_cls": "generate.WorkerMetadata"}


def validate_protocol(settings, frozen):
    """Reject invalid block schedules before creating an engine or any output."""
    split, seeds = settings["split"], settings.get("seeds")
    if split not in {"compatibility", "primary"} or frozen["split"] != split:
        raise ValueError("Requested split differs from the frozen input")
    allowed = (17,) if split == "compatibility" else (17, 29, 43)
    if (type(seeds) is not list or not seeds
            or any(type(seed) is not int or seed not in allowed for seed in seeds)
            or len(set(seeds)) != len(seeds) or "seed" in settings):
        raise ValueError("Requested seeds must be a nonempty unique protocol seed list")
    tasks = frozen["tasks"]
    count = 8 if split == "compatibility" else 128
    if len(tasks) != 2 * count:
        raise ValueError("Requested task count differs from the protocol")
    if not frozen["thinking_enabled"] or frozen["reasoning_effort"] != "xhigh":
        raise ValueError("Unexpected thinking conditions")
    if [row["family"] for row in tasks] != ["coding"] * count + ["ifeval"] * count:
        raise ValueError("Task families differ from the frozen protocol")
    identities = [task["id"] for task in tasks]
    if any(type(identity) is not str or not identity for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError("Frozen request identities must be nonempty and unique")


def main():
    from transformers import AutoTokenizer
    from vllm import LLM

    settings = json.loads(Path("/experiment/settings.json").read_text())
    frozen = json.loads(Path("/inputs/prompts/tasks.json").read_text(encoding="utf-8"))
    tasks = frozen["tasks"]
    validate_protocol(settings, frozen)
    tokenizer = AutoTokenizer.from_pretrained("/inputs/tokenizer", local_files_only=True, trust_remote_code=False)
    for task in tasks:
        encoded = json.dumps(task["prompt_token_ids"], separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest() != task["rendered_prompt_sha256"] or hashlib.sha256(encoded).hexdigest() != task["prompt_tokens_sha256"]:
            raise ValueError(f"Frozen prompt checksum differs: {task['id']}")
        if tokenizer.encode(task["prompt"], add_special_tokens=False) != task["prompt_token_ids"]:
            raise ValueError("Tokenizer changed after prompt freezing")
        if len(task["prompt_token_ids"]) + 16384 > 32768:
            raise ValueError("A frozen prompt exceeds the common context")
    engine_args = engine_arguments(settings)
    started = time.monotonic()
    llm = LLM(**engine_args)
    load_seconds = time.monotonic() - started
    try:
        warmup_seconds = prepare_runtime(llm, tokenizer, frozen, engine_args, load_seconds)
        run_seed_blocks(llm, tokenizer, frozen, settings, engine_args, load_seconds, warmup_seconds)
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=30)


def consume_completed(engine, pending, persist):
    """Persist each finished external request ID before releasing its pending entry."""
    while engine.has_unfinished_requests():
        for result in engine.step():
            if not result.finished or result.request_id not in pending:
                raise RuntimeError("Unexpected or duplicate completion from the engine")
            persist(pending[result.request_id], result)
            del pending[result.request_id]
    if pending:
        raise RuntimeError(f"Engine finished without completing {len(pending)} requested responses")


def prepare_runtime(llm, tokenizer, frozen, engine_args, load_seconds):
    """Validate the resident engine, record its implementation, and warm it once."""
    from vllm import SamplingParams
    import torch

    config = llm.llm_engine.vllm_config
    if config.cache_config.cache_dtype != "bfloat16" or config.cache_config.enable_prefix_caching or config.speculative_config is not None:
        raise ValueError("Actual cache or speculative decoding conditions differ from the protocol")
    if config.model_config.max_model_len != 32768 or config.scheduler_config.max_num_seqs != engine_args["max_num_seqs"]:
        raise ValueError("Actual context or concurrency differs from the recorded conditions")
    runtime = {"engine_arguments": engine_args, "engine_config": str(config), "load_seconds": load_seconds,
               "engine_initialization_seed": engine_args["seed"],
               "model_implementation_paths": llm.collective_rpc("arl_implementation_paths", timeout=60),
               "worker_execution_state": llm.collective_rpc("arl_execution_state", timeout=60),
               "visible_gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda_device_count": torch.cuda.device_count(),
               "gpu_name": torch.cuda.get_device_name(0), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
               "kv_cache_dtype": config.cache_config.cache_dtype, "attention_config": str(config.attention_config),
               "quantization": config.model_config.quantization, "torch": torch.__version__, "cuda": torch.version.cuda}
    save("runtime.json", runtime)
    warmup_prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Reply with READY."}],
                                                  tokenize=False, add_generation_prompt=True, enable_thinking=False)
    started = time.monotonic()
    warmup = llm.generate([{"prompt_token_ids": tokenizer.encode(warmup_prompt, add_special_tokens=False)}],
                          SamplingParams(temperature=0, max_tokens=64, seed=0, stop_token_ids=frozen["stop_token_ids"]), use_tqdm=False)[0].outputs[0]
    warmup_seconds = time.monotonic() - started
    save("warmup.json", {"prompt": warmup_prompt, "thinking_enabled": False, "seed": 0,
                         "text": warmup.text, "token_ids": list(warmup.token_ids), "finish_reason": warmup.finish_reason,
                         "seconds": warmup_seconds})
    if warmup.finish_reason != "stop" or not warmup.text.strip():
        raise ValueError("Warm-up did not produce a normally terminated response; no seed requests were sent")
    return warmup_seconds


def aggregate_results(frozen, settings, blocks, load_seconds, warmup_seconds, failed_seed=None):
    """Count each durable completed block and each resident startup cost once."""
    metrics = {name: sum(block["metrics"][name] for block in blocks)
               for name in ("requested", "recorded", "generation_errors", "generated_tokens", "generation_wall_seconds")}
    metrics.update(load_seconds=load_seconds, warmup_seconds=warmup_seconds)
    completed = [block["details"]["seed"] for block in blocks]
    return {"status": "failed" if failed_seed is not None else ("completed" if completed == settings["seeds"] else "running"),
            "metrics": metrics,
            "details": {"checkpoint": settings["checkpoint"], "split": frozen["split"],
                        "seeds": settings["seeds"], "completed_seeds": completed, "failed_seed": failed_seed,
                        "engine_initialization_seed": settings["seeds"][0],
                        "seed_results": [{"seed": seed, "path": f"seed-{seed}/results.json"} for seed in completed],
                        "runtime_path": "runtime.json", "warmup_path": "warmup.json",
                        "request_latency_includes_queue": True,
                        "throughput_interval": "sum of seed blocks, each first submission through last durable completion"}}


def run_seed_blocks(llm, tokenizer, frozen, settings, engine_args, load_seconds, warmup_seconds):
    """Keep completed receipts available even when a later block aborts the run."""
    blocks = []
    for index, seed in enumerate(settings["seeds"]):
        try:
            block = run_requests(llm, tokenizer, frozen, settings, engine_args, seed,
                                 load_seconds if index == 0 else 0.0, warmup_seconds if index == 0 else 0.0)
        except BaseException:
            with suppress(Exception):
                save("results.json", aggregate_results(frozen, settings, blocks, load_seconds, warmup_seconds, failed_seed=seed))
            raise
        blocks.append(block)
        save("results.json", aggregate_results(frozen, settings, blocks, load_seconds, warmup_seconds))


def run_requests(llm, tokenizer, frozen, settings, engine_args, seed, load_seconds, warmup_seconds):
    """Generate one seed block with explicit external identities and durable rows."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.renderers.inputs.preprocess import parse_model_prompt

    tasks = frozen["tasks"]
    sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
                "max_tokens": settings.get("max_output_tokens", 16384), "seed": seed, "stop_token_ids": frozen["stop_token_ids"], "skip_special_tokens": True}
    errors = recorded = generated_tokens = 0
    engine = llm.llm_engine
    pending, submitted = {}, {}
    tok_params = llm.renderer.default_cmpl_tok_params.with_kwargs()
    params = SamplingParams(**sampling, output_kind=RequestOutputKind.FINAL_ONLY)
    def receipt(status, wall_seconds, failure=None):
        return {"status": status,
                "metrics": {"requested": len(tasks), "recorded": recorded, "generation_errors": errors,
                            "generated_tokens": generated_tokens, "generation_wall_seconds": wall_seconds,
                            "load_seconds": load_seconds, "warmup_seconds": warmup_seconds},
                "details": {"checkpoint": settings["checkpoint"], "repo_id": settings["repo_id"],
                            "revision": settings["revision"], "split": frozen["split"], "seed": seed,
                            "engine_initialization_seed": engine_args["seed"], "sampling": sampling,
                            "thinking_enabled": frozen["thinking_enabled"], "reasoning_effort": frozen["reasoning_effort"],
                            "max_model_len": engine_args["max_model_len"], "kv_cache_dtype": engine_args["kv_cache_dtype"],
                            "runtime_path": "../runtime.json", "warmup_path": "../warmup.json",
                            "request_latency_includes_queue": True,
                            "throughput_interval": "first submission through last durable completion",
                            "failure": failure}}

    block_directory = Path("/workspace", f"seed-{seed}")
    block_directory.mkdir()
    with (block_directory / "generations.jsonl").open("x", encoding="utf-8") as destination:
        def persist(task, result):
            nonlocal errors, recorded, generated_tokens
            if result.prompt_token_ids != task["prompt_token_ids"] or len(result.outputs) != 1:
                raise RuntimeError("Completed response differs from its frozen request")
            output = result.outputs[0]
            tokens = list(output.token_ids)
            error = None
            if output.finish_reason not in {"stop", "length", "repetition"}:
                error = f"Engine terminated the request with {output.finish_reason!r}"
                errors += 1
            row = {key: task[key] for key in ("id", "family", "dataset_id", "prompt_sha256", "rendered_prompt_sha256", "prompt_tokens_sha256")}
            row.update(checkpoint=settings["checkpoint"], repo_id=settings["repo_id"], revision=settings["revision"],
                       seed=seed, thinking_enabled=True, prompt_tokens=len(task["prompt_token_ids"]),
                       text=output.text, raw_decoded_text=tokenizer.decode(tokens, skip_special_tokens=False),
                       token_ids=tokens, finish_reason=output.finish_reason, stop_reason=output.stop_reason,
                       error=error, repeated_128_token_block=repeated_block(tokens),
                       generation_seconds=time.monotonic() - submitted[result.request_id])
            destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
            recorded += 1
            generated_tokens += len(tokens)
            print(json.dumps({"seed": seed, "id": row["id"], "tokens": len(tokens), "finish_reason": row["finish_reason"], "error": error}), flush=True)
            if error:
                raise RuntimeError(error)

        # The core starts work during submission, so timing must start before enqueue.
        batch_started = time.monotonic()
        try:
            for task in tasks:
                identity = f"seed-{seed}/{task['id']}"
                if identity in pending:
                    raise ValueError("Duplicate frozen request identity")
                pending[identity] = task
                submitted[identity] = time.monotonic()
                prompt = {"prompt_token_ids": task["prompt_token_ids"]}
                parsed = parse_model_prompt(llm.model_config, prompt)
                (engine_input,) = llm.renderer.render_cmpl([parsed], tok_params)
                # add_request returns an internal randomized ID; outputs use this external ID.
                engine.add_request(identity, engine_input, params)
            consume_completed(engine, pending, persist)
            generation_wall_seconds = time.monotonic() - batch_started
            engine.do_log_stats()
        except BaseException as error:
            generation_wall_seconds = time.monotonic() - batch_started
            with suppress(Exception):
                engine.abort_request(list(pending))
            with suppress(Exception):
                save(f"seed-{seed}/results.json", receipt("failed", generation_wall_seconds,
                                                       f"{type(error).__name__}: {error}"))
            raise
    result = receipt("completed", generation_wall_seconds)
    save(f"seed-{seed}/results.json", result)
    return result


if __name__ == "__main__":
    main()
