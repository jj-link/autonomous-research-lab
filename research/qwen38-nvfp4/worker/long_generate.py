"""Full held-out coding responses, FP8 KV cache, and cache-managed concurrency."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from generate import WorkerMetadata, engine_arguments, run_seed_blocks, save


class LongMetadata(WorkerMetadata):
    def arl_execution_state(self):
        from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity

        config = self.vllm_config
        runner = self.model_runner
        if not config.use_v2_model_runner or runner.cudagraph_manager is None:
            raise RuntimeError("The declared V2 model runner is not active")
        manager = runner.cudagraph_manager
        capacity, full_context_concurrency = get_kv_cache_capacity(config, runner.kv_cache_config)
        return {
            "model_runner": f"{type(runner).__module__}.{type(runner).__qualname__}",
            "compilation_mode": str(config.compilation_config.mode),
            "cudagraph_mode": config.compilation_config.cudagraph_mode.name,
            "capture_sizes": config.compilation_config.cudagraph_capture_sizes,
            "captured_graphs": len(manager.graphs),
            "captured_token_counts": manager.captured_token_counts(),
            "async_scheduling": config.scheduler_config.async_scheduling,
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "compute_dtype": str(config.model_config.dtype),
            "kv_cache_dtype": config.cache_config.cache_dtype,
            "kv_cache_tokens": capacity,
            "full_context_concurrency": full_context_concurrency,
            "num_gpu_blocks": runner.kv_cache_config.num_blocks,
            "available_kv_cache_memory_bytes": self.available_kv_cache_memory_bytes,
        }


def snapshot(llm):
    return [asdict(metric) for metric in llm.get_metrics()]


def counter(metrics, name):
    values = [row["value"] for row in metrics if row["name"] == name]
    if not values:
        raise RuntimeError(f"Measured runtime counter is absent: {name}")
    return sum(values)


def followup_arguments(settings):
    args = engine_arguments(settings)
    args.update(max_model_len=131072, kv_cache_dtype="fp8",
                safetensors_load_strategy="eager", worker_extension_cls="long_generate.LongMetadata")
    return args


def main():
    settings = json.loads(Path("/experiment/settings.json").read_text())
    os.environ.update(settings["runtime_env"])
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    frozen = json.loads(Path("/inputs/prompts/tasks.json").read_text())
    tasks = frozen["tasks"]
    if (frozen["split"] != "followup" or len(tasks) != 128
            or len({task["id"] for task in tasks}) != 128
            or any(task["family"] != "coding" for task in tasks)
            or not frozen["thinking_enabled"] or frozen["reasoning_effort"] != "xhigh"
            or settings["max_output_tokens"] != 100000
            or not settings["seeds"] or len(set(settings["seeds"])) != len(settings["seeds"])
            or any(type(seed) is not int or seed not in (17, 29, 43) for seed in settings["seeds"])):
        raise ValueError("The requested long-generation protocol differs from the held-out cohort")
    tokenizer = AutoTokenizer.from_pretrained("/inputs/tokenizer", local_files_only=True, trust_remote_code=False)
    for task in tasks:
        encoded = json.dumps(task["prompt_token_ids"], separators=(",", ":")).encode()
        if (hashlib.sha256(encoded).hexdigest() != task["prompt_tokens_sha256"]
                or hashlib.sha256(task["prompt"].encode()).hexdigest() != task["rendered_prompt_sha256"]
                or tokenizer.encode(task["prompt"], add_special_tokens=False) != task["prompt_token_ids"]
                or len(task["prompt_token_ids"]) + settings["max_output_tokens"] > 131072):
            raise ValueError(f"Frozen prompt identity or context differs: {task['id']}")
    args = followup_arguments(settings)
    started = time.monotonic()
    llm = LLM(**args)
    load_seconds = time.monotonic() - started
    try:
        config = llm.llm_engine.vllm_config
        spec = config.speculative_config
        if (config.cache_config.cache_dtype != "fp8" or config.cache_config.enable_prefix_caching
                or config.model_config.max_model_len != 131072
                or config.scheduler_config.max_num_seqs != len(tasks)):
            raise ValueError("Actual cache, context, or concurrency differs from the declared protocol")
        if settings["dflash"]:
            if spec is None or spec.method != "dflash" or spec.num_speculative_tokens != 7:
                raise ValueError("The declared DFlash verifier is not active")
        elif spec is not None:
            raise ValueError("The control arm unexpectedly enabled speculation")
        save("runtime.json", {
            "engine_arguments": args, "engine_config": str(config), "load_seconds": load_seconds,
            "runtime_env": settings["runtime_env"],
            "worker_execution_state": llm.collective_rpc("arl_execution_state", timeout=60),
            "model_implementation_paths": llm.collective_rpc("arl_implementation_paths", timeout=60),
            "visible_gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__})
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Reply with READY."}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
        started = time.monotonic()
        warm = llm.generate([{"prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=False)}],
                            SamplingParams(temperature=0, max_tokens=64, seed=0,
                                           stop_token_ids=frozen["stop_token_ids"]), use_tqdm=False)[0].outputs[0]
        warmup_seconds = time.monotonic() - started
        save("warmup.json", {"seconds": warmup_seconds, "text": warm.text,
                             "token_ids": list(warm.token_ids), "finish_reason": warm.finish_reason})
        if warm.finish_reason != "stop" or not warm.text.strip():
            raise ValueError("Warmup did not finish normally; no study requests were submitted")
        before = snapshot(llm)
        run_seed_blocks(llm, tokenizer, frozen, settings, args, load_seconds, warmup_seconds)
        after = snapshot(llm)
        measured = {"preemptions": counter(after, "vllm:num_preemptions") - counter(before, "vllm:num_preemptions")}
        if settings["dflash"]:
            for key, name in (("draft_steps", "num_drafts"), ("proposed_tokens", "num_draft_tokens"),
                              ("accepted_tokens", "num_accepted_tokens")):
                metric = "vllm:spec_decode_" + name
                measured[key] = counter(after, metric) - counter(before, metric)
            if not 0 <= measured["accepted_tokens"] <= measured["proposed_tokens"] or measured["draft_steps"] <= 0:
                raise ValueError("No valid speculative verification was measured")
            measured["draft_acceptance_rate"] = measured["accepted_tokens"] / measured["proposed_tokens"]
        save("metrics.json", {"before": before, "after": after, "measured": measured})
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=30)


if __name__ == "__main__":
    main()
