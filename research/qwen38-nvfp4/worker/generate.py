"""One checkpoint, sixteen frozen compatibility prompts, no primary generation."""
from __future__ import annotations

from enum import Enum
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback


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


def save(name, value):
    Path("/workspace", name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch

    settings = json.loads(Path("/experiment/settings.json").read_text())
    frozen = json.loads(Path("/inputs/prompts/tasks.json").read_text(encoding="utf-8"))
    tasks = frozen["tasks"]
    if frozen["split"] != "compatibility" or settings["seed"] != 17 or len(tasks) != 16:
        raise ValueError("This worker is restricted to the frozen 16-item compatibility split at seed 17")
    if not frozen["thinking_enabled"] or frozen["reasoning_effort"] != "xhigh":
        raise ValueError("Unexpected thinking conditions")
    if [row["family"] for row in tasks] != ["coding"] * 8 + ["ifeval"] * 8:
        raise ValueError("Compatibility task families differ from the frozen protocol")
    tokenizer = AutoTokenizer.from_pretrained("/inputs/tokenizer", local_files_only=True, trust_remote_code=False)
    for task in tasks:
        encoded = json.dumps(task["prompt_token_ids"], separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest() != task["rendered_prompt_sha256"] or hashlib.sha256(encoded).hexdigest() != task["prompt_tokens_sha256"]:
            raise ValueError(f"Frozen prompt checksum differs: {task['id']}")
        if tokenizer.encode(task["prompt"], add_special_tokens=False) != task["prompt_token_ids"]:
            raise ValueError("Tokenizer changed after prompt freezing")
        if len(task["prompt_token_ids"]) + 16384 > 32768:
            raise ValueError("A frozen prompt exceeds the common context")
    engine_args = {"model": "/inputs/model", "tokenizer": "/inputs/tokenizer", "trust_remote_code": False,
                   "dtype": "bfloat16", "kv_cache_dtype": "bfloat16", "max_model_len": 32768,
                   "tensor_parallel_size": 1, "max_num_seqs": 1, "max_num_batched_tokens": 4096,
                   "enable_chunked_prefill": True, "enable_prefix_caching": False, "speculative_config": None,
                   "enforce_eager": True, "cpu_offload_gb": 0, "offload_group_size": 0,
                   "gpu_memory_utilization": 0.90, "seed": 17, "generation_config": "vllm",
                   "load_format": "safetensors", "safetensors_load_strategy": "lazy",
                   "limit_mm_per_prompt": {"image": 0, "video": 0}, "hf_token": False}
    started = time.monotonic()
    llm = LLM(**engine_args)
    load_seconds = time.monotonic() - started
    config = llm.llm_engine.vllm_config
    if config.cache_config.cache_dtype != "bfloat16" or config.cache_config.enable_prefix_caching or config.speculative_config is not None:
        raise ValueError("Actual cache or speculative decoding conditions differ from the protocol")
    if config.model_config.max_model_len != 32768 or config.scheduler_config.max_num_seqs != 1:
        raise ValueError("Actual context or concurrency differs from the protocol")
    runtime = {"engine_arguments": engine_args, "engine_config": str(config), "load_seconds": load_seconds,
               "model_implementation_paths": llm.apply_model(model_paths),
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
    save("warmup.json", {"prompt": warmup_prompt, "thinking_enabled": False, "seed": 0,
                         "text": warmup.text, "token_ids": list(warmup.token_ids), "finish_reason": warmup.finish_reason,
                         "seconds": time.monotonic() - started})
    if warmup.finish_reason != "stop" or not warmup.text.strip():
        raise ValueError("Warm-up did not produce a normally terminated response; no compatibility requests were sent")
    sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
                "max_tokens": 16384, "seed": 17, "stop_token_ids": frozen["stop_token_ids"], "skip_special_tokens": True}
    errors = 0
    with Path("/workspace/generations.jsonl").open("x", encoding="utf-8") as destination:
        for task in tasks:
            row = {key: task[key] for key in ("id", "family", "dataset_id", "prompt_sha256", "rendered_prompt_sha256", "prompt_tokens_sha256")}
            row.update(checkpoint=settings["checkpoint"], repo_id=settings["repo_id"], revision=settings["revision"],
                       seed=17, thinking_enabled=True, prompt_tokens=len(task["prompt_token_ids"]))
            started = time.monotonic()
            try:
                output = llm.generate([{"prompt_token_ids": task["prompt_token_ids"]}], SamplingParams(**sampling), use_tqdm=False)[0].outputs[0]
                tokens = list(output.token_ids)
                row.update(text=output.text, raw_decoded_text=tokenizer.decode(tokens, skip_special_tokens=False),
                           token_ids=tokens, finish_reason=output.finish_reason, stop_reason=output.stop_reason,
                           error=None, repeated_128_token_block=repeated_block(tokens))
            except Exception as exc:
                errors += 1
                row.update(text="", raw_decoded_text="", token_ids=[], finish_reason=None, stop_reason=None,
                           error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(), repeated_128_token_block=False)
            row["generation_seconds"] = time.monotonic() - started
            destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
            print(json.dumps({"id": row["id"], "tokens": len(row["token_ids"]), "finish_reason": row["finish_reason"], "error": row["error"]}), flush=True)
    save("results.json", {"metrics": {"requested": len(tasks), "recorded": len(tasks), "generation_errors": errors,
                                       "load_seconds": load_seconds},
                          "details": {"checkpoint": settings["checkpoint"], "split": "compatibility", "seed": 17,
                                      "sampling": sampling, "quality_comparison_performed": False}})


if __name__ == "__main__":
    main()
