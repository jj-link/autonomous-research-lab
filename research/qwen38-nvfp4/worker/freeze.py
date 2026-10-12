"""Validate all tokenizer identities and freeze rendered prompts without generation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path



def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def identity(tokenizer, directory):
    vocabulary = tokenizer.get_vocab()
    config = json.loads((Path(directory) / "config.json").read_text())["text_config"]
    return {
        "vocabulary_sha256": digest(vocabulary), "vocabulary_size": len(vocabulary),
        "added_token_definitions": {str(key): {"content": value.content, "special": value.special}
                                    for key, value in tokenizer.added_tokens_decoder.items()},
        "model_token_config": {key: config.get(key) for key in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id")},
        "frontend_defaults": {
            "special_tokens": dict(sorted(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids))),
            "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
    }


def tokenizer_compatibility(reference, candidate):
    """Token identity must match; publisher frontend defaults are normalized."""
    checks = {key: candidate[key] == reference[key] for key in (
        "vocabulary_sha256", "vocabulary_size", "added_token_definitions", "model_token_config"
    )}
    return {"compatible": all(checks.values()), "checks": checks,
            "frontend_defaults_differ": candidate["frontend_defaults"] != reference["frontend_defaults"]}


def main():
    from transformers import AutoTokenizer
    settings = json.loads(Path("/experiment/settings.json").read_text())
    reference = AutoTokenizer.from_pretrained("/inputs/tokenizer_bf16", local_files_only=True, trust_remote_code=False)
    expected = identity(reference, "/inputs/tokenizer_bf16")
    identities = {}
    for label in settings["checkpoints"]:
        candidate = AutoTokenizer.from_pretrained(f"/inputs/tokenizer_{label}", local_files_only=True, trust_remote_code=False)
        identities[label] = identity(candidate, f"/inputs/tokenizer_{label}")
        compatibility = tokenizer_compatibility(expected, identities[label])
        identities[label]["compatibility"] = compatibility
        if not compatibility["compatible"]:
            raise ValueError(f"Tokenizer vocabulary, backend special-token, or model token-config mismatch: {label}: {compatibility['checks']}")
        del candidate
    stops = reference.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    if stops != [248046, 248044]:
        raise ValueError("Original model's pinned stop-token identities changed")
    template_hash = hashlib.sha256(reference.chat_template.encode("utf-8")).hexdigest()
    summary = {"tokenizers": identities, "chat_template_sha256": template_hash, "stop_token_ids": stops,
               "thinking_enabled": True, "reasoning_effort": "xhigh", "splits": {}}
    for split in ("compatibility", "primary"):
        public = json.loads(Path(f"/inputs/public_{split}/tasks.json").read_text(encoding="utf-8"))
        rendered = []
        for task in public["tasks"]:
            if digest(task["messages"]) != task["prompt_sha256"]:
                raise ValueError(f"Public message checksum changed: {task['id']}")
            prompt = reference.apply_chat_template(task["messages"], tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=True, reasoning_effort="xhigh")
            tokens = reference.encode(prompt, add_special_tokens=False)
            if len(tokens) + 16384 > 32768:
                raise ValueError(f"Selected task cannot fit the common context without truncation: {task['id']} ({len(tokens)} tokens)")
            rendered.append({key: task[key] for key in ("id", "family", "dataset_id", "prompt_sha256")} | {
                "prompt": prompt, "rendered_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_token_ids": tokens, "prompt_tokens_sha256": digest(tokens)})
        payload = {"split": split, "chat_template_sha256": template_hash, "stop_token_ids": stops,
                   "thinking_enabled": True, "reasoning_effort": "xhigh", "tasks": rendered}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        Path(f"/workspace/rendered_{split}.json").write_bytes(encoded)
        summary["splits"][split] = {"sha256": hashlib.sha256(encoded).hexdigest(), "items": len(rendered),
                                     "max_prompt_tokens": max(len(task["prompt_token_ids"]) for task in rendered)}
    result = {"metrics": {"tokenizers_compatible": len(identities),
                           "compatibility_items": summary["splits"]["compatibility"]["items"],
                           "primary_items": summary["splits"]["primary"]["items"]},
              "details": summary}
    Path("/workspace/results.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
