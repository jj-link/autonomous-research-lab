"""Freeze held-out coding prompts with the original study's tokenizer identity."""
import hashlib
import json
from pathlib import Path

from freeze import digest, identity, tokenizer_compatibility


def main():
    from transformers import AutoTokenizer

    settings = json.loads(Path("/experiment/settings.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained("/inputs/tokenizer", local_files_only=True, trust_remote_code=False)
    actual = identity(tokenizer, "/inputs/tokenizer")
    if not tokenizer_compatibility(settings["tokenizer_identity"], actual)["compatible"]:
        raise ValueError("The follow-up tokenizer differs from the original study")
    template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    stops = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    if template_hash != settings["chat_template_sha256"] or stops != settings["stop_token_ids"]:
        raise ValueError("The original template or stop tokens changed")
    public = json.loads(Path("/inputs/public/tasks.json").read_text())
    rendered = []
    for task in public["tasks"]:
        if task["family"] != "coding" or digest(task["messages"]) != task["prompt_sha256"]:
            raise ValueError("The selected coding prompt differs from its receipt")
        prompt = tokenizer.apply_chat_template(task["messages"], tokenize=False, add_generation_prompt=True,
                                               enable_thinking=True, reasoning_effort="xhigh")
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) + settings["max_output_tokens"] > settings["max_model_len"]:
            raise ValueError(f"Selected task does not fit without truncation: {task['id']}")
        rendered.append({key: task[key] for key in ("id", "family", "dataset_id", "prompt_sha256")} | {
            "prompt": prompt, "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_token_ids": tokens, "prompt_tokens_sha256": digest(tokens)})
    if len(rendered) != settings["items"] or len({row["id"] for row in rendered}) != settings["items"]:
        raise ValueError("The held-out request inventory is incomplete or duplicated")
    payload = {"split": "followup", "chat_template_sha256": template_hash, "stop_token_ids": stops,
               "thinking_enabled": True, "reasoning_effort": "xhigh", "tasks": rendered}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    Path("/workspace/rendered.json").write_bytes(encoded)
    Path("/workspace/results.json").write_text(json.dumps({
        "metrics": {"items": len(rendered), "max_prompt_tokens": max(len(row["prompt_token_ids"]) for row in rendered)},
        "details": {"sha256": hashlib.sha256(encoded).hexdigest(), "tokenizer": actual,
                    "chat_template_sha256": template_hash, "stop_token_ids": stops}}))


if __name__ == "__main__":
    main()
