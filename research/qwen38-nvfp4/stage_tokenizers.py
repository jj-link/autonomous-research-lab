"""Stage small pinned tokenizer subsets without modifying approved model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from autonomous_research_lab.execution import atomic_json
from autonomous_research_lab.models import LabError, read_json
from autonomous_research_lab.store import Store
from stage_tasks import file_hash

NAMES = {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "merges.txt", "vocab.json",
         "special_tokens_map.json", "added_tokens.json", "tokenizer.model", "config.json", "LICENSE", "LICENSE.txt"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".arl"))
    args = parser.parse_args()
    store = Store(args.root)
    receipt_path = store.root / "studies/qwen38-nvfp4/tokenizers.json"
    inventory = read_json(Path(__file__).with_name("inputs.lock.json"))
    mounted = json.loads(subprocess.check_output(
        ["findmnt", "--mountpoint", "/mnt/d", "--json", "--output", "SOURCE,TARGET"], text=True))["filesystems"]
    if len(mounted) != 1 or mounted[0]["source"] != inventory["model_storage"]["mount_source"]:
        raise LabError("The allocated model HDD is not mounted; no alternate location will be used")
    model_root = Path(inventory["model_storage"]["wsl_root"])
    os.environ["HF_XET_CACHE"] = str(model_root / ".xet")
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(endpoint="https://huggingface.co", token=False)
    receipt = read_json(receipt_path) if receipt_path.exists() else {}
    for item in inventory["checkpoints"]:
        destination = model_root / ".study-tokenizers/qwen38-nvfp4" / item["label"] / item["revision"]
        if any(parent.is_symlink() for parent in (destination, *destination.parents)):
            raise LabError("Tokenizer paths cannot contain symbolic links")
        if item["label"] in receipt:
            previous = receipt[item["label"]]
            if (previous["repo_id"], previous["revision"]) != (item["repo_id"], item["revision"]):
                raise LabError("Existing tokenizer receipt pins a different revision")
            approved = store.verify_input(previous["input_id"])
            if approved["source"] != str(destination.resolve()):
                raise LabError("Existing tokenizer input points to a different directory")
            continue
        info = api.model_info(item["repo_id"], revision=item["revision"], files_metadata=True)
        if info.sha != item["revision"]:
            raise LabError("Tokenizer metadata returned a different revision")
        selected = [row for row in info.siblings if row.rfilename in NAMES]
        if not {"tokenizer.json", "tokenizer_config.json"}.issubset({row.rfilename for row in selected}):
            raise LabError(f"Missing native tokenizer files: {item['label']}")
        expected = {}
        for row in selected:
            path = Path(hf_hub_download(item["repo_id"], row.rfilename, revision=info.sha,
                                       local_dir=destination, endpoint="https://huggingface.co", token=False))
            if path.is_symlink() or not path.is_file():
                raise LabError("Tokenizer source must be an ordinary file")
            sha = file_hash(path)
            if row.lfs is not None:
                valid = sha == row.lfs.sha256
            else:
                if path.stat().st_size > 32 * 1024**2:
                    raise LabError("Unexpectedly large non-LFS tokenizer asset")
                data = path.read_bytes()
                valid = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == row.blob_id
            if not valid:
                raise LabError(f"Publisher checksum mismatch: {item['label']}/{row.rfilename}")
            expected[row.rfilename] = sha
        approved = store.register_input(destination, f"qwen38-{item['label']}-tokenizer", expected_hashes=expected)
        receipt[item["label"]] = {"repo_id": item["repo_id"], "revision": info.sha,
                                  "input_id": approved["id"], "sha256": approved["sha256"], "files": expected}
        atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
