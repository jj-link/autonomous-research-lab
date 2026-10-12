"""Host-side staging for this study; never invoked by generated experiment code."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from autonomous_research_lab.execution import atomic_json
from autonomous_research_lab.models import LabError, read_json, relative_path
from autonomous_research_lab.store import Store


def main():
    inventory = read_json(Path(__file__).with_name("inputs.lock.json"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", choices=[item["label"] for item in inventory["checkpoints"]])
    parser.add_argument("--root", type=Path, default=Path(".arl"))
    args = parser.parse_args()
    checkpoint = next(item for item in inventory["checkpoints"] if item["label"] == args.label)
    storage = inventory["model_storage"]
    mounted = json.loads(subprocess.check_output(
        ["findmnt", "--mountpoint", "/mnt/d", "--json", "--output", "SOURCE,TARGET"], text=True
    ))["filesystems"]
    if len(mounted) != 1 or mounted[0]["source"] != storage["mount_source"]:
        raise LabError("The allocated model-storage drive is not mounted; no alternate destination will be used")
    model_root = Path(storage["wsl_root"])
    destination = model_root / relative_path(checkpoint["repo_id"], "repository") / relative_path(checkpoint["revision"], "revision")
    if not destination.resolve().is_relative_to(model_root.resolve()):
        raise LabError("Checkpoint destination escapes the model store")
    for parent in (destination, *destination.parents):
        if parent.is_symlink():
            raise LabError("Checkpoint paths cannot contain symlinks")
    store = Store(args.root)
    receipt_path = store.root / "studies/qwen38-nvfp4/staged.json"
    receipts = read_json(receipt_path) if receipt_path.exists() else {}
    if args.label in receipts:
        previous = receipts[args.label]
        if (previous["repo_id"], previous["revision"]) != (checkpoint["repo_id"], checkpoint["revision"]):
            raise LabError("The existing study receipt pins a different checkpoint; it was not replaced")
        approved = store.verify_input(previous["input_id"])
        if approved["source"] != str(destination.resolve()):
            raise LabError("The registered input does not match this study's storage destination")
        print(json.dumps(previous, indent=2))
        return

    # The download client is optional and is never installed into an experiment.
    os.environ["HF_XET_CACHE"] = str(model_root / ".xet")
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(endpoint="https://huggingface.co", token=False)
    info = api.model_info(checkpoint["repo_id"], revision=checkpoint["revision"], files_metadata=True)
    if info.sha != checkpoint["revision"]:
        raise LabError("The Hub returned a different revision from the study lock")
    for sibling in info.siblings:
        relative_path(sibling.rfilename, "checkpoint file")
    print(f"Downloading {checkpoint['repo_id']} at {info.sha} to {destination}", flush=True)
    snapshot_download(repo_id=checkpoint["repo_id"], revision=info.sha, local_dir=destination,
                      endpoint="https://huggingface.co", token=False, max_workers=2)

    expected = {}
    for sibling in info.siblings:
        path = destination / sibling.rfilename
        if not path.is_file() or path.is_symlink():
            raise LabError(f"Downloaded checkpoint file is unavailable or is a link: {sibling.rfilename}")
        if sibling.lfs is not None:
            expected[sibling.rfilename] = sibling.lfs.sha256
        else:
            # Non-LFS files have Git blob IDs rather than raw-file SHA-256 hashes.
            if path.stat().st_size > 16 * 1024**2:
                raise LabError("Unexpected large non-LFS file; it was not approved")
            content = path.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if blob != sibling.blob_id:
                raise LabError(f"Git blob checksum mismatch: {sibling.rfilename}")
            expected[sibling.rfilename] = hashlib.sha256(content).hexdigest()
    print("Verifying file hashes and registering the read-only checkpoint input", flush=True)
    approved = store.register_input(destination, f"qwen38-{args.label}", expected_hashes=expected)
    receipt = {"repo_id": checkpoint["repo_id"], "revision": info.sha, "input_id": approved["id"],
               "sha256": approved["sha256"], "bytes": approved["bytes"], "source": approved["source"]}
    receipts[args.label] = receipt
    atomic_json(receipt_path, receipts)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
