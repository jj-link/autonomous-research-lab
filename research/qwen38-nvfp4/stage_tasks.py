"""Stage pinned evaluation inputs locally; never publish problem statements or tests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from autonomous_research_lab.execution import atomic_json
from autonomous_research_lab.models import LabError, read_json, relative_path
from autonomous_research_lab.store import Store
from task_data import prepare_tasks

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / ".arl-cache/qwen38-tasks"
LCB_TERMS = {
    "source": "https://arxiv.org/html/2403.07974v1#A1.SS1",
    "publisher_statement": "The authors rely on research fair use, use the problems for academic purposes only, and do not train on them.",
    "project_use": "Local noncommercial evaluation only; no training or redistribution of problem statements, tests, or copied dataset files.",
    "limitation": "The Hub card says cc without a variant and the loader declares MIT. Neither establishes blanket redistribution rights for source-platform content; this record is not a legal guarantee.",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(item):
    os.environ["HF_XET_CACHE"] = str(DATA / ".xet")
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(endpoint="https://huggingface.co", token=False)
    info = api.dataset_info(item["repo_id"], revision=item["revision"], files_metadata=True)
    if info.sha != item["revision"]:
        raise LabError("Dataset metadata returned a different revision")
    destination = DATA / "raw" / item["label"] / item["revision"]
    if any(parent.is_symlink() for parent in (destination, *destination.parents)):
        raise LabError("Dataset staging paths cannot be symbolic links")
    siblings = {row.rfilename: row for row in info.siblings}
    records = []
    for name in ["README.md", *item["files"]]:
        relative_path(name, "dataset filename")
        sibling = siblings[name]
        print(f"Downloading and checking {item['label']}/{name} ({sibling.size} bytes)", flush=True)
        path = Path(hf_hub_download(item["repo_id"], name, repo_type="dataset", revision=info.sha,
                                   local_dir=destination, endpoint="https://huggingface.co", token=False))
        if path.is_symlink() or not path.is_file():
            raise LabError(f"Dataset input is not a regular file: {name}")
        sha = file_hash(path)
        if sibling.lfs is not None:
            valid = sha == sibling.lfs.sha256
        else:
            digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            valid = digest.hexdigest() == sibling.blob_id
        if not valid:
            raise LabError(f"Publisher hash mismatch: {item['label']}/{name}")
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": sha,
                        "publisher_git_blob": sibling.blob_id})
    return destination, {"repo_id": item["repo_id"], "revision": info.sha, "files": records}


def rows(directory, filenames):
    for filename in filenames:
        with (directory / filename).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def task_filename(task_id):
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest() + ".json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".arl"))
    args = parser.parse_args()
    store = Store(args.root)
    receipt_path = store.root / "studies/qwen38-nvfp4/tasks.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        for split in receipt["inputs"].values():
            for value in split.values():
                store.verify_input(value["id"])
        print(json.dumps(receipt, indent=2))
        return
    inventory = read_json(HERE / "inputs.lock.json")
    datasets = {item["label"]: item for item in inventory["datasets"]}
    coding_dir, coding_record = download_dataset(datasets["livecodebench"])
    ifeval_dir, ifeval_record = download_dataset(datasets["ifeval"])
    prepared = prepare_tasks(rows(coding_dir, datasets["livecodebench"]["files"]),
                             rows(ifeval_dir, datasets["ifeval"]["files"]))
    inputs = {}
    for split in ("compatibility", "primary"):
        public = DATA / "selected" / split / "public"
        private = DATA / "selected" / split / "private"
        if public.exists() or private.exists():
            raise LabError("Unregistered selected-task directories already exist; preserve and inspect them instead of silently replacing the selection")
        atomic_json(public / "tasks.json", {"split": split, "tasks": prepared["public"][split]})
        for row in prepared["private"][split]:
            atomic_json(private / task_filename(row["id"]), row["grading"])
        approved_public = store.register_input(public, f"qwen38-{split}-prompts")
        approved_private = store.register_input(private, f"qwen38-{split}-grading")
        inputs[split] = {"public": {key: approved_public[key] for key in ("id", "sha256", "bytes")},
                         "private": {key: approved_private[key] for key in ("id", "sha256", "bytes")}}
    receipt = {"schema_version": 1, "study": "qwen38-nvfp4", "datasets": [coding_record, ifeval_record],
               "scorers": inventory["scorers"], "usage": {"livecodebench": LCB_TERMS, "ifeval": "Apache-2.0"},
               "selection": prepared["selection"], "inputs": inputs,
               "template_rendering": "Pending separate frozen tokenizer rendering; no primary generation is authorized by this receipt."}
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
