"""Build the study's CPU grading image from pinned source and dependency inputs."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from urllib.request import Request, urlopen
import zipfile

from autonomous_research_lab.execution import atomic_json
from autonomous_research_lab.models import LabError, read_json

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CONTEXT = PROJECT / ".arl-cache/qwen38-grader-build"
NLTK_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
NLTK_BLOBS = {"punkt": "da7ffbd1e6fd6cc5c2f6879c2d4da23c7691944c",
              "punkt_tab": "5e5ff6137d5ee6025e400d1c3a7b21914c48b635"}
IMAGE = "localhost/arl-grading:qwen38-v1"


def fetch(url):
    request = Request(url, headers={"User-Agent": "autonomous-research-lab-study-provisioner"})
    with urlopen(request, timeout=120) as response:
        data = response.read(32 * 1024**2 + 1)
    if len(data) > 32 * 1024**2:
        raise LabError(f"Unexpectedly large grader source: {url}")
    return data


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    if any(parent.is_symlink() for parent in (CONTEXT, *CONTEXT.parents)):
        raise LabError("Grader build context cannot contain symbolic links")
    inventory = read_json(HERE / "inputs.lock.json")
    sources = {item["label"]: item for item in inventory["scorers"]}
    modules = {
        "ifeval": ["instruction_following_eval/evaluation_lib.py", "instruction_following_eval/instructions.py",
                   "instruction_following_eval/instructions_registry.py", "instruction_following_eval/instructions_util.py", "LICENSE"],
        "livecodebench": ["lcb_runner/evaluation/testing_util.py", "LICENSE"],
    }
    records = []
    for label, names in modules.items():
        source = sources[label]
        repository = source["repository"].removeprefix("https://github.com/")
        for name in names:
            url = f"https://raw.githubusercontent.com/{repository}/{source['revision']}/{name}"
            destination = name if name != "LICENSE" else f"licenses/{label}.txt"
            record = save(CONTEXT / "upstream" / destination, fetch(url))
            records.append({"url": url, "image_path": f"/opt/scorers/{destination}", **record})
    for name, blob in NLTK_BLOBS.items():
        url = f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_REVISION}/packages/tokenizers/{name}.zip"
        data = fetch(url)
        if hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != blob:
            raise LabError(f"Pinned NLTK {name} asset checksum differs")
        extracted = []
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                selected = (member.filename in {"punkt/english.pickle", "punkt/PY3/english.pickle", f"{name}/README"}
                            or member.filename.startswith("punkt_tab/english/"))
                if not selected or member.is_dir():
                    continue
                if path.is_absolute() or ".." in path.parts or member.file_size > 8 * 1024**2:
                    raise LabError("Unsafe NLTK archive member")
                record = save(CONTEXT / "nltk_data/tokenizers" / str(path), archive.read(member))
                extracted.append({"path": str(path), **record})
        if not extracted:
            raise LabError(f"Missing English sentence tokenizer assets: {name}")
        records.append({"url": url, "publisher_git_blob": blob, "sha256": hashlib.sha256(data).hexdigest(),
                        "extracted": extracted})
    requirements = HERE / "requirements-grading.txt"
    shutil.copyfile(requirements, CONTEXT / requirements.name)
    build_inputs = {"scorers": inventory["scorers"], "nltk_data_revision": NLTK_REVISION,
                    "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
                    "files": records}
    atomic_json(CONTEXT / "build-inputs.json", build_inputs)
    subprocess.run(["podman", "build", "--pull=never", "--network=host", "--http-proxy=false",
                    "--tag", IMAGE, "--file", str(PROJECT / "Containerfile.grading"), str(CONTEXT)], check=True)
    image = json.loads(subprocess.check_output(["podman", "image", "inspect", IMAGE], text=True))[0]
    receipt = {"image_id": image["Id"], "image_digest": image.get("Digest"), "tag": IMAGE,
               "build_inputs": build_inputs}
    atomic_json(CONTEXT / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
