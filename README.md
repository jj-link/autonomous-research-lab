# Autonomous Research Lab

Autonomous Research Lab is a project to build an AI system that conducts an ongoing research program: developing ideas, running experiments, learning from the results, and deciding what to investigate next.

The goal is to move beyond a single experiment or report. The system should use its accumulated findings to pursue further questions without someone directing every step.

## What it will do

- Read relevant sources and identify research questions worth testing.
- Develop hypotheses and design experiments.
- Write and execute experiment code, retaining the measurements and execution history.
- Interpret results, including negative findings, failed experiments, and unresolved questions.
- Choose follow-up experiments that explain a result, test a change to an idea, or explore a related direction.
- Produce reports grounded in the sources and experiments it actually used.

The initial focus is computational research involving software, data analysis, and machine learning.

## Intended use

A user will provide a research brief: the goal, permitted scope, available data and tools, compute resources, and budget.

The system will make routine research decisions within those limits. Users will be able to inspect its work, pause it, or stop it without needing to approve each experiment. Directions requiring resources or permissions outside the brief will be recorded rather than pursued without authorization.

The research history will retain the designs, code, results, and reasons for subsequent decisions, so a user can follow how the investigation developed.

## Current status

Recorded execution is implemented: immutable input snapshots, rootless Podman isolation, explicit GPU selection, bounded execution, stop/recovery, and inspectable results. Two real CUDA experiments have run on the allocated RTX PRO 6000. Agent-led research and the browser interface are still being developed.

## Record-store commands

Python 3.10+ is required. From this repository in Linux/WSL:

```sh
uv venv .venv-linux --python python3
uv pip install --python .venv-linux/bin/python -e .
.venv-linux/bin/arl submit path/to/experiment.json
.venv-linux/bin/arl list
.venv-linux/bin/arl show EXPERIMENT_ID
```

Replace `EXPERIMENT_ID` with the ID returned by `submit`. A manifest specifies `name`, a dedicated `source` directory, a relative Python `entrypoint`, and optional `question`, `procedure`, `settings`, `args`, `outputs`, `gpu`, and `limits`. Source paths are resolved relative to the manifest. Submission saves inputs; it does not execute code.

State defaults to `.arl/`; `arl --root DIRECTORY ...` selects another store. Source directories must not contain that store. Symlinks, special files, escaping paths, and source bundles over 256 MiB are rejected.

## Run an experiment

Execution requires Linux/WSL with local **rootless Podman**, cgroup v2, and delegated memory and PID controllers. It does not use Docker Desktop or change host resource settings.

From the repository inside Linux/WSL, after installing the CLI above:

```sh
mkdir -p .arl-cache/images
TMPDIR="$PWD/.arl-cache/images" podman build --tmpdir "$PWD/.arl-cache/images" -f Containerfile -t localhost/arl-runtime:0.1 .
.venv-linux/bin/arl configure --image localhost/arl-runtime:0.1 \
  --gpu GPU-6ac06ab6-ac30-f3ad-b246-064a8e6f1309 --gpu-mode wsl
.venv-linux/bin/arl doctor
.venv-linux/bin/arl submit examples/cuda/vector-add.json
.venv-linux/bin/arl run EXPERIMENT_ID --wait
.venv-linux/bin/arl result ATTEMPT_ID
.venv-linux/bin/arl logs ATTEMPT_ID
```

The UUID above identifies this project's allocated RTX PRO 6000, not a portable device identifier. Configure your allocated device explicitly on another worker. CPU-only experiments do not need `--gpu`. `examples/cuda/reduction.json` is a second, numerical-order study. Both examples use the CUDA Driver API directly; the image contains Python, not PyTorch or a CUDA compiler.

`arl stop ATTEMPT_ID --wait` terminates owned work and retains its records. `arl recover --wait` reconciles existing containers after interruption; it does not blindly repeat experiments. `arl list --attempts`, `arl show`, and `arl events` expose the history.

### Enforced boundaries and limits

- No experiment network access, inherited host credentials, writable code bundle, privileged container, or writable root filesystem.
- Hard memory, process-count, wall-time, and per-file limits. Aggregate output size is monitored, not a hard filesystem quota.
- CPU quota is optional, not inferred from allocating a GPU. An explicitly requested quota fails closed if the host does not delegate the CPU controller.
- WSL exposes a shared `/dev/dxg`. `CUDA_VISIBLE_DEVICES` selects the allocated GPU for cooperative code; **it is not a security boundary against malicious code accessing other adapters**. GPU memory is not independently hard-capped. Use a dedicated worker for hostile generated workloads.
- Native Linux CDI mode is implemented but has not been exercised here. The verified path is WSL with read-only driver mounts.
- Execution completion is not scientific success. Measurements remain separate from interpretations; missing expected outputs make an attempt fail.

## Development plan

[PLAN.md](PLAN.md) describes the proposed system in detail: a worked research scenario, component responsibilities, decision process, persistent records, execution and recovery, evaluation, and implementation milestones.
