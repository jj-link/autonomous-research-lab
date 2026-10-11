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

The recorded runner, bounded research coordinator, controlled evaluation suite, and local browser interface are implemented. Two real CUDA experiments have run on the allocated RTX PRO 6000.

The coordinator has also been exercised with scripted HTTP responses and real isolated executions: a failed implementation, a repair, analysis, an evidence-linked follow-up, and budget stopping. **That is software verification, not an autonomous language-model research demonstration.** A live cloud-model run and comparison of research policies remain unverified because the application has no cloud API credential configured.

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

## Start a research program

The director, selector, builder, and analyst use the cloud model configured in the research brief. There is no built-in model choice or automatic local-model fallback. The client expects an OpenAI-compatible `/chat/completions` endpoint and JSON-object message content; provider compatibility must be checked with the model you select.

Set the API key in the **Linux/WSL process environment** before starting the CLI or dashboard. For example, in Bash:

```bash
read -rsp 'Cloud API key: ' ARL_MODEL_API_KEY; echo
export ARL_MODEL_API_KEY
```

Only the environment variable's name goes in the brief. Do not put the key value in source code, manifests, or the browser form. Credentialed requests require HTTPS; redirects and hidden retries are disabled. A missing configured credential blocks immediately, before model requests or execution.

Save a brief such as this as `brief.json`, replacing `CLOUD_MODEL_ID` with your provider's model ID and choosing your own question and limits:

```json
{
  "goal": "How do summation order and precision affect error on cancellation-heavy inputs?",
  "scope": "Controlled synthetic inputs only. Use the supplied standard-library runtime without installing packages.",
  "model": {
    "base_url": "https://api.openai.com/v1",
    "model": "CLOUD_MODEL_ID",
    "api_key_env": "ARL_MODEL_API_KEY"
  },
  "policy": "candidate_comparison",
  "max_experiments": 2,
  "max_model_calls": 12,
  "max_output_tokens_per_call": 8192,
  "max_wall_seconds": 600,
  "max_repairs": 1,
  "resources": {"gpu": true, "wall_seconds": 120}
}
```

`resources` accepts `gpu` plus the runner's `wall_seconds`, `memory_bytes`, `output_bytes`, `log_bytes`, `pids`, and optional `cpus` limits. Unspecified memory/output/log/PID limits use the runner defaults: 2 GiB, 64 MiB, 8 MiB, and 64 respectively; CPU quota remains unset. GPU permission does not impose a CPU quota or guarantee a GPU-memory budget.

Optional `sources` is a list of public HTTPS URLs. Optional `data_files` maps relative filenames to approved UTF-8 text, not host paths; at most 16 files and 256 KiB total. These inputs become read-only files under `/experiment/code/data/`.

```sh
.venv-linux/bin/arl program create brief.json
.venv-linux/bin/arl program start PROGRAM_ID --wait
.venv-linux/bin/arl program show PROGRAM_ID
.venv-linux/bin/arl program report PROGRAM_ID
.venv-linux/bin/arl program pause PROGRAM_ID
.venv-linux/bin/arl program resume PROGRAM_ID --wait
.venv-linux/bin/arl program stop PROGRAM_ID
.venv-linux/bin/arl program recover
```

Replace `PROGRAM_ID` with the returned ID. Omit `--wait` to launch a detached coordinator. There is one initial start, not an approval prompt for every experiment.

### What the coordinator enforces

- Designs stay within the brief's resource permissions. Implementations cannot change the selected question or procedure or replace approved data.
- Adaptive follow-ups cite a prior analysis or execution and explain the connection. Questions retain parent links, status changes, and supporting evidence.
- Failed executions, repairs, and planning calls consume their allowances. `max_experiments` counts execution attempts, including failures, not just successful studies.
- Pause stops new scheduling and lets an active experiment finish. Stop requests termination of owned execution and keeps available outputs. Recovery reconciles recorded work instead of blindly repeating it.
- Interrupted or invalid model requests remain recorded. Explicit resume permits a new, separately charged request; unavailable usage is marked unknown.
- Reports separate recorded measurements from model-authored interpretations. Reaching a budget is a stopping reason, not a scientific finding.

Source retrieval supports public HTTPS text/HTML and Crossref metadata or abstracts, not arbitrary downloads or PDF reading. Excerpts and truncated responses are marked. The generated runtime is Python's standard library plus the optional CUDA Driver API; arbitrary dependencies are not installed.

Accounting retains call counts, known provider token usage, execution attempts, measured wall time, and available CPU/memory observations. Unknown runtime is charged at its reservation. It does not infer complete GPU utilization, energy, monetary billing, or a hard total-dollar limit.

## Browser interface

After configuring the runner and exporting the cloud credential in the same shell:

```sh
.venv-linux/bin/arl serve --host 127.0.0.1 --port 8767
```

Open <http://127.0.0.1:8767/>. The dashboard uses the same records and coordinator as the CLI. It supports brief submission, initial start, pause/resume/stop, question and decision history, sources, model request/response inspection, saved code and settings, measurements, logs, artifact downloads, and reports.

The server is loopback-only, with origin/host checks and a per-server request token. It is not a public or multi-user service. If you add a previously missing credential, restart the server from the environment containing it before resuming the blocked program.

## Controlled evaluation

Five deterministic observational tasks cover a genuine association, confounding, absence, insufficient evidence, and condition-dependent effects. Held-out labels and task identities stay outside the model context and experiment mounts. Numerical scoring checks identifiability, coefficient error, and held-out prediction error against independent references.

Save the brief's `model` object alone as `model.json`, then use:

```sh
.venv-linux/bin/arl evaluate --list
.venv-linux/bin/arl evaluate --model-config model.json --task confounded --prepare-only
.venv-linux/bin/arl evaluate --model-config model.json --task confounded --compare
.venv-linux/bin/arl evaluate --score EVALUATION_ID
```

`--prepare-only` creates records without model calls. `--compare` runs `fixed_plan`, `direct`, and `candidate_comparison` with the same data, model, tools, and budget ceilings. Planning and failed attempts remain charged. Comparison stops if a program is interrupted or blocked.

Dataset preparation and scoring have been exercised for all five cases. **Cloud-agent policy comparisons have not been run.** Numerical scores do not assess every scientific claim: unsupported causality, generality, evidence dependence, and whether follow-ups answer the question still require external review. Repeat across seeds before drawing comparative conclusions.

## Verification

From the repository in Linux/WSL, with the runtime image built:

```sh
ARL_TEST_IMAGE=localhost/arl-runtime:0.1 .venv-linux/bin/python \
  -W error::ResourceWarning -m unittest discover -s tests -v
```

Without `ARL_TEST_IMAGE`, real container integration checks are skipped. Model-client checks use deterministic HTTP fixtures, not local language models.

## Development plan

[PLAN.md](PLAN.md) describes the proposed system in detail: a worked research scenario, component responsibilities, decision process, persistent records, execution and recovery, evaluation, and implementation milestones.
