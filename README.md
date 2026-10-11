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

The coordinator has also been exercised with scripted HTTP responses and real isolated executions: a failed implementation, a repair, analysis, an evidence-linked follow-up, and budget stopping. **That is software verification, not an autonomous language-model research demonstration.** Explicit Codex-account, Spark-cluster, and OpenAI-compatible providers are implemented. Spark discovery and a small real generation have passed; Codex's authentication lifecycle, protected storage, and streaming protocol have been exercised, but live Codex inference still requires this app's independent account sign-in. Live autonomous research and matched-policy comparisons remain unverified.

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

The director, selector, builder, and analyst use the provider and model recorded in the research brief. Provider selection is explicit; an error never switches to a local model or another service. Model calls do not run a Codex CLI agent or give the provider host tools.

### Codex with your ChatGPT account

No API key is required. Sign in once for the selected research store, using the same account you use for Codex:

```sh
.venv-linux/bin/arl provider login
.venv-linux/bin/arl provider status
.venv-linux/bin/arl provider models --provider openai-codex
```

Open the authorization URL printed by `login`; the command waits up to five minutes for the browser callback on `localhost:1455`. Finish or cancel another sign-in if that port is occupied. The dashboard offers the same sign-in, cancellation, and disconnect controls.

This app creates its own OAuth session; it does not import another project's or Codex CLI's rotating refresh grant. Windows and WSL protect the saved session with Windows CurrentUser DPAPI. Native Linux/macOS requires a supported, unlocked OS keyring and installation with the `secure-keyring` extra (`python -m pip install -e '.[secure-keyring]'`). There is no plaintext fallback. Storage and refresh locks are scoped to the selected store and platform: use the same `--root` and runtime for sign-in and research. Moving the store, switching between Windows and WSL, or changing Windows user requires a separate sign-in. DPAPI does not protect against compromise of the same Windows user or an administrator.

`arl provider logout` deletes this store/platform's saved session and cancels pending sign-in. It does not revoke your account globally or terminate a model request already in flight. Credentials are not placed in briefs, experiment environments, or browser storage. The OAuth and Responses protocol follows the [upstream Codex implementation](https://github.com/openai/codex/tree/main/codex-rs); account availability and usage limits still apply.

Codex controls output length. Set `max_output_tokens_per_call` to `null` or omit it; a numeric value is rejected rather than recorded as an unenforced cap. Call budgets, request timeouts, and the response-byte bound still apply. Missing token usage remains unknown.

### Spark cluster or another OpenAI-compatible server

Select **Spark cluster (local models)** to use the cluster explicitly. Its default base URL is `http://100.92.139.82:8888/v1`; it can be overridden in the brief or dashboard. Discover the currently served model rather than assuming a permanent model name:

```sh
.venv-linux/bin/arl provider models --provider spark
```

The verified catalog advertised `GLM-5.3-Flash-EXL3`. A Spark model assignment is `{"provider":"spark","model":"GLM-5.3-Flash-EXL3"}`. The current cluster endpoint is keyless and reachable over the private Tailscale network. Its availability is independent of the RTX PRO 6000 allocated to experiment execution.

For another server, use `{"provider":"openai-compatible","base_url":"https://api.openai.com/v1","model":"YOUR_MODEL_ID","api_key_env":"ARL_MODEL_API_KEY"}`. Spark and API providers use `/chat/completions` and must return JSON-object message content; their model catalogs use `/models`. Set a numeric output-token limit for these providers. Server support must be checked with the selected model.

If an API key is needed, set its value in the **Linux/WSL process environment** before starting the CLI or dashboard:

```bash
read -rsp 'Model API key: ' ARL_MODEL_API_KEY; echo
export ARL_MODEL_API_KEY
```

Only the environment variable's name goes in the brief. Do not put the key value in source code, manifests, or the browser form. Credentialed HTTP-provider requests require HTTPS; the keyless Spark HTTP endpoint cannot accept a bearer credential. Redirects and hidden inference retries are disabled. Missing configured credentials block before requests or execution.

### Submit the brief

Save a brief such as this as `brief.json`, replacing `CODEX_MODEL_ID` with an ID from your account's catalog and choosing your own question and limits:

```json
{
  "goal": "How do summation order and precision affect error on cancellation-heavy inputs?",
  "scope": "Controlled synthetic inputs only. Use the supplied standard-library runtime without installing packages.",
  "model": {
    "provider": "openai-codex",
    "model": "CODEX_MODEL_ID"
  },
  "policy": "candidate_comparison",
  "max_experiments": 2,
  "max_model_calls": 12,
  "max_output_tokens_per_call": null,
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

After configuring the runner, launch the dashboard. Codex sign-in can happen in the browser; only environment-based API credentials must be exported before server startup.

```sh
.venv-linux/bin/arl serve --host 127.0.0.1 --port 8767
```

Open <http://127.0.0.1:8767/>. Select a provider, sign in if using Codex, and use **Discover models** to load that provider's actual catalog. Provider changes affect the draft, not existing programs. The dashboard uses the same records and coordinator as the CLI. It supports brief submission, initial start, pause/resume/stop, question and decision history, sources, model request/response inspection, saved code and settings, measurements, logs, artifact downloads, and reports.

The server is loopback-only, with origin/host checks and a per-server request token. It is not a public or multi-user service. After adding an environment-based API credential, restart the server from that environment before resuming the blocked program. Codex sign-in does not require a server restart.

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
