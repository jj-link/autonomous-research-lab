# Qwen3.8-27B NVFP4: correctness and reliability across released builds

## Question and intended result

Do the Unsloth, NVIDIA, RadixArk, and QUASAR NVFP4 releases of Qwen3.8-27B produce repeatable differences in answer quality under the same inference conditions? How much does each differ from the original BF16 model, and do aggregate scores conceal particular failure modes?

The result should help choose a checkpoint for a workload, with uncertainty and limitations attached. It must not declare a universal winner from a small score difference or describe output similarity to BF16 as proof of correctness. BF16 is a reference, not ground truth; task answers and executable tests supply ground truth.

This is a comparison of published builds. Their retained precision, calibration, and additional training differ. QUASAR uses quantization-aware training. A difference between releases therefore does not, by itself, establish which quantization technique caused it.

## Status and scope

This document is the research design, not a benchmark result or an executable `arl program create` manifest. The NVIDIA checkpoint has been downloaded, verified, and used for the readiness probe recorded below. The other four checkpoints are being staged. No primary evaluation has run. The input revisions in [inputs.lock.json](inputs.lock.json) were obtained from the publishers' metadata on 2026-10-11.

The first comparison is text-only: coding correctness and verifiable instruction following, with completion failures recorded for both. Vision, throughput rankings, retraining, and broad comparisons against other model families are outside this first comparison. Long-context and multi-turn behavior are eligible follow-ups, not properties the initial single-turn evaluation will establish.

## Allocated resources and storage

- Experimental models: the allocated RTX PRO 6000, `GPU-6ac06ab6-ac30-f3ad-b246-064a8e6f1309`. Load one checkpoint at a time.
- Research agents: Codex, using `gpt-6-astra`, confirmed in this account's advertised catalog. No automatic Spark or other local-agent fallback. Evaluating local quantized models does not change the research agents' provider.
- Checkpoints: `D:\models\huggingface` on the existing exFAT HDD, available to `workbench` in NVIDIA-Workbench WSL at `/mnt/d/models/huggingface`.
- Layout: `<model-root>/<owner>/<repository>/<full-commit-sha>/`. Use ordinary files in explicit download directories, not a layout requiring symlinks or hard links. Preserve existing drive contents; do not reformat it.
- Keep research records, credentials, environments, and container layers in their existing locations. Do not move the research database or credentials onto this model-storage drive.
- Cold loading and warm-up are separate from measured inference. Do not count HDD loading time as model generation performance. Do not silently enable CPU/disk offload to make a checkpoint fit.
- GPU allocation does not imply a CPU quota or a host-RAM limit. Do not apply the current generic worker defaults of 2 GiB RAM and 120 seconds to a model-serving job. Determine job reservations from the actual runtime's measured requirements before launching the comparison.
- At setup, WSL reported 33,787,092,992 bytes of host RAM. GPU memory is a separate allocation. Check the loader's peak host-memory use and shard-at-a-time loading before attempting BF16; do not assume the full checkpoint can be materialized in host RAM or change WSL's memory allocation without authorization.

The drive was mounted without changing `/etc/fstab` or restarting WSL. WSL's automount configuration was left unchanged. Before any download, require an actual `/mnt/d` mount whose source is `D:`; directory existence alone is insufficient. If it is absent, stop rather than writing models into the underlying WSL filesystem. The existing drive can be mounted with:

```sh
sudo mount -t drvfs D: /mnt/d -o uid=1000,gid=1000,noatime
findmnt --mountpoint /mnt/d --output TARGET,SOURCE,FSTYPE,OPTIONS
```

On this workstation, `workbench` successfully created, flushed, and reread a probe under the model root; Windows read the same file through `D:`. The probe was then removed. This verifies current access, not persistence after a restart or container access to model files.

Rootless Podman must also see the drive. A namespace created before the HDD was mounted can retain an older mount view even when host reads succeed. Check `podman unshare findmnt --mountpoint /mnt/d --output SOURCE,TARGET`. If the drive is missing there, first check `podman ps`: with **no running containers**, `podman system migrate` refreshes the inactive namespace, after which the mount check must show `D:`. Do not run that command against unrelated active workloads; [Podman's documentation](https://docs.podman.io/en/v5.4.0/markdown/podman-system-migrate.1.html) states that it stops running containers as well as the pause process. The worker checks directory identity across the host and rootless namespace and refuses mismatches; it never performs this refresh automatically.

The study's host-side staging command downloads one selected revision and registers its content in the current `.arl` store:

```sh
uv pip install --python .venv-linux/bin/python -e '.[model-download]'
.venv-linux/bin/python research/qwen38-nvfp4/stage_checkpoint.py nvidia
```

Accepted labels are `bf16`, `unsloth`, `nvidia`, `radixark`, and `quasar`. The script requires the actual `D:` mount, uses public unauthenticated Hub requests and ordinary local files, verifies Git blob IDs and LFS SHA-256 hashes, and records the registered input ID in `.arl/studies/qwen38-nvfp4/staged.json`. It does not install packages into experiments or give generated code download access. A repeat invocation verifies the existing registration rather than replacing it. Use `--root` when preparing a different research store; approvals are store-specific.

## Checkpoints and provenance

The full immutable revisions and metadata sources are in [inputs.lock.json](inputs.lock.json). Do not substitute a later `main` revision or a similarly named repository during execution.

| Build | Role and documented distinction |
|---|---|
| [Qwen BF16](https://huggingface.co/Qwen/Qwen3.8-27B) | Original-model reference |
| [Unsloth NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | Mixed precision; its configuration retains the output head, attention projections, and selected MLP layers in FP8 |
| [NVIDIA NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | Model Optimizer, Local-Hessian calibration on Nemotron data; NVFP4 MLP/output head with FP8 attention |
| [RadixArk NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) | Model Optimizer calibration on CNN/DailyMail; NVFP4 output-head variant |
| [QUASAR NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4) | Quantization-aware training; quantizes transformer linear layers while excluding the output head, vision, and MTP from that conversion |

These descriptions come from model cards and configuration metadata, not inspection of downloaded tensors. The inspected derivative metadata names the base repository but does not establish the exact source-weight revision used for every conversion. Report that limitation; do not assert byte-identical pre-quantization weights.

During staging, record file sizes and SHA-256 hashes for weights, tokenizer, template, generation settings, and quantization configuration. Identify copies using tensor/file identities rather than repository names. Preserve original configurations even where the experiment deliberately overrides a setting. Additional ablated, uncensored, fine-tuned, or repackaged checkpoints do not enter this initial cohort silently.

## What published evidence establishes

The [existing community comparison](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/192) reports differences in distributional fidelity and small, task-dependent accuracy differences. Its deployment used a DGX Spark, vLLM, FP8 KV cache, and MTP. It also reports occasional looping or malformed output; later discussion suggests sensitivity to harness and generation settings. It does not isolate quantization as the cause of those failures.

The [QUASAR card](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4) and [NVIDIA card](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) report their own evaluations. Scores from different protocols are not a head-to-head ranking. These are external reports, not findings from this lab. The initial study is a controlled replication/extension, not a claim that no comparison exists.

## Initial comparison protocol

### Tasks and selection

Use two independently reported task families; do not combine them into an arbitrary overall score:

1. **Coding:** 128 unique problems from the pinned LiveCodeBench `code_generation_lite`, `release_v6`. Score Python solutions using the pinned official evaluator and private tests. Keep tests and expected outputs outside the model's prompt and filesystem. The model receives the question, starter code, and public examples only.
2. **Instruction following:** 128 unique prompts from the pinned `google/IFEval` input file. Use the pinned official strict prompt-level metric as primary. Instruction-level and loose metrics are secondary and must be labeled as such.

Use outcome-independent selection: sort coding IDs (`platform/question_id`) and IFEval keys by the hex SHA-256 of the UTF-8 string `arl-qwen38-nvfp4-v1/<family>/<id>`, breaking ties by the ID. Reserve the first eight in each family for compatibility checks; use the next 128 for the primary comparison. Freeze the selected IDs, exact rendered prompts, dataset hashes, and scorer versions before viewing primary outputs. No replacing difficult or failing questions after seeing a model's answer. If the eligible pool is insufficient, record a protocol amendment before generation rather than silently reducing the sample.

The dataset and scorer revisions are pinned in the input inventory. Both datasets predate this model release; training exposure is unknown, so this is not a contamination-free or novel-problem evaluation. [LiveCodeBench's authors](https://arxiv.org/html/2403.07974v1#A1.SS1) describe collecting publicly visible competition content for academic research under fair use, without training on it. Its Hub card's unspecified `cc` label and loader's MIT label do not establish blanket rights to the source-platform content. This study keeps the dataset local for noncommercial evaluation: no training or redistribution of copied problem statements or tests. Publisher rationale is not a legal guarantee or a license for broader use. IFEval's dataset card declares Apache-2.0.

### Controlled serving and generation

- Use one validated inference-engine build and dependency set for all five checkpoints, recording its immutable image digest, CUDA/driver versions, GPU identity, attention backend, and actual quantization kernel paths. Different weight formats can select different kernels even in the same engine; retain that distinction in interpretation.
- Use the pinned vLLM build for all five checkpoints. The NVIDIA readiness probe succeeded; common-settings compatibility across the full cohort remains to be checked. Do not quietly use a different engine or version for a troublesome candidate.
- Start with a 32,768-token context window, explicit BF16 KV cache, one concurrent request, MTP/speculative decoding disabled, and prefix caching disabled. No engine-selected per-checkpoint cache precision or offload fallback.
- Use the pinned original model's tokenizer and chat template only after verifying the complete token-to-ID map, added-token contents/special flags, and model token configuration for every candidate. Publisher frontend defaults (including padding choice) are recorded separately and normalized by this common tokenizer; they are not token-ID identity. Render identical inputs, with thinking enabled and `reasoning_effort=xhigh`. A vocabulary or backend special-token mismatch blocks the controlled comparison; do not disguise it as a weight-quality difference.
- Fix `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0`, `presence_penalty=0`, and `repetition_penalty=1`. Allow 16,384 generated tokens per response, including reasoning, for every candidate and both task families. Inputs that cannot fit without truncation must be identified before evaluating any candidate.
- Generate one answer per item for each of three shared seeds: 17, 29, and 43. This measures repeated pass@1 behavior, not best-of-three or a repair success rate. Identical seeds do not guarantee equivalent random trajectories across different checkpoints.
- Run seed blocks with checkpoint order rotated from the inventory order by the seed-block index. Warm each loaded checkpoint before evaluation. Save each raw result before scoring; do not repeat a failed request invisibly.
- The compatibility worker uses eager execution, chunked prefill with at most 4,096 batched tokens, GPU memory utilization `0.90`, and the safetensors lazy loader. Both CPU-offload mechanisms are explicitly disabled. Retain the loaded model's implementation/quantization class paths and runtime logs; shared engine versions do not imply identical kernels.
- Compatibility checks use seed 17 on the sixteen reserved items. A separate, non-thinking `Reply with READY.` request warms the engine; save it separately and exclude it from task outcomes. The worker cannot execute the primary split. Its per-job reservations are 26 GiB host RAM, 512 PIDs, 8 GiB output, 32 MiB logs, and six hours; no CPU quota or host memory configuration change.
- Use the original model's stop-token IDs, `[248046, 248044]`. Save generated token IDs and the unfiltered decoded text as well as the response text. Response decoding skips tokenizer-declared special tokens; the original tokenizer marks `<think>` and `</think>` as non-special, so reasoning boundaries remain available for extraction.

This initial design has 256 primary items and three samples per item per checkpoint. It is a bounded sensitivity study, not guaranteed to resolve sub-percentage-point differences. Compatibility items are excluded from the primary estimates.

### Outcomes and failures

Primary outcomes are coding pass rate and strict IFEval prompt-level compliance, each reported separately. For every item retain prompt identity, seed, checkpoint revision, raw reasoning/final output, finish reason, token usage when available, latency, scorer output, and any runtime failure. Keep model-generated code inside an isolated, network-disabled scoring process with no credentials or model-store write access.

Do not grade reasoning text as the final answer. Fix final-answer/code extraction before primary evaluation; do not repair answers by hand or give one checkpoint extra feedback. Empty answers, timeouts, generation errors, and unfinished/truncated outputs remain in the requested-attempt denominator and have explicit failure categories. Report task correctness separately from normal-completion rate: a truncated response is not necessarily the same event as a wrong answer, but it cannot be silently discarded.

The frozen extraction rule takes text after the last `</think>`, allowing the opening `<think>` to have been supplied by the prompt. Missing closure, reopened reasoning, and empty finals are explicit failures. Preserve final-answer whitespace. For coding, choose the last completed line-oriented Python-tagged or untagged backtick/tilde fence; a later eligible unfinished fence fails rather than falling back. If there are no fences, accept the whole final only if it parses as Python with an executable statement. Do not strip prose, repair syntax, or execute code during extraction. Extractable truncated answers are still scored, with truncation reported separately.

Grading uses a separate, pinned CPU-only image and an internal grading record store; it does not change the signed-in dashboard's workspace or inference runner. Each offline container receives one response and that task's private tests, never model weights or credentials. IFEval uses the official strict/loose APIs with evaluator seed 0, including deterministic language detection and pinned offline English sentence-tokenizer assets. LiveCodeBench uses its unmodified official `run_test` in a disposable subprocess: six seconds per test and `7 × test_count + 5` seconds for the whole call. Worker reservations add startup/collection margin, 4 GiB RAM, and 128 PIDs.

LiveCodeBench's `reliability_guard` is **not** a sandbox. Candidate code and expected tests share an interpreter, so official scores are not protected against adversarial scorer introspection or tampering. The enclosing network-disabled container protects the host; do not claim it proves adversarial scoring integrity. Preserve scorer exceptions, child exits, timeouts, and incomplete result inventories as unavailable correctness evidence, not successful tests.

For repetition, record whether any output contains four immediately consecutive copies of an identical block of 128 token IDs. This is a diagnostic flag, not a correctness rule: some instructions legitimately request repetition. Preserve flagged examples for blind inspection. Do not equate exceeding a token allowance with proven reasoning loops, or infer a numerical-instability cause from unusual text alone.

Infrastructure interruptions require a visible event and a predeclared matched-block rerun; preserve the interrupted attempts and costs. Show sensitivity with and without externally caused failures rather than charging an unrelated outage to only one checkpoint or silently erasing it.

### Analysis and decision rules

For each task family, average the three binary outcomes within each unique item, then compare candidates on the same items. Report absolute rates, all pairwise differences, and paired bootstrap confidence intervals resampling whole items with their three samples together. Do not count repeated samples as independent problems.

Use 10,000 bootstrap resamples with analysis seed 20261011. For each family, report nominal 95% intervals and conservative 99.5% intervals for the ten pairwise contrasts among five builds; the latter provide a within-family multiplicity adjustment. Keep the two families' conclusions separate. Predeclare two percentage points as the practical-effect threshold: an adjusted interval entirely above +2 or below -2 supports a practically meaningful difference; one excluding zero but crossing that threshold supports a detectable difference of uncertain practical importance. An interval overlapping zero is inconclusive, not evidence of equality. Claim practical equivalence only if the entire adjusted interval lies inside [-2, +2]. Apply the same caution to reliability rates, especially rare failures.

Boundary guard: a degenerate bootstrap interval is not evidence of equivalence. If fewer than ten independent items show any paired disagreement for a contrast, do not declare equivalence from its bootstrap interval alone; report the discordance count and the limited sensitivity. For rare failures, also report the item-level event "at least one failure in three samples" with an exact binomial interval. Zero observed failures must not produce a zero upper confidence bound.

Report per-item disagreements and task characteristics alongside aggregates. A build can be preferable for one task family and not another. Do not rank ties by a decimal place or treat a single aggregate winner as universally superior. State that inference is limited to these revisions, tasks, sampling settings, hardware, and runtime.

## Result-driven continuation

After the initial analysis, the research agents must propose both (a) a refinement that tests the robustness or explanation of an observed result and (b) a related question opened by that result. These are options, not predetermined findings. Select and execute at most one follow-up in the first research cycle, with its rationale and protocol recorded before new outputs are inspected.

Examples of justified choices:

- If apparent differences are mostly truncation, compare the affected checkpoints and BF16 at a larger common output allowance on held-out items. Ask whether the ranking reflects an output budget rather than persistent correctness loss.
- If repeated/unfinished outputs differ, isolate one serving or sampling factor at a time, retaining BF16 as a control. Do not simultaneously change MTP, temperature, and cache precision and then claim to have identified a cause.
- If effects appear task-specific, test that distinction on a held-out task set rather than reusing only the examples that produced the hypothesis.
- If the main comparison is inconclusive, estimate how many independent items would be needed to distinguish a useful effect. Expansion is a valid follow-up; declaring every build identical is not.
- A long-context or multi-turn comparison is a new question when motivated by the observed failures or the external reports. Its task success criteria, length/turn conditions, and generation limits must be fixed before it runs.

The selected follow-up cites the initial measurements and acknowledges alternative explanations. If none is feasible within the allocation, retain the candidates and stop with an explicit reason; do not substitute an unrelated toy experiment.

## Verified runtime compatibility

On 2026-10-12, experiment `exp_595059e6841b4cc0801f762eec1ea504`, attempt `run_caf1332d8e014809a27dadfce145bbd5`, completed with exit code 0. It loaded the pinned NVIDIA checkpoint from the approved read-only HDD directory and generated `READY` for “Reply with exactly the uppercase word READY and nothing else.” The retained `generation.json` records token IDs `[44061, 248046]` and finish reason `stop`; `results.json` records the runtime, selected GPU, settings, and timings.

The check used vLLM 0.27.1 with BF16 computation, one GPU, eager execution, no prefix caching, a configured 32,768-token context, temperature 0, thinking disabled, and a 64-token output allowance. It did **not** exercise a full-length context or the primary evaluation's thinking/sampling settings. Networking was disabled and opening the checkpoint configuration for writing was rejected.

The attempt allowed 26 GiB host memory, 512 processes, 8 GiB output/cache storage, 32 MiB logs, and 1,800 seconds of container execution, without a CPU quota. Observed peak cgroup memory was 12,879,024,128 bytes; retained outputs and caches totaled 49,332,489 bytes. Model initialization took 1,133.045 seconds and the first generation call took 94.986 seconds. These cold-start observations include initialization/compilation effects and are **not** steady-state throughput measurements or reservations established for the other checkpoints.

Earlier failed attempts remain in the local record: stale rootless drive visibility, initial image user-mapping preparation exceeding the startup deadline, Windows-filesystem socket incompatibility, and read-only shared memory. The final runtime uses filesystem-independent abstract IPC and private writable shared memory while retaining a read-only root filesystem.

This is a one-request infrastructure check, not a benchmark result. No claim about relative quantization quality follows from it.

The separate grading image passed nine synthetic controls through real offline CPU-only worker containers on 2026-10-12: correct/incorrect stdin solutions, a correct function solution, per-test timeout, abnormal child exit, whole-call timeout, passing/failing IFEval responses, and a response that passes loose but fails strict IFEval. The IFEval controls exercised sentence tokenization and language detection without downloads. All nine passed again after the report-storage and worker-startup fixes below. Image/dependency pins and current attempt IDs are recorded in `inputs.lock.json`; detailed records are under `.arl/studies/qwen38-nvfp4/grader-controls-path-fix` and the internal `grading-runner` store. Earlier control series, including interrupted attempts, are retained. These establish the scoring path, not checkpoint quality.

A valid 1,100,002-character IFEval answer exposed the worker's 1 MiB `results.json` observation limit (`run_2f3e7500496946e7b8da250008193ee7`, retained as failed). The adapter now writes a compact summary plus the complete answer and scorer evidence in hash-linked `grading.json`. The same positive control completed after the fix (`run_6d3e00075d6b4b058a761978466ebd19`), scored correct, and preserved the full answer. No response truncation or reduced allowance was used to avoid the limit.

A later control attempt (`run_b46c0aea3d0b487887fb623653cd016f`) failed during Podman inspection. Debug tracing isolated repeated searches of inherited Windows PATH directories for absent optional OCI runtimes. The worker now filters those directories only in Podman's child environment, preserving the parent's Windows credential helpers. The failed series was retained; a separately recorded full nine-control series passed after the fix. Container and client timeouts were not increased.

Tokenizer inspection `run_58b4c15a920f4777af613be68068ad4b` verified that all five releases share 248,077 vocabulary entries, the same added-token definitions/special flags, and the same model token configuration. The vocabulary-map SHA-256 is `467543819daf0b32f85b6b8e9097aa1ac1675affa59836c26acbc4a22eb83f0f`. Their frontend defaults are not identical:

| Releases | Native padding token ID |
|---|---:|
| BF16, Unsloth, QUASAR | 248044 |
| NVIDIA, RadixArk | 248046 |

The comparison deliberately uses the original tokenizer for every release, without editing checkpoint files. The initial overly strict frontend-equality probe (`run_3f934c8361af474e9ef358a2e6c4097c`) remains recorded as failed; the follow-up inspection distinguished configurable defaults from token identity. This does not establish inference compatibility or model quality.

### Frozen task inventory

Production freeze attempt `run_92ca9a65062e432bbc35f755ab2d3e7f` completed before any benchmark generation. It verified token identity across all five releases, then rendered both disjoint task sets with the original tokenizer, thinking enabled, and `reasoning_effort="xhigh"`.

| Split | Items per family | Total items | Largest prompt, tokens |
|---|---:|---:|---:|
| Compatibility | 8 | 16 | 1193 |
| Primary comparison | 128 | 256 | 2854 |

Every selected prompt fits the 32,768-token context with the full 16,384-token output allowance; none was dropped or truncated. The approved public inputs contain the exact messages, rendered prompts, token IDs, and hashes. Private grading data is registered separately and is not mounted for generation. `inputs.lock.json` records the rendered-file and approved-inventory hashes; the local `tasks.json` and `frozen.json` receipts retain dataset-file integrity records and the selected IDs. This validates preparation, not model answers.

## Launch requirements and completion criteria

All five pinned weight snapshots have been downloaded to the verified HDD mount, checked against publisher file hashes, and registered as approved read-only inputs. `inputs.lock.json` records their local IDs, directory-inventory hashes, and byte counts; `.arl/studies/qwen38-nvfp4/staged.json` retains the host receipt. The directory hashes include Hugging Face download metadata and are not parameter-only hashes. This establishes staged inputs, not inference compatibility for every checkpoint. The [runtime setup instructions](../../README.md#pinned-large-model-runtime) cover initial user-mapping preparation and offline execution.

Before submitting the full comparison as a runnable program, the remaining requirements are:

1. Exercise checkpoint inference for all five candidates and a separate isolated grading path on the reserved compatibility items, without inspecting primary results. A package/GPU probe alone does not satisfy this.
2. Bind only approved checkpoint snapshots and public problem inputs. Keep hidden grading data and credentials outside the model-serving environment; large-input support does not authorize unrestricted experiment networking or model-controlled host paths.
3. Runtime-based memory/PID/storage/time reservations and persistent per-request records. No invented CPU quota, silent sample-count reduction, or unrecorded retries. If a budget cannot cover a matched block, stop before beginning that block.
4. A coordinator brief that carries this protocol and input inventory to the Codex research agents, plus enough execution allowance for the initial comparison and one selected follow-up. The worker's attempt count is not the count of checkpoint candidates.

A completed study contains the immutable input/runtime manifest, per-item outputs and scores, failures and resource observations, uncertainty analysis, and the evidence-linked follow-up decision. Neither mounting the drive, catalog discovery, nor a successful single inference request satisfies that criterion.
