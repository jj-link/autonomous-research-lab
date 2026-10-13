# Qwen3.8-27B NVFP4: correctness and reliability across released builds

## Question and intended result

Do the Unsloth, NVIDIA, RadixArk, and QUASAR NVFP4 releases of Qwen3.8-27B produce repeatable differences in answer quality under the same inference conditions? How much does each differ from the original BF16 model, and do aggregate scores conceal particular failure modes?

The result should help choose a checkpoint for a workload, with uncertainty and limitations attached. It must not declare a universal winner from a small score difference or describe output similarity to BF16 as proof of correctness. BF16 is a reference, not ground truth; task answers and executable tests supply ground truth.

This is a comparison of published builds. Their retained precision, calibration, and additional training differ. QUASAR uses quantization-aware training. A difference between releases therefore does not, by itself, establish which quantization technique caused it.

## Scope

This document defines the research design, not results or an executable `arl program create` manifest. [inputs.lock.json](inputs.lock.json) pins the inputs and runtime dependencies.

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

Before any download, require an actual `/mnt/d` mount whose source is `D:`; directory existence alone is insufficient. If it is absent, stop rather than writing models into the underlying WSL filesystem. The existing drive can be mounted with:

```sh
sudo mount -t drvfs D: /mnt/d -o uid=1000,gid=1000,noatime
findmnt --mountpoint /mnt/d --output TARGET,SOURCE,FSTYPE,OPTIONS
```

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
- Use the pinned vLLM build for all five checkpoints. Do not quietly use a different engine or version for a troublesome candidate.
- Use a 32,768-token context window, explicit BF16 KV cache, up to sixteen concurrent requests, MTP/speculative decoding disabled, and prefix caching disabled. No engine-selected per-checkpoint cache precision or offload fallback.
- Use the pinned original model's tokenizer and chat template only after verifying the complete token-to-ID map, added-token contents/special flags, and model token configuration for every candidate. Publisher frontend defaults (including padding choice) are recorded separately and normalized by this common tokenizer; they are not token-ID identity. Render identical inputs, with thinking enabled and `reasoning_effort=xhigh`. A vocabulary or backend special-token mismatch blocks the controlled comparison; do not disguise it as a weight-quality difference.
- Fix `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0`, `presence_penalty=0`, and `repetition_penalty=1`. Allow 16,384 generated tokens per response, including reasoning, for every candidate and both task families. Inputs that cannot fit without truncation must be identified before evaluating any candidate.
- Generate one answer per item for each of three shared seeds: 17, 29, and 43. This measures repeated pass@1 behavior, not best-of-three or a repair success rate. Identical seeds do not guarantee equivalent random trajectories across different checkpoints.
- Preserve the original rotated checkpoint/seed schedule in the cohort plan. Record the approved execution-order amendment separately: retain every already-submitted block, then run each checkpoint's remaining seeds consecutively in inventory order using one loaded engine. Initialize the engine with the first requested seed and explicitly apply each block's sampling seed to its requests. Warm once per loaded checkpoint, retain separate seed receipts, and never repeat a failed request invisibly. CPU grading may overlap later GPU generation, with one grader at a time and only after a completed seed's artifacts have been retained and hash-verified.
- Both splits use normal graph-enabled vLLM execution (`enforce_eager=False`), chunked prefill with at most 8,192 batched tokens, GPU memory utilization `0.90`, and the safetensors lazy loader. Both CPU-offload mechanisms are explicitly disabled. Submit requests concurrently and save each completed response immediately, even when it finishes out of input order. Retain resolved worker graph settings, graph-dispatch statistics, and implementation/quantization class paths; shared engine versions do not imply identical kernels.
- Compatibility checks use seed 17 on the sixteen reserved items. A separate, non-thinking `Reply with READY.` request warms the engine; save it separately and exclude it from task outcomes. Per-job reservations are 26 GiB host RAM, 512 PIDs, 8 GiB output, 32 MiB logs, and six hours; no CPU quota or host memory configuration change.
- A primary seed block executes all 256 frozen items for one checkpoint, with the same memory/PID/storage reservations and generation conditions. After all five preflights are complete and graded, reserve a common wall time per seed of `1.5 × (longest measured load + 256 × 16384 × slowest aggregate seconds per generated token)`, rounded up to an hour and no less than six hours. A resident attempt receives this reservation multiplied by its seed count; memory/PID/storage limits do not increase. Count load and warmup once per resident attempt, not once per seed. Aggregate generation time spans first submission through last durable completion within each seed; never sum overlapping request latencies. Save the measured reservation and original fifteen-block order before primary generation. The reservation covers the full output allowance with headroom; it is not a predicted completion time.
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

## Launch requirements and completion criteria

Before launching the full comparison:

1. Exercise checkpoint inference for all five candidates and a separate isolated grading path on the reserved compatibility items, without inspecting primary results. A package/GPU probe alone does not satisfy this.
2. Bind only approved checkpoint snapshots and public problem inputs. Keep hidden grading data and credentials outside the model-serving environment; large-input support does not authorize unrestricted experiment networking or model-controlled host paths.
3. Runtime-based memory/PID/storage/time reservations and persistent per-request records. No invented CPU quota, silent sample-count reduction, or unrecorded retries. If a budget cannot cover a matched block, stop before beginning that block.
4. A coordinator brief that carries this protocol and input inventory to the Codex research agents, plus enough execution allowance for the initial comparison and one selected follow-up. The worker's attempt count is not the count of checkpoint candidates.

A completed study contains the immutable input/runtime manifest, per-item outputs and scores, failures and resource observations, uncertainty analysis, and the evidence-linked follow-up decision. Neither mounting the drive, catalog discovery, nor a successful single inference request satisfies that criterion.
