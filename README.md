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

The record store and inspection CLI are implemented. Experiments are saved as hashed source/settings snapshots, with SQLite execution history, immutable completed attempts, and per-store GPU reservations. Autonomous research and the execution CLI are still being developed.

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

## Development plan

[PLAN.md](PLAN.md) describes the proposed system in detail: a worked research scenario, component responsibilities, decision process, persistent records, execution and recovery, evaluation, and implementation milestones.
