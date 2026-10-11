"""Controlled observational tasks with hidden outcomes and external numerical scoring."""
from __future__ import annotations

import csv
import io
import math
import random

from .execution import atomic_json
from .models import LabError, read_json
from .model_client import validate_model_config
from .research import POLICIES, ResearchEngine, new_id
from .store import Store, identifier, utc_now

TASKS = {
    "genuine": "A nonzero conditional linear association in both groups.",
    "confounded": "A marginal association explained by the observed covariate.",
    "absent": "No conditional association in either group.",
    "insufficient": "Rank-deficient training data cannot identify separate coefficients.",
    "conditional": "The conditional association differs between groups.",
}
GOAL = """Investigate how x is associated with y after accounting for z, and whether that relationship differs between groups 0 and 1.
These are observational data: do not infer causality. Design comparisons that distinguish a marginal association from a conditional one,
and identify what the available data cannot establish. Use data/train.csv (id,x,z,group,y); data/predict.csv has unlabeled new rows.
Every executed study must write results.json with numeric metrics plus details containing:
identifiable_by_group: {"0": boolean, "1": boolean}, indicating whether intercept/x/z coefficients can be uniquely estimated in each group's training data;
x_slope_by_group: {"0": number or null, "1": number or null}, conditional linear x coefficients (null when not identifiable);
predictions: {row_id: finite_number} for every row in predict.csv. You may investigate other measurements as well.
Provide uncertainty and limitations in analysis. The experiment design and code are your responsibility."""


def _csv(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in rows)
    return stream.getvalue()


def _fit(rows):
    """Small independent OLS reference; None indicates an unidentified design matrix."""
    matrix = [[0.0] * 4 for _ in range(3)]
    for row in rows:
        features = (1.0, row["x"], row["z"])
        for i in range(3):
            for j in range(3):
                matrix[i][j] += features[i] * features[j]
            matrix[i][3] += features[i] * row["y"]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda i: abs(matrix[i][column]))
        if abs(matrix[pivot][column]) < 1e-10:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        matrix[column] = [value / divisor for value in matrix[column]]
        for i in range(3):
            if i != column:
                factor = matrix[i][column]
                matrix[i] = [a - factor * b for a, b in zip(matrix[i], matrix[column])]
    return [matrix[i][3] for i in range(3)]


def dataset(task, seed=1729):
    if task not in TASKS or type(seed) is not int:
        raise LabError("Choose a listed evaluation task and integer seed")
    rng = random.Random(seed)
    coefficients = {"0": 2.0, "1": 2.0}
    if task in {"confounded", "absent"}:
        coefficients = {"0": 0.0, "1": 0.0}
    elif task == "conditional":
        coefficients = {"0": 0.0, "1": 2.0}
    elif task == "insufficient":
        coefficients = {"0": -1.0, "1": 3.0}
    def rows(count, training):
        result = []
        for index in range(count):
            group = str(index % 2)
            z, independent = rng.uniform(-2, 2), rng.uniform(-2, 2)
            x = 0.95 * z + 0.2 * independent if task == "confounded" else independent
            if task == "insufficient" and training:
                x = z
            y = coefficients[group] * x + 2 * z + int(group) + rng.gauss(0, 0.2)
            result.append({"id": ("train-" if training else "predict-") + str(index), "x": x, "z": z, "group": group, "y": y})
        return result
    training = rows(4 if task == "insufficient" else 256, True)
    heldout = rows(64, False)
    fits = {group: _fit([row for row in training if row["group"] == group]) for group in ("0", "1")}
    baseline = None
    if all(fit is not None for fit in fits.values()):
        baseline = sum((sum(a * b for a, b in zip(fits[row["group"]], (1, row["x"], row["z"]))) - row["y"]) ** 2 for row in heldout) / len(heldout)
    data = {"train.csv": _csv(training, ("id", "x", "z", "group", "y")), "predict.csv": _csv(heldout, ("id", "x", "z", "group"))}
    truth = {"task": task, "seed": seed, "population_x_slopes": coefficients, "identifiable_by_group": {g: fit is not None for g, fit in fits.items()},
             "ols_x_slopes": {g: fit[1] if fit else None for g, fit in fits.items()}, "heldout": heldout, "ols_heldout_mse": baseline}
    return data, truth


def score_observations(observations, truth):
    """Score structured claims against independent algebra and withheld labels, not LLM opinions."""
    details = observations.get("details") if isinstance(observations, dict) else None
    if not isinstance(details, dict):
        raise LabError("Evaluation result has no details object")
    flags = details.get("identifiable_by_group")
    estimates = details.get("x_slope_by_group")
    predictions = details.get("predictions")
    if not isinstance(flags, dict) or set(flags) != {"0", "1"} or any(type(v) is not bool for v in flags.values()):
        raise LabError("Result must provide both boolean identifiability assessments")
    if not isinstance(estimates, dict) or set(estimates) != {"0", "1"}:
        raise LabError("Result must provide both conditional slope estimates")
    def finite(value):
        try:
            return type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            return False
    if any(value is not None and not finite(value) for value in estimates.values()):
        raise LabError("Slope estimates must be finite numbers or null")
    expected = {row["id"] for row in truth["heldout"]}
    if not isinstance(predictions, dict) or set(predictions) != expected or not all(finite(v) for v in predictions.values()):
        raise LabError("Predictions must cover exactly the held-out IDs with finite numbers")
    try:
        mse = math.fsum((predictions[row["id"]] - row["y"]) ** 2 / len(expected) for row in truth["heldout"])
    except OverflowError as exc:
        raise LabError("Prediction error exceeds the finite numerical range") from exc
    if not math.isfinite(mse):
        raise LabError("Prediction error exceeds the finite numerical range")
    return {
        "heldout_mse": mse, "independent_ols_heldout_mse": truth["ols_heldout_mse"],
        "identifiability_correct_by_group": {g: flags[g] == truth["identifiable_by_group"][g] for g in ("0", "1")},
        "unsupported_identified_estimates": [g for g in ("0", "1") if not truth["identifiable_by_group"][g] and estimates[g] is not None],
        "absolute_slope_error_against_population": {g: abs(estimates[g] - truth["population_x_slopes"][g]) if estimates[g] is not None and truth["identifiable_by_group"][g] else None for g in ("0", "1")},
        "absolute_slope_difference_from_independent_ols": {g: abs(estimates[g] - truth["ols_x_slopes"][g]) if estimates[g] is not None and truth["ols_x_slopes"][g] is not None else None for g in ("0", "1")},
    }


def prepare(store: Store, model_config, task, policy="candidate_comparison", seed=1729):
    if policy not in POLICIES:
        raise LabError("Unknown comparison policy")
    model_config = validate_model_config(model_config)
    data, truth = dataset(task, seed)
    program = ResearchEngine(store).create({"goal": GOAL, "scope": "Bounded observational analysis and prediction using the supplied CSV files; standard-library Python only.",
        "model": model_config, "data_files": data, "policy": policy, "max_experiments": 2, "max_model_calls": 12,
        "max_output_tokens_per_call": None if model_config["provider"] == "openai-codex" else 8192,
        "max_wall_seconds": 120, "max_repairs": 1,
        "resources": {"gpu": False, "wall_seconds": 30, "memory_bytes": 128 * 1024**2, "output_bytes": 8 * 1024**2, "log_bytes": 1024**2, "pids": 32, "cpus": None}})
    evaluation_id = new_id("eval")
    record = {"id": evaluation_id, "program_id": program["id"], "task": task, "policy": policy, "seed": seed, "created_at": utc_now()}
    folder = store.root / "evaluations" / evaluation_id
    # Neither this directory nor the withheld y values are mounted into experiments or sent to models.
    atomic_json(folder / "truth.json", truth)
    atomic_json(folder / "evaluation.json", record)
    return record


def score(store: Store, evaluation_id):
    folder = store.root / "evaluations" / identifier(evaluation_id, "eval")
    record = read_json(folder / "evaluation.json")
    truth = read_json(folder / "truth.json")
    program = ResearchEngine(store).get(record["program_id"])
    planned = [item for item in program["implementations"] if item.get("attempt_id")]
    known = {a["id"]: a for a in store.list_attempts()}
    attempts = [known[item["attempt_id"]] for item in planned if item["attempt_id"] in known]
    final_attempt = attempts[-1] if attempts else None
    numerical, error = None, None
    if final_attempt and final_attempt["state"] == "completed":
        try:
            numerical = score_observations(final_attempt["result"]["observations"], truth)
        except (LabError, OverflowError) as exc:
            error = str(exc)
    else:
        error = "There is no completed final attempt; earlier successes are not substituted"
    cpu = [a["execution"].get("cpu_seconds_observed") if a.get("execution") else None for a in attempts]
    memory = [a["execution"].get("peak_memory_bytes_observed") if a.get("execution") else None for a in attempts]
    result = {**record, "program_state": program["state"], "stop_reason": program["stop_reason"], "selected_attempt_id": final_attempt["id"] if final_attempt else None,
              "numerical": numerical, "scoring_error": error, "usage": program["usage"],
              "failed_or_incomplete_attempts": sum(a["state"] != "completed" for a in attempts),
              "cpu_seconds_observed_lower_bound": sum(v for v in cpu if v is not None), "attempts_without_cpu_observations": sum(v is None for v in cpu),
              "peak_host_memory_bytes_observed": max((v for v in memory if v is not None), default=None),
              "external_review_required": ["Whether conclusions answer the brief", "Unsupported causal or generality claims", "Whether follow-ups resolve prior questions", "Limitations and evidence dependence"],
              "interpretation_limit": "Numerical checks are external; model analyses do not grade themselves. One run does not establish policy superiority. GPU energy, complete CPU usage and monetary billing are not inferred."}
    atomic_json(folder / "score.json", result)
    return result


def evaluate(store: Store, model_config, task, policy="candidate_comparison", seed=1729):
    record = prepare(store, model_config, task, policy, seed)
    ResearchEngine(store).start(record["program_id"], background=False)
    return score(store, record["id"])


def compare(store: Store, model_config, task, seed=1729):
    results = []
    for policy in ("fixed_plan", "direct", "candidate_comparison"):
        result = evaluate(store, model_config, task, policy, seed)
        results.append(result)
        if result["program_state"] != "completed":
            break  # Respect interruption or an unavailable provider/worker prerequisite.
    return {"task": task, "seed": seed, "results": results, "complete_comparison": len(results) == 3 and all(r["numerical"] is not None and r["program_state"] == "completed" for r in results),
            "limits": "The three policies receive identical data, models, tools and budget ceilings; every planning/repair call and failed execution remains charged. Repeat across seeds and review interpretations before drawing conclusions."}
