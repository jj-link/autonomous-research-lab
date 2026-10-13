"""Paired item-level inference for the predeclared Qwen3.8 comparison."""
from __future__ import annotations

from itertools import combinations
import math
import random

SEEDS = (17, 29, 43)
ANALYSIS_SEED = 20261011
RESAMPLES = 10000
MARGIN = 0.02
FLAGS = ("generation_error", "empty", "empty_final", "truncated", "extraction_failure", "scorer_failure")


def quantile(ordered, probability):
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def binomial_interval(successes, trials, confidence=0.95):
    """Two-sided Clopper-Pearson bounds, including zero/all-event boundaries."""
    if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials or trials < 1:
        raise ValueError("Invalid binomial inventory")
    if not 0 < confidence < 1:
        raise ValueError("Invalid confidence level")
    alpha = (1 - confidence) / 2

    def cdf(k, p):
        return math.fsum(math.comb(trials, i) * p**i * (1 - p)**(trials - i) for i in range(k + 1))

    if successes == 0:
        lower = 0.0
    else:
        low, high = 0.0, 1.0
        for _ in range(80):
            middle = (low + high) / 2
            if 1 - cdf(successes - 1, middle) > alpha:
                high = middle
            else:
                low = middle
        lower = (low + high) / 2
    if successes == trials:
        upper = 1.0
    else:
        low, high = 0.0, 1.0
        for _ in range(80):
            middle = (low + high) / 2
            if cdf(successes, middle) > alpha:
                low = middle
            else:
                high = middle
        upper = (low + high) / 2
    return [lower, upper]


def paired_bootstrap(vectors, *, resamples=RESAMPLES, seed=ANALYSIS_SEED):
    labels = list(vectors)
    counts = {len(values) for values in vectors.values()}
    if len(labels) < 2 or len(counts) != 1 or not next(iter(counts), 0):
        raise ValueError("Paired vectors must have the same nonzero item count")
    if type(resamples) is not int or resamples < 2:
        raise ValueError("At least two bootstrap replicates are required")
    if any(not math.isfinite(value) for values in vectors.values() for value in values):
        raise ValueError("Non-finite paired observation")
    count = next(iter(counts))
    pairs = list(combinations(labels, 2))
    distributions = {pair: [] for pair in pairs}
    rng = random.Random(seed)
    # An index selects an entire item, whose repeated samples were averaged together.
    for _ in range(resamples):
        indices = [rng.randrange(count) for _ in range(count)]
        means = {label: math.fsum(vectors[label][index] for index in indices) / count for label in labels}
        for first, second in pairs:
            distributions[first, second].append(means[first] - means[second])
    estimates = []
    for first, second in pairs:
        ordered = sorted(distributions[first, second])
        estimates.append({
            "a": first, "b": second, "items": count,
            "difference": math.fsum(a - b for a, b in zip(vectors[first], vectors[second])) / count,
            "ci95": [quantile(ordered, 0.025), quantile(ordered, 0.975)],
            "ci995": [quantile(ordered, 0.0025), quantile(ordered, 0.9975)],
        })
    return estimates


def decision(interval, discordant_items, unavailable_items=0):
    lower, upper = interval
    if unavailable_items:
        return "inconclusive_unavailable_scoring"
    if lower > MARGIN:
        return "a_practically_higher"
    if upper < -MARGIN:
        return "b_practically_higher"
    degenerate = math.isclose(lower, upper, rel_tol=0, abs_tol=1e-12)
    if -MARGIN <= lower and upper <= MARGIN and discordant_items >= 10 and not degenerate:
        return "practically_equivalent_within_margin"
    if lower > 0:
        return "a_detectably_higher_practical_importance_unresolved"
    if upper < 0:
        return "b_detectably_higher_practical_importance_unresolved"
    return "inconclusive"


def unavailable(row):
    return bool(row["metrics"]["scorer_failure"] and row["correctness"] is None)


def validate_rows(rows, expected_ids, checkpoints):
    if len(set(checkpoints)) != len(checkpoints) or len(checkpoints) < 2:
        raise ValueError("Checkpoint identities must be distinct")
    if set(expected_ids) != {"coding", "ifeval"}:
        raise ValueError("Both task families are required")
    for family, ids in expected_ids.items():
        if not ids or len(ids) != len(set(ids)) or any(not identity.startswith(family + "/") for identity in ids):
            raise ValueError("Frozen item identities are invalid")
    wanted = {(family, label, identity, seed) for family, ids in expected_ids.items()
              for label in checkpoints for identity in ids for seed in SEEDS}
    indexed = {}
    instruction_counts = {}
    prompt_counts = {}
    for row in rows:
        key = (row["family"], row["checkpoint"], row["id"], row["seed"])
        if key not in wanted or key in indexed:
            raise ValueError(f"Unexpected or duplicate result: {key}")
        metrics = row["metrics"]
        for name in ("correct", "normal_completion", *FLAGS):
            if type(metrics[name]) not in (int, bool) or metrics[name] not in (0, 1):
                raise ValueError(f"Invalid binary outcome: {key}, {name}")
        if row["correctness"] is not None and type(row["correctness"]) is not bool:
            raise ValueError("Correctness must be boolean or unavailable")
        if metrics["correct"] != int(row["correctness"] is True):
            raise ValueError("Pass metric disagrees with the retained correctness evidence")
        if type(row["generated_tokens"]) is not int or row["generated_tokens"] < 0:
            raise ValueError("Invalid generated-token count")
        if type(row["prompt_tokens"]) is not int or row["prompt_tokens"] <= 0:
            raise ValueError("Invalid prompt-token count")
        if prompt_counts.setdefault(row["id"], row["prompt_tokens"]) != row["prompt_tokens"]:
            raise ValueError("Prompt tokenization differs across paired responses")
        if not math.isfinite(row["generation_seconds"]) or row["generation_seconds"] < 0:
            raise ValueError("Invalid generation duration")
        if type(row["repeated_128_token_block"]) is not bool:
            raise ValueError("Invalid repetition flag")
        if row["family"] == "ifeval":
            count = row["instruction_count"]
            if type(count) is not int or count < 1:
                raise ValueError("IFEval instruction inventory is absent")
            previous = instruction_counts.setdefault(row["id"], count)
            if previous != count:
                raise ValueError("IFEval instruction inventory differs across paired responses")
            for name in ("ifeval_strict_instruction", "ifeval_loose_instruction", "ifeval_loose_prompt"):
                value = metrics[name]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Invalid IFEval secondary metric")
        indexed[key] = row
    if set(indexed) != wanted:
        raise ValueError(f"Incomplete comparison: missing {len(wanted - set(indexed))} requested responses")
    return indexed


def analyze(rows, expected_ids, checkpoints, *, resamples=RESAMPLES):
    indexed = validate_rows(rows, expected_ids, checkpoints)
    result = {"schema_version": 1, "analysis_seed": ANALYSIS_SEED, "bootstrap_resamples": resamples,
              "practical_margin": MARGIN, "seeds": list(SEEDS), "checkpoints": list(checkpoints),
              "requested_responses": len(indexed), "families": {}}
    for family, ids in expected_ids.items():
        groups = {label: [[indexed[family, label, identity, seed] for seed in SEEDS] for identity in ids] for label in checkpoints}
        family_result = {"items": len(ids), "samples_per_item": len(SEEDS), "models": {}, "contrasts": {}, "per_item": []}
        for label, items in groups.items():
            flat = [row for item in items for row in item]
            counts = {name: sum(row["metrics"][name] for row in flat) for name in ("correct", "normal_completion", *FLAGS)}
            counts["repeated_128_token_block"] = sum(row["repeated_128_token_block"] for row in flat)
            events = {}
            for name in ("abnormal_completion", *FLAGS, "repeated_128_token_block"):
                def event(row):
                    if name == "abnormal_completion":
                        return not row["metrics"]["normal_completion"]
                    if name == "repeated_128_token_block":
                        return row[name]
                    return row["metrics"][name]
                affected = sum(any(event(row) for row in item) for item in items)
                events[name] = {"items": affected, "rate": affected / len(ids), "ci95": binomial_interval(affected, len(ids))}
            model = {"requested": len(flat), "counts": counts, "observed_pass_rate": counts["correct"] / len(flat),
                     "normal_completion_rate": counts["normal_completion"] / len(flat),
                     "unavailable_scorer_responses": sum(unavailable(row) for row in flat), "item_level_events": events,
                     "summed_request_latency_seconds": math.fsum(row["generation_seconds"] for row in flat),
                     "generated_tokens": sum(row["generated_tokens"] for row in flat)}
            if family == "ifeval":
                instructions = sum(row["instruction_count"] for row in flat)
                model["secondary"] = {"instructions": instructions,
                    "strict_instruction_rate": math.fsum(row["metrics"]["ifeval_strict_instruction"] * row["instruction_count"] for row in flat) / instructions,
                    "loose_instruction_rate": math.fsum(row["metrics"]["ifeval_loose_instruction"] * row["instruction_count"] for row in flat) / instructions,
                    "loose_prompt_rate": math.fsum(row["metrics"]["ifeval_loose_prompt"] for row in flat) / len(flat)}
            family_result["models"][label] = model
        for metric in ("correct", "normal_completion"):
            vectors = {label: [math.fsum(row["metrics"][metric] for row in item) / len(SEEDS) for item in items] for label, items in groups.items()}
            contrasts = paired_bootstrap(vectors, resamples=resamples)
            for contrast in contrasts:
                first, second = contrast["a"], contrast["b"]
                discordant = sum(any(a["metrics"][metric] != b["metrics"][metric] for a, b in zip(one, two))
                                 for one, two in zip(groups[first], groups[second]))
                retained = [index for index in range(len(ids)) if metric != "correct" or
                            not any(unavailable(row) for row in groups[first][index] + groups[second][index])]
                excluded = len(ids) - len(retained)
                contrast.update(discordant_items=discordant, unavailable_items=excluded,
                                decision=decision(contrast["ci995"], discordant, excluded),
                                excludes_zero=contrast["ci995"][0] > 0 or contrast["ci995"][1] < 0,
                                degenerate_interval=math.isclose(*contrast["ci995"], rel_tol=0, abs_tol=1e-12))
                if excluded:
                    sensitivity = {label: [vectors[label][index] for index in retained] for label in (first, second)}
                    contrast["complete_case_sensitivity"] = paired_bootstrap(sensitivity, resamples=resamples)[0] if retained else {"items": 0}
            family_result["contrasts"][metric] = contrasts
        for index, identity in enumerate(ids):
            reference = groups[checkpoints[0]][index][0]
            family_result["per_item"].append({"id": identity, "prompt_tokens": reference["prompt_tokens"],
                "instruction_count": reference["instruction_count"], "test_count": reference["test_count"], "models": {
                label: {"correct": [row["metrics"]["correct"] for row in groups[label][index]],
                        "normal_completion": [row["metrics"]["normal_completion"] for row in groups[label][index]],
                        "unavailable_scoring": [unavailable(row) for row in groups[label][index]]}
                for label in checkpoints}})
        result["families"][family] = family_result
    return result


def render_report(result):
    def percent(value):
        return f"{100 * value:.2f}"

    def interval(values):
        return f"[{percent(values[0])}, {percent(values[1])}]"

    lines = ["# Qwen3.8-27B released-build comparison", "",
             f"Complete paired inventory: {result['requested_responses']} responses; seeds 17, 29, and 43.",
             "This compares the pinned released builds, not quantization algorithms in isolation. BF16 is a reference, not ground truth.", ""]
    for family, data in result["families"].items():
        lines += [f"## {family}", "", f"{data['items']} unique items, with all three samples kept together for paired inference.", "",
                  "| Checkpoint | Observed pass (%) | Normal completion (%) | Unavailable scorer responses | Truncated responses |",
                  "|---|---:|---:|---:|---:|"]
        for label, model in data["models"].items():
            lines.append(f"| {label} | {percent(model['observed_pass_rate'])} | {percent(model['normal_completion_rate'])} | {model['unavailable_scorer_responses']} | {model['counts']['truncated']} |")
        for metric, contrasts in data["contrasts"].items():
            lines += ["", f"### Paired {metric} contrasts", "", "Differences and intervals are percentage points, A minus B. Decisions use the adjusted interval and the predeclared two-point margin.", "",
                      "| A | B | Difference | 95% interval | 99.5% interval | Discordant items | Decision |",
                      "|---|---|---:|---|---|---:|---|"]
            for item in contrasts:
                lines.append(f"| {item['a']} | {item['b']} | {percent(item['difference'])} | {interval(item['ci95'])} | {interval(item['ci995'])} | {item['discordant_items']} | {item['decision'].replace('_', ' ')} |")
        lines += ["", "### Item-level completion failures", "", "Event: at least one abnormal completion among the three responses for an item. Intervals are exact binomial 95% intervals.", "",
                  "| Checkpoint | Affected items | Rate (%) | 95% interval (%) |", "|---|---:|---:|---|"]
        for label, model in data["models"].items():
            event = model["item_level_events"]["abnormal_completion"]
            lines.append(f"| {label} | {event['items']} | {percent(event['rate'])} | {interval(event['ci95'])} |")
        if family == "ifeval":
            lines += ["", "### Secondary IFEval outcomes", "", "Instruction accuracy is weighted by the actual instruction count, not an unweighted mean of per-prompt fractions.", "",
                      "| Checkpoint | Strict instruction (%) | Loose instruction (%) | Loose prompt (%) |", "|---|---:|---:|---:|"]
            for label, model in data["models"].items():
                secondary = model["secondary"]
                lines.append(f"| {label} | {percent(secondary['strict_instruction_rate'])} | {percent(secondary['loose_instruction_rate'])} | {percent(secondary['loose_prompt_rate'])} |")
        lines.append("")
    lines += ["## Interpretation limits", "",
              "- Bootstrap resampling uses whole items, not repeated responses as independent observations. Ten pairwise contrasts per family use 99.5% intervals in addition to nominal 95% intervals.",
              "- An interval crossing zero is not evidence of equality. Equivalence is withheld for degenerate intervals or fewer than ten discordant items.",
              "- Every requested response stays in the observed-pass denominator. Unavailable scorer evidence is not called a wrong answer; affected correctness contrasts remain inconclusive, with paired complete-case sensitivity in analysis.json. Missingness may be informative.",
              "- Completion and correctness are separate outcomes. Truncated but extractable answers can pass; repetition is a diagnostic, not an automatic correctness failure.",
              "- Results apply only to these revisions, tasks, settings, hardware, and runtime. Training exposure is unknown. The QAT release has additional training, and weight formats can select different kernels.",
              "- No universal checkpoint winner or cause of numerical instability follows automatically from these measurements.", ""]
    return "\n".join(lines)
