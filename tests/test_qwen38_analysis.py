import copy
import importlib.util
import math
from pathlib import Path
import unittest


MODULE = Path(__file__).resolve().parents[1] / "research/qwen38-nvfp4/analysis.py"
SPEC = importlib.util.spec_from_file_location("qwen38_analysis", MODULE)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def fixture(items=2, labels=("reference", "candidate")):
    expected = {family: [f"{family}/{index}" for index in range(items)] for family in ("coding", "ifeval")}
    rows = []
    for family, identities in expected.items():
        for label in labels:
            for identity in identities:
                for seed in analysis.SEEDS:
                    metrics = {name: 0 for name in analysis.FLAGS}
                    metrics.update(correct=1, normal_completion=1)
                    if family == "ifeval":
                        metrics.update(ifeval_strict_instruction=1.0, ifeval_loose_instruction=1.0, ifeval_loose_prompt=1)
                    rows.append({"family": family, "checkpoint": label, "id": identity, "seed": seed,
                                 "correctness": True, "metrics": metrics, "instruction_count": 1 if family == "ifeval" else None,
                                 "prompt_tokens": 100, "test_count": 2 if family == "coding" else None,
                                 "generated_tokens": 10, "generation_seconds": 1.0, "repeated_128_token_block": False})
    return expected, rows


class PairedAnalysisTests(unittest.TestCase):
    def test_binomial_boundaries_and_known_interior_interval(self):
        zero = analysis.binomial_interval(0, 128)
        self.assertEqual(zero[0], 0)
        self.assertAlmostEqual(zero[1], 1 - 0.025 ** (1 / 128), places=12)
        all_events = analysis.binomial_interval(128, 128)
        self.assertEqual(all_events[1], 1)
        self.assertAlmostEqual(all_events[0], 1 - zero[1], places=12)
        lower, upper = analysis.binomial_interval(5, 10)
        self.assertAlmostEqual(lower, 0.18708602844739852, places=10)
        self.assertAlmostEqual(upper, 0.8129139715526015, places=10)

    def test_equivalence_requires_non_degenerate_independent_discordance(self):
        self.assertEqual(analysis.decision([-0.01, 0.01], 10), "practically_equivalent_within_margin")
        self.assertEqual(analysis.decision([-0.01, 0.01], 9), "inconclusive")
        self.assertEqual(analysis.decision([0.0, 0.0], 100), "inconclusive")
        self.assertEqual(analysis.decision([0.021, 0.08], 100), "a_practically_higher")
        self.assertEqual(analysis.decision([-0.08, -0.021], 100), "b_practically_higher")
        self.assertEqual(analysis.decision([0.001, 0.05], 100), "a_detectably_higher_practical_importance_unresolved")
        self.assertEqual(analysis.decision([0.1, 0.2], 100, 1), "inconclusive_unavailable_scoring")

    def test_three_samples_stay_together_in_item_bootstrap(self):
        expected, rows = fixture()
        for row in rows:
            correct = row["checkpoint"] == "reference" and row["id"].endswith("/1")
            row["correctness"] = correct
            row["metrics"]["correct"] = int(correct)
        result = analysis.analyze(rows, expected, ["reference", "candidate"])
        contrast = result["families"]["coding"]["contrasts"]["correct"][0]
        self.assertEqual(contrast["items"], 2)
        self.assertEqual(contrast["difference"], 0.5)
        # Treating the six seed rows as independent gives spuriously narrower bounds.
        self.assertEqual(contrast["ci95"], [0.0, 1.0])
        self.assertEqual(contrast["discordant_items"], 1)

    def test_identical_observed_scores_do_not_establish_equality(self):
        expected, rows = fixture(items=12)
        result = analysis.analyze(rows, expected, ["reference", "candidate"], resamples=100)
        family = result["families"]["coding"]
        contrast = family["contrasts"]["correct"][0]
        self.assertEqual(contrast["difference"], 0)
        self.assertEqual(contrast["decision"], "inconclusive")
        self.assertTrue(contrast["degenerate_interval"])
        event = family["models"]["reference"]["item_level_events"]["abnormal_completion"]
        self.assertEqual(event["items"], 0)
        self.assertGreater(event["ci95"][1], 0)

    def test_missing_and_duplicate_requests_are_not_silently_dropped(self):
        expected, rows = fixture()
        with self.assertRaisesRegex(ValueError, "missing 1"):
            analysis.analyze(rows[:-1], expected, ["reference", "candidate"], resamples=10)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analysis.analyze(rows + [copy.deepcopy(rows[0])], expected, ["reference", "candidate"], resamples=10)

    def test_instruction_accuracy_is_weighted_by_instruction_inventory(self):
        expected, rows = fixture()
        for row in rows:
            if row["family"] == "ifeval":
                passed = row["id"].endswith("/1")
                row["instruction_count"] = 3 if passed else 1
                row["correctness"] = passed
                row["metrics"].update(correct=int(passed), ifeval_strict_instruction=float(passed),
                                      ifeval_loose_instruction=float(passed), ifeval_loose_prompt=int(passed))
        result = analysis.analyze(rows, expected, ["reference", "candidate"], resamples=100)
        model = result["families"]["ifeval"]["models"]["reference"]
        self.assertEqual(model["secondary"]["instructions"], 12)
        self.assertEqual(model["secondary"]["strict_instruction_rate"], 0.75)
        self.assertEqual(model["secondary"]["loose_prompt_rate"], 0.5)
        self.assertEqual(model["observed_pass_rate"], 0.5)

    def test_unavailable_scoring_keeps_denominator_and_marks_inference(self):
        expected, rows = fixture()
        rows[0]["correctness"] = None
        rows[0]["metrics"].update(correct=0, scorer_failure=1)
        result = analysis.analyze(rows, expected, ["reference", "candidate"], resamples=100)
        family = result["families"]["coding"]
        reference = family["models"]["reference"]
        self.assertEqual(reference["requested"], 6)
        self.assertEqual(reference["unavailable_scorer_responses"], 1)
        self.assertAlmostEqual(reference["observed_pass_rate"], 5 / 6)
        contrast = family["contrasts"]["correct"][0]
        self.assertEqual(contrast["unavailable_items"], 1)
        self.assertEqual(contrast["complete_case_sensitivity"]["items"], 1)
        self.assertEqual(contrast["decision"], "inconclusive_unavailable_scoring")
        self.assertEqual(family["contrasts"]["normal_completion"][0]["unavailable_items"], 0)

    def test_empty_extraction_is_not_mislabeled_as_unavailable_scorer(self):
        expected, rows = fixture()
        rows[0]["correctness"] = None
        rows[0]["metrics"].update(correct=0, empty_final=1, extraction_failure=1)
        result = analysis.analyze(rows, expected, ["reference", "candidate"], resamples=100)
        reference = result["families"]["coding"]["models"]["reference"]
        self.assertEqual(reference["unavailable_scorer_responses"], 0)
        self.assertEqual(reference["counts"]["correct"], 5)
        self.assertEqual(reference["item_level_events"]["extraction_failure"]["items"], 1)

    def test_nonfinite_timing_and_inconsistent_evidence_are_rejected(self):
        expected, rows = fixture()
        rows[0]["generation_seconds"] = math.nan
        with self.assertRaisesRegex(ValueError, "duration"):
            analysis.validate_rows(rows, expected, ["reference", "candidate"])
        rows[0]["generation_seconds"] = 1.0
        rows[0]["correctness"] = False
        with self.assertRaisesRegex(ValueError, "disagrees"):
            analysis.validate_rows(rows, expected, ["reference", "candidate"])

    def test_input_order_does_not_change_seeded_contrasts(self):
        expected, rows = fixture(items=12)
        for index, row in enumerate(rows):
            correct = index % 4 != 0
            row["correctness"] = correct
            row["metrics"]["correct"] = int(correct)
        original = analysis.analyze(rows, expected, ["reference", "candidate"], resamples=100)
        reversed_rows = analysis.analyze(list(reversed(rows)), expected, ["reference", "candidate"], resamples=100)
        self.assertEqual(original, reversed_rows)


if __name__ == "__main__":
    unittest.main()
