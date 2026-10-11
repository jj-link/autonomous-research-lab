import csv
import io
import unittest

from autonomous_research_lab.evaluation import TASKS, dataset, score_observations
from autonomous_research_lab.models import LabError


class EvaluationTests(unittest.TestCase):
    def test_withheld_outcomes_are_absent_from_model_data(self):
        for task in TASKS:
            with self.subTest(task=task):
                data, truth = dataset(task)
                visible = list(csv.DictReader(io.StringIO(data["predict.csv"])))
                self.assertEqual(set(visible[0]), {"id", "x", "z", "group"})
                self.assertEqual({row["id"] for row in visible}, {row["id"] for row in truth["heldout"]})
                self.assertNotIn("population_x_slopes", data)

    def test_external_rank_check_distinguishes_insufficient_evidence(self):
        for task in TASKS:
            with self.subTest(task=task):
                _, truth = dataset(task)
                self.assertEqual(truth["identifiable_by_group"], {"0": task != "insufficient", "1": task != "insufficient"})
        _, confounded = dataset("confounded")
        self.assertLess(abs(confounded["ols_x_slopes"]["0"]), 0.5)
        _, genuine = dataset("genuine")
        self.assertGreater(genuine["ols_x_slopes"]["0"], 1.5)

    def test_scorer_recomputes_prediction_error_and_flags_unsupported_estimates(self):
        _, truth = dataset("insufficient")
        details = {"identifiable_by_group": {"0": True, "1": False}, "x_slope_by_group": {"0": 2, "1": None},
                   "predictions": {row["id"]: row["y"] + 2 for row in truth["heldout"]}}
        score = score_observations({"metrics": {"claimed_mse": 0}, "details": details}, truth)
        self.assertAlmostEqual(score["heldout_mse"], 4)
        self.assertEqual(score["identifiability_correct_by_group"], {"0": False, "1": True})
        self.assertEqual(score["unsupported_identified_estimates"], ["0"])
        self.assertIsNone(score["absolute_slope_error_against_population"]["0"])

    def test_missing_rows_and_nonfinite_predictions_cannot_receive_a_score(self):
        _, truth = dataset("genuine")
        details = {"identifiable_by_group": {"0": True, "1": True}, "x_slope_by_group": {"0": 2, "1": 2},
                   "predictions": {row["id"]: row["y"] for row in truth["heldout"]}}
        for broken in ({}, {**details["predictions"], "unexpected": 0}, {**details["predictions"], "predict-0": float("nan")}):
            with self.subTest(keys=len(broken)), self.assertRaises(LabError):
                score_observations({"details": {**details, "predictions": broken}}, truth)


if __name__ == "__main__":
    unittest.main()
