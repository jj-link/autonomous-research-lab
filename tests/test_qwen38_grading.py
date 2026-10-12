"""Pure-stdlib grading regressions: these tests never execute model code."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import unittest
import uuid

from autonomous_research_lab.models import read_json


SOURCE = Path(__file__).resolve().parents[1] / "research" / "qwen38-nvfp4" / "worker" / "grade.py"
SPEC = importlib.util.spec_from_file_location("qwen38_grade", SOURCE)
grade = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grade)


def generation(text, finish="stop", error=None, thinking=True):
    return {"text": text, "finish_reason": finish, "error": error,
            "token_ids": [1, 2, 3], "thinking_enabled": thinking}


class FinalExtractionTests(unittest.TestCase):
    def test_prompt_preopened_thinking_has_no_generated_open_tag(self):
        final = grade.extract_final_answer("Let me derive this.\n</think>\nThe answer is 42.\n")
        self.assertEqual(final["status"], "ok")
        self.assertEqual(final["text"], "\nThe answer is 42.\n")

    def test_explicit_thinking_never_grades_reasoning_code(self):
        final = grade.extract_final_answer("<think>\n```python\nprint(1)\n```\n</think>\n```python\nprint(2)\n```\n")
        self.assertEqual(grade.extract_code(final["text"])["text"], "print(2)\n")

    def test_reasoning_only_even_without_opening_tag_is_not_final(self):
        for text in ("I need to solve this problem.", "```python\nprint(1)\n```", "<think>unfinished"):
            with self.subTest(text=text):
                final = grade.extract_final_answer(text)
                self.assertEqual(final["status"], "unfinished_reasoning")
                self.assertEqual(final["text"], "")

    def test_empty_final_after_reasoning(self):
        final = grade.extract_final_answer("The reasoning goes here.</think>\n \t")
        self.assertEqual(final["status"], "empty_final")

    def test_explicit_non_thinking_control_preserves_exact_answer(self):
        text = "  First line.\n\nSecond line.  "
        self.assertEqual(grade.extract_final_answer(text, thinking=False)["text"], text)

    def test_new_unclosed_thinking_after_close_is_not_final(self):
        self.assertEqual(grade.extract_final_answer("old</think><think>new")["status"], "unfinished_reasoning")


class CodeExtractionTests(unittest.TestCase):
    def test_last_eligible_block_not_first_or_concatenation(self):
        text = "Here is an initial draft.\n```python\nprint(0)\n```\nCorrected answer:\n```\nprint(1)\n```\nExplanation follows."
        code = grade.extract_code(text)
        self.assertEqual(code["status"], "ok")
        self.assertEqual(code["text"], "print(1)\n")

    def test_non_python_fence_does_not_become_executable(self):
        result = grade.extract_code("```javascript\nconsole.log(42);\n```\nprint(42)")
        self.assertEqual(result["status"], "no_python_fence")

    def test_later_other_language_does_not_replace_python(self):
        code = grade.extract_code("```python\nprint(42)\n```\n```text\n42\n```")
        self.assertEqual(code["text"], "print(42)\n")

    def test_latest_unclosed_python_block_does_not_reuse_draft(self):
        result = grade.extract_code("```python\nprint(0)\n```\n```python\nprint(1)")
        self.assertEqual(result["status"], "unclosed_fence")
        self.assertEqual(result["text"], "")

    def test_unfenced_code_with_indentation_is_preserved(self):
        text = "def solve():\n    print(int(input()) + 1)\nsolve()\n"
        code = grade.extract_code(text)
        self.assertEqual(code["status"], "ok")
        self.assertEqual(code["source"], "unfenced")
        self.assertEqual(code["text"], text)

    def test_unfenced_prose_is_not_repaired_to_find_python(self):
        result = grade.extract_code("Here is the solution:\nprint(42)\n")
        self.assertEqual(result["status"], "invalid_python")

    def test_invalid_final_block_never_falls_back_to_valid_draft(self):
        result = grade.extract_code("```python\nprint(42)\n```\n```python\nif True\n    print(42)\n```")
        self.assertEqual(result["status"], "invalid_python")

    def test_fence_length_and_case(self):
        result = grade.extract_code("````Python\nprint('```')\n````\n")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["text"], "print('```')\n")

    def test_tilde_fence(self):
        self.assertEqual(grade.extract_code("~~~python\nprint(42)\n~~~\n")["text"], "print(42)\n")

    def test_empty_code_and_literal_prose_do_not_execute(self):
        self.assertEqual(grade.extract_code("```python\n \n```")["status"], "empty_code")
        self.assertEqual(grade.extract_code('"This is merely an explanation."')["status"], "no_executable_code")
        self.assertEqual(grade.extract_code("This is wrong")["status"], "no_executable_code")

    def test_source_is_parsed_never_executed(self):
        # This would immediately raise if extraction used exec/eval.
        source = "raise RuntimeError('candidate code must not run on the host')\n"
        self.assertEqual(grade.extract_code(source)["status"], "ok")
        self.assertEqual(grade.extract_code(source)["text"], source)


class ResponseAssessmentTests(unittest.TestCase):
    def assess(self, text, finish="stop", error=None):
        generated = generation(text, finish, error)
        return grade.assess_response(generated, grade.extract_final_answer(text))

    def test_truncation_retains_extractable_answer_without_normal_completion(self):
        text = "reasoning</think>```python\nprint(42)\n```"
        result = self.assess(text, "length")
        self.assertEqual(result["category"], "truncated")
        self.assertTrue(result["truncated"])
        self.assertFalse(result["normal_completion"])
        self.assertFalse(result["empty_final"])
        self.assertEqual(grade.extract_code(grade.extract_final_answer(text)["text"])["status"], "ok")

    def test_generation_failure_has_priority_over_text(self):
        result = self.assess("reasoning</think>answer", error={"kind": "server_error"})
        self.assertEqual(result["category"], "generation_error")
        self.assertTrue(result["generation_error"])
        self.assertFalse(result["normal_completion"])

    def test_empty_and_truncated_flags_are_independent(self):
        result = self.assess("", "length")
        self.assertTrue(result["empty"])
        self.assertTrue(result["truncated"])
        self.assertFalse(result["normal_completion"])

    def test_missing_finish_reason_is_not_normal_completion(self):
        self.assertEqual(self.assess("reason</think>answer", None)["category"], "abnormal_finish")

    def test_reasoning_stop_is_not_normal_final_completion(self):
        result = self.assess("An unfinished derivation.")
        self.assertEqual(result["category"], "unfinished_reasoning")
        self.assertFalse(result["normal_completion"])


class LcbClassificationTests(unittest.TestCase):
    def test_negative_error_codes_are_never_truthy_successes(self):
        for code, category in ((-1, "global_timeout"), (-2, "wrong_answer"), (-3, "test_timeout"), (-4, "runtime_error"), (-5, "test_runner_error")):
            with self.subTest(code=code):
                result = grade.classify_lcb_results([True, code], 3)
                self.assertFalse(result["correctness"])
                self.assertEqual(result["category"], category)
                self.assertEqual(result["test_passed"], [True, False])
                self.assertEqual(result["unrun_count"], 1)

    def test_official_boolean_wrong_answer_prefix_is_valid(self):
        result = grade.classify_lcb_results([True, False], 4)
        self.assertFalse(result["correctness"])
        self.assertEqual(result["category"], "wrong_answer")
        self.assertEqual(result["unrun_count"], 2)

    def test_official_positive_comparison_not_equality_to_one(self):
        self.assertTrue(grade.classify_lcb_results([True, 1, 2, 0.5], 4)["correctness"])
        self.assertFalse(grade.classify_lcb_results([True, 0], 2)["correctness"])

    def test_empty_tests_and_missing_positive_results_cannot_pass(self):
        for results, count in (([], 2), ([], 0), ([True], 0), ([True], 2), ([True, True], 1), ([True], True), ([True], 1.0)):
            with self.subTest(results=results, count=count):
                with self.assertRaises(ValueError):
                    grade.classify_lcb_results(results, count)

    def test_corrupt_result_types_cannot_pass(self):
        for value in ("True", "-3", None, [True], {"passed": True}, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    grade.classify_lcb_results([value], 1)

    def test_nonempty_equal_length_official_sample(self):
        sample = {"input_output": json.dumps({"inputs": ["1", "2"], "outputs": ["2", "3"], "fn_name": None})}
        self.assertEqual(grade.lcb_test_count(sample), 2)
        self.assertEqual(grade.PER_TEST_SECONDS, 6)
        self.assertEqual(grade.lcb_wall_seconds(2), 19)

    def test_invalid_inventory_rejected_before_candidate_execution(self):
        for inputs, outputs in (([], []), (["1"], []), ("1", "2"), ([1], [2])):
            with self.subTest(inputs=inputs, outputs=outputs):
                with self.assertRaises(ValueError):
                    grade.lcb_test_count({"input_output": json.dumps({"inputs": inputs, "outputs": outputs})})


class ResultSchemaTests(unittest.TestCase):
    def test_generation_error_keeps_denominator_without_invoking_scorer(self):
        # No grading object or scorer installation is needed for a failed request.
        job = {"id": "coding/cf/1", "family": "coding", "generation": generation("reason</think>print(42)", error="request failed")}
        result = grade.grade_job(job)
        self.assertEqual(result["metrics"]["correct"], 0)
        self.assertEqual(result["metrics"]["generation_error"], 1)
        self.assertIsNone(result["details"]["correctness"])
        self.assertEqual(result["details"]["id"], job["id"])
        self.assertEqual(result["details"]["scorer"]["status"], "not_run")
        self.assertEqual(result["metrics"]["scorer_failure"], 0)

    def test_empty_ifeval_result_includes_primary_and_secondary_metrics(self):
        result = grade.grade_job({"id": "ifeval/1", "family": "ifeval", "generation": generation("")})
        self.assertEqual(result["metrics"]["ifeval_strict_prompt"], 0)
        self.assertEqual(result["metrics"]["ifeval_loose_instruction"], 0.0)
        self.assertEqual(result["metrics"]["empty"], 1)
        self.assertIsNone(result["details"]["correctness"])
        self.assertTrue(all(type(value) in (int, float) for value in result["metrics"].values()))
        json.dumps(result, allow_nan=False)

    def test_invalid_code_is_extraction_failure_not_scorer_failure(self):
        result = grade.grade_job({"id": "coding/cf/1", "family": "coding", "generation": generation("reason</think>Here is the solution:\nprint(1)")})
        self.assertEqual(result["metrics"]["extraction_failure"], 1)
        self.assertEqual(result["metrics"]["scorer_failure"], 0)
        self.assertEqual(result["details"]["extraction"]["status"], "invalid_python")

    def test_private_file_failure_preserves_completion_and_unknown_correctness(self):
        result = grade.grade_job({"id": "coding/cf/1", "family": "coding", "generation": generation("print(42)", thinking=False),
                                  "grading_error": grade._failure("grading_job_error", "private grading file unavailable")})
        self.assertEqual(result["metrics"]["scorer_failure"], 1)
        self.assertEqual(result["metrics"]["normal_completion"], 1)
        self.assertIsNone(result["details"]["correctness"])


class LargeReportTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(__file__).resolve().parents[1] / ".arl-cache/tests" / uuid.uuid4().hex
        self.folder.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.folder)

    def test_large_answer_and_error_preserve_failure_without_breaking_worker_summary(self):
        text = "x" * (1024 * 1024 + 1)
        report = grade.grade_job({
            "id": "ifeval/large", "family": "ifeval",
            "generation": generation(text, error=text, thinking=False),
        })
        grade.write_reports(report, self.folder)
        # Exercise the same capped reader used when the real worker finishes.
        summary = read_json(self.folder / "results.json")
        self.assertEqual(summary["metrics"]["generation_error"], 1)
        self.assertEqual(summary["metrics"]["correct"], 0)
        self.assertIsNone(summary["details"]["correctness"])
        artifact = summary["details"]["grading_artifact"]
        content = (self.folder / artifact["path"]).read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(), artifact["sha256"])
        full = json.loads(content)
        self.assertEqual(full["details"]["extraction"]["text"], text)
        self.assertEqual(full["details"]["completion"]["error"], text)


class GradingPathTests(unittest.TestCase):
    def test_relative_approved_filename(self):
        self.assertEqual(grade.validate_grading_filename("coding/abc.json"), ("coding", "abc.json"))

    def test_absolute_traversal_and_ambiguous_paths_rejected(self):
        for filename in ("", "/etc/passwd", "C:/host/job.json", "../job.json", "coding/../job.json", "./job.json", "coding//job.json", "coding\\job.json", "job.json/", "job\x00.json"):
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    grade.validate_grading_filename(filename)


if __name__ == "__main__":
    unittest.main()
