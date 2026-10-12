import base64
import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import random
import unittest
from unittest.mock import patch
import zlib


MODULE_PATH = Path(__file__).resolve().parents[1] / "research" / "qwen38-nvfp4" / "task_data.py"
SPEC = importlib.util.spec_from_file_location("qwen38_task_data", MODULE_PATH)
task_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task_data)


def coding_row(index):
    return {
        "platform": "atcoder",
        "question_id": f"task_{index}",
        "question_content": f"Compute the sum for problem {index}.\nKeep Unicode: 雪.",
        "starter_code": "",
        "public_test_cases": json.dumps([
            {"input": "1 2\n", "output": "3\n", "testtype": "stdin"},
        ]),
        "private_test_cases": json.dumps([
            {"input": f"PRIVATE_INPUT_{index}\n", "output": f"PRIVATE_OUTPUT_{index}\n", "testtype": "stdin"},
        ]),
        "metadata": json.dumps({"private_note": f"PRIVATE_METADATA_{index}"}),
    }


def ifeval_row(index):
    return {
        "key": index,
        "prompt": f"  Exact prompt {index}: 雪\r\n\nDo not trim.\t ",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": [f"PRIVATE_KWARG_{index}"]}],
    }


def pools(count=144):
    return [coding_row(i) for i in range(count)], [ifeval_row(i) for i in range(count)]


def compressed_pickle(value, protocol=4):
    return base64.b64encode(zlib.compress(pickle.dumps(value, protocol=protocol))).decode("ascii")


class Exploit:
    def __reduce__(self):
        return (eval, ("__import__('builtins').__dict__.__setitem__('qwen38_pickle_executed', True)",))


class TaskSelectionTests(unittest.TestCase):
    def test_selection_matches_independent_hash_ranking_and_exact_counts(self):
        coding, ifeval = pools()
        result = task_data.prepare_tasks(coding, ifeval)
        self.assertEqual(set(result), {"public", "private", "selection"})
        for family, rows in (("coding", coding), ("ifeval", ifeval)):
            dataset_ids = [
                f"{row['platform']}/{row['question_id']}" if family == "coding" else str(row["key"])
                for row in rows
            ]
            expected = sorted(dataset_ids, key=lambda value: (
                hashlib.sha256(f"arl-qwen38-nvfp4-v1/{family}/{value}".encode("utf-8")).hexdigest(), value,
            ))
            for split, subset in (("compatibility", expected[:8]), ("primary", expected[8:136])):
                ids = [f"{family}/{value}" for value in subset]
                self.assertEqual(result["selection"][family][split], ids)
                for projection in ("public", "private"):
                    selected = [row["id"] for row in result[projection][split] if row["family"] == family]
                    self.assertEqual(selected, ids)
        self.assertEqual(len(result["public"]["compatibility"]), 16)
        self.assertEqual(len(result["public"]["primary"]), 256)
        for split in ("compatibility", "primary"):
            families = [row["family"] for row in result["public"][split]]
            count = 8 if split == "compatibility" else 128
            self.assertEqual(families, ["coding"] * count + ["ifeval"] * count)

    def test_selection_is_row_order_independent_and_accepts_iterators(self):
        coding, ifeval = pools()
        expected = task_data.prepare_tasks(coding, ifeval)
        random.Random(19).shuffle(coding)
        random.Random(23).shuffle(ifeval)
        self.assertEqual(task_data.prepare_tasks(iter(coding), iter(ifeval)), expected)
        self.assertEqual(task_data.prepare_tasks(reversed(coding), reversed(ifeval)), expected)

    def test_compatibility_and_unselected_rows_are_excluded_from_primary(self):
        coding, ifeval = pools(160)
        result = task_data.prepare_tasks(coding, ifeval)
        for family in ("coding", "ifeval"):
            compatible = set(result["selection"][family]["compatibility"])
            primary = set(result["selection"][family]["primary"])
            self.assertEqual(len(compatible), 8)
            self.assertEqual(len(primary), 128)
            self.assertFalse(compatible & primary)
            all_ids = {
                f"coding/atcoder/task_{index}" if family == "coding" else f"ifeval/{index}"
                for index in range(160)
            }
            self.assertEqual(len(all_ids - compatible - primary), 24)

    def test_exact_minimum_pool_is_supported(self):
        result = task_data.prepare_tasks(*pools(136))
        self.assertEqual(len(result["public"]["primary"]), 256)

    def test_short_pools_fail_instead_of_reducing_sample(self):
        coding, ifeval = pools(136)
        for family, args in (("coding", (coding[:-1], ifeval)), ("ifeval", (coding, ifeval[:-1]))):
            with self.subTest(family=family), self.assertRaises(ValueError):
                task_data.prepare_tasks(*args)

    def test_identical_and_conflicting_duplicate_ids_are_rejected(self):
        coding, ifeval = pools()
        for family in ("coding", "ifeval"):
            for conflict in (False, True):
                extra = copy.deepcopy(coding[0] if family == "coding" else ifeval[0])
                if conflict:
                    extra["question_content" if family == "coding" else "prompt"] = "Conflicting content"
                args = (coding + [extra], ifeval) if family == "coding" else (coding, ifeval + [extra])
                with self.subTest(family=family, conflict=conflict), self.assertRaises(ValueError):
                    task_data.prepare_tasks(*args)

    def test_same_question_id_on_different_platforms_is_distinct(self):
        coding, ifeval = pools(136)
        coding[1]["question_id"] = coding[0]["question_id"]
        coding[1]["platform"] = "codeforces"
        result = task_data.prepare_tasks(coding, ifeval)
        all_ids = result["selection"]["coding"]["compatibility"] + result["selection"]["coding"]["primary"]
        self.assertIn("coding/atcoder/task_0", all_ids)
        self.assertIn("coding/codeforces/task_0", all_ids)

    def test_digest_ties_break_by_dataset_id(self):
        class ConstantDigest:
            def hexdigest(self):
                return "0" * 64
        coding, ifeval = pools(136)
        with patch.object(task_data.hashlib, "sha256", return_value=ConstantDigest()):
            result = task_data.prepare_tasks(reversed(coding), reversed(ifeval))
        expected = [f"ifeval/{key}" for key in sorted(map(str, range(136)))]
        self.assertEqual(result["selection"]["ifeval"]["compatibility"], expected[:8])
        self.assertEqual(result["selection"]["ifeval"]["primary"], expected[8:])

    def test_public_projection_does_not_disclose_grading_information(self):
        coding, ifeval = pools(136)
        result = task_data.prepare_tasks(coding, ifeval)
        visible_json = json.dumps(result["public"])
        for marker in ("PRIVATE_INPUT_", "PRIVATE_OUTPUT_", "PRIVATE_METADATA_", "PRIVATE_KWARG_", "instruction_id_list", "input_output"):
            self.assertNotIn(marker, visible_json)
        for split in ("compatibility", "primary"):
            for row in result["public"][split]:
                self.assertEqual(set(row), {"id", "family", "dataset_id", "messages", "prompt_sha256"})
                self.assertEqual(row["messages"][0]["role"], "user")
                canonical = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                self.assertEqual(row["prompt_sha256"], hashlib.sha256(canonical.encode("utf-8")).hexdigest())
            for row in result["private"][split]:
                self.assertEqual(set(row), {"id", "family", "dataset_id", "grading"})

    def test_hidden_data_changes_neither_prompts_nor_selection(self):
        coding, ifeval = pools(136)
        original_inputs = copy.deepcopy((coding, ifeval))
        before = task_data.prepare_tasks(coding, ifeval)
        self.assertEqual((coding, ifeval), original_inputs)
        coding[0]["private_test_cases"] = json.dumps([
            {"input": "changed hidden input", "output": "changed hidden output", "testtype": "stdin"},
        ])
        ifeval[0]["kwargs"][0]["keywords"] = ["changed hidden instruction argument"]
        after = task_data.prepare_tasks(coding, ifeval)
        self.assertEqual(before["public"], after["public"])
        self.assertEqual(before["selection"], after["selection"])
        self.assertNotEqual(before["private"], after["private"])

    def test_ifeval_prompt_and_original_grading_fields_are_preserved(self):
        coding, ifeval = pools(136)
        original = copy.deepcopy(ifeval)
        result = task_data.prepare_tasks(coding, ifeval)
        for split in ("compatibility", "primary"):
            public = {row["id"]: row for row in result["public"][split]}
            for row in result["private"][split]:
                if row["family"] != "ifeval":
                    continue
                source = original[int(row["dataset_id"])]
                self.assertEqual(row["grading"], source)
                self.assertEqual(public[row["id"]]["messages"], [{"role": "user", "content": source["prompt"]}])
        ifeval[0]["kwargs"][0]["keywords"].append("caller mutation")
        saved = next(row for split in result["private"].values() for row in split if row["id"] == "ifeval/0")
        self.assertEqual(saved["grading"], original[0])

    def test_official_sample_layout_for_plain_and_compressed_private_tests(self):
        coding, ifeval = pools(136)
        public = [{"input": "[1, 2]\n", "output": "3", "testtype": "functional"}]
        private = [{"input": "[4, 5]\n", "output": "9", "testtype": "functional"}]
        expected = {"input_output": json.dumps({
            "inputs": [test["input"] for test in public + private],
            "outputs": [test["output"] for test in public + private],
            "fn_name": "add",
        })}
        envelopes = [json.dumps(private)]
        envelopes += [compressed_pickle(json.dumps(private), protocol) for protocol in range(6)]
        envelopes += [compressed_pickle(json.dumps(private).encode("utf-8"), protocol) for protocol in range(3, 6)]
        for encoded in envelopes:
            row = coding[0]
            row.update({
                "platform": "leetcode", "starter_code": "class Solution:\n    def add(self, a, b):\n        pass",
                "public_test_cases": json.dumps(public), "private_test_cases": encoded,
                "metadata": json.dumps({"func_name": "add"}),
            })
            with self.subTest(encoded=encoded[:20]):
                result = task_data.prepare_tasks(coding, ifeval)
                selected = next(item for split in result["private"].values() for item in split if item["id"] == "coding/leetcode/task_0")
                self.assertEqual(selected["grading"], expected)

    def test_stdin_sample_has_null_function_name_and_preserves_whitespace(self):
        coding, ifeval = pools(136)
        result = task_data.prepare_tasks(coding, ifeval)
        selected = next(row for split in result["private"].values() for row in split if row["id"] == "coding/atcoder/task_0")
        self.assertEqual(selected["grading"], {"input_output": json.dumps({
            "inputs": ["1 2\n", "PRIVATE_INPUT_0\n"], "outputs": ["3\n", "PRIVATE_OUTPUT_0\n"], "fn_name": None,
        })})


    def test_private_payload_cannot_execute_pickle_code(self):
        coding, ifeval = pools(136)
        self.assertFalse(hasattr(builtins, "qwen38_pickle_executed"))
        coding[0]["private_test_cases"] = compressed_pickle(Exploit())
        try:
            with self.assertRaises(ValueError):
                task_data.prepare_tasks(coding, ifeval)
            self.assertFalse(hasattr(builtins, "qwen38_pickle_executed"))
        finally:
            if hasattr(builtins, "qwen38_pickle_executed"):
                delattr(builtins, "qwen38_pickle_executed")

    def test_malformed_private_envelopes_and_nonliteral_pickles_are_rejected(self):
        valid_pickle = pickle.dumps("[]", protocol=4)
        payloads = [
            "not-json-or-base64", "@@@@", base64.b64encode(b"not zlib").decode("ascii"),
            compressed_pickle([]), compressed_pickle(5), compressed_pickle({"tests": []}),
            base64.b64encode(zlib.compress(valid_pickle + b"trailing")).decode("ascii"),
            base64.b64encode(zlib.compress(valid_pickle) + b"trailing").decode("ascii"),
            base64.b64encode(zlib.compress(valid_pickle)[:-2]).decode("ascii"),
            base64.b64encode(zlib.compress(b"\x80\x04")).decode("ascii"),
            compressed_pickle("not json"), compressed_pickle("null"),
        ]
        coding, ifeval = pools(136)
        for payload in payloads:
            coding[0]["private_test_cases"] = payload
            with self.subTest(payload=payload[:30]), self.assertRaises(ValueError):
                task_data.prepare_tasks(coding, ifeval)

    def test_malformed_coding_rows_are_not_silently_dropped(self):
        changes = [
            {"platform": "unknown"}, {"question_id": 123}, {"question_id": ""}, {"question_id": "a/b"},
            {"question_content": None}, {"starter_code": 123},
            {"public_test_cases": "null"}, {"public_test_cases": "[]", "private_test_cases": "[]"},
            {"public_test_cases": '[{"input": "x", "output": "y"}]'},
            {"public_test_cases": '[{"input": [], "output": "y", "testtype": "stdin"}]'},
            {"private_test_cases": '[{"input": "x", "output": "y", "testtype": "other"}]'},
            {"metadata": "[]"}, {"metadata": '{"func_name": 10}'},
            {"metadata": '{"func_name": null, "func_name": "f"}'},
            {"metadata": '{"value": NaN}'},
        ]
        for change in changes:
            coding, ifeval = pools(137)
            # With 137 rows, the maximum-ranked identity is guaranteed excluded.
            excluded = max(coding, key=lambda row: hashlib.sha256(
                f"arl-qwen38-nvfp4-v1/coding/{row['platform']}/{row['question_id']}".encode("utf-8")
            ).hexdigest())
            excluded.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                task_data.prepare_tasks(coding, ifeval)
        coding, ifeval = pools(136)
        del coding[0]["metadata"]
        with self.assertRaises(ValueError):
            task_data.prepare_tasks(coding, ifeval)
        coding, ifeval = pools(136)
        with self.assertRaises(ValueError):
            task_data.prepare_tasks(coding + [None], ifeval)

    def test_malformed_ifeval_rows_are_rejected(self):
        changes = [
            {"key": "0"}, {"key": True}, {"key": 1.0}, {"prompt": None}, {"prompt": ""},
            {"instruction_id_list": []}, {"instruction_id_list": [1]},
            {"kwargs": []}, {"kwargs": [None]}, {"kwargs": [{"value": float("nan")}]},
            {"kwargs": [{"value": object()}]}, {"kwargs": [{1: "changed key"}]},
            {"kwargs": [{"value": (1, 2)}]},
        ]
        for change in changes:
            coding, ifeval = pools(136)
            ifeval[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                task_data.prepare_tasks(coding, ifeval)
        coding, ifeval = pools(136)
        del ifeval[0]["kwargs"]
        with self.assertRaises(ValueError):
            task_data.prepare_tasks(coding, ifeval)


if __name__ == "__main__":
    unittest.main()
