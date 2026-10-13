from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


MODULE = Path(__file__).resolve().parents[1] / "research/qwen38-nvfp4/worker/generate.py"
SPEC = importlib.util.spec_from_file_location("qwen38_generate", MODULE)
generate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate)


class RepeatedBlockTests(unittest.TestCase):
    def test_exact_four_blocks_and_unaligned_final_offset(self):
        block = list(range(128))
        self.assertTrue(generate.repeated_block(block * 4))
        self.assertTrue(generate.repeated_block([900, 901, 902] + block * 4))

    def test_short_or_interrupted_repetition_is_not_flagged(self):
        block = list(range(128))
        self.assertFalse(generate.repeated_block(block * 3))
        self.assertFalse(generate.repeated_block((block * 4)[:-1]))
        changed = block * 4
        changed[256 + 47] = 999
        self.assertFalse(generate.repeated_block(changed))

    def test_matching_boundary_tokens_do_not_establish_repetition(self):
        blocks = [[0] + list(range(start, start + 127)) for start in (1, 201, 401, 601)]
        self.assertFalse(generate.repeated_block(sum(blocks, [])))


class CompletionPersistenceTests(unittest.TestCase):
    class Engine:
        def __init__(self, batches):
            self.batches = list(batches)

        def has_unfinished_requests(self):
            return bool(self.batches)

        def step(self):
            batch = self.batches.pop(0)
            if isinstance(batch, Exception):
                raise batch
            return batch

    @staticmethod
    def result(identity):
        return SimpleNamespace(request_id=identity, finished=True)

    def test_out_of_order_completions_retain_original_task_identity(self):
        pending = {"external-a": {"id": "coding/a"}, "external-b": {"id": "ifeval/b"}}
        engine = self.Engine([[self.result("external-b")], [self.result("external-a")]])
        persisted = []
        generate.consume_completed(engine, pending, lambda task, result: persisted.append(task["id"]))
        self.assertEqual(persisted, ["ifeval/b", "coding/a"])
        self.assertEqual(pending, {})

    def test_engine_failure_does_not_discard_an_earlier_completed_response(self):
        pending = {"a": {"id": "a"}, "b": {"id": "b"}}
        engine = self.Engine([[self.result("b")], RuntimeError("engine stopped")])
        persisted = []
        with self.assertRaises(RuntimeError):
            generate.consume_completed(engine, pending, lambda task, result: persisted.append(task["id"]))
        self.assertEqual(persisted, ["b"])
        self.assertEqual(set(pending), {"a"})

    def test_persistence_failure_never_marks_the_response_durable(self):
        pending = {"a": {"id": "a"}}
        engine = self.Engine([[self.result("a")]])

        def disk_full(task, result):
            raise OSError("disk full")

        with self.assertRaises(OSError):
            generate.consume_completed(engine, pending, disk_full)
        self.assertEqual(set(pending), {"a"})

    def test_missing_and_duplicate_completions_cannot_report_success(self):
        with self.assertRaises(RuntimeError):
            generate.consume_completed(self.Engine([]), {"a": {"id": "a"}}, lambda *_: None)
        persisted = []
        engine = self.Engine([[self.result("a"), self.result("a")]])
        with self.assertRaises(RuntimeError):
            generate.consume_completed(engine, {"a": {"id": "a"}}, lambda task, result: persisted.append(task["id"]))
        self.assertEqual(persisted, ["a"])


class SeedProtocolTests(unittest.TestCase):
    @staticmethod
    def frozen(split="primary"):
        count = 8 if split == "compatibility" else 128
        return {"split": split, "thinking_enabled": True, "reasoning_effort": "xhigh",
                "tasks": [{"id": f"{family}/{index}", "family": family}
                          for family in ("coding", "ifeval") for index in range(count)]}

    def test_nonempty_unique_integer_protocol_seed_list_is_required(self):
        frozen = self.frozen()
        for seeds in (None, [], 17, (17,), [17, 17], [17, 29, 29], [17, 99], [True], ["17"]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                generate.validate_protocol({"split": "primary", "seeds": seeds}, frozen)
        with self.assertRaises(ValueError):
            generate.validate_protocol({"split": "primary", "seed": 17}, frozen)
        with self.assertRaises(ValueError):
            generate.validate_protocol({"split": "primary", "seed": 17, "seeds": [17]}, frozen)

    def test_compatibility_rejects_primary_only_seeds(self):
        frozen = self.frozen("compatibility")
        generate.validate_protocol({"split": "compatibility", "seeds": [17]}, frozen)
        for seeds in ([29], [43], [17, 29]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                generate.validate_protocol({"split": "compatibility", "seeds": seeds}, frozen)

    def test_primary_subset_keeps_requested_order(self):
        settings = {"split": "primary", "seeds": [43, 17]}
        generate.validate_protocol(settings, self.frozen())
        self.assertEqual(settings["seeds"], [43, 17])

    def test_duplicate_frozen_identity_is_rejected_before_submission(self):
        frozen = self.frozen()
        frozen["tasks"][-1]["id"] = frozen["tasks"][0]["id"]
        with self.assertRaises(ValueError):
            generate.validate_protocol({"split": "primary", "seeds": [17, 29]}, frozen)


class SeedReceiptStateTests(unittest.TestCase):
    @staticmethod
    def block(seed, load_seconds=0, warmup_seconds=0):
        return {"status": "completed",
                "metrics": {"requested": 256, "recorded": 256, "generation_errors": 0,
                            "generated_tokens": seed * 100, "generation_wall_seconds": seed,
                            "load_seconds": load_seconds, "warmup_seconds": warmup_seconds},
                "details": {"seed": seed, "sampling": {"seed": seed, "max_tokens": 16384}}}

    def test_aggregate_counts_disjoint_blocks_and_startup_once(self):
        settings = {"checkpoint": "checkpoint", "seeds": [17, 29, 43]}
        blocks = [self.block(17, 12, 3), self.block(29), self.block(43)]
        result = generate.aggregate_results({"split": "primary"}, settings, blocks, 12, 3)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["metrics"], {"requested": 768, "recorded": 768, "generation_errors": 0,
                                            "generated_tokens": 8900, "generation_wall_seconds": 89,
                                            "load_seconds": 12, "warmup_seconds": 3})
        self.assertEqual(result["details"]["completed_seeds"], [17, 29, 43])

    def test_later_failure_preserves_completed_receipt_and_stops_schedule(self):
        settings = {"checkpoint": "checkpoint", "seeds": [17, 29, 43]}
        frozen = {"split": "primary"}
        durable = {}
        attempted = []
        completed = self.block(17, 12, 3)
        first_rows = [{"id": "coding/original", "seed": 17, "token_ids": [101, 102]}]

        def persist(name, value):
            durable[name] = deepcopy(value)

        def generate_block(llm, tokenizer, frozen, settings, engine_args, seed, load_seconds, warmup_seconds):
            attempted.append(seed)
            if seed == 17:
                persist("seed-17/generations.jsonl", first_rows)
                persist("seed-17/results.json", completed)
                return deepcopy(completed)
            persist("seed-29/results.json", {"status": "failed", "metrics": {"recorded": 1}})
            raise RuntimeError("GPU failed after one response in the second block")

        with patch.object(generate, "save", side_effect=persist), patch.object(generate, "run_requests", side_effect=generate_block):
            with self.assertRaisesRegex(RuntimeError, "GPU failed"):
                generate.run_seed_blocks(None, None, frozen, settings, {"seed": 17}, 12, 3)
        self.assertEqual(attempted, [17, 29])
        self.assertEqual(durable["seed-17/generations.jsonl"], first_rows)
        self.assertEqual(durable["seed-17/results.json"], completed)
        root = durable["results.json"]
        self.assertEqual(root["status"], "failed")
        self.assertEqual(root["details"]["completed_seeds"], [17])
        self.assertEqual(root["details"]["failed_seed"], 29)
        self.assertEqual(root["metrics"], completed["metrics"])
        self.assertNotIn("seed-43/results.json", durable)


if __name__ == "__main__":
    unittest.main()
