import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


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


if __name__ == "__main__":
    unittest.main()
