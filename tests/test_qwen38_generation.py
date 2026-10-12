import importlib.util
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
