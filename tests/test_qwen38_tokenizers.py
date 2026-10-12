import copy
import importlib.util
from pathlib import Path
import unittest


MODULE = Path(__file__).resolve().parents[1] / "research/qwen38-nvfp4/worker/freeze.py"
SPEC = importlib.util.spec_from_file_location("qwen38_freeze", MODULE)
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)


def signature():
    return {
        "vocabulary_sha256": "reference-token-map", "vocabulary_size": 3,
        "added_token_definitions": {"1": {"content": "</think>", "special": False}},
        "model_token_config": {"vocab_size": 3, "bos_token_id": 0, "eos_token_id": 2, "pad_token_id": None},
        "frontend_defaults": {"pad_token_id": 0, "eos_token_id": 2, "special_tokens": {"stop": 2}},
    }


class TokenizerCompatibilityTests(unittest.TestCase):
    def test_frontend_padding_preferences_are_recorded_not_token_identity(self):
        reference = signature()
        candidate = copy.deepcopy(reference)
        candidate["frontend_defaults"] = {"pad_token_id": 2, "eos_token_id": 2, "special_tokens": {}}
        result = freeze.tokenizer_compatibility(reference, candidate)
        self.assertTrue(result["compatible"])
        self.assertTrue(result["frontend_defaults_differ"])
        self.assertTrue(all(result["checks"].values()))

    def test_mapping_backend_flags_and_model_token_config_cannot_be_normalized_away(self):
        reference = signature()
        changes = {
            "vocabulary_sha256": "different-token-map",
            "vocabulary_size": 4,
            "added_token_definitions": {"1": {"content": "</think>", "special": True}},
            "model_token_config": {"vocab_size": 3, "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": None},
        }
        for key, value in changes.items():
            with self.subTest(field=key):
                candidate = copy.deepcopy(reference)
                candidate[key] = value
                result = freeze.tokenizer_compatibility(reference, candidate)
                self.assertFalse(result["compatible"])
                self.assertFalse(result["checks"][key])
                self.assertFalse(result["frontend_defaults_differ"])


if __name__ == "__main__":
    unittest.main()
