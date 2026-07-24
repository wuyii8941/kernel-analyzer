from __future__ import annotations

import sys
import unittest
from pathlib import Path


THEORY_DIR = Path(__file__).resolve().parents[1] / "theory_oracle"
if str(THEORY_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_DIR))

from evaluate_qwen3_common_t1b_smoke_v0_1 import (  # noqa: E402
    arithmetic_fields,
    bank_identity,
    build_bank,
)


class CharacterTokenizer:
    eos_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = True):
        del add_special_tokens
        return {"input_ids": [ord(value) for value in text]}


class CommonT1bSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "start_index_inclusive": 9008,
            "stop_index_exclusive": 9072,
            "prompt_template": "Q {start} {added} {removed}",
            "prompt_completion_separator": "\n",
            "completion_template": "A {result}",
            "include_eos_in_target": False,
            "require_prompt_token_prefix_stability": True,
        }

    def test_arithmetic_definition_matches_builtin_dataset(self) -> None:
        fields = arithmetic_fields(9008)
        self.assertEqual(fields["start"], 9015)
        self.assertEqual(fields["added"], 13)
        self.assertEqual(fields["removed"], 4)
        self.assertEqual(fields["result"], 9024)

    def test_static_bank_has_declared_scope_and_completion_mask(self) -> None:
        bank = build_bank(CharacterTokenizer(), self.config)
        self.assertEqual(len(bank), 64)
        self.assertEqual(bank[0]["dataset_index"], 9008)
        self.assertEqual(bank[-1]["dataset_index"], 9071)
        self.assertEqual(
            len(bank[0]["completion_token_ids"]),
            len(bank[0]["completion"]),
        )
        self.assertTrue(all(row["prefix_stable"] for row in bank))

    def test_bank_identity_is_deterministic_and_content_sensitive(self) -> None:
        first = build_bank(CharacterTokenizer(), self.config)
        second = build_bank(CharacterTokenizer(), self.config)
        self.assertEqual(bank_identity(first), bank_identity(second))
        second[0]["completion"] = "changed"
        self.assertNotEqual(bank_identity(first), bank_identity(second))


if __name__ == "__main__":
    unittest.main()
