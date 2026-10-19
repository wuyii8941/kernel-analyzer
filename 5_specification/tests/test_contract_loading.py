"""Reject ambiguous explicit fields while preserving existing YAML inheritance."""
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernel_spec.contracts_io import ContractLoader, load_contract


class ContractLoading(unittest.TestCase):
    def test_existing_contracts_keep_the_same_meaning(self):
        paths = sorted((ROOT / "contracts").glob("*.yaml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(contract=path.stem):
                self.assertEqual(load_contract(path), yaml.safe_load(path.read_text()))

    def test_aliases_and_explicit_merge_overrides(self):
        source = "base: &b {x: 1, y: 2}\na: {<<: *b, x: 3}\nb: {<<: *b, y: 4}\n"
        self.assertEqual(yaml.load(source, Loader=ContractLoader), yaml.safe_load(source))

    def test_duplicate_keys_in_merge_sources_and_invalid_keys(self):
        for source in ("a: {<<: &b {x: 1, x: 2}}", "a: {<<: {x: 1}, <<: {y: 2}}",
                       "a: {true: 1, 1: 2}", "? [a, b]\n: 1"):
            with self.subTest(source=source), self.assertRaises(yaml.YAMLError):
                yaml.load(source, Loader=ContractLoader)
