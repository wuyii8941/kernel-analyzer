import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import Invalid


class RequiredInputs(unittest.TestCase):
    required = {"GDR": ("q", "k", "v", "g", "beta"),
                "SSU": ("state", "x", "dt", "A", "B", "C", "D", "z", "dt_bias",
                        "dt_softplus", "state_batch_indices", "dtype"),
                "LORA": ("x", "weights", "mapping", "scaling", "out_before",
                         "input_dtype", "output_dtype"),
                "SAMPLING": ("logits", "k", "p", "dtype"),
                "BSR": ("shape", "blocksize", "crow", "col", "values", "dtype")}

    def test_missing_required_fields_are_invalid_inputs(self):
        for cid, fields in self.required.items():
            contract = registry.load(cid)
            example = contract.data["examples"][0]
            for field in fields:
                inp = copy.deepcopy(example["input"])
                del inp[field]
                with self.subTest(cid=cid, field=field), self.assertRaises(Invalid):
                    contract.adapter.reference(example["profile"], inp)

    def test_inputs_must_be_a_dictionary(self):
        for cid in self.required:
            contract = registry.load(cid)
            profile = contract.data["examples"][0]["profile"]
            for inp in (None, [], "state"):
                with self.subTest(cid=cid, inp=inp), self.assertRaises(Invalid):
                    contract.adapter.reference(profile, inp)

    def test_lora_mapping_is_a_vector_not_an_arbitrary_iterable(self):
        c = registry.load('LORA')
        profile = 'vllm016_shrink_f32out_real'
        base = dict(x=[[2]], weights=[[[[3]]]], mapping=[0], scaling=1,
                    out_before=[[[9]]], input_dtype='f16', output_dtype='f32')
        self.assertEqual(c.adapter.reference(profile, base).fields['y'], [[[6]]])
        for value in ('0', {0: 'ignored'}, 0, None, [True], [0.5], [], [0, 0]):
            with self.subTest(value=value), self.assertRaises(Invalid):
                c.adapter.reference(profile, dict(base, mapping=value))

    def test_sampling_thresholds_are_per_row_vectors(self):
        c = registry.load('SAMPLING')
        profile = 'vllm016_topk_topp_keep_real'
        base = dict(logits=[[2, 1]], k=None, p=None, dtype='f32')
        for field in ('k', 'p'):
            expected = c.adapter.reference(profile, dict(base, **{field: [1]})).fields
            self.assertEqual(c.adapter.reference(profile, dict(base, **{field: (1,)})).fields, expected)
            for value in ('1', {1: 'ignored'}, 1, [], [1, 1], [True]):
                with self.subTest(field=field, value=value), self.assertRaises(Invalid):
                    c.adapter.reference(profile, dict(base, **{field: value}))

    def test_sparse_matrix_requires_all_structure_fields(self):
        contract = registry.load("BSR")
        example = contract.data["examples"][0]
        for field in ("shape", "blocksize", "crow", "col", "values"):
            inp = copy.deepcopy(example["input"])
            del inp[field]
            with self.subTest(field=field), self.assertRaises(Invalid):
                contract.adapter.reference(example["profile"], inp)

    def test_sparse_structure_requires_integer_vectors(self):
        contract = registry.load("BSR")
        example = contract.data["examples"][0]
        cases = [(field, value) for field in ("shape", "blocksize", "crow", "col")
                 for value in (None, 1, "12", [[1]], [3.9], [True])]
        cases += [(field, value) for field in ("shape", "blocksize")
                  for value in ([], [1], [1, 2, 3])]
        for field, value in cases:
            inp = copy.deepcopy(example["input"])
            inp[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(Invalid):
                contract.adapter.reference(example["profile"], inp)

    def test_field_names_must_be_strings(self):
        for cid in self.required:
            contract = registry.load(cid)
            example = contract.data["examples"][0]
            inp = copy.deepcopy(example["input"])
            inp.update({1: 0, "unknown": 0})
            with self.subTest(cid=cid), self.assertRaises(Invalid):
                contract.adapter.reference(example["profile"], inp)


if __name__ == "__main__":
    unittest.main()
