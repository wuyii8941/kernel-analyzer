"""Input-boundary and paged-attention regression checks."""
import copy
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import evaluate


class AttentionGuarantees(unittest.TestCase):
    def test_attention_requires_complete_input_records(self):
        cases = (
            ("ATTN", "torch210_sdpa_defined_real",
             ("q", "k", "v", "dtype", "attn_mask", "attn_mask_dtype", "is_causal", "scale",
              "enable_gqa", "dropout_p"), "unsupported"),
            ("PACK", "flashattn283_varlen_causal_real",
             ("q", "k", "v", "cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k",
              "dropout_p", "softmax_scale", "causal", "window_size", "softcap", "alibi_slopes",
              "deterministic", "return_attn_probs", "block_table", "dtype"), "invalid_input"),
            ("PAGEDATTN", "vllm016_causal_2d_real", ("query", "cu_seqlens_q", "seq_lens", "key_cache",
                    "value_cache", "block_table", "scale", "causal", "window", "softcap", "dtype"),
             "invalid_input"),
        )
        for cid, profile, required, extra_outcome in cases:
            contract = registry.load(cid)
            complete = copy.deepcopy(contract.data["examples"][0]["input"])
            for field in required:
                inp = {key: value for key, value in complete.items() if key != field}
                with self.subTest(contract=cid, missing=field):
                    self.assertEqual(evaluate(contract.adapter.reference, profile, inp).outcome, "invalid_input")
            for inp in (None, [], "task"):
                with self.subTest(contract=cid, input_type=type(inp).__name__):
                    self.assertEqual(evaluate(contract.adapter.reference, profile, inp).outcome, "invalid_input")
            with self.subTest(contract=cid, extra=True):
                self.assertEqual(evaluate(contract.adapter.reference, profile,
                                          dict(complete, unexpected=True)).outcome, extra_outcome)
            bad = dict(complete)
            bad[1] = "not a field name"
            self.assertEqual(evaluate(contract.adapter.reference, profile, bad).outcome, "invalid_input")

    def test_paged_index_vectors_are_not_arbitrary_iterables(self):
        contract = registry.load("PAGEDATTN")
        inp = copy.deepcopy(contract.data["examples"][0]["input"])
        # These strings/dictionaries would otherwise be iterated as valid indices,
        # silently manufacturing a different input tensor.
        replacements = (("cu_seqlens_q", "04"), ("cu_seqlens_q", {0: None, 4: None}),
                        ("seq_lens", "4"), ("seq_lens", {4: None}),
                        ("block_table", ["10"]), ("block_table", [{1: None, 0: None}]),
                        ("cu_seqlens_q", None), ("seq_lens", 4), ("block_table", [None]))
        for field, value in replacements:
            with self.subTest(field=field, value=value):
                result = evaluate(contract.adapter.reference, "vllm016_causal_2d_real", dict(inp, **{field: value}))
                self.assertEqual(result.outcome, "invalid_input", result.note)

    def test_paged_softcap_does_not_coerce_invalid_falsy_inputs(self):
        contract = registry.load("PAGEDATTN")
        inp = copy.deepcopy(contract.data["examples"][0]["input"])
        for cap in (False, "", [], {}):
            with self.subTest(softcap=cap):
                result = evaluate(contract.adapter.reference, "vllm016_causal_2d_real", dict(inp, softcap=cap))
                self.assertEqual(result.outcome, "invalid_input")

    def test_paged_outputs_against_explicit_logical_sequences(self):
        from itertools import permutations
        adapter = registry.load("PAGEDATTN").adapter
        # Independent task data before packing. Two KV heads, three coordinates
        # and unequal positive sequence/query lengths.
        values = [
            [[[2, -1, 7], [11, 3, -2]], [[5, 4, 1], [13, 6, 9]],
             [[17, -3, 8], [19, 2, 5]]],
            [[[23, 8, -4], [29, -5, 3]], [[31, 7, 2], [37, 9, -6]]],
        ]
        zero = [[0, 0, 0], [0, 0, 0]]
        blocks = [[values[0][0], values[0][1]], [values[0][2], zero], values[1]]
        for order in permutations(range(3)):
            physical = {logical: order.index(logical) for logical in range(3)}
            cache = [copy.deepcopy(blocks[i]) for i in order]
            keys = [[[[0, 0, 0] for _ in range(2)] for slot in block] for block in cache]
            for window in ([-1, -1], [0, -1], [1, -1]):
                inp = dict(query=[[[0, 0, 0] for _ in range(4)] for _ in range(3)],
                           cu_seqlens_q=[0, 2, 3], seq_lens=[3, 2],
                           key_cache=keys, value_cache=cache,
                           block_table=[[physical[0], physical[1]], [physical[2], physical[2]]],
                           scale=F(1, 2), causal=True, window=window, softcap=0, dtype="f32")
                before = copy.deepcopy(inp)
                expected = []
                # Absolute query positions are 1, 2 in sequence 0 and 1 in sequence 1.
                for sequence, positions in ((values[0], (1, 2)), (values[1], (1,))):
                    for position in positions:
                        selected = [row for j, row in enumerate(sequence)
                                    if j <= position and (window[0] < 0 or position-j <= window[0])]
                        expected.append([[sum(F(row[h//2][d]) for row in selected)/len(selected)
                                          for d in range(3)] for h in range(4)])
                with self.subTest(order=order, window=window):
                    result = adapter.reference("vllm016_causal_2d_real", inp)
                    self.assertEqual(result.fields, {"out": expected})
                    self.assertEqual(inp, before)


if __name__ == "__main__":
    unittest.main()
