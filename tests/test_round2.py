import gzip
import json
import unittest
from pathlib import Path

from scripts.round2_vl_math import (
    _verify_arithmetic_composite,
    _verify_elementary_unit,
)


ROOT = Path(__file__).resolve().parents[1]


def _node(
    name,
    target,
    args,
    shape,
    *,
    origin=None,
    source=None,
):
    edges = []
    for index, value in enumerate(args):
        if isinstance(value, dict) and "node" in value:
            edges.append(
                {
                    "argument_path": ["args", str(index)],
                    "source_node": value["node"],
                    "source_op": "call_function",
                }
            )
    return {
        "name": name,
        "target": target,
        "arguments": {"args": args, "kwargs": {}},
        "input_edges": edges,
        "tensor_meta": [shape, "torch.float32", False, [], None, False, {}],
        "source_fn_stack": source,
        "fwd_source_fn_stack": origin,
    }


class Round2ResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with gzip.open(ROOT / "results/final/vl.json.gz", "rt") as handle:
            cls.results = json.load(handle)["results"]

    def test_full_math_coverage(self) -> None:
        for name, expected_units in (("bf16_math", 3589), ("fp32_math", 3211)):
            result = self.results[name]
            self.assertEqual(
                result["status"],
                "COMPLETE_AOT_FORWARD_BACKWARD_DERIVATION",
            )
            self.assertEqual(
                result["denominator"]["semantic_forward_backward_units"],
                expected_units,
            )
            self.assertEqual(
                result["denominator"][
                    "units_with_complete_real_arithmetic_vjp_proof"
                ],
                expected_units,
            )
            self.assertTrue(all(result["gates"].values()))
            self.assertEqual(
                result["coverage"][
                    "semantic_unit_real_arithmetic_vjp_proof"
                ],
                1.0,
            )
            self.assertEqual(
                result["coverage"]["auxiliary_backward_node_derivation"],
                1.0,
            )

    def test_silu_causal_intervention_and_negative_directional_gate(self) -> None:
        for name in ("bf16_cause", "fp32_cause"):
            comparison = self.results[name]["comparison"]
            self.assertEqual(
                comparison["candidate_intervention_exact_parameter_count"],
                comparison["parameter_count"],
            )
            self.assertEqual(
                comparison["candidate_intervention_residual_l2"], 0.0
            )
            self.assertEqual(
                comparison["candidate_intervention_delta_cosine"], 1.0
            )
        bias = self.results["bias"]
        self.assertEqual(bias["state_count"], 6)
        self.assertTrue(
            bias["verdict"][
                "candidate_is_exactly_mediated_by_silu_backward_decomposition"
            ]
        )
        self.assertTrue(bias["verdict"]["all_losses_exact"])
        self.assertFalse(
            bias["verdict"][
                "cross_state_error_has_positive_mean_pairwise_inner_product"
            ]
        )
        self.assertFalse(bias["verdict"]["property_stage_entered"])
        self.assertLess(
            bias["global_direction"]["cross_state_error_inner_product"], 0
        )

    def test_structural_verifier_fails_closed_on_wrong_transpose_axis(self) -> None:
        source_stack = [["transpose", "transpose"]]
        source = _node("x", "placeholder", [], [2, 3, 4])
        forward = _node(
            "transpose",
            "aten.transpose.int",
            [{"node": "x"}, 0, 2],
            [4, 3, 2],
            source=source_stack,
        )
        backward = _node(
            "transpose_backward",
            "aten.transpose.int",
            [{"node": "q"}, 0, 1],
            [2, 3, 4],
            origin=source_stack,
        )
        proof = _verify_elementary_unit(
            [forward], [backward], {"x": source}
        )
        self.assertIsNotNone(proof)
        self.assertFalse(proof["passed"])
        self.assertFalse(proof["checks"]["backward_uses_same_axes"])

    def test_structural_verifier_fails_closed_on_wrong_saved_pow_input(self) -> None:
        source_stack = [["pow", "pow"]]
        source = _node("x", "placeholder", [], [2, 4])
        forward = _node(
            "pow",
            "aten.pow.Tensor_Scalar",
            [{"node": "x"}, 2],
            [2, 4],
            source=source_stack,
        )
        backward = [
            _node(
                "pow_b",
                "aten.pow.Tensor_Scalar",
                [{"node": "wrong_x"}, 1.0],
                [2, 4],
                origin=source_stack,
            ),
            _node(
                "scale",
                "aten.mul.Scalar",
                [{"node": "pow_b"}, 2.0],
                [2, 4],
                origin=source_stack,
            ),
            _node(
                "product",
                "aten.mul.Tensor",
                [{"node": "q"}, {"node": "scale"}],
                [2, 4],
                origin=source_stack,
            ),
        ]
        proof = _verify_arithmetic_composite(
            [forward], backward, {"x": source}
        )
        self.assertIsNotNone(proof)
        self.assertFalse(proof["passed"])
        self.assertFalse(
            proof["checks"]["saved_input_is_exact_forward_input"]
        )


if __name__ == "__main__":
    unittest.main()
