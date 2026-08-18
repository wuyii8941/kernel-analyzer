from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def test_softmax_vjp_formula_matches_autograd() -> None:
    torch.manual_seed(41)
    logits = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(3, 7, dtype=torch.float64)
    probability = torch.softmax(logits, dim=-1)
    (probability * upstream).sum().backward()
    analytic = probability.detach() * (
        upstream - (upstream * probability.detach()).sum(dim=-1, keepdim=True)
    )
    torch.testing.assert_close(logits.grad, analytic, rtol=1e-12, atol=1e-12)


def test_softmax_proof_closes_forward_and_backward_programs() -> None:
    payload = json.loads(
        (ROOT / "results/coverage/cases/qwen128_softmax_fb.json").read_text()
    )
    proof = payload["concrete_program_proof"]
    boolean_gates = [value for value in proof.values() if isinstance(value, bool)]
    assert boolean_gates and all(boolean_gates)
    unit = payload["forward_backward_unit"]
    assert len(unit["forward"]) == 4
    assert len(unit["backward"]) == 4
    assert unit["aot_program"]["q_vjp_output_path"][-1] == "view_516"
    assert unit["aot_program"]["k_vjp_output_path"][-1] == "view_513"
    assert "reconstructs P" in unit["lowered_saved_state_transform"]


def test_softmax_proof_uses_full_state_vectors_and_exact_decomposition() -> None:
    payload = json.loads(
        (ROOT / "results/coverage/cases/qwen128_softmax_fb.json").read_text()
    )
    assert payload["numerical"]["states"] == 32
    assert payload["numerical"]["coordinates_per_state"] == 262144
    assert payload["numerical"]["two_repeats_exact"]
    total_pass = (
        payload["numerical"]["sources"]["semantic_total"]["direction_status"] == "PASS"
    )
    assert payload["semantic_total_coherent"] is total_pass
    assert (payload["verdict"] == "COHERENT_BIAS") is total_pass
    assert set(payload["numerical"]["sources"]) == {
        "kernel", "output_rounding", "saved_state_reconstruction",
        "forward_probability_kernel", "forward_probability_rounding", "semantic_total",
    }
    assert payload["bindings"]["sequence_nr"] == 14308
