import copy
import gzip
import json
from pathlib import Path

from scripts.build_concrete_fb_witnesses import (
    direct_theorem,
    source_output_matches,
)
from scripts.build_fb_proof_unit_ledger import build_components, digest, proof_unit


ROOT = Path(__file__).resolve().parents[1]


def load_gzip(relative: str):
    with gzip.open(ROOT / relative, "rt") as handle:
        return json.load(handle)


def test_all_twelve_cells_have_concrete_witnesses_for_the_full_primary_denominator():
    paths = [
        f"results/coverage/concrete_fb_witnesses/{architecture}_seq{seq}.json.gz"
        for architecture in ("qwen", "mamba", "phi4", "deepseek8b")
        for seq in (64, 128, 256)
    ]
    payloads = [load_gzip(path) for path in paths]
    assert all(row["status"] == "COMPLETE_CONCRETE_FB_WITNESSES" for row in payloads)
    assert sum(row["denominator"]["primary_fb_units"] for row in payloads) == 186807
    assert sum(row["denominator"]["analytically_proved"] for row in payloads) == 186807
    assert sum(row["denominator"]["unresolved"] for row in payloads) == 0
    assert all(len(row["witnesses"]) == row["denominator"]["primary_fb_units"] for row in payloads)


def test_parametric_theorems_reject_wrong_backward_programs():
    assert direct_theorem((("aten.mm.default",), ("aten.neg.default",))) is None
    assert direct_theorem((("aten.bmm.default",), ("aten.bmm.default",))) is None
    assert direct_theorem((("aten.silu.default",), ("aten.mul.Tensor",))) is None
    assert direct_theorem((("aten._softmax.default",), ("aten.sum.dim_IntList",))) is None
    assert direct_theorem((("aten.embedding.default",), ("aten.index.Tensor",))) is None
    assert direct_theorem((("aten.view.default",), ("aten.mul.Tensor",))) is None


def test_runtime_tensor_source_binding_rejects_identity_changes():
    output = {
        "storage_id": 7, "storage_offset": "0", "shape": ["2", "3"],
        "stride": ["3", "1"], "dtype": "torch.bfloat16", "device": "cuda:0",
        "layout": "torch.strided",
    }
    source = {"output_tensors": [output]}
    assert source_output_matches(dict(output), source)
    for field, replacement in (
        ("storage_id", 8), ("storage_offset", "1"),
        ("shape", ["3", "2"]), ("stride", ["1", "2"]),
        ("dtype", "torch.float32"),
    ):
        changed = dict(output)
        changed[field] = replacement
        assert not source_output_matches(changed, source)


def test_multishape_ledger_rejects_tampered_component_witnesses():
    ledger = load_gzip("results/coverage/phi4_seq64_invocation_ledger.json.gz")
    witnesses = load_gzip("results/coverage/concrete_fb_witnesses/phi4_seq64.json.gz")
    contract = json.loads((ROOT / "results/coverage/coverage_contract.json").read_text())
    components, audit = build_components("phi4_mini_3p8b", ledger)
    assert audit["dangling_origin_links"] == []
    witness_by_members = {
        row["member_row_ids_sha256"]: row for row in witnesses["witnesses"]
    }
    members = next(
        value for value in components
        if any(row["invocation"]["phase"] == "FORWARD" for row in value)
    )
    key = digest(sorted(row["row_id"] for row in members))
    witness = witness_by_members[key]
    args = (
        "phi4_mini_3p8b", contract["models"]["phi4_mini_3p8b"], members,
        ledger["result_sha256"], "batch1_seq64",
    )
    assert proof_unit(*args, witness)["gates"]["FB_ANALYTICALLY_PROVED"] is True

    for flag in (
        "saved_tensor_origins_exact", "cotangent_edge_exact",
        "backward_program_matches_analytic_vjp", "non_tensor_arguments_exact",
        "output_edges_exact",
    ):
        tampered = copy.deepcopy(witness)
        tampered["concrete_program_proof"][flag] = False
        assert proof_unit(*args, tampered)["gates"]["FB_ANALYTICALLY_PROVED"] is False

    tampered = copy.deepcopy(witness)
    tampered["member_row_ids_sha256"] = "0" * 64
    assert proof_unit(*args, tampered)["gates"]["FB_ANALYTICALLY_PROVED"] is False

    tampered = copy.deepcopy(witness)
    tampered["concrete_program_proof"]["analytic_derivation_sha256"] = ""
    assert proof_unit(*args, tampered)["gates"]["FB_ANALYTICALLY_PROVED"] is False
