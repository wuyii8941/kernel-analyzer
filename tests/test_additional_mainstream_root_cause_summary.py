from scripts.build_additional_mainstream_root_cause_summary import build


def _stage(rms, decision="SYSTEMATIC_BIAS_NOT_CONFIRMED"):
    return {
        "total_rms": rms,
        "decision": decision,
        "aligned_ratio_of_sums": 0.0,
        "aligned_statewise_gain_interval": {"interval": [0.0, 0.0]},
        "systematic_bias_reasons": [],
    }


def test_summary_keeps_nonzero_rows_and_source_statuses():
    scan = {
        "schema": "scan",
        "selected_models": ["pythia_410m", "gpt2"],
        "samples": 16,
        "dtype": "bfloat16",
        "scope": "generated",
        "cases": {
            "pythia_410m": {
                "operators": {
                    "rotary_position_embedding": {
                        "stages": {"OUTPUT": _stage(0.002, "SYSTEMATIC_BIAS_CONFIRMED")}
                    },
                    "linear_projection": {"stages": {"OUTPUT": _stage(0.0)}},
                }
            },
            "gpt2": {
                "operators": {
                    "activation": {"stages": {"BACKWARD": _stage(0.003)}},
                    "layernorm": {"stages": {"BACKWARD": _stage(1e-6)}},
                }
            },
        },
    }
    sources = {
        "rotary": {
            "models": {
                "pythia_410m": {
                    "all_reference_matches_bf16_formula": True,
                    "all_candidate_matches_fp32_formula": True,
                }
            }
        },
        "activation": {
            "models": {
                "gpt2": {"all_candidate_backward_differs": True}
            }
        },
        "layernorm": {"models": {"gpt2": {"all_candidate_backward_matches_analytic": False}}},
        "softmax": {"analysis": {"all_reference_matches_fp32_formula": True}},
    }
    result = build(
        scan,
        activation=sources["activation"],
        layernorm=sources["layernorm"],
        rotary=sources["rotary"],
        softmax=sources["softmax"],
    )
    assert result["counts"]["nonzero_stage_rows"] == 3
    assert result["counts"]["direction_confirmed_rows"] == 1
    rows = {(row["model"], row["operator"], row["stage"]): row for row in result["rows"]}
    assert rows[("pythia_410m", "rotary_position_embedding", "OUTPUT")]["root_cause_status"] == (
        "INTERMEDIATE_ARITHMETIC_MATERIALIZATION_BOUNDARY"
    )
    assert rows[("gpt2", "layernorm", "BACKWARD")]["root_cause_status"].startswith(
        "LAYERNORM_BACKWARD_REDUCTION"
    )

