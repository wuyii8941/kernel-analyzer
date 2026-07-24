from scripts.phase8_matched_step_counterfactual import build_analysis


def test_build_analysis_classifies_forks_and_normalizes_gradient_gap() -> None:
    def arm(name: str, clips: list[bool], norms: list[float]) -> dict:
        return {
            "arm": name,
            "trajectory": [
                {
                    "step": index + 1,
                    "target_clip_active": clip,
                    "target_loss_gradient": 0.0 if clip else -0.1,
                    "full_gradient_norm": norm,
                    "loss": float(index),
                    "target_logp": float(index),
                }
                for index, (clip, norm) in enumerate(zip(clips, norms, strict=True))
            ],
        }

    merged = {
        "fork_id": "fork-1",
        "arms": [
            arm("A_reference", [False, False, True, True], [10.0, 10.0, 10.0, 10.0]),
            arm("B_alternative", [True, False, False, True], [8.0, 10.0, 12.0, 10.0]),
        ],
        "distances": [{"step": 1, "A_B": {"l2": 1.0}, "A_C": {"l2": 0.5}, "recovery_ratio_A_C_over_A_B": 0.5}],
    }

    result = build_analysis(merged)

    assert result["summary"]["fork_steps"] == [1, 3]
    assert result["summary"]["target_gradient_semantic_fork_count"] == 2
    assert result["rows"][0]["normalized_gradient_norm_gap"] == 2.0 / 9.0
    assert result["summary"]["metrics"]["normalized_gradient_norm_gap"]["nonfork_mean"] == 0.0
