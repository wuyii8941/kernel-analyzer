import hashlib
import json

from kernel_analyzer.training_equivalence import exact_binomial_one_sided_bounds
from scripts.verify_adamw8bit_trajectory_response_audit import verify


def test_trajectory_verifier_recomputes_summary(tmp_path):
    source = tmp_path / "source"
    source.write_text("frozen")
    protocol = {
        "streams": 1, "checkpoints": [0, 1024],
        "source_sha256": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "streams").mkdir()
    checkpoints = {}
    for step in (0, 1024):
        checkpoints[str(step)] = {
            "evaluation_loss": {
                "FP32_ADAMW": 1.0,
                "ADAMW8BIT_BLOCK64": 1.1,
                "ADAMW8BIT_BLOCK256": 1.2,
            },
            "parameter_distance_to_fp32": {
                "ADAMW8BIT_BLOCK64": {"relative_l2": 0.1},
                "ADAMW8BIT_BLOCK256": {"relative_l2": 0.2},
            },
        }
    row = {
        "status": "COMPLETE", "stream_index": 0, "checkpoints": checkpoints,
        "final_block64_parameter_closer": True, "final_block64_loss_closer": True,
    }
    (tmp_path / "streams/stream-00.json").write_text(json.dumps(row))
    bounds = exact_binomial_one_sided_bounds(1, 1, alpha=0.05)
    summary = {
        "block64_final_parameter_closer_count": 1,
        "block64_final_parameter_closer_probability_bounds": list(bounds),
        "block64_parameter_proximity_result": "NOT_CONFIRMED",
        "block64_final_loss_closer_count": 1,
        "mean_final_relative_parameter_distance": {
            "ADAMW8BIT_BLOCK64": 0.1, "ADAMW8BIT_BLOCK256": 0.2,
        },
        "mean_final_evaluation_loss": {
            "FP32_ADAMW": 1.0, "ADAMW8BIT_BLOCK64": 1.1,
            "ADAMW8BIT_BLOCK256": 1.2,
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert verify(tmp_path)["status"] == "VERIFIED"
