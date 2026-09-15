import json
import subprocess
import sys
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_bias_proof_audit_closes_declared_gates(tmp_path):
    output = ROOT / "results/property/bias_proof_plan_v1/test-output.json"
    if output.exists():
        output.unlink()
    try:
        subprocess.run([
            sys.executable, "scripts/build_bias_proof_audit.py", "--output", str(output)
        ], cwd=ROOT, check=True)
        result = json.loads(output.read_text())
        assert result["status"] == "COMPLETE"
        assert all(result["gates"].values())
        assert result["zero_mean_high_energy_control"]["total_rms"] == pytest.approx(.02)
        assert result["adamw8bit"]["bias_result"]["aligned"]["vector_mean_nonzero_established"] is False
        assert result["second_family"]["operator_family"].startswith("LIGER")
    finally:
        if output.exists():
            output.unlink()
