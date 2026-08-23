import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_direct_persistence_v4_phase_optimizer.py"


def test_phase_optimizer_entrypoint_fails_closed_when_phases_are_missing(tmp_path):
    import subprocess
    output = tmp_path / "phase.json"
    subprocess.run(
        ["python3", str(SCRIPT), "--output", str(output)],
        check=True,
        cwd=ROOT,
    )
    result = json.loads(output.read_text())
    assert result["status"] == "ABSTAIN_MISSING_PHASE_CAPTURE"
    assert result["missing_phases"] == ["early", "middle", "late"]
