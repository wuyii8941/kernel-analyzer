import gzip
import json
import subprocess
import sys
from pathlib import Path


def test_ledger_retains_every_forward_and_backward_invocation(tmp_path: Path):
    inventory = tmp_path / "inventory.json.gz"
    payload = {
        "trace": {"events": [
            {"invocation_id": "forward:0", "phase": "FORWARD", "overload": "aten.add.Tensor", "category": "ELEMENTWISE", "sequence_binding_status": "EXACT"},
            {"invocation_id": "backward:1", "phase": "BACKWARD", "overload": "aten.mul.Tensor", "category": "ELEMENTWISE", "sequence_binding_status": "EXACT"},
        ]}
    }
    with gzip.open(inventory, "wt") as handle:
        json.dump(payload, handle)
    output = tmp_path / "ledger.json.gz"
    subprocess.run([
        sys.executable, "scripts/build_tcmp_invocation_ledger.py",
        "--cell-id", "cell", "--inventory", str(inventory), "--output", str(output),
    ], check=True)
    with gzip.open(output, "rt") as handle:
        ledger = json.load(handle)
    assert ledger["counts"] == {"invocations": 2, "forward": 1, "backward": 1, "unique_overloads": 2}
    assert len(ledger["invocation_ids"]) == 2
    assert all(row["disposition"] == "UNRESOLVED_PROOF" for row in ledger["rows"])
