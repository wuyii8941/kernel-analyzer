import json

import pytest

from scripts import verify_liger_language_confirmation as verifier


@pytest.mark.parametrize("bad_evaluation", [False, True])
def test_verifies_checkpoint_and_rejects_nonfinite_evaluation(tmp_path, monkeypatch, bad_evaluation):
    root = tmp_path / "repo"
    out = tmp_path / "results"
    source = root / "scripts/run_liger_language_confirmation.py"
    source.parent.mkdir(parents=True)
    source.write_text("# frozen test source\n")
    directory = out / "pair0"
    directory.mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps({"runner_sha256": verifier.digest(source),
        "initialization_seeds": [12], "steps": 1}))
    data = {}
    for split in ("train", "validation"):
        path = directory / f"{split}.npy"
        path.write_bytes(b"frozen-input")
        data[split] = {"encoded_sha256": verifier.digest(path)}
    (directory / "protocol.json").write_text(json.dumps({"model": {"initialization_seed": 12},
        "source_sha256": {}, "data": data, "eval_every": 1}))
    for condition in ("candidate", "reference"):
        folder = directory / condition
        folder.mkdir()
        (folder / "final.pt").write_bytes(b"checkpoint")
        (folder / "steps.jsonl").write_text(json.dumps({"step": 1, "loss": 2.0}) + "\n")
        (folder / "status.json").write_text(json.dumps({"status": "COMPLETE_DEVELOPMENT_PILOT",
            "evaluations": [{"step": 1, "shared_evaluation_loss": float("nan") if bad_evaluation else 2.0}],
            "final_checkpoint_sha256": verifier.digest(folder / "final.pt")}))
    monkeypatch.setattr(verifier, "ROOT", root)
    monkeypatch.setattr(verifier, "OUT", out)
    monkeypatch.setattr(verifier, "N", 1)
    if bad_evaluation:
        with pytest.raises(SystemExit):
            verifier.verify()
    else:
        verifier.verify()
    result = json.loads((out / "execution_verification.json").read_text())
    assert (result["status"] == "VERIFIED_RECORDED_EXECUTION") is not bad_evaluation
