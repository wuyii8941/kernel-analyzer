import json

import pytest

from scripts.run_numerical_coverage import save_idempotent


def test_interrupted_freeze_can_resume_with_identical_content(tmp_path):
    path = tmp_path / "source_snapshot.json"
    save_idempotent(path, {"source": "one"})
    save_idempotent(path, {"source": "one"})
    assert json.loads(path.read_text()) == {"source": "one"}


def test_interrupted_freeze_rejects_changed_content(tmp_path):
    path = tmp_path / "source_snapshot.json"
    save_idempotent(path, {"source": "one"})
    with pytest.raises(ValueError, match="partial freeze differs"):
        save_idempotent(path, {"source": "two"})
