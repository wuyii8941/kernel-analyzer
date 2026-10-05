"""The installed package must carry everything the engine imports or reads (a non-editable install from a clean
clone is part of the WP1 freeze check)."""
import filecmp
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_subpackage_is_listed_for_installation():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]
    listed = set(cfg["packages"])
    found = {"kernel_analyzer" + "".join("." + p for p in d.relative_to(ROOT / "src" / "kernel_analyzer").parts)
             for d in (ROOT / "src" / "kernel_analyzer").rglob("*") if d.is_dir() and (d / "__init__.py").exists()}
    assert found | {"kernel_analyzer"} <= listed


def test_packaged_op_registry_matches_the_repository_copy():
    assert filecmp.cmp(ROOT / "results/reference_eval/ttir_op_registry.json",
                       ROOT / "src/kernel_analyzer/reference_eval/data/ttir_op_registry.json", shallow=False)
