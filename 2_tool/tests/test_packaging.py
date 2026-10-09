"""The installed package must carry everything the engine imports or reads (a non-editable install from a clean
clone is part of the freeze check)."""
import json
import tomllib
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1]
REPO = TOOL.parent


def test_every_subpackage_is_listed_for_installation():
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]
    listed = set(cfg["packages"])
    src = TOOL / "src" / "kernel_analyzer"
    found = {"kernel_analyzer" + "".join("." + p for p in d.relative_to(src).parts)
             for d in src.rglob("*") if d.is_dir() and (d / "__init__.py").exists()}
    assert found | {"kernel_analyzer"} <= listed
    assert cfg["package-dir"]["kernel_analyzer"] == "2_tool/src/kernel_analyzer"


def test_the_op_registry_is_packaged_and_used():
    from kernel_analyzer.reference_eval import ttir_mapping as M
    packaged = TOOL / "src/kernel_analyzer/reference_eval/data/ttir_op_registry.json"
    assert Path(M.REGISTRY_PATH).resolve() == packaged.resolve()
    assert json.loads(packaged.read_text())["triton"] == M.LOCKED_TRITON
