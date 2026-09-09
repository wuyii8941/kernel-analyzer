from pathlib import Path

from scripts.build_all_release_source_pool import build, triton_symbols


SOURCE = '''\
kernel_a = async_compile.triton("kernel_a", """
def kernel_a():
    pass
""")
kernel_b = async_compile.triton("kernel_b", """
def kernel_b():
    pass
""")
'''


def test_triton_symbols_ignores_non_triton_assignments():
    assert triton_symbols(SOURCE + "other = 1\n") == ["kernel_a", "kernel_b"]


def test_build_keeps_release_qualified_duplicates_and_missing_trace(tmp_path):
    releases = [tmp_path / "r1", tmp_path / "r2", tmp_path / "missing"]
    for release in releases[:2]:
        source = release / "trace" / "model" / "output_code.py"
        source.parent.mkdir(parents=True)
        source.write_text(SOURCE)
    inventory = {"records": [
        {"release": str(releases[0]), "task_id": "a"},
        {"release": str(releases[0]), "task_id": "b"},
        {"release": str(releases[1]), "task_id": "c"},
        {"release": str(releases[2]), "task_id": "d"},
    ]}
    result = build(inventory)
    assert result["release_count"] == 3
    assert result["source_count"] == 2
    assert result["source_with_triton_count"] == 2
    assert result["source_path_occurrences"] == 2
    assert result["unique_source_digests"] == 1
    assert result["definition_records"] == 4
    assert len({(row["release"], row["symbol"]) for row in result["records"]}) == 4
    assert result["release_status_counts"] == {
        "SOURCE_INVENTORIED": 2, "NO_SAVED_TRACE_SOURCE": 1}
    assert result["selection_uses_numerical_results"] is False
    assert result["all_kernel_support_established"] is False


def test_build_deduplicates_symlinked_source_paths(tmp_path):
    real = tmp_path / "real"
    source = real / "trace" / "model" / "output_code.py"
    source.parent.mkdir(parents=True)
    source.write_text(SOURCE)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    result = build({"records": [
        {"release": str(real), "task_id": "a"},
        {"release": str(alias), "task_id": "b"},
    ]})
    # Release identities resolve to the same execution package and source.
    assert result["release_count"] == 1
    assert result["source_count"] == 1
    assert result["source_with_triton_count"] == 1
    assert result["definition_records"] == 2
