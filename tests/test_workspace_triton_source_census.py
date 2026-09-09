from scripts.build_workspace_triton_source_census import build, embedded_definitions


SOURCE = '''\
kernel = async_compile.triton("kernel", """
@triton.jit
def kernel(in_ptr0):
    value = tl.load(in_ptr0)
    tl.store(in_ptr0, value)
""")
'''


def test_semantic_hash_ignores_embedded_decorator_metadata():
    first = embedded_definitions(SOURCE)[0]
    changed = embedded_definitions(SOURCE.replace("@triton.jit", "@different.metadata"))[0]
    assert first["function_semantic_ast_sha256"] == changed["function_semantic_ast_sha256"]


def test_workspace_census_marks_registered_and_unregistered_runtime_releases(tmp_path):
    results = tmp_path / "results"
    registered = results / "coverage" / "runtime_releases" / "r1"
    unregistered = results / "experiment" / "runtime_release"
    other = results / "standard_aot"
    for root in (registered, unregistered, other):
        path = root / "trace" / "model" / "output_code.py"
        path.parent.mkdir(parents=True)
        path.write_text(SOURCE)
    result = build(results, {registered})
    assert result["source_paths"] == 3
    assert result["unique_source_digests"] == 1
    assert result["unique_function_semantic_ast_digests"] == 1
    assert result["release_qualified_definition_occurrences"] == 3
    assert result["provenance_counts"] == {
        "REGISTERED_RELEASE": 1,
        "UNREGISTERED_RUNTIME_RELEASE": 1,
        "OTHER_SAVED_COMPILED_SOURCE": 1,
    }
    assert result["selection_uses_numerical_results"] is False
    assert result["all_kernel_support_established"] is False
