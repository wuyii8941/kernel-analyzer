from scripts.select_workspace_unique_source_pool import select


def test_select_prefers_registered_copy_without_dropping_distinct_source():
    census = {
        "source_records": [
            {"source": "/z", "source_sha256": "same", "provenance_status": "UNREGISTERED_RUNTIME_RELEASE"},
            {"source": "/a", "source_sha256": "same", "provenance_status": "REGISTERED_RELEASE"},
            {"source": "/b", "source_sha256": "different", "provenance_status": "OTHER_SAVED_COMPILED_SOURCE"},
        ],
        "definition_records": [
            {"source": "/z", "source_sha256": "same", "symbol": "k", "status": "TRITON_FUNCTION_INSPECTED", "provenance_status": "UNREGISTERED_RUNTIME_RELEASE"},
            {"source": "/a", "source_sha256": "same", "symbol": "k", "status": "TRITON_FUNCTION_INSPECTED", "provenance_status": "REGISTERED_RELEASE"},
            {"source": "/b", "source_sha256": "different", "symbol": "q", "status": "TRITON_FUNCTION_INSPECTED", "provenance_status": "OTHER_SAVED_COMPILED_SOURCE"},
        ],
    }
    result = select(census)
    assert result["source_count"] == 2
    assert {(row["source"], row["symbol"]) for row in result["records"]} == {
        ("/a", "k"), ("/b", "q")}
    assert result["selected_provenance_counts"] == {
        "REGISTERED_RELEASE": 1, "OTHER_SAVED_COMPILED_SOURCE": 1}
    assert result["selection_uses_numerical_results"] is False
