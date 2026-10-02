"""Read-only maintenance checks; never regenerate research evidence."""

from scripts import check_research_docs as documents


def test_maintained_document_contract_and_links():
    assert documents.check_doc_contract() == []
    assert documents.check_links() == []


def test_retained_figures_agree_with_existing_records():
    assert documents.check_figures() == []


def test_missing_document_is_reported_without_link_check_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "ROOT", tmp_path)
    monkeypatch.setattr(documents, "CURRENT_DOCS", ("missing.md",))
    monkeypatch.setattr(documents, "RETIRED_DUPLICATE_DOCS", ())
    assert documents.check_doc_contract() == ["Missing current document: missing.md"]
    assert documents.check_links() == []


def test_retired_snapshot_cannot_silently_return(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "obsolete.md").write_text("old status\n")
    monkeypatch.setattr(documents, "ROOT", tmp_path)
    monkeypatch.setattr(documents, "CURRENT_DOCS", ())
    monkeypatch.setattr(documents, "RETIRED_DUPLICATE_DOCS", ("obsolete.md",))
    assert documents.check_doc_contract() == [
        "Retired duplicate document is present: docs/obsolete.md"
    ]
