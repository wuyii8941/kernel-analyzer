"""Registry adapter for the source-audited pure embedding lookup family."""

from kernel_analyzer.embedding_lookup_reference import select_output
from kernel_analyzer.embedding_lookup_source import check_source


def reference(metadata, candidate, contract):
    pointers = metadata.get("runtime_pointers") or {}
    preserved = {name: pointers[name] for name in ("in_ptr0", "in_ptr1") if name in pointers}
    return select_output(
        preserved, candidate, tokens=contract["tokens"], width=contract["width"],
        vocabulary_size=contract["vocabulary_size"],
    )
