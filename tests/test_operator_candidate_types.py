import ast
import hashlib
import pytest
from scripts.audit_operator_candidate_types import inspect_definition, audit


def definition(dtype):
    inner = f"@decorate(triton_meta={{'signature': {{'in_ptr0': '{dtype}', 'out_ptr0': '*i1'}}}})\ndef k(in_ptr0, out_ptr0):\n    pass\n"
    return f'k = compile.triton("k", {inner!r})'


def test_types_do_not_claim_family_or_bias():
    for dtype, prefix in [('*i64', 'INTEGER_BOOLEAN'), ('*bf16', 'FLOATING'), ('*fp8', 'UNKNOWN')]:
        result = inspect_definition(ast.parse(definition(dtype)).body[0], 'k')
        assert result['status'].startswith(prefix)
        assert result['numerical_family_verified'] is False
        assert result['runtime_verified'] is False


def test_source_integrity_and_rebinding(tmp_path):
    p = tmp_path / 'source.py'
    p.write_text(definition('*i64'))
    row = dict(source=str(p), symbol='k', source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    assert audit([row])['counts'] == {'INTEGER_BOOLEAN_POINTERS_REQUIRES_CONTROL_REVIEW': 1}
    p.write_text(definition('*i64') + '\n' + definition('*bf16'))
    with pytest.raises(ValueError, match='changed'):
        audit([row])
    row['source_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    assert audit([row])['counts'] == {'AMBIGUOUS_OR_MISSING_DEFINITION': 1}
