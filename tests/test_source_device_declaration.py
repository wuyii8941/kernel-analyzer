import ast
import pytest
from scripts.declare_source_device import declare


def test_only_device_index_changes_not_math_or_other_options():
    kernel='@wrap(device=DeviceProperties(type="cuda",index=0,cc=86),enable_fp_fusion=False)\ndef k(p):\n    tl.store(p, tl.load(p)*2)\n'
    source='k=async_compile.triton("k",'+repr(kernel)+')'
    result,changes=declare(source,3)
    actual=ast.parse(ast.parse(result).body[0].value.args[1].value)
    expected=ast.parse(kernel.replace('index=0','index=3'))
    assert ast.dump(actual)==ast.dump(expected)
    assert changes[0]['old_index']==0
    assert 'index=0' in source


def test_missing_or_dynamic_index_rejected():
    with pytest.raises(ValueError):declare('x=1',0)
    kernel='@wrap(device=DeviceProperties(index=unknown))\ndef k(): pass'
    with pytest.raises(ValueError):declare('k=async_compile.triton("k",'+repr(kernel)+')',1)
