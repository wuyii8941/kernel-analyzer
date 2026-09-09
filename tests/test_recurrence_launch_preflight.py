import pytest
from scripts.preflight_recurrence_launch import declared_device, preflight


def source(index):
    inner=f'@wrap(device=DeviceProperties(index={index}))\ndef kernel(): pass'
    return f'kernel=async_compile.triton("kernel",{inner!r})'


def test_declared_logical_device():
    assert declared_device(source(3),'kernel')==3


@pytest.mark.parametrize('index',['unknown','True'])
def test_dynamic_or_boolean_device_rejected(index):
    with pytest.raises(ValueError): declared_device(source(index),'kernel')


def test_ambiguous_function_rejected():
    with pytest.raises(ValueError): declared_device(source(0)+'\n'+source(1),'kernel')


def test_unrelated_launch_unchanged():
    assert preflight(['python','-c','print(1)']) is None


def test_missing_plan_rejected_before_execution():
    with pytest.raises(ValueError): preflight(['python','scripts/run_decayed_recurrence_capture.py'])
