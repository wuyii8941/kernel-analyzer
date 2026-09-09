import pytest
import torch

from kernel_analyzer.fixed_suite_implementation_capture import FixedSuiteImplementationCapture


def test_generic_fixed_suite_capture_uses_declared_stages():
    capture=FixedSuiteImplementationCapture([str(i) for i in range(32)],('LOCAL','PARAMETER_WRITE'))
    for i in range(32):
        reference=torch.arange(1,9,dtype=torch.float32)+i
        capture.append({'LOCAL':(reference*1.02,reference),'PARAMETER_WRITE':(reference,reference)})
    rows,stages=capture.finish()
    assert len(rows['LOCAL'])==32
    assert stages['LOCAL']['EXACT']['profile']['suite']['repair_aligned_effect']==pytest.approx(.02,abs=1e-7)
    assert stages['PARAMETER_WRITE']['EXACT']['profile']['suite']['total_effect_rms']==0.0


def test_generic_fixed_suite_rejects_missing_stage():
    capture=FixedSuiteImplementationCapture([str(i) for i in range(32)],('LOCAL','PARAMETER_WRITE'))
    with pytest.raises(ValueError,match='all declared stages'):
        capture.append({'LOCAL':(torch.ones(2),torch.ones(2))})
