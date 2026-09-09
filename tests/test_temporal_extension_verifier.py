import json
import pytest
from scripts import summarize_liger_temporal_extension as verifier


@pytest.mark.parametrize('wrong_pairing',[False,True])
def test_temporal_summary_checks_complete_windows_and_paired_inputs(tmp_path,monkeypatch,wrong_pairing):
    monkeypatch.setattr(verifier,'OUT',tmp_path);monkeypatch.setattr(verifier,'N',1)
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'evaluation_steps':[2048,3072,4096]}))
    for condition in ('candidate','reference'):
        root=tmp_path/'pair0'/condition;root.mkdir(parents=True)
        (root/'checkpoint_4096.pt').write_bytes(b'fixed checkpoint')
        offset=1 if wrong_pairing and condition=='reference' else 0
        (root/'steps.jsonl').write_text(''.join(json.dumps({'step':i,'loss':2.,'offsets':[offset]})+'\n' for i in range(1025,4097)))
        (root/'direct.jsonl').write_text(''.join(json.dumps({'step':i,'effect_energy':1.,'repair_energy':2.})+'\n' for a,b in verifier.WINDOWS for i in range(a,b+1)))
        windows=[{'effect_energy_sum':32.,'repair_energy_sum':64.,'effect_mean_energy':1.,'split_half_mean_inner_product':1.} for _ in verifier.WINDOWS]
        (root/'summary.json').write_text(json.dumps({'status':'COMPLETE_TEMPORAL_EXTENSION','plan_sha256':verifier.file_sha256(plan),'checkpoint_sha256':verifier.file_sha256(root/'checkpoint_4096.pt'),'window_mean_gram':[[1.]*4 for _ in range(4)],'windows':windows,'evaluations':[{'step':i,'shared_evaluation_loss':2.} for i in (2048,3072,4096)]}))
    if wrong_pairing:
        with pytest.raises(RuntimeError,match='Paired inputs differ'):verifier.main()
        assert not (tmp_path/'summary.json').exists()
    else:
        verifier.main()
        result=json.loads((tmp_path/'summary.json').read_text())
        assert result['status']=='COMPLETE_VERIFIED_TEMPORAL_EXTENSION'
        assert result['final_nonzero_loss_gaps']==0
