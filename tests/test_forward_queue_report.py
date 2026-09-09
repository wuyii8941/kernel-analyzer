import hashlib
import json
import pytest
from scripts.join_forward_measurements import queue_roots


def test_only_successful_exit_is_eligible(tmp_path):
    jobs = []
    for i, code in enumerate((None, 1, 0)):
        root = tmp_path/f'job_{i}'
        root.mkdir()
        if code is not None:
            (root/'exit.json').write_text(json.dumps({'exit_code':code}))
        plan = tmp_path/f'plan_{i}.json'
        plan.write_text(json.dumps({'cases':[{'case_id':str(i)}]}))
        if code == 0:
            (root/'raw').mkdir()
            (root/'raw/family_execution_protocol.json').write_text(json.dumps({
                'schema':'residual-rms-forward-capture-v1',
                'source_sha256':{str(plan.resolve()):hashlib.sha256(plan.read_bytes()).hexdigest()}}))
        jobs.append(dict(root=str(root), plan=str(plan),
                         plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(), case_ids=[str(i)]))
    (tmp_path/'queue_protocol.json').write_text(json.dumps(dict(schema='family-plan-queue-v2',jobs=jobs)))
    roots, statuses = queue_roots(tmp_path)
    assert roots == [tmp_path/'job_2']
    assert statuses[0]['status'] == 'NO_TERMINAL_RECORD_NOT_A_LIVENESS_CLAIM'
    assert statuses[1]['status'] == 'CAPTURE_FAILED'
    capture = tmp_path/'job_2/raw/family_execution_protocol.json'
    recorded = json.loads(capture.read_text())
    recorded['source_sha256'] = {}
    capture.write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match='queued partition'):
        queue_roots(tmp_path)
