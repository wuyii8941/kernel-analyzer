import gzip
import json
import pytest
from scripts.run_numerical_coverage import sha
from scripts.include_historical_release_tasks import extend


def test_existing_verdict_unchanged_and_missing_package_not_supported(tmp_path):
    path=tmp_path/'same_dtype_tasks.json.gz'
    path.write_bytes(gzip.compress(json.dumps({'rows':[dict(task_id='forward:1',symbol='gelu')]}).encode()))
    row=dict(release='/old',task_id='forward:1',eligibility='CHECKED',runtime_measurement_status='VERIFIED')
    inventory={'records':[row]}
    package=dict(release=str(tmp_path),task_file=str(path),task_sha256=sha(path),
        status='TASK_PACKAGE_READ',included_in_current_inventory=False,missing_package_files=['capture.json'])
    audit=dict(rows=[package],duplicate_file_groups=[])
    report=extend(inventory,audit)
    assert report['records'][0] == row
    added=report['records'][1]
    assert added['eligibility']=='HISTORICAL_PACKAGE_INCOMPLETE'
    assert added['reference_candidates']==[]
    assert added['runtime_measurement_status']=='NOT_ASSESSED_BY_THIS_INVENTORY'
    assert report['new_measurement_count']==0
    assert len(inventory['records'])==1
    with pytest.raises(ValueError,match='Duplicate historical'):
        extend(inventory,dict(audit,rows=[package,package]))
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        extend(inventory,audit)
