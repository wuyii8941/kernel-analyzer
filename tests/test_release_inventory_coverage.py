import gzip
import json
from scripts.audit_release_inventory_coverage import audit


def test_omitted_and_duplicate_packages_are_not_new_support(tmp_path):
    results = tmp_path/'results'
    first = results/'current'
    second = results/'historical'
    bad = results/'invalid'
    for directory in (first, second, bad):
        directory.mkdir(parents=True)
    raw = gzip.compress(json.dumps({'rows':[{'task_id':'forward:1','symbol':'kernel'}]}).encode())
    (first/'same_dtype_tasks.json.gz').write_bytes(raw)
    (second/'same_dtype_tasks.json.gz').write_bytes(raw)
    (bad/'same_dtype_tasks.json.gz').write_bytes(gzip.compress(b'{}'))
    report = audit(tmp_path, {'records':[{'release':str(first)}]})
    assert report['discovered_packages'] == 3
    assert report['included_packages'] == 1
    assert report['omitted_packages'] == 2
    assert report['distinct_task_file_digests'] == 1
    assert report['duplicate_file_groups'][0]['copies'] == 2
    assert not report['runtime_support_inferred']
    assert any(r['status']=='UNREADABLE_OR_INVALID' for r in report['rows'])
