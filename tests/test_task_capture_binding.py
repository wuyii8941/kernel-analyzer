from scripts.verify_task_capture_binding import verify


def test_all_graphs_and_both_identities_required():
    tasks=dict(bindings=dict(proof_capture_result_sha256='proof'))
    proof=dict(result_sha256='proof',standard_aot_capture=dict(capture_sha256='capture',graphs=[{'node':1}]))
    capture=dict(capture_sha256='capture',graphs=[{'node':1}])
    assert verify(tasks,proof,capture)['status']=='CAPTURE_BOUND_TO_TASK_PROOF'
    assert not verify(tasks,proof,capture)['runtime_measurement_complete']
    capture['graphs']=[{'node':2}]
    assert verify(tasks,proof,capture)['status']=='BINDING_MISMATCH'
    capture['graphs']=[{'node':1}]; tasks['bindings']['proof_capture_result_sha256']='other'
    assert verify(tasks,proof,capture)['status']=='BINDING_MISMATCH'


def test_empty_graphs_never_count_as_a_match():
    tasks=dict(bindings=dict(proof_capture_result_sha256='proof'))
    capture=dict(capture_sha256='capture',graphs=[])
    proof=dict(result_sha256='proof',standard_aot_capture=capture)
    assert verify(tasks,proof,capture)['status']=='BINDING_MISMATCH'
