from scripts.check_detached_experiments import classify, process


def state(value): return dict(status=value)


def test_child_survival_prevents_restart_when_worker_gone():
    assert classify(state('NOT_PRESENT'),state('LIVE_MATCHING_COMMAND'),None)=='EXPERIMENT_PROCESS_LIVE'


def test_successful_exit_is_not_verified_experiment():
    assert classify(state('NOT_PRESENT'),state('NOT_PRESENT'),dict(exit_code=0))=='PROCESS_EXIT_ZERO_REQUIRES_RESULT_VERIFICATION'


def test_no_handle_is_not_a_completed_job():
    assert 'REQUIRES_DIAGNOSIS' in classify(state('NOT_PRESENT'),state('PID_IDENTITY_DIFFERS'),None)


def test_running_worker_precedes_missing_child_record():
    assert classify(state('LIVE_MATCHING_COMMAND'),state('NO_PROCESS_RECORD'),None)=='WORKER_LIVE'


def test_another_pid_namespace_is_unknown_not_dead():
    result=process(dict(pid=999999999,command=['fake'],pid_namespace='different-namespace'))
    assert result['status']=='NAMESPACE_DIFFERS_CANNOT_CHECK_PID'
    assert classify(result,state('NO_PROCESS_RECORD'),None)=='PROCESS_LIVENESS_UNKNOWN_REQUIRES_DIAGNOSIS'


def test_worker_failure_reported_only_after_liveness_check():
    error=dict(error='Dependency failed')
    assert classify(state('NOT_PRESENT'),state('NO_PROCESS_RECORD'),None,error)=='WORKER_STOPPED_WITH_RECORDED_ERROR'
    assert classify(state('NOT_PRESENT'),state('LIVE_MATCHING_COMMAND'),None,error)=='EXPERIMENT_PROCESS_LIVE'
    assert classify(state('NAMESPACE_DIFFERS_CANNOT_CHECK_PID'),state('NO_PROCESS_RECORD'),None,error)=='PROCESS_LIVENESS_UNKNOWN_REQUIRES_DIAGNOSIS'
