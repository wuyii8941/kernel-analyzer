from scripts.bind_backward_rescreen_carriers import endpoint_reachability


def test_all_backward_nodes_can_use_existing_dataflow_not_case_names():
    capture={'graphs':[
        {'phase':'FORWARD','nodes':[{'name':'p','op':'placeholder'}]},
        {'phase':'BACKWARD','nodes':[
            {'name':'a','op':'call_function','input_edges':[]},
            {'name':'b','op':'call_function','input_edges':[{'source_node':'a'}]},
            {'name':'unrelated','op':'call_function','input_edges':[]},
            {'name':'output','op':'output','input_edges':[{'source_node':'b','argument_path':[0]}]},
        ]}]}
    mapping={'p':{'name':'model.weight','aliases':['model.weight'],'shape':[2]}}
    endpoints={'backward:graph0:a','backward:graph0:b','backward:graph0:unrelated','backward:graph0:absent'}
    result=endpoint_reachability(capture,mapping,endpoints)
    assert result['backward:graph0:a']['parameters'][0]['name']=='model.weight'
    assert result['backward:graph0:a']['parameters'][0]['aot_distance']==1
    assert result['backward:graph0:b']['parameters'][0]['aot_distance']==0
    assert not result['backward:graph0:unrelated']['parameters']
    assert result['backward:graph0:absent']['status']=='UNRESOLVED_ENDPOINT_NODE'
