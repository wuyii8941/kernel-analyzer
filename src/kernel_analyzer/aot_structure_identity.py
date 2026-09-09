"""Strict recorded graph comparison; not proof of identical runtime values."""
FIELDS = ('op', 'target', 'arguments', 'input_edges', 'tensor_meta', 'original_aten')


def compare(left, right):
    def indexed(capture):
        result = {}
        for graph in capture['graphs']:
            for node in graph['nodes']:
                key = (graph['phase'], graph.get('graph_index', 0), node['name'])
                if key in result:
                    raise ValueError('Duplicate graph node identity')
                result[key] = node
        return result
    a, b = indexed(left), indexed(right)
    missing_left, missing_right = sorted(set(b)-set(a)), sorted(set(a)-set(b))
    differences = []
    for key in sorted(set(a) & set(b)):
        fields = [field for field in FIELDS if a[key].get(field) != b[key].get(field)]
        # A missing edge/argument record is not a positive identity witness.
        if a[key]['op'] == 'call_function':
            fields += [field + ':missing' for field in ('target', 'arguments', 'input_edges')
                       if field not in a[key] or field not in b[key]]
        if fields:
            differences.append(dict(node=list(key), fields=fields))
    return dict(left_nodes=len(a), right_nodes=len(b), missing_left=missing_left,
        missing_right=missing_right, differences=differences,
        structure_identical=bool(a) and not (missing_left or missing_right or differences),
        runtime_values_or_execution_identity_proved=False)
