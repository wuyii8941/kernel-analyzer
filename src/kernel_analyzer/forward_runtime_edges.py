"""Extract observed forward/backward identity edges, without name guessing."""


def observed_edges(runs):
    """Require consistent unique source identity in every observation of a graph.

    Returns observed links only, not parameter reach or a derivative guarantee.
    Ambiguous aliases remain unresolved rather than choosing a convenient source.
    """
    groups = {}
    for run in runs:
        graph = run['backward_phase']['graph_index']
        if type(graph) is not int or graph < 0:
            raise ValueError('Invalid backward graph index')
        observation = {}
        for row in run['backward_inputs']:
            name = row['placeholder']
            if name in observation:
                raise ValueError('Duplicate backward placeholder')
            matches = row.get('global_forward_matches', [])
            sources = set()
            for match in matches:
                if match.get('identity_mode') not in ('EXACT_PYTHON_OBJECT', 'EXACT_STORAGE_VIEW'):
                    continue
                index = match['phase_graph_index']
                source = match['source_node']
                if type(index) is not int or index < 0 or not isinstance(source, str) or not source:
                    raise ValueError('Invalid forward identity')
                sources.add((index, source))
            observation[name] = sources
        groups.setdefault(graph, []).append(observation)
    edges, unresolved = [], []
    for graph, observations in sorted(groups.items()):
        names = set().union(*(set(row) for row in observations))
        for name in sorted(names):
            choices = [row.get(name, set()) for row in observations]
            if (any(len(choice) != 1 for choice in choices)
                    or any(choice != choices[0] for choice in choices)):
                unresolved.append(dict(backward_graph=graph, placeholder=name,
                    reason='MISSING_AMBIGUOUS_OR_CHANGING_RUNTIME_IDENTITY'))
                continue
            source_graph, source = next(iter(choices[0]))
            edges.append(dict(forward_graph=source_graph, source_node=source,
                backward_graph=graph, placeholder=name, observations=len(observations)))
    return dict(edges=edges, unresolved=unresolved,
                claim_scope='OBSERVED_RUNTIME_IDENTITY_NOT_PARAMETER_REACH')


def forward_backward_entries(capture, endpoints):
    """Find saved-value paths, not complete derivative or parameter reach."""
    from collections import defaultdict, deque
    nodes, adjacency = {}, defaultdict(set)
    for graph in capture['graphs']:
        phase, index = graph['phase'], graph.get('graph_index', 0)
        for node in graph['nodes']:
            key = (phase, index, node['name'])
            if key in nodes:
                raise ValueError('Duplicate graph node identity')
            nodes[key] = node
        for node in graph['nodes']:
            for edge in node.get('input_edges', []):
                source = (phase, index, edge['source_node'])
                if source not in nodes:
                    raise ValueError('Unknown graph edge source')
                adjacency[source].add((phase, index, node['name']))
    bridge = observed_edges(capture.get('cross_phase_runtime_bridge', {}).get('runs', []))
    for edge in bridge['edges']:
        source = ('FORWARD', edge['forward_graph'], edge['source_node'])
        target = ('BACKWARD', edge['backward_graph'], edge['placeholder'])
        if source not in nodes or target not in nodes or nodes[target]['op'] != 'placeholder':
            raise ValueError('Invalid runtime identity edge')
        adjacency[source].add(target)
    result = []
    for graph, name in endpoints:
        start = ('FORWARD', graph, name)
        if start not in nodes:
            result.append(dict(graph=graph, node=name, status='UNKNOWN_FORWARD_NODE', entries=[]))
            continue
        distance, queue, entries = {start: 0}, deque([start]), []
        while queue:
            current = queue.popleft()
            if current[0] == 'BACKWARD':
                entries.append(dict(graph=current[1], placeholder=current[2], distance=distance[current]))
                continue
            for target in sorted(adjacency[current]):
                if target not in distance:
                    distance[target] = distance[current] + 1
                    queue.append(target)
        result.append(dict(graph=graph, node=name, entries=entries,
            status='OBSERVED_SAVED_VALUE_PATH' if entries else 'NO_OBSERVED_SAVED_VALUE_PATH'))
    return dict(rows=result, unresolved_identity=bridge['unresolved'],
        claim_scope='SAVED_VALUE_PATH_ONLY_NOT_COMPLETE_DERIVATIVE_OR_RUNTIME_PARAMETER_REACH')
