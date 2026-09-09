"""Normalize recorded output tokens, never infer identity from token spelling."""
from copy import deepcopy


def normalize_capture(capture):
    result = deepcopy(capture)
    for run in result.get('cross_phase_runtime_bridge', {}).get('runs', []):
        index = run['forward_phase']['graph_index']
        if type(index) is not int or index < 0:
            raise ValueError('Invalid forward graph index')
        outputs = {}
        for output in run.get('forward_outputs', []):
            token = output['runtime_token']
            source = output['source_node']
            if token in outputs and outputs[token] != source:
                raise ValueError('Conflicting output token identity')
            outputs[token] = source
        for row in run.get('backward_inputs', []):
            if 'global_forward_matches' in row:
                continue
            matches = []
            for match in row.get('forward_output_matches', []):
                if match.get('identity_mode') not in ('EXACT_PYTHON_OBJECT', 'EXACT_STORAGE_VIEW'):
                    continue
                token = match['runtime_token']
                if token not in outputs:
                    raise ValueError('Identity token lacks recorded output')
                matches.append(dict(phase_graph_index=index, source_node=outputs[token],
                    identity_mode=match['identity_mode']))
            row['global_forward_matches'] = matches
            row['normalization_provenance'] = 'RECORDED_FORWARD_OUTPUT_TOKEN_LOOKUP_ONLY'
    return result
