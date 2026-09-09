"""Check a release-bound input bank before loading a training model (CPU only)."""
import argparse
import hashlib
import json
from pathlib import Path


def check(bank_path, capture_path, count, warmup=0):
    bank_bytes = Path(bank_path).read_bytes()
    bank = json.loads(bank_bytes)
    capture = json.loads(Path(capture_path).read_text())
    identity = capture['input']
    digest = hashlib.sha256(bank_bytes).hexdigest()
    if digest != identity['input_bank_sha256']:
        raise ValueError('Input bank differs from the release-bound bank')
    if count < 1 or warmup < 0:
        raise ValueError('Positive state count and nonnegative warmup required')
    states = bank.get('states', bank.get('records'))
    if not isinstance(states, list) or len(states) < count + warmup:
        raise ValueError('Insufficient input states')
    tokens = lambda state: state.get('input_ids', state.get('token_ids'))
    anchor = hashlib.sha256(json.dumps(tokens(states[0]), sort_keys=True,
                            separators=(',', ':')).encode()).hexdigest()
    if anchor != identity['token_ids_sha256']:
        raise ValueError('Release anchor token sequence differs')
    for state in states[:count + warmup]:
        value = tokens(state)
        if not isinstance(value, list) or len(value) != identity['sequence_length']:
            raise ValueError('Input state changes the declared sequence shape')
    return dict(status='INPUT_IDENTITY_CHECKED_NOT_EXECUTION_PROOF',
                input_bank_sha256=digest, anchor_token_sha256=anchor,
                states=count, warmup_steps=warmup)


def check_command(command):
    """Apply only to the capture interface audited here, not arbitrary commands."""
    audited = {
        'run_residual_rms_forward_capture.py',
        'run_grouped_causal_softmax_capture.py',
    }
    if not any(Path(arg).name in audited for arg in command):
        return None
    def argument(name, default=None):
        values = []
        for i, arg in enumerate(command):
            if arg == name:
                if i + 1 == len(command) or command[i + 1].startswith('--'):
                    raise ValueError('Missing argument: ' + name)
                values.append(command[i + 1])
            elif arg.startswith(name + '='):
                values.append(arg.split('=', 1)[1])
        if not values and default is not None:
            return default
        if len(values) != 1 or not values[0]:
            raise ValueError('Unique explicit argument required: ' + name)
        return values[0]
    if any(arg == '--state-bank' or arg.startswith('--state-bank=') for arg in command):
        raise ValueError('Separate state bank is not yet audited for this capture interface')
    capture = Path(argument('--release-dir')) / 'capture.json'
    result = check(argument('--input-bank'), capture, int(argument('--states')),
                   int(argument('--warmup-steps', '0')))
    result['capture_sha256'] = hashlib.sha256(capture.read_bytes()).hexdigest()
    result['checker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-bank', required=True)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--states', type=int, required=True)
    parser.add_argument('--warmup-steps', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(check(args.input_bank, args.capture, args.states, args.warmup_steps)))
