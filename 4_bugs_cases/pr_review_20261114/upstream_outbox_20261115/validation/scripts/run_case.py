"""Run isolated upstream pytest against an exact baseline module or local fix."""
import argparse
import importlib.abc
import importlib.util
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PACK = Path(__file__).resolve().parent
WORK = ROOT / '.cache/pr_review_20261114'
MODULES = {
    'B007': ('fla.ops.linear_attn.utils', 'fla/ops/linear_attn/utils.py'),
    'B038': ('flag_gems.ops.flash_attention_backward', 'src/flag_gems/ops/flash_attention_backward.py'),
    'B047': ('attorch.p_loss_kernels', 'attorch/p_loss_kernels.py'),
    'B049': ('liger_kernel.ops.kl_div', 'src/liger_kernel/ops/kl_div.py'),
    'B054': ('trident.operation.masked_softmax', 'trident/operation/masked_softmax.py'),
}


class BaselineFinder(importlib.abc.MetaPathFinder):
    def __init__(self, name, path):
        self.name, self.path = name, path

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name:
            return importlib.util.spec_from_file_location(fullname, self.path)
        return None


def child(args):
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    if args.variant == 'before':
        name, path = MODULES[args.case]
        source = args.baseline_module or PACK / 'baseline_modules' / args.case / Path(path).name
        sys.meta_path.insert(0, BaselineFinder(name, str(source)))
    def execute():
        if args.tests in (['@benchmark'], ['@dropout_benchmark'], ['@mode_benchmark']):
            if args.tests == ['@mode_benchmark']:
                from microbench_modes import measure
            elif args.tests == ['@dropout_benchmark']:
                from microbench_dropout import measure
            else:
                from microbench import measure
            measure(args.case, Path(os.environ['KA_PR_OUTPUT_DIR']))
            return 0
        import pytest
        return pytest.main(args.tests)
    if args.case == 'B054':
        # Names/signatures only, as in the earlier independent Trident replay.
        sys.path.insert(0, str(ROOT/'1_experiments/mainstream_rescan_20261025'))
        from original_trident_runtime import original_trident
        with original_trident():
            return execute()
    return execute()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--case', choices=MODULES, required=True)
    parser.add_argument('--variant', choices=['before', 'after'], required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--source', type=Path, help='Optional isolated upstream worktree')
    parser.add_argument('--memory-fraction', type=float, default=0.16)
    parser.add_argument('--baseline-module', type=Path, help='Explicit original or controlled module for a before run')
    parser.add_argument('--child', action='store_true')
    parser.add_argument('tests', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= 1:
        parser.error('--memory-fraction must be in (0, 1]')
    if args.tests[:1] == ['--']:
        args.tests = args.tests[1:]
    if args.child:
        return child(args)
    if args.out is None or not args.tests:
        parser.error('fresh --out and explicit test selection required')
    args.out.mkdir(parents=True, exist_ok=False)
    baseline = args.baseline_module.resolve() if args.baseline_module else PACK/'baseline_modules'/args.case/Path(MODULES[args.case][1]).name
    if not baseline.exists():
        baseline.parent.mkdir(parents=True, exist_ok=True)
        data = subprocess.check_output(['git', '-C', str(WORK/args.case), 'show', 'HEAD:'+MODULES[args.case][1]])
        baseline.write_bytes(data)
    cache = WORK/'gpu_cache'
    cache.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(ROOT/'.cache/tmp'), XDG_CACHE_HOME=str(cache/'xdg'),
               TRITON_CACHE_DIR=str(cache/'triton'), TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
               CUDA_CACHE_PATH=str(cache/'cuda'), FLAGGEMS_CACHE_DIR=str(cache/'flaggems'),
               FLA_CI_ENV='0', FLA_CACHE_RESULTS='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    env['LD_LIBRARY_PATH'] = '/data1/tzh/envs/ka_main/lib' + (':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    source_root = args.source.resolve() if args.source else WORK/args.case
    imports = [source_root/'src', source_root, WORK/'deps',
               ROOT/'.cache/mainstream_rescan_deps', ROOT/'.cache/qwen_ce_amp_reference_deps_20261103']
    env['PYTHONPATH'] = ':'.join(str(p) for p in imports)
    env['KA_PR_OUTPUT_DIR'] = str(args.out.resolve())
    junit = args.out.resolve()/'junit.xml'
    selected = ['-q', '-p', 'no:cacheprovider', '-o', 'addopts=', '--junitxml', str(junit)] + args.tests
    if args.tests in (['@benchmark'], ['@dropout_benchmark'], ['@mode_benchmark']):
        selected = args.tests
    command = [sys.executable, str(Path(__file__).resolve()), '--child', '--case', args.case,
               '--variant', args.variant, '--memory-fraction', str(args.memory_fraction), '--'] + selected
    if args.baseline_module:
        position = command.index('--')
        command[position:position] = ['--baseline-module', str(baseline)]
    record = dict(case=args.case, variant=args.variant, gpu=args.gpu, command=command,
                  source=str(source_root), baseline_module=str(baseline),
                  note='Original complete numerical module imported for baseline; tests and other source unchanged. Trident uses legacy API-name compatibility only.')
    numerical_source = baseline if args.variant == 'before' else source_root/MODULES[args.case][1]
    record['numerical_module_sha256'] = hashlib.sha256(numerical_source.read_bytes()).hexdigest()
    record['base_head'] = subprocess.check_output(['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True).strip()
    record['memory_fraction'] = args.memory_fraction
    (args.out/'declaration.json').write_text(json.dumps(record, indent=2))
    start = time.monotonic()
    with (args.out/'run.log').open('x') as log:
        process = subprocess.run(command, cwd=source_root, env=env, stdout=log, stderr=subprocess.STDOUT)
    record.update(returncode=process.returncode, wall_seconds=time.monotonic()-start, junit_exists=junit.exists())
    (args.out/'process.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in ['case','variant','returncode','wall_seconds','junit_exists']}))
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
