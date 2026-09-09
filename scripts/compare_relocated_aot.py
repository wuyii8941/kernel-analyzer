"""Compare saved graphs allowing only an explicitly declared device-argument change."""
import argparse
import copy
from pathlib import Path
from kernel_analyzer.aot_structure_identity import compare
from scripts.run_numerical_coverage import read, save, sha


def relocated_compare(reference, captured, source_device, target_device):
    if source_device == target_device or not source_device.startswith('cuda:') or not target_device.startswith('cuda:'):
        raise ValueError('Declare distinct CUDA devices')
    adjusted = copy.deepcopy(captured)
    changes = []
    for graph in adjusted['graphs']:
        for node in graph['nodes']:
            arguments = node.get('arguments')
            kwargs = arguments.get('kwargs') if isinstance(arguments, dict) else None
            if isinstance(kwargs, dict) and kwargs.get('device') == source_device:
                kwargs['device'] = target_device
                changes.append(dict(phase=graph['phase'],graph_index=graph.get('graph_index',0),
                                    node=node['name'],field='arguments.kwargs.device'))
    return dict(strict=compare(reference,captured),
                after_declared_device_change=compare(reference,adjusted),
                declared_change=dict(source=source_device,target=target_device),
                changed_fields=changes,
                runtime_values_or_execution_identity_proved=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference','captured','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--captured-device', required=True)
    parser.add_argument('--reference-device', required=True)
    args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Choose new output under /data1/tzh')
    left,right=read(args.reference),read(args.captured)
    result=relocated_compare(left.get('capture',left),right.get('capture',right),
                             args.captured_device,args.reference_device)
    result['source_sha256']={str(p.resolve()):sha(p) for p in
                            (args.reference,args.captured,Path(__file__))}
    save(args.output,result)
    print({'structure_matches_after_declared_device_change':result['after_declared_device_change']['structure_identical'],
           'changed_device_fields':len(result['changed_fields'])})
