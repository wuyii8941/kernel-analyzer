"""Automatically scan all literal kernels using the complete forward contract."""
import argparse
from pathlib import Path
from scripts import scan_recurrence_sources
from scripts.run_numerical_coverage import save, sha
from kernel_analyzer import grouped_causal_softmax_source


def scan(source):
    return scan_recurrence_sources.scan(source, checker=grouped_causal_softmax_source.check_source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Choose a new output under /data1/tzh')
    sources = []
    for path in args.source:
        rows = scan(path.read_text())
        sources.append(dict(path=str(path.resolve()), sha256=sha(path), rows=rows))
        print(dict(source=str(path), definitions=len(rows),
                   checked=sum(r['status'] == 'SOURCE_CHECKED' for r in rows)), flush=True)
    dependencies = [Path(__file__), Path(scan_recurrence_sources.__file__),
                    Path(grouped_causal_softmax_source.__file__)]
    save(args.output, dict(schema='grouped-causal-softmax-source-scan-v1', sources=sources,
         numerical_results_read=False, runtime_measurement_complete=False,
         source_sha256={str(p.resolve()): sha(p) for p in dependencies}))


if __name__ == '__main__':
    main()
