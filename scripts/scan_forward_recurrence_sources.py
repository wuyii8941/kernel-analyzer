"""Check every literal Triton definition against the forward recurrence family."""
import argparse
from pathlib import Path
from scripts.scan_recurrence_sources import scan
from scripts.run_numerical_coverage import save, sha
from kernel_analyzer import forward_state_recurrence_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose a new output under /data1/tzh')
    sources = []
    for path in a.source:
        rows = scan(path.read_text(), checker=forward_state_recurrence_source.check_source)
        sources.append(dict(path=str(path.resolve()), sha256=sha(path), rows=rows))
        print(dict(source=str(path), definitions=len(rows),
                   checked=sum(r['status'] == 'SOURCE_CHECKED' for r in rows)), flush=True)
    save(a.output, dict(schema='forward-recurrence-source-scan-v1', sources=sources,
                        numerical_results_read=False, runtime_measurement_complete=False,
                        source_sha256={str(p.resolve()): sha(p) for p in (
                            Path(__file__), Path(forward_state_recurrence_source.__file__))}))


if __name__ == '__main__':
    main()
