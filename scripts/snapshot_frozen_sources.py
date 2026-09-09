#!/usr/bin/env python3
"""Preserve still-matching declared Python sources before a new implementation version."""
import argparse
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha, ROOT


def preserve(protocol, output):
    """Save only still-matching sources; never reconstruct a changed version."""
    output = Path(output)
    if output.exists() or not output.resolve().is_relative_to(Path('/data1/tzh')):
        raise ValueError('Use new output under /data1/tzh')
    sources = {}
    for name, expected in read(protocol)['source_sha256'].items():
        path = (ROOT / name).resolve()
        if sha(path) != expected:
            raise ValueError('Source changed; do not reconstruct history: ' + name)
        if path.suffix == '.py':
            sources[str(path)] = path.read_text()
    save(output, sources)
    return {'preserved_python_sources': len(sources)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Use new output under /data1/tzh')
    print(preserve(args.protocol, args.output))
