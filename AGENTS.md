# Agent instructions (kernel-analyzer)

The single current index is [`CURRENT.json`](CURRENT.json): release / frozen tag, versions, entry points, capability,
status documents and pending reviewer decisions. Do not copy current conclusions into other files; link to it.

Working rules (same as `CLAUDE.md`):
- Read and write only inside `/data1/tzh/kernel-analyzer`; never use `/home` (no home-directory caches). Point caches to
  the repository: `export HOME=$PWD/.cache XDG_CACHE_HOME=$PWD/.cache TRITON_CACHE_DIR=$PWD/.cache/triton
  PIP_CACHE_DIR=$PWD/.cache/pip TMPDIR=$PWD/.cache/tmp PYTHONPATH=src`.
- Main environment: `/data1/tzh/envs/ka_main` (torch 2.10.0+cu128, Triton 3.6.0); system `python` is 2.7.
- Frozen records (protocols, answers, raw results, tags such as `general-v3.1`) are never overwritten, renamed or
  recomputed in place; new runs append. Specs change only through the reviewer.
- Tests must not modify or rewrite tracked files (the session guard in `tests/conftest.py` fails the run); write
  outputs under `tmp_path`.
- Changing reference semantics, default statistics or verdict rules requires a version bump and regression; tidying
  never changes algorithms, and algorithm fixes are committed separately from tidying.
