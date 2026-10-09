# Agent instructions (kernel-analyzer)

The single current index is [`CURRENT.json`](CURRENT.json). The repository has four parts, each keeping only the
latest, most complete version: [1_experiments](1_experiments/README.md) (complete experiments),
[2_tool](2_tool/README.md) (the current tool), [3_audits](3_audits/README.md) (audits, open findings, pending external
materials), [4_bugs_cases](4_bugs_cases/README.md) (bugs found and important cases). Material removed on 2026-10-09
is recoverable from tag `pre-reorg-20261009`; do not restore it into the tree.

Working rules (same as `CLAUDE.md`):
- Read and write only inside `/data1/tzh/kernel-analyzer`; never use `/home`. Point caches to the repository:
  `export HOME=$PWD/.cache XDG_CACHE_HOME=$PWD/.cache TRITON_CACHE_DIR=$PWD/.cache/triton PIP_CACHE_DIR=$PWD/.cache/pip
  TMPDIR=$PWD/.cache/tmp PYTHONPATH=2_tool/src`.
- Main environment `/data1/tzh/envs/ka_main` (torch 2.10.0+cu128, Triton 3.6.0); official main Triton in
  `/data1/tzh/envs/triton_main`; system `python` is 2.7.
- Frozen records (protocols, answers, raw results, tags such as `general-v3.1`) are never overwritten or recomputed in
  place. Specs change only through the reviewer. Results are reported in three states only.
- Tests (`python -m pytest -q 2_tool/tests`) must not modify tracked files (session guard in `2_tool/tests/conftest.py`).
- Open external-audit findings (`3_audits/README.md`) bound what the tool's conclusions may claim; fix them test-first.
- Upstream issues are drafts only (submitted by the user); public pushes need the user's confirmation.
