# Resource policy

The root filesystem is not an experiment workspace.  All model weights,
environments, caches, compiler products, temporary files, and raw results must
live below `/data1/tzh`.

Every new model/configuration must pass `scripts/resource_preflight.py` before
loading weights.  The preflight records free disk, host memory, all visible GPU
memory, cache paths, and a conservative weight/gradient/activation/compiler
budget.  A measured smoke peak supersedes the estimate before longer shapes
are admitted.

The minimum free-space floor is 500 GiB.  Compact mathematical ledgers,
protocols, frozen inputs, and case evidence are permanent.  Raw tensors,
compiler traces, failed worker outputs, and regenerable caches are deleted
after their compact digest-bearing result is validated.
