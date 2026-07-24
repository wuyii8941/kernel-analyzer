# Qwen3 backward repeated-family proxy sham contract v0.1

The sham installs the same module-level Python proxy used by the frozen
early/middle/late repair campaign for
`triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_32`.  It forwards
all 27 calls to the original compiled kernel without changing arguments or
outputs.

Validity requires the frozen candidate and scorer identities, one backward
hook, one generated module, exactly 27 family calls and exactly 27 forwards to
the original kernel.  The clipped-gradient and AdamW-update tensor hashes must
then be compared with the uninstrumented compiled baseline.

An exact-null sham rules out module monkey-patching and proxy dispatch as the
source of the earlier endpoint effects.  It does not control the additional
kernel launches, temporary allocations, fusion-boundary change or possible
synchronization changes introduced by an eager local replacement.  A non-null
sham invalidates the repair campaign's attribution interpretation until the
instrumentation effect is removed.
