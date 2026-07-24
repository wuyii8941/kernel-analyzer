# Qwen3 original compiled operator/kernel inventory contract v0.2

This revision preserves every v0.1 realization and scorer gate, but corrects a
missing trace-materialization gate discovered when the ordinary Inductor cache
returned an exact compiled artifact without populating the requested debug
files.

## Additional protocol

- compile in a fresh temporary `TORCHINDUCTOR_CACHE_DIR`;
- force-disable the FX graph cache for this observational run;
- delete the temporary compilation cache after the trace is copied;
- continue to prohibit real-tensor trace dumps.

The cache location is not treated as model state. Nevertheless, equality of the
frozen graph family and target scorer hash remains mandatory, so a cache-induced
realization change fails closed.

## Additional validity gate

For both forward graph specializations, the trace must contain:

- readable FX graph;
- transformed FX graph;
- pre-fusion IR;
- post-fusion IR;
- generated output code.

An empty trace is classified `INCOMPLETE_KERNEL_INVENTORY`, even if graph and
scorer identities are exact.

All purposes, interpretation limits, expected hashes and the no-causal-claim
rule from v0.1 remain unchanged.
