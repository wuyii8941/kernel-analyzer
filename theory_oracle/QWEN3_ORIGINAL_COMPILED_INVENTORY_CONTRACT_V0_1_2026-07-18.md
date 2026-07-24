# Qwen3 original compiled operator/kernel inventory contract v0.1

## Purpose

Observe the unchanged, frozen Qwen3-0.6B whole-compiled scorer realization and
materialize the correspondence evidence needed to design valid operator
interventions.

This inventory is descriptive. It is neither an Oracle verdict nor causal
operator attribution.

## Frozen realization

- matched state: heldout transport run B, optimizer step 29;
- model/input/FP16/SDPA-math/training protocol: identical to the validated natural
  transition and operator pilots;
- ordered compiler input history: the ten captured minibatches in snapshot
  metadata;
- expected ordered unique Dynamo graph family:
  - `31ec1dd1b3689460c96b5b7882e5cbcb3a33dea614ef07111b47f48373caef04`,
    455 nodes;
  - `ee4053cb35f6351f6303e6b9922ccf0fa2189246fc5bcbee31d4793241164e5b`,
    457 nodes;
- expected target scorer SHA256:
  `1107b4ac9c2662b34572cee3b4b4e1bf454a4b6d0a6def0c427d84f9944a09f2`.

## Observations

- Dynamo graph node target and available module/source metadata;
- Inductor pre-fusion IR, post-fusion IR and generated code emitted by PyTorch's
  trace facility;
- no real tensors (`save_real_tensors = false`);
- graph hashes/node counts, runtime invocation counts and target scorer hashes.

## Fail-closed gates

The inventory is valid for the original candidate only if:

1. the ordered unique graph hashes and node counts equal the frozen family;
2. the two target scorer repetitions equal the frozen candidate hash;
3. target repetitions are bit-exact;
4. trace instrumentation stores no real tensor dump.

Failure means the observer changed or failed to reproduce the treatment. In that
case its op/kernel correspondence must not guide causal claims.

## Allowed interpretation

A valid inventory can identify standalone versus fused operator boundaries and
select candidate interventions. It cannot establish that any listed operator
caused the observed discrepancy. Fused membership only shows non-identifiability
of a naive single-op swap under the original realization.
