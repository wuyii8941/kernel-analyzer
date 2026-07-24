# Qwen3 original-candidate weight-copy kernel repair contract v0.1

## Separate subjects

The following are three distinct generated treatment families and remain
separate in all outputs and coverage accounting:

- `_to_copy_1`: 56 calls, q-projection and attention-output weights;
- `_to_copy_3`: 56 calls, k-projection and v-projection weights;
- `_to_copy_13`: 84 calls, MLP gate/up/down weights.

Each kernel converts a contiguous FP32 weight buffer to a preallocated FP16
buffer.  The replacement checks equal element count, reshapes only to the live
destination shape, performs eager dtype conversion and copies into that buffer.

## Selection

For the two-role families, test both roles at layers 0, 14 and 27: call indices
`0,1,28,29,54,55`.  For the three-role family, test all roles at those layers:
`0,1,2,42,43,44,81,82,83`.

## Gates and claims

Use the common original-candidate gates: exact eager/candidate anchors and graph
family, exact family call count with one selected replacement, exact repeats,
no repair-time backend compilation and exact restoration.  Null effects are
valid evidence and must not be dropped.

Each passing result covers only its named generated family representatives.
Even a null result is not automatically transferable across the other copy
families, untested states or other dtypes.  Eager is not a correctness authority.
