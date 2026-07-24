# TVM Gather fresh rerun — invalid for frozen-core scoring

The historical TVM Gather negative-index case was rerun after the generic
localization core freeze, using the locally checked-out buggy revision
`/data1/tzh/tvm_bug_19436` and the patch-excluded case package.

Fresh artifacts:

- `results/operator_oracle/tvm_gather_negative_v1/buggy.json`
- `results/operator_oracle/tvm_gather_negative_v1/buggy_r2.json`
- `results/operator_oracle/tvm_gather_negative_v1/pre_reveal_certificate.json`

Both runs violate the ONNX Runtime reference and both positive-index controls
are exact.  But they are not repeatable: the first negative-index run has
maximum absolute error `12.0`, while the second has approximately
`3.745e32`.  The pre-reveal certificate therefore correctly emits `INVALID`
because its complete witness and null-repeatability gates fail.

The most plausible mechanism is an out-of-bounds/undefined read caused by the
un-normalized negative index.  That is consistent with the historical
frontend patch, but this report does **not** use that mechanism to rescue the
certificate: a non-repeatable witness is unsuitable for the frozen-method
external scoring protocol.

Consequences:

- Previous Case 004 artifacts remain historical evidence, not a fresh Phase-3
  score under the frozen core.
- The case cannot presently serve as the higher-level external validation.
- No fixed checkout, patch, or repair was used in this rerun before the blind
  certificate was generated.

