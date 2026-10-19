# Inductor SiLU call binding review — 2026-10-18

This review covers `corpus/calls/inductor_silu.py` and
`corpus/call_bindings/inductor_silu.json`. It applies only to the public
out-of-place SiLU call declared there.

## Task correspondence

The wrapper accepts exactly one floating tensor and returns its complete SiLU
result. An independent `make_fx` trace on the installed PyTorch 2.10.0+cu128
package represents `torch.nn.functional.silu(x)` as exactly
`torch.ops.aten.silu.default(x)`. The installed schema is
`aten::silu(Tensor self) -> Tensor`; the separate in-place operation is
`aten::silu_(Tensor(a!) self) -> Tensor(a!)` and is not selected here.

This establishes the plan's `aten.silu` operation from the public call rather
than from agreement with generated code. The admitted ACT `torch` profile
defines each result as `x/(1+exp(-x))` for finite floating inputs and supplies a
rigorous enclosure. Device exponential approximation and output rounding stay
in the program-numerics comparison.

## Complete boundary

The plan maps the complete input pre-state, the complete returned tensor, and
requires the input to remain unchanged. The wrapper has no tensor parameters,
buffers, scalar task arguments, or closed-over tensor state.

An earlier isolated experiment used the same wrapper and mapping on eight
development and sixteen confirmation inputs. The baseline established a
complete reference for every coordinate. A controlled `silu(x)+0.01` wrapper
made all 16,728 sampled coordinates exclude zero in the task residual, while
the baseline task residual excluded zero at no coordinate. The ordinary
program-numerics comparison remained separately visible. This experiment
supports the distinction between task and numerical effects but does not by
itself retain the current files.

After retention, the repository measurement declaration passed the formal
entry without a temporary registry override. All 24 calls used the reviewed
binding and an admitted ACT reference. Each call captured one TTIR kernel with
complete IR coverage; its loaded tensor was identified as the declared input,
every public output coordinate was written, the input remained unchanged, and
no hidden source or failure class was reported. The producer trace is expected
not to align because the unmeasured trace runs the compiled function under an
eager stance; this single-launch call has no carried state, and input identity,
bytes, coverage, and version counters establish its provenance.

The reference covered all 16,728 sampled coordinates and met the declared
resolution. The task residual excluded zero at no coordinate, with projection
means around `1e-17`. The program-numerics comparison found a small negative
mean on R1 (about `-8.03e-8`, Holm-adjusted p about `0.000307`) and R2 (about
`-7.15e-8`, adjusted p about `0.000832`); complete `K-f` carries the same
finding. This is a numerical result for the declared normal-input population,
not a task-formula error. The complete report is
`.cache/spec_trial_20261018_diverse/inductor_silu_controlled_20261018/production_reviewed_report_20261018.json`.

## Scope and limits

The recorded population uses finite float32 values of shape 17×41 sampled from
a centered normal distribution. It does not test every admitted shape or
floating dtype, extreme finite magnitudes, infinities, NaN, backward SiLU, an
in-place call, or a fused SiLU composite. Those cases require their own valid
input population or call binding.

A non-detection under the declared projections is not proof that every input
is unbiased.
