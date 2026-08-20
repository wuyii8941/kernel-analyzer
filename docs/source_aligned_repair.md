# Source-aligned MM repair

## Why the old arm was insufficient

For a low-precision MM output, Kernel Analyzer uses the exact same-operand
identity

\[
Y_{\mathrm{low}}-Y_{32}
=
\left(Y_{\mathrm{low}}-Q(Y_{32})\right)
+
\left(Q(Y_{32})-Y_{32}\right).
\]

The first term is the generated MM kernel difference. The second is output
rounding. Computing the MM in FP32 and then casting its result back to BF16
removes only the first term. It cannot remove the second term because the cast
is precisely that source.

Therefore candidate--repair trajectory separation under the historical
`FP32 MM -> BF16 cast` arm proves only that kernel arithmetic affects the real
F+B path. It is not evidence that a separately identified output-rounding
source was repaired.

## Implemented arms

`scripts/run_mm_source_aligned_repair.py` binds one exact generated MM call and
implements four modes:

| Arm | Kernel arithmetic | Output materialization |
|---|---|---|
| `SHAM` | natural | exact reconstruction of the natural output |
| `KERNEL_ONLY` | FP32 reference | deterministic nearest low-dtype rounding |
| `ROUNDING_ONLY` | preserve measured kernel residual | coordinate-wise unbiased stochastic rounding |
| `JOINT` | FP32 reference | coordinate-wise unbiased stochastic rounding |

For FP32 value \(y\) bracketed by adjacent low-precision values
\(y_-\le y\le y_+\), the rounding repair returns \(y_+\) with probability

\[
p=\frac{y-y_-}{y_+-y_-}
\]

and \(y_-\) otherwise. Thus

\[
\mathbb{E}[Q_{\mathrm{repair}}(y)\mid y]=y.
\]

This is an exact conditional-centering statement. A finite number of random
materializations still has Monte Carlo residual, which is reported separately
and is not used to deny the analytic centering property.

## Case binding

- Qwen seq128 `v_proj`: `ROUNDING_ONLY`, because only output rounding is a
  coherent source.
- Qwen seq64 `v_proj`: `JOINT`, because the completed 32-state decomposition
  finds both kernel arithmetic and output rounding coherent.
- Mamba seq64 `in_proj`: `KERNEL_ONLY`, `ROUNDING_ONLY`, and `JOINT` factorial
  arms; `JOINT` is the full observed-source repair.

## Claim boundary

The rounding modes preserve the declared BF16/FP16 ABI and remove deterministic
rounding bias in conditional expectation. They do not make a single BF16 value
equal to an arbitrary FP32 value. They also do not establish end-to-end
equivalence to a full FP32 training step. Such an equivalence claim requires a
separate high-precision complete-reference arm.

The source-repair population reports three independent facts:

1. whether the declared source is centered after repair;
2. whether the removed source is directional across the frozen population;
3. whether that removed source reaches the declared parameter-gradient carrier.

Trajectory drift is a downstream consequence and is never reinterpreted as
proof that the repair itself remains erroneous.

## Conditional downstream audit

Local centering is not an end-to-end debiasing theorem.  For a fixed training
state (s), stochastic materialization guarantees

\[
\mathbb{E}[R(x)-x\mid s]=0,
\]

but a nonlinear backward or optimizer map (F_s) can still satisfy

\[
\mathbb{E}[F_s(R(x))-F_s(x)\mid s]\ne 0.
\]

The runner therefore has a `--conditional-debias` mode.  It changes only the
repair random draw while holding weights, input, ordinary model RNG, and the
declared optimizer condition fixed.  It produces two deliberately different
estimands:

1. `REPAIR_RESIDUAL` for the declared local source component, relative to its
   exact same-operand zero point (the joint arm is the total local residual;
   a source-specific arm may deliberately retain other measured sources);
2. `CANDIDATE_MINUS_REPAIR_ENSEMBLE` after backward and optimizer mapping.

The second estimand proves that the source-debiased ensemble removes a
systematic candidate effect.  It does **not** prove that the repair's own
downstream residual is zero, because that requires an exact downstream
reference (F_s(x)).  Stateless SGD and zero-moment AdamW-step1 are reported
separately; the latter is an optimizer susceptibility probe, not a substitute
for mature natural optimizer moments.

Cross-state carrier coherence remains a useful stronger property, but its
failure is not used as evidence that a repair is conditionally unbiased.

## Qwen seq128 confirmation

The first formal conditional campaign uses 16 predeclared Qwen seq128 states
and 8 independent rounding-repair draws per state.  All matched shams preserve
the loss and the complete declared `v_proj.weight` gradient bitwise.

| Estimand | Centered | Biased | Conditions |
|---|---:|---:|---:|
| repair local rounding residual | 16 | 0 | 16 |
| candidate local effect removed | 0 | 16 | 16 |
| candidate gradient effect removed | 0 | 16 | 16 |
| candidate stateless-SGD update effect removed | 0 | 16 | 16 |
| candidate zero-moment AdamW-step1 effect removed | 0 | 16 | 16 |

Thus deterministic nearest BF16 rounding is a fixed-state conditional source
bias whose removed effect reaches the actual backward and both declared update
maps.  This result was missed by the former cross-state fixed-direction gate.
The subsequent aligned 32-step `ROUNDING_ONLY` trajectory resolves this source
as diffusive/canceling rather than persistent: the local coherence
amplification is `0.999`, the actual drift amplification is `1.825`, and the
local resultant is `3.011` times the independent split-ensemble Monte Carlo
resultant. The downstream repair residual remains
`NOT_IDENTIFIABLE_MISSING_EXACT_REFERENCE` in the absolute-reference sense.

The compact certificate is
`results/property/conditional_debias/qwen128_vproj.json`; the full per-condition
Gram evidence is
`results/coverage/cases/qwen128_vproj_conditional_debias.json.gz`.

## Qwen seq64 independent confirmation

Qwen seq64 uses `JOINT`, because its exact decomposition identifies both MM
kernel arithmetic and deterministic output rounding as coherent local source
components.  An initial 8-draw campaign left one of 16 gradient/SGD conditions
unresolved.  No threshold was changed.  A new seed bank and 16 repair draws
were therefore applied to all 16 fixed conditions.

| Estimand | Centered | Biased | Conditions |
|---|---:|---:|---:|
| repair local joint-source residual | 16 | 0 | 16 |
| candidate local effect removed | 0 | 16 | 16 |
| candidate gradient effect removed | 0 | 16 | 16 |
| candidate stateless-SGD update effect removed | 0 | 16 | 16 |
| candidate zero-moment AdamW-step1 effect removed | 0 | 16 | 16 |

This closes the fixed-state conditional source-formation result for Qwen64.
As above, the downstream rows identify the systematic candidate effect removed
relative to the source-debiased ensemble; they do not provide an absolute
high-precision reference certificate for the repaired downstream computation.
The compact result is
`results/property/conditional_debias/qwen64_vproj.json`; the full result is
`results/coverage/cases/qwen64_vproj_conditional_debias_r16.json.gz`.

## Mamba cross-architecture confirmation

Mamba seq64 `in_proj` was evaluated under the `JOINT` arm on 16 fixed
conditions with 16 independent draws per condition.  The model's optimized
selective-scan path is unavailable in this environment, so the four
condition-disjoint shards use the same sequential implementation and are
merged only after each condition is independently certified.  No
cross-condition direction is reconstructed.

| Estimand | Centered | Biased | Unresolved | Conditions |
|---|---:|---:|---:|---:|
| repair local joint-source residual | 16 | 0 | 0 | 16 |
| candidate local effect removed | 0 | 16 | 0 | 16 |
| candidate gradient effect removed | 0 | 13 | 3 | 16 |
| candidate stateless-SGD update effect removed | 0 | 13 | 3 | 16 |
| candidate zero-moment AdamW-step1 effect removed | 0 | 16 | 0 | 16 |

This establishes conditional local source formation across a different model
architecture, but it does not pass the frozen all-layer gate.  In particular,
the three unresolved gradient/SGD conditions are neither imputed centered nor
converted to positive by increasing the margin.  AdamW-step1 is a bounded
response probe and does not replace mature optimizer-state evidence.

The compact certificate is
`results/property/conditional_debias/mamba_seq64_input_proj.json`; the merged
full certificate is
`results/coverage/cases/mamba_seq64_input_proj_conditional_debias.json.gz`.
