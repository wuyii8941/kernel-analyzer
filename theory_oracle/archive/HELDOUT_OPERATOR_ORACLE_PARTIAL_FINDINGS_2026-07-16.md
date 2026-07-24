# Held-out Operator Oracle Partial Findings — 2026-07-16

> Contract: `HELDOUT_OPERATOR_ORACLE_VALIDATION_CONTRACT_V0_1_2026-07-16.md`. H1--H3, H5 and H6 have been executed or attempted; H4 was attempted but the intended compiled-bf16 realization was not applicable on the available T4 path. This is a partial confirmation report; unexecuted cells remain missing. The later preregistration audit establishes that v0.1 froze case families but not every executable field, so “held-out” here must not be read as a complete immutable-manifest claim.

## 1. H1 — explicit fp32/bf16/fp32 cast preservation

Source artifacts: `paper/confirmed_bugs/pytorch/bug_004_bf16_cast_elided/original_bug19.py` and the issue-draft minimal reproducer.

The frozen minimal relation was executed with seed 0 and shape `(4, 32, 32)` on PyTorch `2.11.0+cu126` and `2.13.0.dev20260609+cu126`. Both environments produced the same distinguishing measurements:

```text
eager vs compiled max       = 0.10945892333984375
compiled vs no-cast max     = 1.52587890625e-05
eager vs no-cast max        = 0.10945892333984375
```

The explicit `fp32 -> bf16 -> fp32` conversion is an observable representation operation, not an algebraic identity. On the covered composite and input, the candidate behaves like the program with that operation removed.

Verdict:

```text
REJECT exact representation-conversion relation
subject: compiled composite program
constituent fusion-pass attribution: NOT_IDENTIFIABLE from this witness alone
```

Both available environments are positive candidates for this case. The precommitted fixed-negative role is therefore missing; nightly status cannot be inferred from issue metadata.

## 2. H2 — expanded `index_add` exact indexing relation

Artifact: `paper/confirmed_bugs/pytorch/bug_033_index_family_expanded_wrong_result/minimal_repro.py`.

### PyTorch 2.11 result

For the minimal input, eager produced rows:

```text
row 0 = 2, row 1 = 1, rows 2/3 = 0
```

Compiled produced `3` in every row, with max difference `3.0`. Across additional shapes max differences ranged from about `4.52` to `14.33`; dim-1 expansion, fp64, fp16 and CPU also showed wrong results. Materializing with `.clone()` or using a no-op same-size expand removed the mismatch. Backward with respect to `src` matched in the tested case.

Verdict:

```text
REJECT exact indexing/expanded-view conformance
subject: composite API computation
effect: deterministic and trigger/shape conditioned
```

### Current nightly result

PyTorch `2.13.0.dev20260609+cu126` still produced `3` in every row for the minimal case, with max difference `3.0`, and reproduced across GPU dtype/shape variants. Therefore this environment is also a positive candidate, not the precommitted fixed-negative role.

The local bug mapping records the issue as fixed later, but this June 9 nightly evidently predates or lacks that fix. Version status is not inferred from issue status.

### Contract consequence

- H2 positive detection: pass on both applicable environments;
- H2 matched fixed negative: missing;
- an absent fix cannot be scored as a negative-control failure;
- exact indexing relation rejects without any floating tolerance.

## 3. H3 — `slice_scatter(...).sum(same_dim)` backward relation

Artifact: `paper/confirmed_bugs/pytorch/bug_007_slice_scatter_backward/minimal_repro.py`.

### PyTorch 2.11 result

Forward max difference was only `4.7684e-07`. Nevertheless:

```text
eager x.grad sum = 140.6300
compiled x.grad sum = 0
x.grad max difference = 3.6680
y.grad max difference = 0
```

The bug fired when reduction used the same dimension as `slice_scatter`, including another slice range. It did not fire for a different reduction dimension or the tested mean case.

Verdict:

```text
REJECT backward/transition contract
forward numerical conformance alone is insufficient
```

### Current-nightly result

Forward max difference remained nonzero at `7.1526e-07`, but eager and compiled `x.grad` and `y.grad` matched exactly in the primary and characterized cases. The reproducer's printed prose still says “WRONG” because it is static historical text; the numerical fields show the fix.

Verdict for covered cases:

```text
ACCEPT exact backward relation / no historical bug witness
nonzero forward floating delta remains a separate numerical measurement
```

### Incremental-value consequence

H3 is a real case where forward raw delta is tiny and nearly unchanged between broken and fixed versions, while the backward semantic relation changes from grossly wrong to exact. A forward-only raw-delta detector does not identify the violation; the transition/backward contract does.

This is evidence of endpoint/contract incremental value, not yet an aggregate detector-performance claim.

## 4. H4 — `hardtanh` bf16 boundary backward applicability

The issue-draft minimal reproducer was attempted on both PyTorch `2.11.0+cu126` and `2.13.0.dev20260609+cu126` using a Tesla T4. Both runs emitted a warning that the T4 does not support native bfloat16 compilation and that this compilation path was skipped. Eager and the returned callable both produced gradient `1.0`.

This equality is not a fixed/non-trigger negative: the frozen validation contract requires the candidate compiled-bf16 realization to be established. Under its predeclared version-role rule, unsupported hardware or failure to instantiate the historical bug condition is `INAPPLICABLE`, not `ACCEPT`.

```text
intended candidate identity: not established
behavioral bit: undefined
verdict: INAPPLICABLE for H4 confirmation scoring
```

## 5. H5 — higher-order gradient metadata

Artifact: `paper/confirmed_bugs/pytorch/bug_019_create_graph_requires_grad_dropped/minimal_repro.py`.

### PyTorch 2.11 result

Eager result had `requires_grad=True` and a `SumBackward0` `grad_fn`. Compiled result had `requires_grad=False`, `grad_fn=None`, and backward failed because the tensor did not require grad.

Verdict:

```text
REJECT exact observable autograd-metadata/state relation
```

### Current-nightly result

Compiled result now had `requires_grad=True` and a `CompiledFunctionBackward` `grad_fn`, so the historical metadata-drop witness is fixed. Calling backward then raised an explicit error that `torch.compile` with AOTAutograd does not currently support double backward.

Contract interpretation:

- historical metadata relation: covered-case `ACCEPT`/no witness;
- double-backward execution: must be evaluated against a documented capability contract;
- if the API declares it unsupported, report `UNSUPPORTED`, not correctness `REJECT`;
- if the claimed domain promises double backward, the explicit failure rejects that broader capability contract.

The legacy script labels any double-backward exception as “FAILS,” but script exit/prose is not the Oracle. The expected relation and claimed domain determine the verdict.

## 6. H6 — eager-wrong / XLA-right SELU numerical case

Artifact: `paper/confirmed_bugs/tensorflow/tf_selu_elu_expm1/repro.py`.

Environment: TensorFlow `2.22.0-dev20260618`, Tesla T4.

For negative float32 inputs:

| Input | Eager SELU | Stable mathematical/XLA result |
|---:|---:|---:|
| `-1e-7` | `-2.095818e-7` | `-1.758099e-7` |
| `-1e-10` | `0` | `-1.758099e-10` |
| `-1e-20` | `0` | `-1.758099e-20` |
| `-1e-30` | `0` | `-1.758099e-30` |

Eager ELU similarly returned zero for inputs at and below `-1e-8`. The root-cause contrast showed float32 `exp(x)-1` rounds to zero at small magnitude while stable `expm1(x)` preserves the result. XLA matched the stable mathematical SELU result for the covered small inputs.

Oracle interpretation:

```text
truth reference: stable high-precision expm1-based mathematical relation
eager: REJECT truth-relative numerical contract on covered tail inputs
XLA candidate: ACCEPT covered truth-relative relation within observed rounding
differential-only result: disagreement, but cannot identify the correct side
```

This is direct empirical evidence that implementation-relative bias can point from a wrong baseline toward a correct compiled result. Eager cannot be the default mathematical truth.

## 7. V1/V2 — invalid and subject-scope controls from existing evidence

### V1 fallback identity

The earlier online scan's nonzero discrepancy ended exactly when Dynamo reached its recompile limit; later records lacked a per-call compiled-path canary and became exactly equal to eager. Under the frozen rule, post-limit rows are `INVALID` implementation comparisons rather than zero-discrepancy passes.

### V2 segmented region scope

All six BERT segmented compiled endpoints failed the predeclared exact monolithic-parity requirement. The observed effects remain valid for the segmented regions but are classified `I1 REGION EFFECT`; original-program constituent-operator causal claims are `NOT_IDENTIFIABLE`.

These are empirical evidence that the Oracle's validity/scope gates refuse attractive but unsupported conclusions.

## 8. Raw-delta comparison already identified

The partial confirmation contains two forms of incremental information:

1. H3 broken forward max delta was `4.7684e-07`, while the fixed nightly forward max delta was slightly larger at `7.1526e-07`. A forward raw-delta ranking therefore orders the fixed case as worse, while the backward contract correctly distinguishes an all-zero gradient from an exact gradient match.
2. H6 eager/XLA absolute disagreement becomes arbitrarily tiny with inputs such as `-1e-30`, yet eager has 100% relative truth error while XLA follows the stable mathematical relation. An absolute threshold can miss the violation, and differential magnitude cannot determine which side is correct.

This supports incremental validity of endpoint/specification-aware contracts over one raw forward/absolute delta. It is not yet an aggregate comparison against all predeclared baselines.

## 9. Partial scoring

| Cell | Status |
|---|---|
| H1 applicable positive detection | `2/2` environments rejected |
| H1 fixed negative | missing; current nightly still exhibits the cast-elision fingerprint |
| H2 applicable positive detection | `2/2` environments rejected |
| H2 fixed negative | missing |
| H3 broken positive detection | `1/1` rejected |
| H3 current fixed negative | `1/1` covered case accepted |
| H5 broken metadata positive | `1/1` rejected |
| H5 current metadata negative | `1/1` historical witness fixed; double-backward capability unresolved by this contract |
| H6 eager-wrong/XLA-right truth relation | executed; truth-relative contract selects XLA rather than eager |
| H4 positive/fixed pair | attempted; intended compiled-bf16 path `INAPPLICABLE` on available T4 runs |
| N2 dedicated non-trigger strata | not executed as a unified control |
| N3 set-valued control | synthetic mechanics pass; no natural candidate pair |
| N4 legal reassociation | synthetic mechanics pass plus separately frozen real CUDA eager/compiled `ACCEPT` |
| V1 fallback invalidation | supported by earlier fallback/recompile-limit evidence |
| V2 region/operator downgrade | supported by six segmented-parity failures |
| U1 softmax no-margin refusal | measured; correctly `UNINSTANTIATED` |
| U2 stochastic finite-resolution control | original v0.1 row uninstantiated; separately versioned 100-draw control correctly `INDETERMINATE` |
| all four hard raw-delta quadrants | mechanics represented; unified cross-case scoring not run |

No aggregate success rate is reported because the contract's confirmation set is incomplete.

### N3 contract-level large-but-conforming control

The predeclared set-valued `topk` control can be instantiated without a numerical tolerance. Consider a length-`1,000,001` input whose only equal maxima occur at indices `0` and `1,000,000`, with `k=1`. The documented PyTorch contract does not guarantee a stable tied index. Therefore both

```text
(value=max, index=0)
(value=max, index=1,000,000)
```

belong to the allowed set. A raw index delta is `1,000,000`, while both membership bits are `fail=0`.

This is a synthetic semantic negative control that validates the set-valued verdict mechanics and realizes a large-but-conforming **output-representation** quadrant. It is not evidence that the available compiler naturally selects the two alternatives, so it does not count as compiler external-validity coverage.

### N4 contract-level legal reduction-order control

For float32 operands

```text
[2^24, 1, -2^24]
```

two admitted sequential evaluation orders produce:

```text
(2^24 + 1) + (-2^24) = 0
2^24 + (1 + -2^24)   = 1
exact real sum        = 1
```

The standard `gamma_2 * sum(abs(x))` forward-error enclosure is approximately `4.000000596`. Both results therefore belong to the declared analytical envelope, despite raw output delta `1.0` and an unbounded/undefined relative discrepancy if normalized by the zero result.

```text
both membership bits: fail=0
contract verdict: ACCEPT for the covered reduction-order alternatives
```

This is a synthetic arithmetic control, not evidence that a particular compiled reduction selected each order. Together with N3, it proves that “large raw delta implies violation” is not a valid Oracle rule in either set-valued discrete or floating semantics.

### U1 empirical measurement with no accuracy allowance

On PyTorch `2.13.0.dev20260609+cu126`, a fixed `7 x 257` float32 CUDA softmax input produced:

```text
eager vs compiled max       = 1.7881393432617188e-07
eager vs fp64 truth max     = 1.879416912098364e-07
compiled vs fp64 truth max  = 1.539405395378779e-07
max normalization residual  = 1.1920928955078125e-07 for both
```

These are valid measurements, and the candidate happened to be closer to the fp64 target under max error. However, the API supplies a formula but no quantitative float32 accuracy allowance, and no application margin was declared. Necessary invariants such as normalization are not sufficient. Consequently:

```text
truth-relative measurements: reported
numerical acceptance bit: undefined
verdict: UNINSTANTIATED
```

Choosing a tolerance from these values would invalidate the control.

### U2 preregistration defect

The frozen contract names “a declared equivalence boundary” but does not actually give the boundary, finite draw budget, simultaneous confidence construction or target law. Therefore U2 is not executable as a preregistered `INDETERMINATE` control. As currently written, its own contract is `UNINSTANTIATED`.

Any concrete U2 instantiation must be versioned and frozen before draws. It cannot be retroactively counted in v0.1 confirmation scoring.

A separately versioned manifest subsequently fixed target `p=(0.5,0.5)`, TV margin `0.01`, `n=100` and a 95% Hoeffding interval before draws. The compiled candidate produced counts 53/47 and interval `[0.3941898, 0.6658102]`, which crosses the acceptable interval `[0.49,0.51]`. It correctly returned `INDETERMINATE`. This validates the mechanics but does not repair the original v0.1 preregistration retrospectively.

### Zero implementation discrepancy with shared wrong behavior

TensorFlow `2.22.0-dev20260618` was evaluated at float32 `x=-1e-10` for eager SELU and `tf.function(..., jit_compile=False)` graph execution. Both returned exactly `0.0`, so their implementation-relative absolute delta was zero. The stable `expm1` mathematical result was `-1.7580993642326155e-10`.

```text
eager vs graph delta: 0
eager truth error:     1.7580993642326155e-10
graph truth error:     1.7580993642326155e-10
truth-relative bit:    fail=1 for both covered implementations
```

This is the required shared-wrong control: no eager/candidate discrepancy statistic—bias, variance or raw delta—can detect it. Independent semantics adds information unavailable to differential testing.

For the same operator and input, XLA returned the stable nonzero result while the non-XLA graph matched eager's zero. Relative to eager, the candidate ordering is therefore inverted:

```text
non-XLA graph: eager delta = 0                    -> truth REJECT
XLA:           eager delta ≈ 1.758099e-10         -> covered truth ACCEPT
```

This is not a cross-operator scale artifact. Any rule monotone in eager discrepancy ranks the wrong candidate ahead of the conforming candidate on the identical SELU contract instance.

## 10. Hard-quadrant mechanics audit

| Quadrant/control | Evidence | Contract result | Coverage status |
|---|---|---|---|
| small + conforming | H3 fixed backward is exact even though forward delta is nonzero | covered `ACCEPT` | real candidate, endpoint-specific |
| large + conforming | frozen CUDA sum: eager `2`, compiled `0`, default allclose false, both inside ±`8.000001` envelope | both `fail=0` | real candidate confirmation; N3/N4 synthetic controls also pass |
| small/zero discrepancy + violating | SELU eager vs non-XLA graph delta `0`, both wrong vs stable truth | `REJECT` both | real same-operator/input control |
| large + violating | H1 cast elision and H2 expanded indexing | `REJECT` | real candidates |
| invalid apparent pass | H4 intended bf16 path skipped | no bit / `INAPPLICABLE` | real applicability control |

The mechanics quadrants are now represented, including a separately frozen real-candidate large-conforming pair and a separately frozen stochastic indeterminate result. The original confirmation success gate is still not met because those supplements cannot retroactively fill missing v0.1 manifest fields, matched fixed/non-trigger rows remain incomplete, and no unified immutable cross-case baseline scoring has been run.

## 11. New theory correction from evidence

The two cases reinforce contract precedence:

1. exact index/option/state obligations must be evaluated before a broad numerical envelope;
2. forward, backward and state-transition contracts are separate—even an excellent forward match does not license a backward pass;
3. issue/fix metadata cannot replace execution on the actual candidate version;
4. historical repro scripts may contain static assertions/prose that invert meaning on a fixed version; semantic outputs, not script exit status alone, determine the contract witness.
5. fixing a silent state bug can expose an explicitly unsupported capability; these require two contract fields rather than one pass/fail label.
6. relative implementation shift has no inherent error direction: a compiled candidate can differ because it is more accurate than eager.
7. output equality is not negative evidence when the intended candidate realization is skipped or unsupported.
8. a validation plan that names a case class but omits its acceptable boundary and uncertainty rule is itself uninstantiated; “frozen” prose does not substitute for frozen estimands.
9. an eager/candidate pair can agree exactly and still share wrong behavior; the truth/specification axis is not reducible to discrepancy decomposition.
10. even within one operator and one input, smaller eager discrepancy can rank a wrong candidate ahead of a conforming one.

## 12. Remaining gate

Obtain an applicable H4 environment or retain it as missing coverage, and issue a unified versioned confirmation manifest before applying an aggregate success gate. Validation still needs H1/H2 matched fixed negatives, dedicated N2 strata and cross-case baseline scoring. Real-candidate large-conforming and fully instantiated stochastic-indeterminate mechanics now pass in separately frozen supplements, but they do not close compiler-family coverage.
