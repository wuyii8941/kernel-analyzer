# Signed Transport Coherence

> Historical candidate, superseded for current bias-formation claims by
> `docs/effective_antithetic_symmetry.md`.  Signed transport coherence remains
> a useful special case of event/pairing asymmetry and a downstream carrier
> description; it is not the current general property.

## Result

The property target is one exact forward plus its actual backward.  It asks
whether implementation arithmetic creates a stable directional error in a
complete parameter-gradient carrier.  Optimizer and multi-step T4 behavior are
consequences and are not labels or predictors.

The current claim is:

`NO_REFERENCE_ONLY_PROPERTY_CLAIM_YET`.

The exhaustive F+B population contains 1,562 endpoints.  At the present audit
snapshot, 57 pass T3 coherent-carrier evidence, 588 are completed normal
references at T1--T3, and 917 remain unresolved.  All endpoints are retained;
no representative replaces endpoint analysis.

## Candidate property

For state (s) and arithmetic event (e), derive the signed local residual
from reference operands and the declared candidate schedule:

\[
\epsilon_{s,e}=Q_e(f_e(x_{s,e}))-f_e(x_{s,e}).
\]

Let (J_{s,e}) be the complete analytic reference F+B transport from that
event to fixed parameter-gradient coordinates.  Define

\[
v_s=\sum_e J_{s,e}\epsilon_{s,e},\qquad
A=\mathbb E_s\|v_s\|^2,\qquad
D=\|\mathbb E_s v_s\|^2,\qquad C=D/A.
\]

Signed Transport Coherence holds when the cross-state U-statistic for (v_s)
has a positive frozen lower bound and

\[
\sqrt D-\mathbb E_s\rho_s>\tau,
\]

where (ho_s) bounds nonlinear transport remainder and (	au) is a frozen
reference-only numerical margin.  Affine regions have (ho_s=0); nonlinear
regions without a valid bound abstain.

Flash Attention is the special case

\[
\epsilon_T=(\delta_{lp}-\delta_{hp})[T],\qquad
J_T=\alpha(PK)[T]^\top X[T].
\]

Thus the paper's biased coefficient times similar low-rank carrier is one
instance of signed source--transport alignment.  The proposed property does
not require every carrier to be low rank.

## Evidence boundary

The previous 41 T4-pass versus 15 T4-fail comparison is invalid for this
property target: all 56 already pass T3 and are positive F+B-bias evidence.
Observed T3 carrier geometry may describe a mechanism but cannot be a
candidate-value-blind predictor.

The property becomes a contribution only if event residuals and transports are
computed without candidate tensor values or historical verdicts, then predict
unseen endpoints and respond correctly to semantic-preserving arithmetic
interventions.  The identity \(\Delta g=\sum_eJ_e\epsilon_e\) alone is not a
novel result.

Machine-readable population and the all-endpoint factor queue are in
`results/property/hypothesis_matrix.json` and
`results/property/signed_transport_queue.json.gz`.
