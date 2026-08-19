# Bias property search: current scientific result

## Candidate explanation

The strongest non-tautological candidate is **effective antithetic symmetry**.
For local implementation residuals `epsilon_j` and their real F+B/optimizer
influence `A_j`, the relevant event is `w_j=A_j epsilon_j`.  Harmless variance
cancels when the conditional `w` ensemble is closed under sign reversal.  Bias
forms when event/pairing asymmetry or a non-odd downstream response breaks that
symmetry.

This yields the falsifiable bias budget:

```text
Delta u = A_bar(epsilon) epsilon

E[A_bar epsilon | c]
  = E[A_bar|c] E[epsilon|c]       # source asymmetry
  + Cov_c(A_bar, epsilon).        # transport/influence coupling
```

`A_bar` is the exact path-averaged downstream derivative, so nonlinear and
optimizer effects are included rather than appended as unconstrained stages.

Equivalently, splitting the conditional event density and the actual response
into symmetric/antisymmetric and even/odd parts gives the exact parity budget:

```text
E[F(epsilon)|c]
  = integral p_s(epsilon) F_e(epsilon)   # response rectification
  + integral p_a(epsilon) F_o(epsilon).  # event/pairing asymmetry
```

Thus bias has two irreducible channels under a predeclared semantic sign
operation: unmatched antithetic events, or a non-odd F+B/optimizer response.

## Evidence

- Liger: a same-real-semantics padding/rechunk orbit is directional with BF16
  accumulation (24/24 signs) and centered with FP32 accumulation (13/11).
- Phi: natural backward pairing is biased; a local-norm-preserving row-pairing
  shuffle is centered.  The exact analytic subfactor remains open, so the
  supported object is composite F+B influence.
- Layer-23 attention: restoring `S_bwd` removes a 27/32-sign direction through
  `G_q=S_bwd K`; this is consistent evidence, but not yet a marginal-preserving
  symmetry intervention.
- Qwen v-projection and Mamba input-projection decompositions provide supporting
  source observations: Qwen isolates a directional output-rounding term while
  its same-operand kernel term centers; Mamba has directional kernel and output-
  rounding terms.  Their source/trajectory contrasts are not yet fully matched.
- Qwen saved-P: head-pairing shuffle changes accumulated effective-update resultant by -13.73%. The resultant increased, so the tested head-specific pairing mechanism is rejected. The exact +delta_g/-delta_g Adam test has accumulated oddness ratio 0.6817 and mean per-step ratio 0.1998; its oddness resultant aligns 0.6871 with the natural update resultant. Only 1.87% of active coordinates cross gradient sign, yet they carry 81.49% of response-even energy under an unweighted step average. Energy weighting raises that concentration to 99.48%, and 99.51% of even-response energy occurs in the first two steps.

- Qwen3-VL SiLU: the independent exact +delta_g/-delta_g Adam test has accumulated oddness ratio 0.6956 and mean per-step ratio 0.0354; the response-even resultant aligns 0.6969 with the natural update resultant. Mean sign-crossing fraction is 0.33%, carrying 6.38% of response-even energy. Energy-weighted sign-crossing concentration is 99.87%, with 100.00% of even-response energy in the first two steps.

## What this rules out

Error magnitude, raw tensor signed mean, BF16 dtype, and a fixed global rank-1
carrier do not individually explain the observed split.  SEUP remains the
downstream persistence test.

## Current boundary

The evidence supports one exact formation map across source/schedule,
composite-transport, and optimizer-response mechanisms.  It is not yet a
universal predictor for unseen operators.  Saved-P rejects head-specific pairing, while exact antithetic-gradient experiments in saved-P and Qwen3-VL SiLU independently produce large accumulated non-odd Adam responses.  This repeats optimizer response rectification across two closed F+B cases; unseen-case prediction remains open.
