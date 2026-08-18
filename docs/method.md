# Method

For each concrete invocation, the scientific unit is one forward and its actual
backward:

\[
U_i=(F_i,x_i,y_i,q_i,J_{F_i}(x_i)^Tq_i,B_i).
\]

`seq_nr` establishes runtime origin only. Analytic closure additionally proves
that operands, saved tensors, upstream cotangent, output edge, shapes, dtypes,
non-tensor arguments and the arithmetic of the executed backward program
implement the derived VJP. Formula registration, autograd and finite differences
are checks, not substitutes for that proof.

Numerical contrasts are kept distinct:

\[
e_{precision}=R_{low}-R_{32},\qquad
e_{optimization}=C_{low}-R_{low}.
\]

The first can support a precision-caused case; the second can attribute a local
difference to an optimized implementation. Total error is explanatory only and
cannot assign a cause without those arms. Every certificate records
`PRECISION`, `OPTIMIZATION`, `MIXED`, or `OTHER`. All arms require exact semantic
boundaries and program identities. A directional bias case then follows the
paper-style ordered gates:

1. T1: nonzero, finite, repeat-stable attributable local error with a predeclared
   complete local endpoint;
2. T2: exact local reference replacement, exact sham and parameter reach;
3. T3: independent states, complete fixed parameter-gradient coordinates and
   family-wise-controlled raw/relative/factor coherence hypotheses;
4. T4: paired same-weight updates with directional accumulation and live-weight
   divergence, used only to certify a training consequence.

Missing repair or trajectory blocks Flash-style promotion. Failure of the
separate cross-state gate does not revoke a complete trajectory-local case.
The cross-operator property target is T3 F+B bias; T4 is never its label or
predictor. Precision is the first searched mechanism, not the only recorded cause;
layout, fusion/materialization,
reduction order, ABI, alias/mutation and other mechanisms retain explicit fields.
