# Denominator and verdict layers

Every actual invocation stays in the denominator.  Repeated states do not
create extra operator samples, and unresolved or invalid records are not
dropped.

1. **Execution census** — every eager forward/backward invocation.
2. **F+B origin unit** — exact runtime origin binds a forward invocation to the
   backward program that consumes its saved tensors and cotangent.
3. **Analytic F+B proof** — the concrete forward map, saved values, output
   edge, shapes, dtypes, and executed VJP arithmetic are closed.
4. **Candidate binding** — generated/fused regions bind to exact F+B units
   without reducing either denominator.
5. **Numerical Oracle** — a valid precision or same-dtype comparison with
   exact boundary, ABI, repeats, and state provenance.
6. **Flash-style case** — T1 local difference, T2 causal repair and parameter
   reach, T3 complete coherent carrier, and T4 paired accumulation.
7. **Generalizable property** — independent-state evidence for a declared
   mechanism; it is not inferred from a T1 screen or a single trajectory.

The current full-coordinate denominator is 1,562 directional endpoints.  A
pending endpoint remains pending; it is never relabeled as a negative case.
