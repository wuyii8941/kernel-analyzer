# Qwen3 candidate masked safe-softmax repair findings v0.1

The generic audit in
`results/operator_oracle/qwen3_candidate_masked_softmax_repair_v0_1/audit.json`
passes every artifact, anchor, graph, treatment-integrity, repeat,
no-recompilation, restoration and arithmetic gate.

All early/middle/late repairs have deterministic nonzero scorer effects.  None
reduces whole-eager distance at this state: calls 0, 14 and 27 increase L2
distance by about 4.04%, 0.45% and 0.12%.  The last-layer repair is nearly
orthogonal to the candidate-to-eager vector.

This is not evidence that eager safe-softmax is wrong.  It shows that replacing
one generated invocation with its eager semantic fragment inside an otherwise
compiled execution need not monotonically approach whole eager.  Upstream and
downstream discrepancies and nonlinear propagation make the repair contrast
intervention-dependent.

Coverage credit belongs to the fused mask-add plus safe-softmax invocation, not
to max reduction, sum reduction, exponent, division or all-masked-row handling
separately.
