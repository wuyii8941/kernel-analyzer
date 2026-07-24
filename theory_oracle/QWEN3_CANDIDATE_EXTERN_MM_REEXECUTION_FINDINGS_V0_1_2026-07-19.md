# Qwen3 external MM reexecution findings v0.1

The independent audit in
`results/operator_oracle/qwen3_candidate_extern_mm_reexecution_v0_1/audit.json`
passes the frozen-artifact, eager/candidate anchor, graph-family, live external
object identity, exact call accounting, operand shape/stride/dtype, repeat,
no-recompile and restoration gates.

All seven projection roles at layers 0, 14 and 27, plus the singleton LM head,
are exact null effects at the scorer endpoint when the candidate external
wrapper is re-executed through eager `torch.mm` on identical candidate
operands. This covers 22 predeclared representatives across the distinct
linear roles and positions.

This supports only a shared-path conclusion. Candidate and eager ATen may
dispatch to the same CUDA library and algorithm, so this is not an independent
arithmetic repair. It does not establish equality of eager and candidate
projection inputs, general MM correctness, cross-state transport or
sufficiency.
