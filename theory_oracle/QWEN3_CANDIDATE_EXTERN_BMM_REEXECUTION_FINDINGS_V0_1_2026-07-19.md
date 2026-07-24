# Qwen3 external BMM reexecution findings v0.1

The independent audit in
`results/operator_oracle/qwen3_candidate_extern_bmm_reexecution_v0_1/audit.json`
passes the frozen-artifact, eager/candidate anchor, graph-family, live external
object identity, exact call accounting, operand shape/stride/dtype, repeat,
no-recompile and restoration gates.

The query-key and probability-value calls at layers 0, 14 and 27 are all exact
null effects at the scorer endpoint when the candidate external wrapper is
re-executed through eager `torch.bmm` on the identical candidate operands.

This supports a shared-path conclusion, not a general BMM correctness or
upstream-equivalence conclusion. Candidate and eager ATen may dispatch to the
same CUDA library and algorithm; the experiment does not show that eager and
candidate supply identical q/k/probability/value inputs. It is one-state
reexecution evidence without an independent arithmetic reference, injection or
population transport.
