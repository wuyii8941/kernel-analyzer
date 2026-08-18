# Next

1. Close the remaining Qwen seq64 layer-0 `v_proj` T1-positive endpoint with
   invocation-local F+B proof, causal repair, and paired trajectory. Do not
   infer its result from the completed seq128 invocation.
2. Complete strict analytic F+B proof for all 186,807 origin-bound
   units, generating concrete saved-tensor/cotangent/backward-program witnesses.
3. Keep all 83 non-Triton candidates in the frozen denominator. Their live
   full-coordinate follow-up is complete: all six DeepSeek candidates were
   rejected and no candidate remains pending live measurement.
4. Regenerate Triton references with independently typed FP32 programs and add
   same-dtype eager-vs-candidate optimization contrasts. The invalid ABI replay
   cannot be repaired by relabeling its existing values.

Property induction remains blocked: six strict Flash-style cases exist, but
two are semantic-region cases rather than single-kernel labels, only two
concrete mechanisms pass cross-state confirmation, and no held-out cross-
operator property has been established.
