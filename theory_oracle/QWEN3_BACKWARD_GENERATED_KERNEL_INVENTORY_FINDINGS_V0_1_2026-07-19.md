# Qwen3 backward generated-kernel inventory findings v0.1

The audited generated AOT backward trace contains 39 generated Triton families
with 1,126 static call sites, plus 563 external `mm` and 168 external `bmm` call
sites. Thus the static backward generated-treatment denominator has 41 family
names and 1,857 generated/external call sites. It reconciles with the previously
audited 9,471 static ATen/prims nodes across 40 target types.

This result corrects the intuition that full backward coverage means testing
only 40 operator names. The 40 names are a source/IR vocabulary; the 41
generated family names are candidate intervention strata; and the 1,857 call
sites are only a conservative static denominator. Repeated family names may
reduce that denominator only after semantic role, shape/fusion context and
cross-state transport are validated.

The source inventory compiled but did not execute backward, so no call site is
yet confirmed live for the target loss. All current backward entries remain
descriptive only. The census gives no repair, injection, causal-attribution,
population or correctness credit.
