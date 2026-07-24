# Phase A2 Near-Boundary Concentration Audit

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- iteration-2 only: PASS (2)
- zero-advantage rows excluded: PASS
- case-to-prompt alignment: PASS

## Delta Self Control
Not applicable to the Phase 0 margin distribution; Phase A1 audits path self consistency.

## External Validity
These concentration statistics describe the T4 FP16 GRPO run. They do not establish how BF16 kernels redistribute near-boundary decisions; BF16-hardware replication remains required.

## Concentration
| dimension | groups | groups_with_near_boundary | near_boundary_tokens | groups_for_80_percent | fraction_groups_for_80_percent | max_group_share |
| --- | --- | --- | --- | --- | --- | --- |
| prompt | 60 | 53 | 165 | 32 | 0.5333333333333333 | 0.05454545454545454 |
| response | 312 | 123 | 165 | 90 | 0.28846153846153844 | 0.03636363636363636 |

## Top Prompts
| prompt | near_boundary_tokens |
| --- | --- |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 53 items, receives 5, then gives away 2. How many remain? | 9 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 51 items, receives 3, then gives away 5. How many remain? | 8 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 25 items, receives 10, then gives away 4. How many remain? | 7 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 7 items, receives 3, then gives away 1. How many remain? | 6 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 13 items, receives 9, then gives away 2. How many remain? | 5 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 9 items, receives 5, then gives away 3. How many remain? | 5 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 26 items, receives 11, then gives away 5. How many remain? | 5 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 23 items, receives 8, then gives away 2. How many remain? | 5 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 24 items, receives 9, then gives away 3. How many remain? | 4 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 54 items, receives 6, then gives away 3. How many remain? | 4 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 42 items, receives 5, then gives away 1. How many remain? | 4 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 35 items, receives 9, then gives away 4. How many remain? | 4 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 49 items, receives 12, then gives away 3. How many remain? | 4 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 18 items, receives 3, then gives away 2. How many remain? | 4 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 20 items, receives 5, then gives away 4. How many remain? | 4 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 67 items, receives 8, then gives away 1. How many remain? | 4 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 56 items, receives 8, then gives away 5. How many remain? | 4 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 32 items, receives 6, then gives away 1. How many remain? | 4 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 64 items, receives 5, then gives away 3. How many remain? | 4 |
| Solve the problem. Show concise reasoning and end with the numeric answer.

A box starts with 19 items, receives 4, then gives away 3. How many remain? | 4 |

## Conclusion
Near-boundary mass is not confined to only two or three prompts.
