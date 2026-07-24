# Analytic Linear Training-Step Findings — 2026-07-17

## Verdict

The numerical correctness ledger is now instantiated and validated for one
complete, scoped training transition with an independent exact reference.

| Candidate mode | Identity valid | Exact structure | Numerical result | Repeat stability |
|---|---:|---:|---:|---:|
| correct compiled reduction | 32/32 | 32/32 `ACCEPT` | 32/32 `ACCEPT` | 32/32 |
| reverse-term compiled reduction | 32/32 | 32/32 `ACCEPT` | 32/32 `ACCEPT` | 32/32 |
| drop-last negative control | 32/32 | 32/32 `ACCEPT` | 31 `REJECT`, 1 `ACCEPT` | 32/32 |

All runs used a Tesla T4, CUDA 12.6 and PyTorch
`2.13.0.dev20260609+cu126`. Each mode compiled one tracked graph and recorded 65
runtime invocations: one discarded warmup plus 32 states times two repeats.

## Independent correctness relation

The accepted set was not derived from eager/candidate differences. Exact rational
arithmetic computed the real-valued prediction, loss, complete gradient and next SGD
parameter vector. A predeclared IEEE-754 forward-error envelope used `u=2^-24`,
operation counts and exact input scales.

The largest candidate error/bound ratio was about `0.139` in both permitted modes.
Thus the observed values remained comfortably inside, rather than defining, the
acceptance boundary.

## Legal reassociation control

The reverse-term candidate compiled a different 10-node graph from the normal
9-node graph and was accepted on all states. This demonstrates that the numerical
Oracle is not bitwise-equality disguised as correctness. The contract admits
ordinary reduction reassociation within the independently fixed error envelope.

## Wrong-program control

The drop-last candidate also compiled successfully and preserved gradient shape,
dtype and optimizer structure. It nevertheless violated all four numerical fields
on 31 states. Its maximum error/bound ratio exceeded `5,000`.

State 6 accepted because its omitted last product was exactly zero:

```text
w[-1] = 0.1005859375
x[-1] = 0
w[-1] * x[-1] = 0
```

For that state, the mutated program has the same observable mathematical transition.
This is expected input-conditioned Oracle behavior, not a false negative. A wrong
implementation need not fail on every state.

## What this proves

- `ACCEPT` and `REJECT` can be based on an external mathematical relation rather
  than eager-as-truth;
- legal floating reassociation can be accepted;
- a candidate can pass execution identity and exact structure while failing the
  numerical transition;
- verdicts are state-conditioned;
- the numerical ledger covers prediction, loss, gradient and next parameter state,
  not one selected scalar.

## What this does not prove

- The same envelope applies to transformer attention, softmax, normalization,
  AdamW or mixed precision.
- Qwen/BERT numerical transition correctness is resolved.
- Inductor has a natural bug; the rejecting mode is an injected negative control.
- The finite deterministic state bank defines a deployment population.
- No semantic decision boundary or long-run training consequence is studied here.

## Evidence

- contract: `ANALYTIC_LINEAR_STEP_CONTRACT_V0_1_2026-07-17.md`;
- executor: `analytic_linear_step_oracle.py`;
- correct results:
  `results/training_step_oracle/analytic_linear_correct_v0_1`;
- legal reverse results:
  `results/training_step_oracle/analytic_linear_reverse_v0_1`;
- negative-control results:
  `results/training_step_oracle/analytic_linear_drop_last_v0_1`.

