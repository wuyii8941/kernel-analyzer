# Phase 0 Margin Histogram

## Confound Checklist
- fixed_response_tokens: N/A for margin dump
- real_old_logp_present: PASS
- advantage_sign_present: PASS
- deterministic_env_recorded: external training dump must provide metadata
- late_minibatch_gate_used: PASS

## Delta Self Control
N/A in Phase 0; delta_self is checked in Phase 1.

## Summary
GO: late-minibatch near-boundary mass is sufficient for clipping fork scan.

## Overall
| group | n | p0.1 | p1 | p5 | p50 | P(margin<0.0001) | P(margin<0.001) | P(margin<0.01) | P(margin<0.05) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 2 | 5.551115123125783e-20 | 5.551115123125783e-19 | 2.7755575615628915e-18 | 2.7755575615628914e-17 | 1.0 | 1.0 | 1.0 | 1.0 |

## Late Minibatches
| group | n | p0.1 | p1 | p5 | p50 | P(margin<0.0001) | P(margin<0.001) | P(margin<0.01) | P(margin<0.05) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| late_epoch=0 | 2 | 5.551115123125783e-20 | 5.551115123125783e-19 | 2.7755575615628915e-18 | 2.7755575615628914e-17 | 1.0 | 1.0 | 1.0 | 1.0 |

## By Minibatch
| group | n | p0.1 | p1 | p5 | p50 | P(margin<0.0001) | P(margin<0.001) | P(margin<0.01) | P(margin<0.05) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epoch=0,minibatch=0 | 2 | 5.551115123125783e-20 | 5.551115123125783e-19 | 2.7755575615628915e-18 | 2.7755575615628914e-17 | 1.0 | 1.0 | 1.0 | 1.0 |
