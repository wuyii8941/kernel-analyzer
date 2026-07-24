# Generic localization core freeze v0.1

Freeze date: 2026-07-23.  This freeze is required before external historical
validation.  Any change below requires a new version and cannot be silently
tuned on a held-out case.

## Frozen contract

The core consumes only an execution adapter with declared stage contract rows,
region IDs, a symptom predicate, provenance and evidence.  It:

1. reports a supported stage candidate rather than a presumed first-bad pass;
2. runs name-agnostic ddmin over supplied region IDs;
3. records every subset query;
4. emits a certificate with an empty manual-decision ledger by default;
5. never derives a claim level from numerical delta magnitude.

The adapter is allowed to know how to restore a subject's state and execute a
substitution.  It is not allowed to choose candidates or claim levels.  The
historical-evaluation protocol will bind adapter source and case manifest before
the issue/patch is revealed.

## Bound implementation hashes

| file | SHA-256 |
|---|---|
| `src/forkcert/localization.py` | `2ec30d6a81559ec058c4100188d6abee79d88dffa2835cea1b4b1831d3688849` |
| `src/forkcert/localization_runtime.py` | `129e692eab91f99436d375e0c7d631b334286b1e3f50c1d33222ed004d869885` |
| `tests/test_localization.py` | `076db3ea4e5c135e29dee982f7d12c0a7f78a605b58c6e9ae725c1550d540548` |

## Calibration evidence bound to this freeze

- Kernel plumbing: `results/calibration/kernel_plumbing_v0_2/seeded_fault/seeded_fault_record.json`.
- One-step training: `results/calibration/tiny_training_v0_1/training_calibration_record.json`.
- Focused tests: `13 passed` for `tests/test_localization.py` and
  `tests/test_operator_evidence.py` under the nightly Python environment.

## Limits retained deliberately

This freeze does not claim that ddmin identifies a unique causal producer, or
that a seeded calibration predicts historical-bug accuracy.  It only freezes
the mechanics required to score that question on hidden independent patches.

