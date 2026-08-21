# Bias-property search: completion audit

Status: `COMPLETE_WITH_EXPLICIT_SCOPE` (2026-08-22)

This audit closes the development → short-screen → freeze → unseen-validation
phase.  It does **not** claim a universal all-operator property.

## Requirement audit

| requirement | evidence | disposition |
|---|---|---|
| Profile four development properties | `results/property/bias_property_search/development_property_profile.json` | Complete on 13 development records (8 known/partial rows and 5 centered controls). |
| Require centered controls before promotion | `development_property_separation_audit_v1.json` | Applied fail-closed. Missing control measurements are not negative labels. |
| Freeze before held-out reveal | `property_freeze_v1.json` | Frozen before the Gemma validation. |
| Shared low-cost ordered screen | `src/kernel_analyzer/short_persistence.py` | 256-dimensional signed CountSketch, ordered prefix growth, short-lag correlation and 2,000-draw sign-flip null. |
| Escalate only risk candidates | `scripts/select_short_screen_escalations.py` | `RISK_CANDIDATE` escalates; null-like and malformed rows abstain and remain in the denominator. |
| Validate an unseen implementation | `heldout_validation_v3_gemma_disjoint.json` | One new implementation class, with disjoint formation and consequence state banks. |
| Verify implementation and tests | targeted property/Oracle test set | 38 tests passed. |

## Frozen property decisions

1. **Conditional source asymmetry** is retained as a formation prior.  It is
   admissible only when the source boundary and its centered control are both
   measured under the frozen rule.
2. **Source–transport coupling** remains a case-level mechanism.  It is not
   admitted to a cross-case predictor because no centered control has a valid
   marginal-preserving pairing intervention.
3. **Transport concentration** is a supporting ranking feature only.  Its
   development ranges overlap centered controls, so it cannot issue a verdict.
4. **Carrier stability** is retained as a consequence/persistence screen, not
   as a formation label.  A null-like short screen means “do not escalate”; it
   never means SAFE.  Exact trajectories remain the confirmation boundary.

## Unseen implementation result

Gemma-4 E2B is the current authoritative disjoint validation record.  The
source prediction was frozen before the consequence trajectory:

```text
source amplification  = 0.99834
odd/even cosine        = 0.00306
source verdict         = NO_SOURCE_PERSISTENCE_UNDER_PROTOCOL
```

On the independent consequence bank, the local path was null-like
(`1.0260` versus null upper `1.0337`), while feedback and actual paths were
risk candidates (`2.6583`/`2.6457` versus null upper about `1.63`).  This is a
feedback-sustained, out-of-domain record.  It is not a new Flash-style
source-persistence case.

## Claim boundary

The completed phase supports a reproducible workflow for prioritising exact
trajectory analysis and preserves unresolved rows in the denominator.  It
does not establish universal accuracy: source–transport and carrier-control
gaps remain explicitly scoped, and one new implementation class is not a
cross-implementation generalisation proof.
