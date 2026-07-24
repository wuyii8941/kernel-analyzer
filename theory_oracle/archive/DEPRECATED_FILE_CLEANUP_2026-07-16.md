# Deprecated File Cleanup — 2026-07-16

## Policy

This cleanup removes documents whose active conclusions were explicitly replaced by Operator Oracle v0.1 and Training-Step Oracle v0.1. It does not delete raw/compact evidence, confirmed-bug mappings, empirical findings, framework contract records or reproducibility scripts.

## Removed top-level file

| Removed | Reason | Authoritative replacement |
|---|---|---|
| `CODEX_PLAN.md` | fork-centered execution story and novelty claims no longer describe the project | `theory_oracle/README.md`, Operator Oracle v0.1, Training-Step active plan |

## Removed theory plans/status documents

| Removed | Reason | Replacement |
|---|---|---|
| `DISCREPANCY_ORACLE_PLAN.md` | old discrepancy/calibration plan; bias/variance profile was not a correctness Oracle | Operator Oracle definition and contract instantiation |
| `GRANULAR_DISCREPANCY_ORACLE_PLAN_2026-07-16.md` | abandoned coordinate/site-primary branch | operator instance/signature primary scale in Operator Oracle v0.1 |
| `ORACLE_SCALE_SELECTION_PLAN_2026-07-16.md` | superseded scale-selection exercise | Operator Oracle definition and realization identity contract |
| `ORACLE_SCALE_SELECTION_FINDINGS_2026-07-16.md` | historical inference-only scale conclusion contradicted the later operator study | completion audit and evidence reclassification |
| `ORACLE_V2_SCALE_DEFINITION_2026-07-16.md` | incorrectly relegated operator analysis to attribution only | Operator Oracle v0.1 and Training-Step Oracle v0.1 |
| `OPERATOR_SCALE_STATUS_2026-07-16.md` | pre-definition status snapshot | Operator Oracle completion audit |
| `ORACLE_DEFINITION_GAP_AUDIT_2026-07-16.md` | gap list was resolved into the normative contract | Operator Oracle definition/0–1 process |
| `GENERAL_ORACLE_FRAMEWORK.md` | semantic-impact-first framework was superseded by contract conformance as the primary correctness object | Operator Oracle definition and Training-Step definition |
| `EXPERIMENT_FRAMEWORK.md` | old discrepancy→boundary→transition progression no longer gives the active gates | Training-Step active plan and validation standard |
| `ORACLE_DISCOVERY_PROTOCOL.md` | partial relational semantic-impact Oracle is no longer the main Oracle definition | Operator Oracle validation standard |
| `PLAN_DEVIATIONS_2026-07-15.md` | deviations were already merged into later definitions; retaining it created dead references to removed plans | current normative documents and this cleanup manifest |

## Retained historical/evidence material

Retained on purpose:

- `DISCREPANCY_ORACLE_CORE.md` as the measurement/decomposition module;
- theoretical audit, literature survey and falsifiable working hypotheses;
- confirmed-bug mappings and broken/fixed findings;
- matched-state inference, transition, sampling and attribution evidence;
- manifests/findings for real large-but-conforming and stochastic-indeterminate controls;
- scripts needed to reproduce retained evidence;
- original reports/results outside `theory_oracle`, even when they no longer define the main Oracle.

The retained files may describe historical experiments, but they no longer compete with the normative reading order.

Disposable `theory_oracle/__pycache__/` bytecode was also removed after verification; source scripts remain.
