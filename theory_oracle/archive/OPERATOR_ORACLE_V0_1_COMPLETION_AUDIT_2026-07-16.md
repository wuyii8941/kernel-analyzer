# Operator Oracle v0.1 Completion Audit — 2026-07-16

## 1. Audit question

Has the project moved from “a list of discrepancy measurements” to a real, understandable and technically defensible testing Oracle?

## 2. Requirement-by-requirement evidence

| Requirement | Authoritative definition | Status | Limitation |
|---|---|---|---|
| explicit subject | operator instance/signature in `OPERATOR_ORACLE_V0_1_DEFINITION_2026-07-16.md` §2 | defined | candidate realization must still be proven per contract |
| expected relation | exact/numerical/distributional/transition registry, §3 | defined | relation must be selected per operator |
| distinction between truth error and relative discrepancy | definition §3.2 and instantiation §1--2 | defined | many existing data use baseline only |
| input-conditioned permitted behavior | semantic envelope in definition §5 and instantiation §3.1 | defined | framework semantics may be too weak to instantiate sharply |
| acceptable behavior | acceptable set, definition §5 | parameterized | numerical bounds remain operator-specific |
| bias/heterogeneity/runtime roles | definition §4 and instantiation §9 | defined | estimator choice remains protocol-specific |
| sampling uncertainty | simultaneous confidence-set verdict, definition §6--7 | defined | concrete confidence/power convention required |
| explicit finite verdict | definition §7 | defined | no retrospective verdict for data lacking a contract |
| operational 0/1 process | `OPERATOR_ORACLE_0_1_DECISION_PROCESS_V0_1_2026-07-16.md` | defined with real result cards | bit is deliberately undefined when contract/identity/evidence gates fail |
| exact/discrete usable core | instantiation §3--4 | directly instantiable | only where governing semantics are actually specified |
| numerical contract | instantiation §5 | decision form defined | needs truth geometry and independent error budget |
| stochastic contract | instantiation §6 | decision form defined | needs target law and acceptable distance |
| state-transition contract | instantiation §7 | decision form defined | must enumerate complete claimed state |
| semantic impact | definition §8--9 and instantiation §8 | separated | requires external impact tolerance and valid intervention |
| operator-family claim | definition §10 and instantiation §10--11 | defined | external validity to unseen signatures remains empirical |
| input population | instantiation §11 | defined choices | project must select nominal/stress populations |
| randomness/configuration | instantiation §12 | defined roles | concrete protocol must freeze each source |
| operator/region/kernel boundary | instantiation §13 | fail-closed rule defined | fused operators may remain unidentifiable |
| candidate realization/causal identity | `OPERATOR_REALIZATION_IDENTITY_CONTRACT_V0_1_2026-07-16.md` | R0--R4 and I0--I3 claim levels defined | concrete evidence must establish the claimed level |
| common-sense counterexamples | `OPERATOR_ORACLE_V0_1_SANITY_AUDIT_2026-07-16.md` | passed conceptually | must be rerun for every instantiated contract |
| initial cross-semantic contracts | `INITIAL_OPERATOR_CONTRACT_SUITE_V0_1_2026-07-16.md` | archetypes defined | governing-framework mapping and empirical witness validation remain |
| real PyTorch contract probes | `PYTORCH_CONTRACT_RECORDS_V0_1_2026-07-16.md` | P1--P4 semantically instantiated | numerical-family breadth and compiled realization evidence remain |
| numerical family geometry | `NUMERICAL_OPERATOR_CONTRACT_CATALOG_V0_1_2026-07-16.md` | main family contract forms defined | operator-specific quantitative envelopes remain uneven |
| PyTorch numerical records | `PYTORCH_NUMERICAL_CONTRACT_RECORDS_V0_1_2026-07-16.md` | sum/mm envelopes and softmax/LayerNorm refusal boundaries defined | empirical candidate realization and wider family records remain |
| incremental-value validation | `OPERATOR_ORACLE_VALIDATION_STANDARD_V0_1_2026-07-16.md` | controls, baselines, metrics and kill criteria defined | partial held-out evidence exists; unified success gate not yet run |
| first broken/fixed witnesses | `FIRST_EMPIRICAL_CONTRACT_WITNESSES_2026-07-16.md` | two confirmed real REJECT/matched-negative pairs executed | not held-out, no hard-quadrant coverage or general detector comparison |
| precommitted-schema partial confirmation | `HELDOUT_OPERATOR_ORACLE_PARTIAL_FINDINGS_2026-07-16.md` | H1/H2/H3/H5/H6 include cast, indexing, backward, metadata and eager-wrong truth cases; H3 beats forward raw-delta ordering; H4 correctly refuses an inapplicable path; shared-wrong evidence defeats delta=0 pass | matched fixed/non-trigger rows, H4 applicable environment and unified scoring remain |
| real large-but-conforming confirmation | sum manifest/findings | frozen CUDA candidate differs by `2`, fails default allclose, but both outputs are inside the precomputed ±`8.000001` envelope | one reduction signature; no prevalence or cross-family claim |
| stochastic abstention confirmation | multinomial U2 manifest/findings | frozen target/margin/budget/CI returns `INDETERMINATE` on 53/47 draws | mechanics control only; no distributional correctness conclusion |
| preregistration completeness | `HELDOUT_VALIDATION_PREREGISTRATION_AUDIT_2026-07-16.md` | case families and anti-post-hoc rules were frozen | exact U2 margin/budget/CI, N4 envelope and hard-quadrant inputs were not; v0.1 is a schema rather than a full manifest |

## 3. What “defined” now means

The Oracle is no longer the statement “report bias and variance.” It is the following decision procedure:

```text
1. identify the semantic subject and candidate realization;
2. select an externally meaningful expected relation;
3. define its semantic loss and acceptable set;
4. collect valid evidence under fixed input/randomness contracts;
5. construct simultaneous uncertainty for every required constraint;
6. return UNINSTANTIATED, INVALID, ACCEPT, REJECT or INDETERMINATE;
7. label the result as correctness, compatibility or impact.
```

This procedure can give a determinate result for exact specified obligations today. For approximate floating behavior, a determinate acceptance verdict remains impossible until the tolerance source is justified.

## 4. Claims that remain prohibited

- “compiled is correct because it is close to eager”;
- “no significant difference means equivalent”;
- “floating point is variance”;
- “mean relative bias summarizes operator correctness”;
- “a region repair identifies a constituent operator”;
- “sampled pass proves a universal family property”;
- “stress-test frequency estimates deployment prevalence”;
- “the tolerance can be selected from the same candidate results it judges.”

## 5. Completion judgment

The **zero-to-one Oracle core is now achieved** in the scoped sense required by this audit:

1. a bit has a fixed meaning—membership violation of an independently sourced, input-conditioned semantic envelope;
2. missing contract, invalid identity and insufficient evidence cannot be silently converted to pass;
3. exact, numerical, distributional and transition relations have explicit decision forms;
4. real evidence exercises `REJECT`, covered `ACCEPT`, `UNINSTANTIATED`, `INAPPLICABLE/INVALID` and `INDETERMINATE`;
5. a real frozen sum case shows large raw delta/default-allclose failure with conformance, while same-input SELU evidence shows zero eager delta with shared wrong behavior;
6. operator/region/kernel identity and correctness/compatibility/impact scopes are fail-closed;
7. bias, input heterogeneity, runtime variability and sampling uncertainty are explanatory fields rather than circular correctness definitions.

What remains is a different and larger claim: a **validated general-purpose framework-specific contract suite**. That requires broader operator-specific envelopes, a nominal/stress population policy, matched fixed/non-trigger rows, an immutable unified held-out manifest, aggregate baseline scoring and cross-family external validity.

Therefore the defensible artifact name is:

> **Operator Oracle v0.1 core with validated decision mechanics and partial framework contracts—not a universal PyTorch/compiler correctness suite.**
