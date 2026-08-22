# Gate and counting history

This file records why reported counts changed.  It distinguishes new evidence,
methodology corrections, and different denominators so that a count cannot be
mistaken for post-hoc threshold fitting.

| date | commit | gate/count at that point | change and reason |
|---|---|---|---|
| 2026-08-03 | `5e45dd8` | 2 project cases plus the literature FlashAttention anchor | Initial case document required a complete F+B relation, a real parameter/weight carrier, and a causal intervention. Project cases were Qwen `lm_head dX` and Liger fused CE. |
| 2026-08-03 | `3b84992` | still 2 project cases | Added Liger’s paired 32-step accumulator-repair trajectory. This strengthened an existing case; it did not add one. |
| 2026-08-18 | `7ee6811` | 6 strict Flash-style project cases; 2 cross-state concrete-mechanism passes | Introduced the explicit T1–T4 Flash-style track and separated it from the independent-state `GENERALIZABLE_BIAS` track. The evidence re-audit added Phi, Mamba, layer-23, and saved-P to the strict registry. Qwen `lm_head` remained strict despite failing the separate cross-state gate. This was new/rejoined evidence plus a transparent dual-track correction, not a relaxed numerical threshold. |
| 2026-08-20 | `ec308a6` | strict count unchanged | Corrected the formation interpretation: fixed absolute direction across unrelated states is stronger than trajectory-local bias. Conditional and trajectory-aware formation were introduced without promoting rows missing repair or trajectory evidence. |
| 2026-08-20 | `7d969a8` | 8-record Bias Formation roster | Created an eight-record audit denominator for mechanism analysis. “8” meant paired F+B trajectory records, not eight strict or source-persistent positives. |
| 2026-08-20 | `f3c0cd3`, `d9dc839` | matched formation evidence rose from 4 to 6 within the eight-record roster | Added antithetic/source-aligned interventions for Qwen v-proj, saved-P/SiLU response geometry, and Mamba’s bounded conditional result. The strict six-case registry did not change. |
| 2026-08-20 | `3ca5666` | strict count unchanged | DeepSeek layer-35 `dV` passed a conditional moving-frame formation test. It lacked live-weight persistence and remained `PARTIAL_CONDITIONAL_BIAS_NO_FLASH_STYLE_PERSISTENCE`. |
| 2026-08-20 | `cd5e509` | 8/8 paired separation, 6/8 matched formation, 4/8 same-contrast full chain | Separated mere trajectory distance growth from directional persistence and same-contrast closure. This exposed denominator differences instead of collapsing them into one count. |
| 2026-08-20 | `b2c1af2` | 7/8 broad directional persistence; 6 strict Flash-style cases | Qwen128 v-proj was resolved as source-formation-positive but trajectory-diffusive under its aligned rounding contrast. SiLU was resolved as feedback-sustained rather than persistent-local. The broad roster count became 7/8; the strict registry stayed six. |
| 2026-08-22 | `301e746`, `79148eb` | four property candidates profiled, then frozen before new validation | Source asymmetry was admitted as a conditional formation prior; source–transport stayed case-level; concentration became supporting-only; carrier stability became a short consequence screen. Missing controls remained missing. |
| 2026-08-22 | `0b898db` | short-screen protocol v2 | Replaced the v1 consecutive-late-prefix rule with final-versus-four-step-warmup growth. Previous observations were not relabeled under v2. |
| 2026-08-22 | `16afc8d` | short-screen protocol v3 | Increased CountSketch dimension from 64 to 256 after an engineering power audit. The Ministral diagnostic was not promoted to held-out evidence; v3 was frozen for subsequent validation. |
| 2026-08-22 | `36958d5` | one disjoint `NEW_IMPL` v3 validation | Gemma formation and consequence were moved to separate state banks. Local source was null-like; feedback/actual paths were risk candidates. This added a feedback-sustained validation record, not a strict Flash-style source case. |
| 2026-08-22 | `75a3fba` | property-search phase closed with explicit scope | Recorded that the workflow is complete as a frozen, fail-closed prioritization system while universal-property, control, and cross-implementation limitations remain explicit. |

## Current authoritative counts

- Strict project Flash-style cases: **6**.
- Bias Formation roster denominator: **8**.
- Ordered-trajectory directional persistence in that roster: **7/8**.
- Matched formation mechanisms in that roster: **6/8**.
- Same-contrast formation-plus-persistence chains in that roster: **4/8**.
- Cross-state concrete-mechanism passes in the strict re-audit: **2**.
- Universal cross-operator properties: **0**.
- New-implementation source-persistence positives after freeze: **0**.

The machine-readable sources are
`results/coverage/existing_case_reaudit.json`, `case.md`, and
`results/property/bias_property_search/completion_audit_v1.json`.

