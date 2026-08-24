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

## 历史审计计数

- 历史 strict registry：**6** 条记录。
- Bias Formation 历史 roster：**8** 条记录。
- 该 roster 中历史 ordered-trajectory persistence：**7/8**。
- 该 roster 中 matched formation evidence：**6/8**。
- 同一 contrast 完成 formation 与 persistence 的历史链：**4/8**。

这些数字描述不同的旧协议，不能当成当前统一 AdamW 下的正例数。

## 当前统一 AdamW 计数

- 回溯评估：**15** 行。
- 确认的直接持续正例：**2** 行，Liger 和 Phi `lm_head dX`。
- Qwen `lm_head dX`：AdamW 下直接更新抵消。
- `0543`：总体校正后保持未决。
- 新 Gemma 实现检查：**3** 行，0 个直接持续正例、1 个状态反馈对照、2 个不适用。
- 通用跨算子 property：**0**。

The machine-readable sources are
`results/coverage/existing_case_reaudit.json`, `case.md`, and
`results/property/bias_property_search/completion_audit_v1.json`.
