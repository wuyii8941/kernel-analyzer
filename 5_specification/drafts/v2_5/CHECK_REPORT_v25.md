| contract | clause | source identity | guarantee | examples | misreadings (rejected by) | status | conclusion supported |
|---|---|---|---|---|---|---|---|
| A1_LINEAR | A1.C1 | paper | exact | E1:established, E2:established, E3:established, E4:established | M1->E4, M4->E2 | established | meets the task definition |
| A1_LINEAR | A1.C2 | model_impl/paper | rigorous_enclosure | E7:established, E8:established | M5->E8 | established | meets the task definition |
| A1_LINEAR | A1.C3 | model_impl | exact | E5:established | M2->E5 | established | agrees with a reference program |
| A1_LINEAR | A1.C4 | paper | exact | E9:established | M6->E9 | established | meets the task definition |
| A1_LINEAR | A1.C5 | paper | exact | E6:established | M3->E6 | established | meets the task definition |
| B1_PAGED | B1.C1 | engine_ref | exact | E1:established | - | established | agrees with a reference program |
| B1_PAGED | B1.C2 | engine_ref/torch_doc | rigorous_enclosure | E2:established, E7:established | M1->E2/E3/E4/E8 | established | meets the task definition |
| B1_PAGED | B1.C3 | model_impl | rigorous_enclosure | E3:established | - | established | agrees with a reference program |
| B1_PAGED | B1.C4 | model_impl | rigorous_enclosure | E4:established | M2->E4/E8 | established | agrees with a reference program |
| B1_PAGED | B1.C5 | torch_doc | exact | E5:established | - | established | meets the task definition |
| B1_PAGED | B1.C6 | standard | rigorous_enclosure | E6:established, E7:established | M3->E6/E8, M4->E7/E8 | established | meets the task definition |
| B1_PAGED | B1.C7 | engine_ref | allowed_set | E8:established | M5->E8 | established | agrees with a reference program |
| C1_SPARSE | C1.C1 | paper | exact | E1:established, E4:established | M1->E4 | established | meets the task definition |
| C1_SPARSE | C1.C2 | engine_ref/paper | allowed_set | E1:established, E2:established, E3:established, E5:established | M2->E2/E5, M3->E2 | established | meets the task definition |
| C1_SPARSE | C1.C3 | paper | allowed_set | E6:established | - | established | meets the task definition |
| C1_SPARSE | C1.C4 | engine_ref | exact | E7:established | - | established | agrees with a reference program |
| F1_FORMATS | F1.C1 | standard | exact | E1:established, E2:established | M1->E1/E7, M2->E2/E4/E6/E7 | established | meets the task definition |
| F1_FORMATS | F1.C2 | standard | exact | E3:established, E4:established, E5:established, E6:established | M3->E3 | established | meets the task definition |
| F1_FORMATS | F1.C3 | standard | exact | E7:established, E8:established, E9:established | M4->E9 | established | meets the task definition |
| F1_FORMATS | F1.C4 | standard | exact | E10:established | - | established | meets the task definition |
| F2_MX | F2.C1 | standard | exact | E1:established, E2:established, E3:established | M1->E1/E2/E3, M3->E1 | established | meets the task definition |
| F2_MX | F2.C2 | standard | exact | E1:established, E2:established | M2->E1 | established | meets the task definition |
| F2_MX | F2.C3 | standard | allowed_set | E4:established | - | established | meets the task definition |
| F2_MX | F2.C4 | standard | exact | E3:established | - | established | meets the task definition |
| F3_NVFP4 | F3.C1 | standard | exact | E1:established, E3:established | M1->E3 | established | meets the task definition |
| F3_NVFP4 | F3.C2 | standard | exact | E1:established, E2:established | M2->E1/E2/E3/E4 | established | meets the task definition |
| F3_NVFP4 | F3.C3 | standard | exact | E1:established, E2:established | - | established | meets the task definition |
| F3_NVFP4 | F3.C4 | standard | exact | E4:established | M3->E4 | established | meets the task definition |
| F4_INTQ | F4.C1 | engine_ref/torch_doc | exact | E1:established, E2:established | M1->E2, M2->E1/E2 | established | meets the task definition |
| F4_INTQ | F4.C2 | engine_ref | exact | E3:established, E4:established | M3->E4 | established | agrees with a reference program |
| F4_INTQ | F4.C3 | engine_ref | allowed_set | E5:established | M4->E5 | established | agrees with a reference program |
| F5_PACK | F5.C1 | engine_ref | exact | E1:established | M2->E1/E2 | established | agrees with a reference program |
| F5_PACK | F5.C2 | engine_ref | exact | E2:established, E3:established | M1->E2/E3 | established | agrees with a reference program |
| F6_DELAYED | F6.C1 | standard | exact | E1:established | M1->E1/E2/E3/E4 | established | meets the task definition |
| F6_DELAYED | F6.C2 | standard | exact | E3:established, E4:established | - | established | meets the task definition |
| F6_DELAYED | F6.C3 | standard | exact | E1:established, E2:established | M2->E2 | established | meets the task definition |
| F7_FP8SCALE | F7.C1 | engine_ref/standard | exact | E1:established, E2:established | M2->E2 | established | meets the task definition |
| F7_FP8SCALE | F7.C2 | engine_ref | exact | E1:established, E3:established | M1->E1 | established | agrees with a reference program |
| F7_FP8SCALE | F7.C3 | standard | exact | E4:established | - | established | meets the task definition |
| H1_ROUTING | H1.C1 | model_impl | rigorous_enclosure | E2:established | M3->E2 | established | agrees with a reference program |
| H1_ROUTING | H1.C2 | engine_ref/paper | rigorous_enclosure | E3:established, E4:established | M1->E4, M2->E3 | established | meets the task definition |
| H1_ROUTING | H1.C3 | model_impl/paper | allowed_set | E1:established | - | established | meets the task definition |
| H2_ALIGN | H2.C1 | engine_ref | exact | E1:established, E2:established | M1->E1/E2, M3->E2 | established | agrees with a reference program |
| H2_ALIGN | H2.C2 | engine_ref | allowed_set | E1:established | - | established | agrees with a reference program |
| H2_ALIGN | H2.C3 | model_impl | rigorous_enclosure | E3:established | M2->E3 | established | agrees with a reference program |
| K14_COLL | K14.C1 | standard | exact | E1:established | M1->E1 | established | meets the task definition |
| K14_COLL | K14.C2 | standard | exact | E2:established | M2->E2 | established | meets the task definition |
| K14_COLL | K14.C3 | engine_ref/standard | exact | E3:established | - | established | meets the task definition |
| K14_COLL | K14.C4 | paper | rigorous_enclosure | E4:established | M3->E4 | established | meets the task definition |
| K14_COLL | K14.C5 | engine_ref | exact | E5:established | M4->E5 | established | agrees with a reference program |
| R1_RANDOM | R1.C1 | torch_doc | exact | E1:established | - | established | meets the task definition |
| R1_RANDOM | R1.C2 | torch_doc | exact | E2:established, E3:established | M1->E2/E3, M2->E3 | established | meets the task definition |
| R1_RANDOM | R1.C3 | torch_doc | allowed_set | E5:established, E6:established | M3->E5 | established | meets the task definition |
| R1_RANDOM | R1.C4 | paper | exact | E4:established | - | established | meets the task definition |

| contract | kind | survey class | dimensions defined / n.a. / open | readings | schema problems |
|---|---|---|---|---|---|
| A1_LINEAR | K7 | A | 6 / 5 / 1 | - | none |
| B1_PAGED | K5 | B | 4 / 6 / 2 | E3[inclusive]=established, E5[modulo]=established | none |
| C1_SPARSE | K6 | C | 4 / 6 / 2 | E6[max]=established, E7[closed]=established | none |
| F1_FORMATS | K11 | F | 3 / 9 / 0 | E10[before_rounding]=established | none |
| F2_MX | K11 | F | 3 / 8 / 1 | E1[ceil_pow2]=established | none |
| F3_NVFP4 | K11 | F | 2 / 8 / 2 | - | none |
| F4_INTQ | K11 | F | 3 / 9 / 0 | - | none |
| F5_PACK | K1 | G | 1 / 11 / 0 | - | none |
| F6_DELAYED | K11 | F | 3 / 8 / 1 | - | none |
| F7_FP8SCALE | K11 | F | 4 / 8 / 0 | E4[fp32]=established | none |
| H1_ROUTING | K10 | H | 3 / 8 / 1 | - | none |
| H2_ALIGN | K10 | H | 4 / 7 / 1 | - | none |
| K14_COLL | K14 | none | 4 / 7 / 1 | - | none |
| R1_RANDOM | K15 | R | 3 / 8 / 1 | E4[ge]=established | none |

Open dimensions (reported as "not established" by the tool):
- A1_LINEAR D11: padding slots of the state pool and partially filled chunks need each engine's convention
- B1_PAGED D8: quantized KV: the default target is attention over the dequantized cache; comparing against attention on the original tensors needs them captured
- B1_PAGED D12: backward of paged attention not specified in v2.5
- C1_SPARSE D5: fp8 / fp4 KV byte layouts and UE8M0 scale rounding in indexer caches need per-engine declarations (F contracts give the arithmetic)
- C1_SPARSE D7: compressed-KV ring buffers not specified in v2.5
- F2_MX D9: stochastic rounding of MX elements in training recipes must be declared per tensor; F1.C3 applies once the kernel's uniforms are captured
- F3_NVFP4 D3: rowwise vs columnwise block axis and the swizzled scale layout must be declared per kernel
- F3_NVFP4 D9: TE applies stochastic rounding to some tensors; checked with F1.C3 once the uniforms are captured
- F6_DELAYED D11: behaviour when the history amax is 0 is not confirmed from the source (the reference keeps the previous scale)
- H1_ROUTING D12: router backward (gradient through gating values only) not specified in v2.5
- H2_ALIGN D5: quantized expert weights follow F and G contracts; per-kernel format declaration needed
- K14_COLL D7: symmetric-memory buffers and signal protocols are execution details; only their results are specified
- R1_RANDOM D2: temperature floor and the order of top-k / top-p filtering need per-engine declarations

TOTAL: 14 contracts, 54 clauses (54 established), 76 examples, misreadings rejected 47/47
