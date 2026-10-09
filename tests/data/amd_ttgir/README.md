# AMD TTGIR fixtures (DSL v2 increment 12)

TTIR dumped by the official main build (Triton e50b186e) while running official unit tests on sm_86 (W0 observed
inventory, `.cache/dsl_v2/w0_dump`), and the TTGIR the official AMD backend's own stages produce from it
(`scripts/dsl_v2/amd_stages.py`: `make_backend` + `add_stages`, stages `ttir` and `ttgir` only).  Location annotations
removed (`#loc` lines and ` loc(...)` suffixes; layout aliases kept); otherwise unchanged.  Variants used by tests/test_signatures_amd.py are made in
the test and say so.

| fixture | source TTIR (dump dir / file) | target | sha256[:16] of the TTIR | sha256[:16] of the TTGIR as produced |
| --- | --- | --- | --- | --- |
| `argmax_masked_other` | `WMRB47JAFXZCYHWKU3W3ORCEDCKQE5LRNMV7O6Y4K6QWRELDNQ3A/argmax_fast_kernel.ttir` | gfx942 | `49a0ce61ecc55418` | `f523f64a30010860` |
| `masked_copy_bf16` | `CZ4G4SAEZEFPUXW6O4AE7S7OCWOEU7BYIB6IES4L5LJS5KC67HYQ/kernel.ttir` | gfx942 | `054ee47d86e4272b` | `52c7cf68dd94973d` |
| `atomic_fadd_pairs` | `ODV475H4OXLHUWP4NOTBIXJENX5JHWO33SRKZ7B6MAOCI7LKQJMA/kernel.ttir` | gfx942 | `657e63eabc3c5b13` | `5ba98fcd62990b70` |
| `atomic_cas_rows` | `EQLAMA6CZULRPTJ5JTXTT53WU5K2NL66XU445Y7CAZV75XPWJBAQ/kernel.ttir` | gfx942 | `b8bff4aea60591e3` | `7a047539e0835f09` |
| `simple_dot_transpose` | `SA6TBEGUKIQSIJEFTVK2ZMKYVEEFA5UUMLDTFIOC5UXKU3RSV3PQ/simple_dot.ttir` | gfx942 | `56cc0a0fca737332` | `acb3e4ca8f8acdfb` |
| `simple_dot_mxfp8` | `Y5W22EAGKNJSI4U3ZWFFOOUFMO3FTHOK34BQGXXC3VZKEC4MPWHQ/simple_dot_mxfp.ttir` | gfx942 | `f92758ff84b7ae61` | `f1d86847507fcb94` |
| `mxfp8_mxfp4_matmul` | `AC7H67ID5IHQKKA2CFSP5IZZQXBS2SOSOSOKZYOF2OJJ2QOAEZKA/mxfp8_mxfp4_matmul_tma.ttir` | gfx942 | `c8aa269d36ca1310` | `99b8d5e88b54a15d` |
| `matmul_pipelined` | `4HWPURTIJ5SK2FBW26OGB2EZB7FK3SQI47QX7HV4D6VOR4MZ4CVQ/matmul_kernel.ttir` | gfx942 | `4ba16424ba73be40` | `cf30def81db2ac71` |
| `gather_dot_pipeline` | `WTWAG6XUOV7RFC64IDTYPZXT2EMWMZ3Y7HWR4Q4VCV2IBKVXTA7Q/tma_gather_dot_pipeline.ttir` | gfx950 | `8f0165ecc0248c79` | `1ffff154c0c4230b` |
