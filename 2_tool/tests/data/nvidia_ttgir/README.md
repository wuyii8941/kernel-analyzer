# NVIDIA sm_100 TTGIR fixtures (DSL v2 increment 13)

TTIR dumped by the official main build (Triton e50b186e) while running official unit tests on sm_86 (W0 observed
inventory, `.cache/dsl_v2/w0_dump`), and the TTGIR `triton.compile` of the same build produces from it for
`GPUTarget("cuda", 100, 32)` (compile only; nothing runs on sm_100).  Location annotations removed (`#loc` lines and
` loc(...)` suffixes; layout aliases kept); otherwise unchanged.  Variants used by 2_tool/tests/test_signatures_nvidia_scaled.py
are made in the test and say so.

| fixture | source TTIR (dump dir / file) | target | sha256[:16] of the TTIR | sha256[:16] of the TTGIR as produced |
| --- | --- | --- | --- | --- |
| `simple_dot_mxfp` | `Y5W22EAGKNJSI4U3ZWFFOOUFMO3FTHOK34BQGXXC3VZKEC4MPWHQ/simple_dot_mxfp.ttir` | sm_100 | `f92758ff84b7ae61` | `5bfbcb9c91ee26b0` |
| `mxfp4_matmul_n128_s1` | `AC7H67ID5IHQKKA2CFSP5IZZQXBS2SOSOSOKZYOF2OJJ2QOAEZKA/mxfp8_mxfp4_matmul_tma.ttir` | sm_100 | `c8aa269d36ca1310` | `a2d4ceb61fa05c59` |
| `mxfp4_matmul_n128_s3` | `LYHUDZDXLPEI4XKXGDXWUN4VYJBZ4D7ZIFUI2Y6XGUXTHZJHPJ6A/mxfp8_mxfp4_matmul_tma.ttir` | sm_100 | `86ce482dd75f0a07` | `510c161ab8527ec0` |
| `mxfp4_matmul_n256_s1` | `A3JZIJU2EC5LPSO72L4IGSJXWQCRT6FB4OFRANAZJF2J66BNIS4Q/mxfp8_mxfp4_matmul_tma.ttir` | sm_100 | `c690083dddc0a00b` | `ab52e924a62a6d40` |
| `mxfp4_matmul_n256_s2` | `MJKEBBGYEZXSUXD4DJKEJDPKDU56CLI34M6MAF6VKCFFOCCA6K6A/mxfp8_mxfp4_matmul_tma.ttir` | sm_100 | `9d3e496e09041d11` | `a2fdcbadf63741d6` |
