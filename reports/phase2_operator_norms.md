# Phase 2 Operator Input Norms

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- real model inputs measured: PASS
- exact reduction lengths from Linear modules: PASS
- conservative product-sum inequality: PASS
- exact kernel reduction order: FAIL / not yet known
- cross-source local independence: FAIL / not yet established

## Delta Self Control
This is a one-path norm measurement; Phase 1/A4 self controls remain authoritative.

## External Validity
Measured on the exact step-5 T4 FP16 snapshot with FP32 master weights and FP16 autocast.

## Largest Conservative Injection Terms
| module | in_features_reduction_length | out_features | weight_abs_max | input_l1_max | sum_abs_product_upper | invocations |
| --- | --- | --- | --- | --- | --- | --- |
| model.layers.27.mlp.down_proj | 3072 | 1024 | 0.5546875 | 13699.48046875 | 7598.930572509766 | 4 |
| model.layers.2.mlp.down_proj | 3072 | 1024 | 0.5820279717445374 | 11578.869140625 | 6739.225721013383 | 4 |
| model.layers.27.self_attn.k_proj | 1024 | 1024 | 0.6835964918136597 | 9068.083984375 | 6198.910399190383 | 4 |
| model.layers.25.self_attn.q_proj | 1024 | 2048 | 0.39453527331352234 | 9981.8974609375 | 3938.2106429385312 | 4 |
| model.layers.27.self_attn.q_proj | 1024 | 2048 | 0.42772984504699707 | 9068.083984375 | 3878.6901575098746 | 4 |
| model.layers.24.self_attn.q_proj | 1024 | 2048 | 0.43945637345314026 | 8743.560546875 | 3842.4134089976433 | 4 |
| model.layers.21.self_attn.k_proj | 1024 | 1024 | 0.6289107799530029 | 5317.15087890625 | 3344.0135063807247 | 4 |
| model.layers.26.self_attn.q_proj | 1024 | 2048 | 0.3437500596046448 | 9686.0224609375 | 3329.5707982791937 | 4 |
| model.layers.27.mlp.up_proj | 1024 | 3072 | 1.234375 | 2218.78662109375 | 2738.8147354125977 | 4 |
| model.layers.23.self_attn.q_proj | 1024 | 2048 | 0.3828144371509552 | 6709.15673828125 | 2568.362060522675 | 4 |
| model.layers.27.mlp.gate_proj | 1024 | 3072 | 1.15625 | 2218.78662109375 | 2565.4720306396484 | 4 |
| model.layers.26.mlp.down_proj | 3072 | 1024 | 0.6406295299530029 | 4002.390869140625 | 2564.0497811857495 | 4 |
| model.layers.27.self_attn.v_proj | 1024 | 1024 | 0.2617204487323761 | 9068.083984375 | 2373.303009533498 | 4 |
| model.layers.26.self_attn.k_proj | 1024 | 1024 | 0.23827794194221497 | 9686.0224609375 | 2307.9654975982558 | 4 |
| model.layers.25.self_attn.v_proj | 1024 | 1024 | 0.2197284698486328 | 9981.8974609375 | 2193.30705527775 | 4 |
| model.layers.24.self_attn.k_proj | 1024 | 1024 | 0.2451210618019104 | 8743.560546875 | 2143.2308451792924 | 4 |
| model.layers.21.self_attn.q_proj | 1024 | 2048 | 0.40234023332595825 | 5317.15087890625 | 2139.3037252484646 | 4 |
| model.layers.22.self_attn.q_proj | 1024 | 2048 | 0.34374552965164185 | 5887.44921875 | 2023.7843499963637 | 4 |
| model.layers.25.mlp.down_proj | 3072 | 1024 | 0.6093794703483582 | 3317.169921875 | 2021.415250047692 | 4 |
| model.layers.25.self_attn.k_proj | 1024 | 1024 | 0.20215333998203278 | 9981.8974609375 | 2017.8739110866882 | 4 |

These measurements close the input-norm evidence gap only. They do not make the Phase 2 bound semi-certified until algorithm order and local independence are justified.
