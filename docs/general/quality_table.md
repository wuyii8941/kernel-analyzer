# 第 3 项：参照精度与成本的家族表（工具 3.0 对 2.3 累加方式）

每行四个数：完整参照率；宽度 / 输出 dtype 在 |G| 处的 ulp（各输出中位数的中位数、各输出 90% 分位的中位数、最大值）；达到分辨目标（≤ 1/8 ulp）的元素比例；分段耗时（秒：捕获含编译预热 / 参照 / 统计含规格）。「2.3 累加方式」用同一代码、`KA_ACCUMULATION=gamma` 重跑，只换回 γₙ 求和与点积界。

| 方式 | 家族 | 来源 | 输出 | 完整率 | 宽度/ulp 中位 | 90% | 最大 | 达到 1/8 ulp | 捕获 / 参照 / 统计 s |
|---|---|---|---|---|---|---|---|---|---|
| 3.0 (SumK / DotK) | activations | tier-1 mode B (3 units) | 25 | 1 | 3.7e-32 | 3.73e-09 | 3.35e+07 | 0.939 | 2.9 / 1.2 / 0.7 |
| 3.0 (SumK / DotK) | attention | tier-1 mode B (3 units) | 23 | 0.975 | 3.2e-07 | 9.45e-07 | 0.0817 | 1 | 14.6 / 522.0 / 2.1 |
| 3.0 (SumK / DotK) | embedding | tier-1 mode B (3 units) | 8 | 1 | 0 | 0 | 2.38e-07 | 1 | 2.6 / 0.5 / 0.3 |
| 3.0 (SumK / DotK) | gather_layout | tier-1 mode B (3 units) | 23 | 1 | 0 | 0 | 7.45e-09 | 1 | 1.6 / 0.3 / 0.4 |
| 3.0 (SumK / DotK) | matmul_linear | tier-1 mode B (3 units) | 2 | 1 | 0 | 0 | 0 | 1 | 6.5 / 0.0 / 0.9 |
| 3.0 (SumK / DotK) | moe | tier-1 mode B (3 units) | 10 | 1 | 0 | 0 | 2.51e-06 | 1 | 2.5 / 12.0 / 0.4 |
| 3.0 (SumK / DotK) | normalization | tier-1 mode B (3 units) | 31 | 1 | 9.31e-09 | 1.86e-08 | 0.0201 | 1 | 3.3 / 5.2 / 1.0 |
| 3.0 (SumK / DotK) | optimizers | tier-1 mode B (3 units) | 72 | 1 | 1.86e-09 | 1.86e-09 | 2.86e-06 | 1 | 4.8 / 1.3 / 2.6 |
| 3.0 (SumK / DotK) | packing | tier-1 mode B (3 units) | 9 | 1 | 1.62e-07 | 3.65e-07 | 0.019 | 1 | 4.9 / 111.1 / 0.6 |
| 3.0 (SumK / DotK) | reductions | tier-1 mode B (3 units) | 46 | 0.893 | 0 | 0 | 8.94e-08 | 1 | 12.5 / 2.1 / 1.0 |
| 3.0 (SumK / DotK) | numerical stream (bf16 / fp16 Inductor) | quality probe (3 units) | 20 | 1 | 2.13e-13 | 2.56e-13 | 6.19e-06 | 1 | 9.8 / 9.4 / 1.2 |
| 3.0 (SumK / DotK) | G6 compositions (float32 Inductor, fwd + bwd) | quality probe (3 units) | 27 | 1 | 2.05e-08 | 2.85e-08 | 0.000412 | 1 | 20.6 / 19.1 / 0.4 |
| 3.0 (SumK / DotK) | new operators (torch.sparse Triton kernels) | quality probe (3 units) | 3 | 1 | 1.3e-08 | 5.96e-08 | 0.000244 | 1 | 0.3 / 1.4 / 0.1 |
| 2.3 accumulation (gamma) | activations | tier-1 mode B (3 units) | 25 | 1 | 3.7e-32 | 3.73e-09 | 3.35e+07 | 0.939 | 3.2 / 1.5 / 0.8 |
| 2.3 accumulation (gamma) | attention | tier-1 mode B (3 units) | 23 | 0.975 | 3.5e-06 | 1.22e-05 | 5.72 | 1 | 14.1 / 452.9 / 2.1 |
| 2.3 accumulation (gamma) | embedding | tier-1 mode B (3 units) | 8 | 1 | 0 | 0 | 7.22e-06 | 1 | 2.3 / 0.3 / 0.2 |
| 2.3 accumulation (gamma) | gather_layout | tier-1 mode B (3 units) | 23 | 1 | 0 | 0 | 1.03e-06 | 1 | 1.7 / 0.4 / 0.5 |
| 2.3 accumulation (gamma) | matmul_linear | tier-1 mode B (3 units) | 2 | 1 | 0 | 0 | 0 | 1 | 7.0 / 0.0 / 0.9 |
| 2.3 accumulation (gamma) | moe | tier-1 mode B (3 units) | 10 | 1 | 0 | 0 | 5.11e-06 | 1 | 1.8 / 9.6 / 0.3 |
| 2.3 accumulation (gamma) | normalization | tier-1 mode B (3 units) | 31 | 1 | 2.79e-08 | 4.73e-08 | 0.12 | 1 | 3.4 / 5.2 / 1.1 |
| 2.3 accumulation (gamma) | optimizers | tier-1 mode B (3 units) | 72 | 1 | 1.86e-09 | 1.86e-09 | 2.86e-06 | 1 | 3.9 / 1.1 / 2.6 |
| 2.3 accumulation (gamma) | packing | tier-1 mode B (3 units) | 9 | 1 | 2.28e-06 | 5.54e-06 | 0.287 | 1 | 5.0 / 100.4 / 0.6 |
| 2.3 accumulation (gamma) | reductions | tier-1 mode B (3 units) | 46 | 0.893 | 0 | 0 | 0.00443 | 1 | 12.9 / 1.4 / 1.0 |
| 2.3 accumulation (gamma) | numerical stream (bf16 / fp16 Inductor) | quality probe (3 units) | 20 | 1 | 8.9e-12 | 1.47e-11 | 6.19e-06 | 1 | 9.6 / 5.8 / 1.3 |
| 2.3 accumulation (gamma) | G6 compositions (float32 Inductor, fwd + bwd) | quality probe (3 units) | 27 | 1 | 2.36e-07 | 4.55e-07 | 0.00323 | 1 | 4.2 / 16.3 / 0.4 |
| 2.3 accumulation (gamma) | new operators (torch.sparse Triton kernels) | quality probe (3 units) | 3 | 1 | 3.02e-07 | 1.55e-06 | 0.00266 | 1 | 0.4 / 1.0 / 0.1 |
| 3.0 (SumK / DotK) | unified entry: bsr_dense_mm {'x.dtype': 'float32', 'd.dtype': 'float32'} | unified entry (96 units) | 1 | 1 | 1.12e-08 | 5.96e-08 | 0.00244 | 1 | 2.9 / 7.1 / 1.3 |
| 3.0 (SumK / DotK) | unified entry: bsr_dense_mm {'x.dtype': 'bfloat16', 'd.dtype': 'bfloat16'} | unified entry (96 units) | 1 | 1 | 1.71e-13 | 9.09e-13 | 7.45e-09 | 1 | 0.2 / 6.5 / 0.5 |
| 3.0 (SumK / DotK) | unified entry: bsr_softmax  | unified entry (96 units) | 1 | 1 | 1.3e-08 | 1.68e-08 | 2.42e-08 | 1 | 2.9 / 22.8 / 1.5 |
| 3.0 (SumK / DotK) | unified entry: sampled_addmm  | unified entry (96 units) | 1 | 1 | 1.49e-08 | 6.71e-08 | 0.0156 | 1 | 2.9 / 10.3 / 1.5 |
