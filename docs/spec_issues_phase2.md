# 规格问题（执行方发现，交审阅方；规格代码未改）

按任务书第 0、9 节：规格语义只由审阅方修改。执行方在运行中发现的规格问题记在这里；涉及的条款**停止裁决**，数据照常记录，
结论栏标「条款待审阅」（`scripts/closure/contract_v2.py::PENDING`）。

## SPEC-ISSUE-1 `spec_optimizers` O-D2：SGD 的 maximize 位置与锁定版本文档不符（2026-10-08，F 组运行中发现）

- **文档原文**（torch 2.10.0 `torch.optim.SGD` docstring，逐字存于 `results/closure/doc_check/8_torch_optim_SGD.txt`）：
  算法框第一步 `if maximize: g_t ← −∇f_t(θ_{t−1}) else g_t ← ∇f_t(θ_{t−1})`，随后加权重衰减、更新 b_t，最后
  `θ_t ← θ_{t−1} − γ g_t`。即 maximize **先**取负梯度，权重衰减和动量都作用在取负后的 g_t 上。
- **规格**：O-D2 docstring 写 `θ_t ← θ_{t−1} − γ g_t (+ if maximize)`；`sgd_step` 中 b 累积未取负的梯度，maximize 只在最后一步
  改成 `x + lr·g`。同一文件的 OPT-A2 写「the boxes apply maximize to g_t first and weight decay unchanged」，与 O-D2 的实现不一致。
- **两者的差别**：
  - weight_decay = 0：参数轨迹在实数上相同；动量缓冲 b 符号相反（文档的 b_t 累积 −∇f）。
  - weight_decay = λ ≠ 0：文档 θ_t = θ − γ(−∇f + λθ)（仍是衰减），规格 θ_t = θ + γ(∇f + λθ)（衰减方向反了），参数轨迹不同。
- **观测**（2b 条件 `opt_sgd_nesterov_cold_max_wd0.0`，3 个种子，全部 torch 实现与 float64 配对）：每个 torch 候选参数 0 / 408 超出；
  `momentum_buffer` 306 / 306（该条件全部缓冲元素）超出（float32 实现；14688 / 3672 是全家族合计）；float64 for_loop CPU 的缓冲与「规格 b 取负」相差 ≤ 1.5e-16（相对），
  与规格 b 本身相差 1.59（相对）。精度不变性把这些缓冲元素判为语义（102 / 102，每个配对每个种子）。
  `results/closure/f_eval/optimizers.json`。2b 没有 SGD + maximize + λ ≠ 0 的条件，参数轨迹的分歧没有被运行覆盖。
- **执行方处理**：O-D2 的 maximize 部分停止裁决；该条件的 `momentum_buffer` 比对记「条款待审阅」，不计入第 1 栏。
- **请审阅方决定**：O-D2 是否改为文档框的顺序（maximize 先取负，b 累积取负后的梯度）；改后执行方重跑这一条件，并补一个
  SGD + maximize + λ ≠ 0 的条件。
