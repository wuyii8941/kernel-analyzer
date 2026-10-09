# 运行环境

所有环境都在 `/data1/tzh/envs/` 下。运行前按 `CLAUDE.md` 把 HOME、pip、Triton 缓存与
临时目录指到 `kernel-analyzer/.cache`。

机器：4 × NVIDIA RTX A6000（sm_86），驱动 535.129.03，驱动支持到 CUDA 12.2。
因此 cu130 的 wheel（如 torch 2.13+cu130）在这台机器上不能用 GPU。

## 主线环境 `ka_main`

用于参照求值器、TTIR 映射和之后的自动参照实验。版本锁定信息（Triton 绑定的 LLVM
哈希、编译选项）由 `pre-reorg-20261009:scripts/record_version_lock.py` 写入
`pre-reorg-20261009:results/reference_eval/version_lock.json`。

| 组件 | 版本 |
|---|---|
| Python | 3.11（conda-forge） |
| torch | 2.10.0+cu128 |
| triton | 3.6.0（wheel，无 git 提交号；以 wheel 版本与 LLVM 哈希锁定） |
| gmpy2 / python-flint | MPFR 区间与精确有理数 / Arb |
| z3-solver 4.13.4.0 | scan 结合性证书与 CAS 自旋锁可交换性证书（未安装时不发证书，相应结果保持「前提下完整」） |
| numpy、scipy、pytest | — |

不装 liger-kernel、bitsandbytes、torchao、transformers。需要它们的旧案例用下面的
测试环境复现。

## 测试环境（保留，不删）

它们可作为跨版本对照：同一 kernel 在不同 Triton 版本下的 K 与 G（见阶段总结第 5 节
版本比较恒等式）。

| 环境 | Python | torch | triton | 备注 |
|---|---|---|---|---|
| liger | 3.10 | 2.10.0+cu128 | 3.6.0 | Liger、bitsandbytes、torchao、transformers 4.57；旧案例与现有测试套件；Liger/torchao 内核的捕获（libtriton 为 py3.10 构建，哈希与 ka_main 不同，捕获包各自记录） |
| pt271_alias_bug | 3.11 | 2.7.1+cu126 | 3.3.1 | transformers 4.57 |
| mamba_scan | 3.11 | 2.7.1+cu126 | 3.3.1 | Mamba 扫描 |
| kernelfuzz | 3.10 | 2.5.1+cu121 | 3.1.0 | 较老的 Triton |
| vllm_0160_qwen35412 | 3.12 | 2.9.1+cu128 | 3.5.1 | vLLM |
| pt_nightly_transformers5 | 3.11 | 2.13.0.dev+cu126 | 3.1.0+cf34004b8a | transformers 5 |
| pt213_scan | 3.11 | 2.13.0+cu130 | 3.7.1 | cu130，本机驱动下 GPU 不可用；能否只编译到 TTIR 未验证 |
| pt211_operator_tomography / pt220_operator_tomography | 3.11 | CPU 版 | — | 只能做 CPU 测试 |
| kernel_analyzer | 3.8 | — | — | 空环境，未使用 |
| fpcore | 3.11（由 ka_main 的解释器建立的 venv） | — | — | 只装 titanfp 0.1.2 与 numpy：FPCore 交叉复算的第三方求值器（`pre-reorg-20261009:scripts/fpcore_crosscheck.py evaluate`），与工具代码隔离 |
