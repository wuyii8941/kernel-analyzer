# 来源与适用范围

[R1–R7]与[P1–P6,D1–D4]继承上传rc2的来源记录，本轮不重新认证全部仓库与文献。新增官方核对见[S01–S21]及evidence/official_sources.json。
文献用于支撑方法依据，不表示它们已经证明本工具实现正确。

## 仓库

[R1] [program.py](https://github.com/wuyii8941/kernel-analyzer/blob/5b9ac32578733c52ee6b98030bf7af2b21783165/src/kernel_analyzer/reference_eval/program.py)。Program/Op的现有骨架。

[R2] [evaluator.py](https://github.com/wuyii8941/kernel-analyzer/blob/5b9ac32578733c52ee6b98030bf7af2b21783165/src/kernel_analyzer/reference_eval/evaluator.py)；[ttir_eval.py](https://github.com/wuyii8941/kernel-analyzer/blob/5b9ac32578733c52ee6b98030bf7af2b21783165/src/kernel_analyzer/reference_eval/ttir_eval.py)。读取的是相关片段，并非本轮全文件独立审计。

[R3] [ttir_mapping.py](https://github.com/wuyii8941/kernel-analyzer/blob/5b9ac32578733c52ee6b98030bf7af2b21783165/src/kernel_analyzer/reference_eval/ttir_mapping.py)。版本锁定、名字映射与拒绝项。

[R4] [rule_registry.py](https://github.com/wuyii8941/kernel-analyzer/blob/5b9ac32578733c52ee6b98030bf7af2b21783165/src/kernel_analyzer/reference_eval/rule_registry.py)。规则状态及尚未检查的前提；不是全域支持证明。

[R5] [ttir_op_registry.json](https://github.com/wuyii8941/kernel-analyzer/blob/5b9ac32578733c52ee6b98030bf7af2b21783165/results/reference_eval/ttir_op_registry.json)。登记229个名字；本次附表按工具返回内容转录，不声称独立重新枚举libtriton。

[R6] [结构验收说明](https://github.com/wuyii8941/kernel-analyzer/blob/5b9ac32578733c52ee6b98030bf7af2b21783165/docs/acceptance/structure_v1_1_rc1_status_20261008.md)。23/28的范围以及待评分状态。

[R7] 用户提供：《Triton 编译产物指令覆盖表：K_R 的构造依据》，2026-10-02；以及历史数值实现偏差讲稿。作为设计演进依据，正文明确标出的旧约定不再沿用。

## 原始论文及官方文档

[P1] Patrick Cousot, Radhia Cousot. Abstract interpretation: a unified lattice model for static analysis of programs by construction or approximation of fixpoints. POPL 1977, 238–252. DOI: 10.1145/512950.512973。
[作者页面](https://www.di.ens.fr/~cousot/COUSOTpapers/POPL77.shtml)。支持局部可靠抽象运算到组合可靠性的论证；不保证任意抽象精确、也不保证我们的实现符合模型。

[P2] George C. Necula. Proof-carrying code. POPL 1997, 106–119. DOI: 10.1145/263699.263712。
[出版页面](https://doi.org/10.1145/263699.263712)。借鉴证据生成与证据核验分工；本稿并未实现一个完整PCC系统。

[P3] Adam Betts, Nathan Chong, Alastair F. Donaldson, Shaz Qadeer, Paul Thomson. GPUVerify: a verifier for GPU kernels. OOPSLA 2012, 113–132. DOI: 10.1145/2384616.2384625。
[作者工具页](https://fastpl.doc.ic.ac.uk/tools/GPUVerify/index.php)。并发访问与屏障需要独立验证；并不意味着该工具可无需适配地证明Triton/新硬件的全部访存。

[P4] Andreas Maurer, Massimiliano Pontil. Empirical Bernstein Bounds and Sample Variance Penalization. COLT 2009.
[原论文](https://arxiv.org/abs/0907.3740)。有界条件下的均值集中与数据依赖区间；本稿的Hoeffding展示使用更简单的有界集中路线。

[P5] D. Richard Kuhn, Raghu N. Kacker, Yu Lei. Combinatorial Coverage Measurement. NISTIR 7878, 2012.
[官方页面](https://www.nist.gov/publications/combinatorial-coverage-measurement)。输入和配置t-way覆盖的度量，不是全程序正确性的替代。

[D1] [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)；[Side Effects & Speculation](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/)。操作、类型、区域与效应不同层次，不能只按操作名生成可靠分析。在线文档读取日2026-10-09，具体转换仍须锁定构建。

[D2] [TritonOps](https://triton-lang.org/main/dialects/TritonOps.html)；[associative_scan](https://triton-lang.org/main/python-api/generated/triton.language.associative_scan.html)；[atomic_add](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html)。区域、前缀语义、原子返回旧值等事实依据；动态main文档不能替代锁定版本定义。

[D3] [GNU MPFR manual](https://www.mpfr.org/mpfr-current/mpfr.html)。正确舍入与任意精度原语依据；将点函数用于区间输入还需正确的范围/极值处理，组合宽度不自动收敛。

[P6] Wassily Hoeffding. Probability Inequalities for Sums of Bounded Random Variables. JASA 58(301), 13–30, 1963.
[原论文出版页面](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830)。独立有界变量的集中界；不适用于用样本最大值替换总体范围。

[D4] [Triton教程总览](https://triton-lang.org/main/getting-started/tutorials/)；[Gluon介绍](https://triton-lang.org/main/gluon/index.html)；[Warp Specialization](https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html)。支持现实覆盖不应只围绕已有测试集：持久化、块缩放、显式布局与异步分工是独立结构维度。当前文档不代表锁定3.6.0的逐条能力。

## 本轮新增的官方范围核对

固定源码提交为e50b186e8bd2d16ae3f564311ef868aa3bd45a2d；main文档只作此次调查快照，不能直接等同于该提交的安装注册信息。

[S01] [官方方言索引](https://triton-lang.org/main/dialects/dialects.html)。官方页面列出的十个方言；不是安装注册全集的独立证明。

[S02] [triton.language 公共导出](https://github.com/triton-lang/triton/blob/e50b186e8bd2d16ae3f564311ef868aa3bd45a2d/python/triton/language/__init__.py)。本次读取固定提交的公共导出列表与入口类型处理。

[S03] [Gluon 公共导出](https://github.com/triton-lang/triton/blob/e50b186e8bd2d16ae3f564311ef868aa3bd45a2d/python/triton/experimental/gluon/language/__init__.py)。本次读取固定提交；包括布局、共享存储、原子、区域及后端命名空间。

[S04] [NVIDIA Gluon 子模块](https://github.com/triton-lang/triton/blob/e50b186e8bd2d16ae3f564311ef868aa3bd45a2d/python/triton/experimental/gluon/language/nvidia/__init__.py)。导出 ampere/hopper/blackwell/rubin；不表示每个硬件版本已实测。

[S05] [AMD Gluon 子模块](https://github.com/triton-lang/triton/blob/e50b186e8bd2d16ae3f564311ef868aa3bd45a2d/python/triton/experimental/gluon/language/amd/__init__.py)。导出 CDNA、RDNA、gfx1250 等子模块；不表示设备可用或完整实现已认证。

[S06] [TritonOps](https://triton-lang.org/main/dialects/TritonOps.html)。tt 的数据、区域、原子和descriptor契约。

[S07] [TritonGPUOps](https://triton-lang.org/main/dialects/TritonGPUOps.html)。布局、共享内存、async、warp specialization与逐线程inline asm。

[S08] [TritonNvidiaGPUOps](https://triton-lang.org/main/dialects/TritonNvidiaGPUOps.html)。NVIDIA特有矩阵、tensor memory及同步操作的目标面。

[S09] [TritonAMDGPUOps](https://triton-lang.org/main/dialects/TritonAMDGPUOps.html)。AMD buffer/LDS、异步、TDM、MFMA与缩放转换的目标面。

[S10] [Hopper APIs](https://triton-lang.org/main/gluon/api/nvidia.hopper.html)。cluster、mbarrier、TMA、warpgroup MMA及等待。

[S11] [Blackwell APIs](https://triton-lang.org/main/gluon/api/nvidia.blackwell.html)。CLC、tensor memory、packed算术等。

[S12] [CDNA4 APIs](https://triton-lang.org/main/gluon/api/amd.cdna4.html)。buffer、MFMA、块缩放和格式转换。

[S13] [elementwise inline asm](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html)。pack、constraints、side effects与元素分组约束。

[S14] [cat](https://triton-lang.org/main/python-api/generated/triton.language.cat.html)。can_reorder会改变可保证的元素对应，不能一律按固定concat解释。

[S15] [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)。误差可按ulp、绝对值、相对值或分段给出；特殊值、内存序及target qualifiers须逐条匹配。

[S16] [TritonInstrumentOps](https://triton-lang.org/main/dialects/TritonInstrumentOps.html)。GSan/ConSan/FpSan等插桩相关操作；不能默认全是nop。

[S17] [ProtonGPUOps](https://triton-lang.org/main/dialects/ProtonGPUOps.html)。计数器读取与日志写入等具有状态效应。

[S18] [NVGPUOps](https://triton-lang.org/main/dialects/NVGPUOps.html)。acquire加载、低层矩阵指令与tensor memory地址。

[S19] [Gluon overview](https://triton-lang.org/main/gluon/index.html)。官方将Gluon定位为直接暴露布局、共享内存和目标特征的低层模型。

[S20] [triton.language 文档](https://triton-lang.org/main/python-api/triton.language.html)。高层tensor API、随机数、编译提示、调试与控制相关入口。

[S21] [GluonOps](https://triton-lang.org/main/dialects/GluonOps.html)。Gluon导入层。

