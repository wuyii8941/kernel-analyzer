# 检查点迁移：保留全部125项，不以删除难项获得通过

旧语义或范围错误用更正/扩展记录替代，旧措辞仍保存在JSON和history/input_rc2.zip。链接锚点验证只证明文档有对应位置，不证明规则正确。

| ID | 处理 | rc3要求 | 文档 |
|---|---|---|---|
| A1 | retained | 模式 A 不需要任务规格 f | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| A2 | retained | 模式 B：e_num、e_sem、e_total 分开 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| A3 | retained | 接口项 e_interface（发布值与接收值） | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| A4 | retained | 编译期常数以存储值进入 G；归因分「重算核实 / 机制推断」 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| A5 | retained | K − G 不先验称为「只舍入」 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| A6 | retained | f 缺失不令 f := g | [design/01_repository_basis.md](design/01_repository_basis.md) |
| A7 | retained | 模式 B 的规格库与五类输入契约 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| B1 | retained | 输出 / 梯度 / 公式更新 / 实际写入的下游映射 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| B2 | retained | 实际写入区分 CPU/GPU、foreach/fused、dtype | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| B3 | retained | 状态序列：零梯度与 grad=None 不同、跳步、保存恢复 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| B4 | retained | 影子状态 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| B5 | corrected | D_m需可靠区间扩展；跨舍入边界保留集合，来源缺失才未建立 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| B6 | retained | 训练后果不在 DSL 保证内，需独立配对训练 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| C1 | retained | Real / Format / BV 不隐式互换 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| C2 | retained | 特殊值、Poison、定义域未定义的结果类型 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| C3 | retained | 合法的 −inf 与 mask 流（如 masked attention） | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| C4 | retained | if 与 select 的区别 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| C5 | retained | 精确核心 L_Q | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| C6 | retained | 可收紧扩展 L_C，初等函数须严格包围库 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| C7 | expanded | 集合包围不删除目标，可能足以支持稳健均值 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| C8 | retained | 离散使用点的判定证据 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| C9 | retained | 统一的分析策略 π，不事后改 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D1 | corrected | 任意有类型combine纳入；树绑定需可信证据，非关联不冒充顺序无关 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D2 | retained | Welford 的 Φ 证书（取代按写法识别） | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D3 | retained | 无保护的 0/0 不被代数修复 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D4 | retained | scan 保持前缀顺序 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D5 | retained | 参照循环的终止证书 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D6 | corrected | 近似契约必须按ulp/absolute/relative/piecewise及域和target区分 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D7 | retained | libdevice / extern_elementwise 语义表 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D8 | expanded | 两种内联asm均纳入目标，未知语义是插件/证据欠账 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D9 | retained | 浮点融合（mul+add → fma） | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D10 | expanded | 枚举当前官方dot属性与厂商matrix变体，不限旧三个取值 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D11 | retained | 格式转换与 MX 块缩放 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D12 | corrected | divf具体下降依编译器/target/选项，不普遍假定近似除法 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D13 | expanded | reshape与cat均按重排权限和实际元素关系解释 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D14 | retained | maximumf 与 maxnumf 的 NaN 语义 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D15 | retained | gather 索引逐样本检查 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D16 | corrected | 并列先查源契约，规定tie-break时不是任意集合 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D17 | expanded | RNG位生成器与浮点分布变换分离，随机源不排语言外 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D18 | retained | 排序、top-k、直方图 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D19 | retained | block pointer（make_tensor_ptr / advance） | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D20 | corrected | 高层descriptor逻辑访问与低层TMA/TDM的异步token分层 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D21 | retained | dot_scaled | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D22 | retained | map_elementwise | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D23 | retained | 整数位宽、溢出标志、poison、除零 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D24 | retained | 位级操作走格式桥，不回填实际值 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| E1 | retained | 来源按存储身份，不按数值相等 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| E2 | retained | kernel 级与调用级分开 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| E3 | expanded | 调用生产者以图和存储证明；主张scope与外部边界明确 | [design/04_implementation_plan.md](design/04_implementation_plan.md) |
| E4 | corrected | 检查thread/lane、scope和happens-before，实际与参照路径分别覆盖 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| E5 | corrected | 同program无屏障不是充分竞争条件；同线程顺序读写可合法 | [design/04_implementation_plan.md](design/04_implementation_plan.md) |
| E6 | corrected | 逐位不一致只触发诊断，不直接证明race或允许随机平均 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| E7 | retained | 原子折叠条件 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| E8 | retained | 原子程序重复执行、输入内平均 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| E9 | corrected | 已知竞争不作均值；有效原子调度随机性可以分析 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| E10 | retained | 同步的效应必须解释 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| E11 | retained | 缓存键包含来源证明 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| F1 | retained | 接纳判断 Γ; Φ; Π | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| F2 | corrected | per_sample须包含参照可达义务；不能只认证K实际轨迹 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| F3 | corrected | 逐样本承诺仍需参照侧路径/域有效，不外推全域 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| F4 | retained | 证书形式的限制（不接受未核查的 CAS/SMT 结果） | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| F5 | corrected | 总性、收紧是预先定义片段的条件命题，非全部语言无条件承诺 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| F6 | retained | 资源失败不回填实际值 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| F7 | expanded | 测试和证据按完整签名与模式，JSON引用不等于语义证明 | [design/04_implementation_plan.md](design/04_implementation_plan.md) |
| G1 | retained | 查询五项 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G2 | retained | 固定方向与对齐分开 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G3 | retained | 独立单位：batch、历史、run；checkpoint 不独立 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G4 | retained | 默认输入来源：真实训练状态，按步数分层 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G5 | retained | 开发 / 确认分离与检验族事前登记 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G6 | corrected | 端点截断是可选证明路线；空交集必须报证据矛盾 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G7 | corrected | M界住总体残差；输入无界不代表没有M | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G8 | retained | Hoeffding 推导 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G9 | retained | 经验 Bernstein 作后续选项 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G10 | retained | 近似路线的条件与失效范围 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G11 | retained | 两个判定轴；δ 的来源 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G12 | retained | 最小可检出效应是设计灵敏度 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G13 | retained | Holm；确认与复现分开校正 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G14 | retained | 整向量盒子 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G15 | corrected | 无有效观测未建立；s=0不禁止有界区间且不自动给p=0 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G16 | retained | 过滤改变总体须声明 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G17 | retained | 序贯追加样本须事先声明 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G18 | retained | 八类校准分布 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| G19 | retained | δ 的标定（可接受做法尺子、剂量反应） | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| H1 | corrected | 四服务路径不等于可靠性等级；完整kernel不自动完整调用 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| H2 | corrected | κ、scope、来源、精度和统计分别记录 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| H3 | retained | 模式 B 四栏 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| H4 | corrected | 改为精度响应诊断，不能恢复缺失G或作语义裁决 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| H5 | retained | 成本四段与人工 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| H6 | retained | K_R 字段到 G 与 ℛ 的映射 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| H7 | retained | 结果类别分开计数 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| I1 | expanded | 官方U、语言L、分析实现A与现实样本B独立 | [design/00_official_scope.md](design/00_official_scope.md) |
| I2 | expanded | 公共API加源码、构建、实际路径四来源链核账 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| I3 | superseded | 229/165仅历史profile，不是新DSL分母 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| I4 | expanded | 旧35项的实施难度分组不限定新版目标 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| I5 | retained | 已映射名字的属性守卫 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| I6 | retained | 覆盖单位是带约束的签名 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| I7 | retained | 现实语料三层，失败也记录 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| I8 | retained | 外部 kernel 语料（如 Correctness Illusion） | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| I9 | retained | 结构交互维度 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| I10 | retained | 随机生成器独立于被测工具 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| I11 | retained | E / F / P / FR 四组 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| I12 | retained | 普通参照加同样统计（第 3 层） | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| I13 | retained | 容差基线：默认与登记扫描、同等预算 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| I14 | retained | 既有证据（盲测、校准）不直接继承 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| I15 | expanded | 官方登记、规则实现、兑现、分辨率、现实可用性分开 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| I16 | corrected | 标签含scope/保证类别/执行有效性，合法非交换不作为非法 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| I17 | retained | 外部人员接入 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| I18 | retained | 开发 / 保留按实现谱系划分 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| J1 | retained | 严格部件须附病态探针 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| J2 | corrected | 激活与竞争独立核对；执行不一致不是竞争的充分证据 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| J3 | retained | 无类别暗示；答案先封存；测量后更正单列非盲 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| J4 | retained | 冻结结果只读 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| J5 | retained | 「精确」只用于精确路径 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| J6 | retained | 偏离先登记再运行 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| J7 | retained | 工作精度按宽度提高，不按显著性 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| K1 | superseded | Gluon、ttg/ttng、AMD全部进入官方目标，只可分期实施 | [design/00_official_scope.md](design/00_official_scope.md) |
| K2 | superseded | CAS/load/store/poll作为事件关系纳入，不因非交换拒绝语言 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| K3 | retained | 不要求先写完 14 份任务规格 | [design/04_implementation_plan.md](design/04_implementation_plan.md) |
| K4 | corrected | 可借用并发验证器；不能宣称简单逐样本最后写入表已充分 | [design/04_implementation_plan.md](design/04_implementation_plan.md) |
| K5 | retained | 不做通用 bias 预测器 | [design/04_implementation_plan.md](design/04_implementation_plan.md) |
| K6 | corrected | 官方支持及实际依赖决定范围，普查没有出现不构成排除理由 | [design/03_coverage_and_evaluation.md](design/03_coverage_and_evaluation.md) |
| B7 | retained | 测量点的推进顺序：先算子输出，再保存值、梯度、实际写入 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| C10 | retained | 形状、启动与硬件配置（含 autotune 选中的配置）是任务的一部分 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D25 | retained | fp8 等格式的编码：显式舍入、饱和与次正规约定 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
| D26 | retained | 多 kernel 的启动序列与跨启动状态 | [design/02_language_and_guarantees.md](design/02_language_and_guarantees.md) |
