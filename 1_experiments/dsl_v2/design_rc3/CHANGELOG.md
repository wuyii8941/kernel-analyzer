# rc2 → rc3

- 取消把Gluon/AMD/低层路径列为DSL范围外；实施可分期，官方目标不能删。
- 新增官方范围主文档与61项能力族；保留旧229路由但撤销其作为新版总分母的地位。
- 补NVIDIA/AMD布局、共享/tensor memory、TMA/TDM、mbarrier、warp specialization、cluster/CLC。
- 将CAS/原子旧值/合法非交换交错纳入关系语义；加集合稳健均值，不能捏造调度分布。
- 更正竞争守卫、per_sample参照路径义务、近似指令误差metric、cat重排、tie-break及RNG浮点变换。
- 补编译假设、时钟、插桩/profiling、pack/constraints、runtime参数绑定与外部库符号。
- 更正M、零方差、端点截断和κ/scope混用。
- 规则schema按模式/条件记录精度保证，四个例子均为planned。
- 125个旧检查点保留或显式更正，没有静默删除。
- 新增核账/声明检查工具及35项CPU设计检查。不包含新GPU或生产支持声明。
