# Qwen3 GRPO natural transition findings v0.2

## 有效查询

selected state 为预先抽取的 run-B optimizer-step-29 pre-minibatch snapshot。A1/A2 为 eager；B1/B2 为 final-state 上按捕获 inputs 顺序构建的 fresh tracked-Inductor scorer。四个 arms 均从相同 model、optimizer、scheduler、GradScaler、RNG 和 minibatch 开始。

construction validity 全部通过：scorer bitwise anchors、ordered unique graph family、compiled runtime invocation、parameter-object identity、history state preservation、finite gradients/updates 和四臂 pre-state identity 均有效。

## 结果

| Endpoint | 观测 |
|---|---:|
| loss B signed effect | `+3.0979514e-4` |
| loss paired N | `0` |
| clipped-gradient B L2 | `9.6042462e-3` |
| clipped-gradient relative B | `0.9604%` |
| gradient paired N L2 | `0` |
| parameter-update B L2 | `3.1111398e-5` |
| parameter-update relative B | `0.4103%` |
| update paired N L2 | `0` |
| eager self next-state | exact |
| candidate self next-state | exact |
| eager/candidate next-state | different |

所有已声明 semantic events 均无 disagreement：per-token clipping decisions、global gradient clipping trigger、AMP step skip、nonfinite gradient/update 与 scaler transition 均相同。

因此 selected-state exact transition compatibility verdict 为 `REJECT_EXACT_SELECTED_STATE`。它只表示 scorer-forward implementation treatment 改变了该 state 的自然下一状态。

## 不能推出

- H：只有一个 transition state，`UNIDENTIFIABLE_ONE_STATE`；
- U/population：selection 不是 state population sample，`NOT_ESTIMATED_ONE_SELECTED_STATE`；
- correctness：eager 无 truth authority，`UNINSTANTIATED`；
- operator attribution：treatment 是 whole scorer forward，`NOT_CLAIMED`；
- long-run harm：一步 update effect 不证明不收敛、精度下降或训练变长；
- historical compile lineage：snapshot 没有十份历史模型/optimizer states，不能重建原训练的十次 compile events。

## 对 Oracle 的意义

这个实例直接否定“只有发生 fork 才有训练影响”。它也否定“观察到稳定 update difference 就能称为 bug”。规范 Oracle 必须并列保留 continuous、semantic 和 transition ledgers，并让 correctness/long-run/attribution 维持独立 authority。

原始结果：`results/training_step_oracle/qwen3_grpo_natural_transition_v0_2/evaluation.json`。
