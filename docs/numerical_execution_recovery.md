# 中断后的实验恢复

本次会话中断后，原会话句柄失效；随后在主机上核查，原实验进程不在运行，
四张 GPU 均已释放。退出原因和退出码没有可靠记录，不写成数值失败或成功退出。
旧目录、日志、协议和已完成结果全部保留，没有覆盖或删除。

## 恢复范围

- SiLU 全计划36位置中，首项和两个原队列的首项已完整复算，共3项，不重跑。
  其余33项保持原顺序，进入 `deepseek128_silu_queue_gpu2_retry1` 和
  `deepseek128_silu_queue_gpu3_retry1`。原队列的未完成记录不重复计数。
- 完整均值复测尚未完成，在 `deepseek128_silu_full_mean_replay_retry1`
  重新执行，仍要求与已完成首项的原始 JSON 逐字节一致。不是新增独立案例。
  此项现已完成并通过该核验，完整坐标结果见 [SiLU 说明](silu_backward_common_input.md)。
- RMS 原35位置批次没有完整结果文件，部分状态进度不能恢复为完整测量。
  保留全35位置，按原顺序拆成7批，每批5位置，使用新目录。
  `deepseek256_rms_batch000_retry1` 完成且复算通过后，自动执行剩余6批。
  新采集增加执行前声明的 rtol=1e-5、atol=1e-8 对照与成本记录，
  不替换历史协议、不修改原1%参数更新RMS范围。

## 防止把进程状态当实验结果

`launch_detached_experiment.py` 独立启动实验，保存完整命令、进程身份、日志和
实际退出码。`check_detached_experiments.py` 检查 `/proc` 中 PID 对应的真实
命令，区分仍在运行、退出、身份改变以及无法核验；不根据日志或锁文件推断存活。
独立进程减少对话会话中断的影响，但不承诺抵御主机重启或外部终止。

后续任务必须等待前一任务真实退出并核验结果，不能仅凭退出码0继续签发完成。
当前进程记录在 `numerical_coverage_v1/detached_runs/`；启动后的实测存活记录
为 `initial_live_verification.json`。它是一次观察，不是永久存活保证。

`deepseek128_silu_execution_audit_recovered.json` 按原36项核验恢复后的完整分母。
尚未完成、未保存足够数据和需要其他参考的记录都继续保留；主线没有因此缩小。
