# Qwen3 U2 directional confirmation：必须避免的自适应方差陷阱

U2 是完整有符号 parameter-update discrepancy field。其每 state L2 只有大小，没有方向，因此不能作为
Bias。推荐的 confirmation estimand 是：先用四条 calibration 的等权 mean field 冻结单位方向 `v_cal`，
再在全新 confirmation trajectories 上估计 `<delta_U, v_cal>` 的 B/H/N/U。

但有一个额外陷阱：若 `v_cal` 由四条 calibration 选择，又直接用这四条在 `v_cal` 上的 in-sample
projections 估计 planning variance，dispersion 会因方向朝着同一数据选择而可能偏小。

v0.1 因此使用 leave-one-trajectory-out planning diagnostic：对每条 calibration trajectory `j`，只用其余
三条的 mean field 定义方向，再投影被留出的 `j`。得到四个 cross-fitted projections；其 sample variance
仍需小样本单侧上界和独立 variance floor，但比 full-direction in-sample variance 更接近新 trajectory
projection 的 planning target。

该诊断仍不自动实例化方向。freeze 前还必须独立声明：

1. vector measurement floor；不能因为 deterministic repeats 的 N=0 就自动设为 0；
2. leave-one-out direction stability threshold；不能看到 cosine 后临时放宽；
3. desired projection half-width 与 resource cap；
4. U2 是 joint endpoint family 的成员还是明确命名的 secondary claim。

若 full mean norm 不超过 measurement floor，或 leave-one-out direction 不稳定，U2 direction verdict 应是
`UNINSTANTIATED_DIRECTION`，不能从 confirmation outcomes 重新选择方向。即使独立 confirmation 成功，
结论也只是“calibration 学到的这个固定 update 方向可复现”，不是 full-vector omnibus claim、correctness
error 或长期训练危害。
