# 冷启动 update：负对齐不等于更新范数缩小

对同一状态的 candidate update c、reference update r，令 u=c−r。
恒等式为 `2<u,r> = ||c||² − ||r||² − ||u||²`。
对同一固定集合求和并除以 reference 能量，可得
`2β = E_candidate/E_reference − 1 − Q`。
因此两种 update 能量相近时，β≈−Q/2；负对齐可以来自方向变化，
不能单独解释成 candidate 的整个更新范数变小。

零 moments、第一步、无 weight decay 的理想实数 AdamW 更新为
`−lr*g/(|g|+epsilon)`。在非零梯度且 epsilon 相对很小时，接近逐坐标符号更新。
若 candidate/reference 的有效坐标和更新幅度相同，仅部分坐标符号不同，
变号比例 p 对应 `Q=4p`、`β=−2p`。这是限定条件下的推导，
不是有限精度实际参数写入的无条件公式。

新 RMSNorm 数据记录了每个状态都重置 moments。上述推导因此是值得检验的解释，
但现有 X/B/A 的恒等式不能独立证明发生了多少符号翻转，也不能证明真实训练
中偏差持续或造成 loss 分叉。完整验证还需相同状态的梯度变号统计、实际
update 写入与非零 moments 对照，不能由本代数关系代替。

自动汇总现在额外保留由原坐标量重建的 candidate/reference 更新能量比例。
它是解释现有 Q 和 β 所需的量，不增加新的判定阈值或修改历史统计协议。
