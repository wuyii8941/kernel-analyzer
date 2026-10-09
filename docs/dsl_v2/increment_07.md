# DSL v2 第七个增量：登记（2026-10-09）

分支 `dsl-v2`，工具 4.0（开发中）。对应 rc3 正式版 04 的 W4（CAS 与同步：「用有限事件关系、偏序归约、协议摘要或声明顺序；
未能证明进展是证明缺口，不等于非法」），依据 02 §6.3、§6.9。本登记写于实现之前。

## 1. 选这组的理由

- 官方教程 05（layer-norm）的反向 kernel `_layer_norm_bwd_dx_fused`：同组的 program 用 CAS 自旋锁（`while atomic_cas(Lock, 0, 1) == 1`）保护
  对 DW/DB 部分和的读改写，最后 `atomic_xchg(Lock, 0)` 释放。现在报为跨 program 竞争，参照未建立。
- 官方 `test_atomic_cas`：2000 个 program 在锁内对同一组元素加 1，10 次启动全部中止（「循环条件无法判定」：CAS 的返回值在有竞争时记为未建立）。
- Liger 等外部库的 layer-norm / RMSNorm 反向沿用同一模式。

## 2. 做法

| 项 | 内容 |
| --- | --- |
| A 串行化语义 | 含有竞争 CAS 的启动按「临界区的某个串行顺序」求值。program 依次执行，CAS 看到前面 program 留下的状态，所以自旋锁在第一次尝试就获得。这只是一个合法串行化的参照 |
| B 次序证据 | 同一启动再按逆序（program 号从大到小）从启动前的内存重算一遍，逐元素比较两遍的写入。两遍都建立且区间重叠：取两者的凸包，并记声明前提「启动结果不依赖竞争 CAS 的次序（程序序与逆序一致）」。不重叠或状态不同：该元素记未建立，原因写「依赖 CAS 次序」。两种次序一致只是证据，不是对所有串行化的证明，所以登记为声明前提（DECLARED_PREMISE），交审阅方 |
| C happens-before（向量时钟） | release（原子更新 sem 为 release / acq_rel）记录发布方程序、release 序号，以及发布方当时已获得的 happens-before 快照。acquire（CAS 成功或原子更新 sem 为 acquire / acq_rel，读到某个 release 写入的值）把发布方的 release 序号和快照并入自己的边，从而支持传递。普通 load 读到的写入若被这些边覆盖，不算竞争 |
| D 原子写入者 | 原子更新也记录写入的程序和 release 序号，这样普通 load 读到另一个程序原子写入的值（例如 layer-norm 的 Count）也能按 happens-before 判断 |
| E relaxed | sem 为 relaxed 的 CAS 不产生 acquire。锁内的普通访存没有 happens-before，按内存模型仍报竞争（未建立），不因为程序看起来像锁就放宽 |

不在本增量内：对所有串行化的证明（例如交换律证书）；超过两种次序的枚举；跨 kernel 启动的锁。

## 3. 预算

实现约半天；测试与捕获复跑约 1 h。

## 4. 预期（实现前写定）

- `test_atomic_cas` 的 serialized_add：sem 为默认（acq_rel）或 acquire 时，data 参照完整（全部为 2000），记声明前提；sem 为 relaxed 或 release 时，
  锁内读写报竞争、参照未建立。change_value（单 program）完整。
- 教程 05 的 `_layer_norm_bwd_dx_fused`：DX 完整；DW、DB（锁内累加的部分和）完整并记声明前提；Lock、Count 完整。
- 逐签名测试：锁内交换（累加）在两种次序下一致，完整；锁内「先到者写入自己的编号」两种次序不同，未建立；relaxed 锁报竞争；
  不释放锁的 program 让后续 program 永远自旋，报未建立（循环不终止），不挂起。
- 回归：已有全部测试不变。
