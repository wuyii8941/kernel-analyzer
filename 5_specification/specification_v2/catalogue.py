"""Catalogue v2: three generators with provenance. Nothing here asserts completeness from a file's existence.
G1 documents -> algorithm specs (per role: source, readings, reference kind, accepted domain, validation, known issues)
G2 mathematics -> relations derived by fixed rules (derive.py), each with method and status
G3 implementation paths -> path-equivalence axes enumerated from framework feature axes (provenance: EAGLE rule or doc)"""
import json
from collections import Counter

G1 = [  # role, source, readings, reference kind, accepted domain, validation, known issues (status carried from reviews)
 dict(id='CE', source='torch.nn.functional.cross_entropy 文档（2.x）', readings='ignore_index 与权重并存时的分母：按文档，忽略行不计', reference='逐点严格区间（Decimal）；全部忽略且 mean → NaN（集合语义：未定义）', domain='有限 logits；类别数 ≥ 2', validation='第一阶段独立复算；D3 符号证明梯度公式（n=3）', issues='无'),
 dict(id='AVGPOOL', source='torch.nn.functional.avg_pool 文档', readings='ceil_mode 下窗口越界的分母：R1/R2 两种读法登记', reference='精确有理数', domain='1D/2D；padding ≤ kernel/2', validation='第一阶段独立复算；伴随恒等式精确', issues='无'),
 dict(id='MAXPOOL', source='torch.nn.functional.max_pool 文档', readings='并列时返回集合', reference='集合（并列）', domain='同上', validation='第一阶段', issues='无'),
 dict(id='INDEXREDUCE', source='torch.Tensor.index_reduce_ / scatter_reduce 文档', readings='NaN 传播读法', reference='精确；amax/amin 并列为集合', domain='下标在范围内', validation='第一阶段；B020 由此发现', issues='无'),
 dict(id='ACCUM', source='任务定义：窗口内 token 平均', readings='无', reference='精确', domain='至少一个 token', validation='HF 问题召回', issues='无'),
 dict(id='ATTN', source='torch SDPA 文档；FlashAttention 的 window 约定', readings='causal 对齐两种；softcap；window=(left,right) 相对对齐后的对角线', reference='严格区间', domain='至少一个允许的 key（否则按声明拒绝或置零）', validation='v2 复核：Lq=1、Lk=5、bottom_right、window=(1,0)、V_j=j → 7/2', issues='v1 的 window 只支持 upper_left 且按查询局部坐标，已改'),
 dict(id='GQA', source='GQA 论文', readings='无', reference='精确映射', domain='H_q 整除 H_kv', validation='第二阶段', issues='无'),
 dict(id='PACK', source='任务定义：文档独立', readings='无', reference='精确 mask 与位置', domain='长度 ≥ 1', validation='D2：跨文档权重恒为零', issues='无'),
 dict(id='ROPE', source='RoFormer；两种布局', readings='rotate_half 与 interleaved', reference='严格区间', domain='偶数维', validation='D4：只依赖相对位置', issues='无'),
 dict(id='MATMUL', source='数学定义', readings='无', reference='精确', domain='形状匹配', validation='D5、D3 精确', issues='无'),
 dict(id='REDUCE', source='数学定义；torch 的 correction 参数', readings='var 的 correction', reference='精确', domain='非空', validation='D7 精确', issues='无'),
 dict(id='SOFTMAX', source='数学定义', readings='全 −inf 行按声明拒绝', reference='严格区间（200 bit）', domain='至少一个有限元素', validation='D4、D6、D3 符号证明（n=3）', issues='无'),
 dict(id='ACT', source='torch 激活函数文档', readings='GELU-erf 使用声明的高精度近似（不是严格区间）；ReLU 零点两种 profile（torch：0；math：次梯度区间）', reference='区间 / 声明近似 / 集合', domain='有限输入', validation='第二阶段', issues='v1 的 ReLU 零点集合 {0, upstream} 不对应任何文档，已改为两种 profile'),
 dict(id='GATHER', source='torch.gather / index_select 文档', readings='无', reference='精确', domain='下标在范围内', validation='伴随恒等式', issues='无'),
 dict(id='EMB', source='torch.nn.Embedding / EmbeddingBag 文档', readings='padding_idx 行梯度为零', reference='精确 / 区间（renorm）', domain='下标在范围内', validation='第二阶段', issues='无'),
 dict(id='MOE', source='Switch Transformer / Mixtral 的路由描述；容量按 token 顺序消耗', readings='并列时所有合法 top-k 集合均允许', reference='moe_forward_joint：按容量状态枚举全部允许的联合轨迹（有预算，超出则 not_established）；moe_forward 只给代表值并标 representative_only', domain='k ≤ E', validation='v2.3：两 token 并列、容量 1 → {(1,0),(1,3),(3,0),(3,1)}；无并列、容量 1 → [1,0]', issues='v2.2 的逐 token 枚举丢失容量状态，已改为联合枚举'),
 dict(id='NORM', source='torch.nn.LayerNorm / RMSNorm / GroupNorm 文档', readings='无', reference='严格区间', domain='方差分母非零', validation='D4/D5：LN 平移不变（任意 eps）、缩放不变（仅 eps=0）；RMSNorm 平移不变不成立（反例）', issues='无'),
 dict(id='BN', source='torch.nn.BatchNorm 文档（running 统计、momentum=None 的累计平均）', readings='训练用有偏方差归一、running_var 用无偏方差', reference='严格区间', domain='N ≥ 2（训练）', validation='v2 复核：首批后计数 1、running_mean=[2,12]、running_var=[2,8]', issues='v1 的批计数在通道循环内多加，已改'),
 dict(id='ADAM', source='torch.optim.Adam / AdamW 算法框', readings='amsgrad 的两种读法登记', reference='严格区间（sqrt）', domain='超参数在文档范围内；theta/grad 长度一致（否则拒绝），grad=None 按文档跳过', validation='D8：零有效梯度衰减；coupled decay 下不成立（反例）', issues='无'),
 dict(id='SGD', source='torch.optim.SGD 算法框', readings='无', reference='精确', domain='nesterov 需 momentum>0、dampening=0；theta/grad 长度一致（否则拒绝），grad=None 按文档跳过', validation='v2 复核：θ=1, g=0, lr=wd=1/10, maximize → 99/100', issues='v1 的 maximize 在 weight decay 之后取负，已改'),
 dict(id='RMSPROP', source='torch.optim.RMSprop 算法框', readings='无', reference='严格区间', domain='超参数在文档范围内；theta/grad 长度一致（否则拒绝），grad=None 按文档跳过', validation='v2.3：centered + weight_decay 两步可求值，第二步包围含 −0.68617（与 PyTorch 2.10 一致）', issues='v2.2 的区间类型缺平方，centered 路径第二步抛错，已改'),
 dict(id='SCHED', source='torch.optim.lr_scheduler；HF get_cosine_schedule_with_warmup', readings='无', reference='严格区间（cos）', domain='步数在范围内', validation='D5：对 base_lr 线性', issues='无'),
 dict(id='CLIP', source='torch.nn.utils.clip_grad_norm_ 文档（分母含 1e-6）', readings='无', reference='严格区间', domain='范数有限', validation='D5：不齐次（反例）', issues='无'),
 dict(id='SCALER', source='torch.amp.GradScaler 文档', readings='无', reference='精确状态机', domain='—', validation='第二阶段', issues='无'),
 dict(id='LABELS', source='任务定义：标签右移与计数', readings='无', reference='精确', domain='—', validation='第二阶段', issues='无'),
 dict(id='RANKMEAN', source='任务定义：跨 rank 的 token 平均', readings='无', reference='精确', domain='至少一个 token', validation='第二阶段', issues='无'),
 dict(id='CKPT', source='数学恒等：重算等于保存', readings='无', reference='比较两份结果的性质，不是独立参照', domain='两份结果长度一致（否则拒绝）', validation='—', issues='v2.2 的长度不一致被静默截短，已改为拒绝'),
 dict(id='CAST', source='IEEE 754-2019；OCP MX / NVIDIA FP8（E4M3FN 无 inf、最大 448；E5M2 有 inf、最大 57344）', readings='溢出：RNE → inf/NaN；定向 → 同号最大有限值；saturate=True 强制饱和', reference='精确', domain='全部有限输入', validation='test_oracle：与 numpy 4000 组一致；定向溢出与 FP8 边界逐项核对', issues='v1 定向舍入溢出与 E4M3 最大值错误，已改'),
 dict(id='EMA', source='torch.optim.swa_utils.AveragedModel、get_ema_avg_fn、get_swa_avg_fn 文档', readings='第一次更新为复制；n_averaged 属于状态', reference='精确有理数', domain='decay ∈ [0, 1]；参数长度一致', validation='phase3：SWA 等于算术平均、EMA 展开式（随机 50 组）；随机验证 200 组', issues='无'),
 dict(id='KVCACHE', source='任务定义：因果解码的增量输出等于同一序列的因果 prefill', readings='RoPE 在缓存前按绝对位置旋转', reference='严格区间（连续、分页两种 cache）', domain='至少一个位置', validation='phase3：decode = prefill（连续、分页、有无 RoPE）；按位置 0 旋转的变异必须失败', issues='无'),
 dict(id='KVCOMP', source='arXiv 2609.36322 §3.1 的压缩算法族（窗口 W、步长 S、负载映射、门控、聚合）', readings='只压缩完整窗口；因果可见性：窗口末位小于查询位置', reference='精确（均匀聚合）/ 严格区间（门控）', domain='L ≥ W', validation='phase3：平移步长等变（四种组合）；平移一位与按偏移编号参数下的窗口内置换给出反例；随机验证 200 组', issues='无'),
]

G3 = [  # path-equivalence axes: two implementation paths of the same framework whose declared semantics must agree
 dict(axis='编译', paths='eager 对 torch.compile/Inductor', provenance='EAGLE 规则 1', ours='K − G 两栏拆分；Inductor 生成的 Triton 在工具范围内'),
 dict(axis='布局', paths='连续对转置/步幅视图；batch-major 对 time-major', provenance='EAGLE 规则 10', ours='声明层精确比较'),
 dict(axis='dtype', paths='共同可表示范围内的两种 dtype', provenance='EAGLE 规则 12', ours='CAST 规格给出可表示集合，精确判定'),
 dict(axis='稀疏', paths='稠密对 COO 稀疏表示（含重复下标）', provenance='EAGLE 规则 8；torch.sparse 文档', ours='spec_sparse：spmm 等于稠密矩阵乘、coalesce 与条目次序不改变稠密值（精确）；工具侧稀疏 kernel 尚不在 DSL 范围'),
 dict(axis='批大小', paths='整批对拆批', provenance='EAGLE 规则 15；D7', ours='CE sum 精确；BN 训练模式不成立（反例已记录）'),
 dict(axis='cache', paths='prefill 对 decode；连续 cache 对分页 cache', provenance='任务定义', ours='spec_kvcache：逐位置相等（严格区间），RoPE 位置错误的变异必须失败'),
 dict(axis='检查点', paths='重算对保存', provenance='数学恒等', ours='CKPT 性质'),
 dict(axis='分布式', paths='单卡对跨 rank 平均', provenance='任务定义', ours='RANKMEAN 精确'),
 dict(axis='保存加载', paths='保存后加载再继续对直接继续', provenance='EAGLE 规则 16；state_dict 文档', ours='spec_state_io：Adam 与 EMA 恢复后等于直接继续；遗漏 t 或 n_averaged 的变异必须失败'),
 dict(axis='内置对外部', paths='内置 padding/mask 对显式构造', provenance='EAGLE 规则 2、14', ours='PACK、ATTN 的 mask 精确比较'),
]

if __name__ == '__main__':
    D = json.load(open('derivation_log.json'))
    c = Counter(x['reference'].split('；')[0].split('（')[0] for x in G1)
    M = dict(roles=len(G1), roles_implemented=sum(1 for x in G1 if x['reference'] != '待写'), roles_pending=sum(1 for x in G1 if x['reference'] == '待写'),
             roles_with_v2_fix=sum(1 for x in G1 if '已改' in x['issues']), derivation_rules=D['summary']['rules'], derivation_checks=D['summary']['checks'],
             derivation_holds=D['summary']['holds'], derivation_fails=D['summary']['fails'], derivation_by_method=D['summary']['by_method'],
             grid_cells=D['summary']['grid_cells'], grid=D['summary']['cells'], path_axes=len(G3), path_axes_executable_now=sum(1 for x in G3 if not x['ours'].startswith('待')))
    json.dump(dict(G1=G1, G3=G3, counts=M), open('catalogue.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(M, ensure_ascii=False))
