# 实验 exp413：PSCCR（Pose-Semantic Complementary Coverage Ranking）

## 动机

exp411 zero-owner说明“三个等支持视图对全部身份集合排序”本身有效，但pose×CLIP owner multiplicity没有归因；
exp412又表明把身体token梯度集中到可靠视图会降低mAP。两条证据共同指出：不能再让外部证据接管identity geometry、
support权重或难图梯度，但可以把已证有效的完整支持集改写为**逐步获得互补证据时都要正确排序**的训练对象。

PSCCR不丢弃任何support、不搬运feature、不蒸馏CLIP坐标，也不改变backbone forward。pose与identity-free CLIP只给
同PID三图support一个确定性互补覆盖顺序；student global descriptor分别面对长度1/2/3的prefix做all-identity
ranking。长度3严格等于sealed zero-owner，因此新变量只是在同一完整集合上增加两个部分支持排序约束。

## 核心假设

遮挡ReID的query在gallery证据尚不完整时也必须取得正确全列表排序。若先进入prefix的support分别覆盖不同可靠身体槽，
长度1/2的身份集合距离可迫使student学习“少量但互补的同ID证据”而不是只在三图平均后抵消错误；长度3同时保留
zero-owner已验证的完整支持结构。只有correct在mAP与R1同时严格胜zero-owner及三条matched control，才说明
pose与CLIP共同定义的coverage chain有独立价值。

## 技术方案

### 1. 固定宿主与单变量

- 宿主固定为sealed exp411 zero-owner：Swin-Tiny、batch64、`P×K=16×4`、seed1234、learned CE、D0 pose loss、
  三图leave-one-position-out支持、all-identity logmeanexp公式、训练recipe和eval全部不变；
- 固定`PCMPSR_ENABLED=True`、`PCMPSR_CONTROL_MODE=zero_owner`，不恢复失败的owner multiplicity；
- 新增默认关闭的`PSCCR_ENABLED`与`PSCCR_CONTROL_MODE`，开启时只用coverage-chain ranking替换zero-owner set ranking；
- 复用只读exp411 region-CLIP cache与exp412 text-axis asset，禁止重建、改prompt或改cache；
- PSGC关闭，不修改model forward、descriptor、score、CE、pose auxiliary或测试路径。

### 2. leave-one-position-out之后的严格序数

先沿用zero-owner的occurrence位置：对每个batch query anchor `a`，每个身份`p`都排除与`a`相同类内位置的图，得到
三图support `S[p,a]`。所有序数必须在这个三图support内部重新计算，禁止先用K=4排名再排除；因此被排除图（正类时
即anchor）的pose/CLIP/valid不可能影响链。沿用exp411的增强后pose visibility `v[i,r]`，并由exp412冻结文本轴计算：

`q[i,r] = <c[i,r], t_visible[r]> - <c[i,r], t_occluded[r]>`。

对每个`(p,a,r)`，只在`S[p,a]`三图内部计算严格序数：

`rank_v[i,r] = sum_{j in S[p,a]} 1[v[i,r] > v[j,r]]`，

对依赖q的correct/q-only/text-shuffle臂：

`rank_q[i,r] = sum_{j in S[p,a], clip_valid[j,r]} 1[q[i,r] > q[j,r]]`。

若`clip_valid[i,r]=False`，则该图的`rank_q[i,r]`与所有q-dependent `u[i,r]`固定为0；valid图只与valid peer比较，
不把invalid的伪q值纳入序数。correct可靠度为`u=min(rank_v,rank_q)`；pose-only完全忽略`clip_valid/q`并令
`u=rank_v`；q-only完全忽略visibility并令`u=rank_q`。序数仅取`0..2`，不设阈值、温度、连续权重、top-k或
可调scale；并列自然获得相同序数。即使某槽全部CLIP-invalid，三support仍会按其他槽覆盖与最终batch-index并列规则
全部进入链，不设置可调fallback。

### 3. 无丢弃互补覆盖链

排除同位置图既保证各身份等支持，也构成严格的support-only边界；真正loss query始终是anchor `a`的student global
descriptor。对任意被排除图修改`v/q/valid`，`S[p,a]`的序数与chain必须完全不变。

对任意已选prefix `A`定义五槽覆盖：

`C(A) = sum_r max_{i in A} u[i,r]`，并令`C(empty)=0`。

从三support中每步选择使`C(A union {i}) - C(A)`最大的图；并列时取最小绝对batch index。选中图从候选删除，直到
三图全部进入，得到严格排列`pi[p,a,1:3]`。因此support不丢失、不重复、不加multiplicity；`C(prefix1) <=
C(prefix2) <= C(prefix3)`确定成立。所谓complementary只指五槽最大可靠度覆盖，不声称贪心/submodular原子新颖。

### 4. 逐前缀all-identity ranking

令`D[a,i]`为student global descriptor的原欧氏距离。对`k=1,2`：

`d_k(a,p) = (1/k) * sum_{t=1..k} D[a, pi[p,a,t]]`。

`k=3`不按排列后的浮点顺序重新求和，而显式复用sealed zero-owner原`support_indices`与原均值路径计算`d_3`，保证
distance与loss bit-exact。每个prefix沿用原all-identity目标：

`L_k(a) = softplus(logmeanexp_{p != y_a}(d_k(a,y_a) - d_k(a,p)))`，

最终`L_PSCCR = mean_{a,k in {1,2,3}} L_k(a)`。CE权重、triplet-loss权重位置和D0 pose loss均保持宿主配置；
不新增margin、temperature、loss weight或参数。query/难图始终保留完整梯度，CLIP只决定离散support次序。

### 5. matched controls

- `correct`：`u=min(rank_v,rank_q)`；
- `pose-only`：`u=rank_v`，其余链与loss完全相同；
- `q-only`：`u=rank_q`，其余链与loss完全相同；
- `text-shuffle`：五槽visible/occluded文本轴固定循环错位一槽，再计算`rank_q`与`u`；
- `zero-owner`：sealed exp411三support单次完整集合排序；
- `clean D0`：sealed原batch-hard基线。

首轮只运行correct。三control只在correct性能GO后依次串行运行，禁止为负结果补跑归因臂。

## 对照组与性能门

- sealed zero-owner e120：`58.9 mAP / 70.3 R1 / 81.9 R5 / 86.2 R10`；
- sealed clean D0 e120：`57.6/67.7/80.8/84.6`；
- exp411 correct与wrong-RGB仅作为失败归因背景，不作PSCCR宿主。

correct自然e120必须在mAP与R1同时严格胜sealed zero-owner，才记`EXP413 PERFORMANCE GO`并启动controls；任一不
严格胜即`EXP413 PERFORMANCE NO-GO / MATCHED CONTROLS NO-START`。性能GO后，correct还须在mAP/R1同时严格
胜pose-only、q-only、text-shuffle三者，才记`POSE+CLIP SCIENTIFIC GO`。

## 一次性真实PK64合同

正式训练前只允许一次固定MMPOSE-ABU真实PK64 CUDA/AMP合同：

1. 默认关闭时state、forward、loss及Python/NumPy/Torch CPU/CUDA RNG与zero-owner exact；
2. 每个chain是原三support的严格排列，无重复、无跨PID、无anchor self；
3. coverage对prefix单调不降，确定性并列规则成立；
4. prefix3的set distance与loss对sealed zero-owner bit-exact；
5. 同一runner前半用手算micro-oracle逐元素核对三图strict rank方向、负q、tie、部分/全部invalid、greedy
   batch-index并列，以及correct/pose-only/q-only/text-shuffle的预期链；
6. 任意修改被排除图的`v/q/valid`，对应support的rank、coverage与chain bit-exact不变；
7. runner后半真实PK64中，correct相对pose-only、q-only、text-shuffle的链改变率均非零；
8. isolated PSCCR梯度finite且进入Stage-3/backbone，并与zero-owner梯度不同；
9. production combined loss使用原生GradScaler取得一次真实参数update。

任一机械门FAIL只修致命实现bug/变量混淆；PASS后不追加preflight，立即建立fresh formal/output并正式e120。

## 风险与失败解释

1. strict ordinal在LOO三support内分辨率只有三级，链可能大量并列；若control改变率为0，机制直接NO-START。
2. greedy prefix可能优先选择易图而非真正互补图；pose-only或q-only不低于correct即联合语义归因失败。
3. 长度1/2集合对负身份也使用同样排序，可能放大单图噪声；correct不胜zero-owner即说明部分支持约束伤害完整排序。
4. prefix3虽保留zero-owner项，但三prefix等权平均会改变总体优化对象；这正是唯一核心变量，不能通过调prefix权重救臂。
5. 当前在线检索受网络超时限制；创新只按C类窄候选推进，投稿前必须补正式查新，不能做绝对首次声明。
