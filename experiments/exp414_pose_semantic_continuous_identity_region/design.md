# 实验 exp414：PSCIR（Pose-Semantic Continuous Identity-Region Ranking）

## 动机

exp411 zero-owner证明，严格leave-one-position-out的三图身份支持集能够稳定改善clean D0；但exp411
pose×CLIP owner、exp412监督机会路由与exp413互补prefix均未建立联合归因。与此同时，PCVT、SKC、
PC-MSC、PSC-JEPA、CASD与CAVT已经从增强、一致性、点特征补全、预训练蒸馏、跨图teacher和donor运输等角度反复
说明：被遮挡图并不能可靠预测一个唯一的缺失身份纹理点。

PSCIR因此不再“补出一个点”，也不再给support加权或排序。它把同PID三张真实support的student descriptor视为
三个可观测身份状态，并只在pose×identity-free CLIP共同认可的两条跨视图边上允许连续插值，形成一条两段式
identity region。anchor直接对全部身份region排序；最终测试仍只使用原RGB global descriptor。

## 核心假设

遮挡图对应的身份状态通常不是一个可从单图唯一恢复的完整点，而是位于若干真实同ID观察之间的兼容区域。若两个
support在pose可见性与identity-free CLIP遮挡语义上呈互补状态，则它们之间的student-space线段比三个离散距离的
简单均值更可能覆盖anchor的合法身份变化；错误pose、错误文本轴或任意完整图会改变合法连边拓扑。

只有correct自然e120同时严格胜sealed zero-owner与clean D0的mAP/R1，才保留性能资格；随后还必须严格胜
pose-only、q-only、text-shuffle与all-edges matched controls，才能把收益归因于pose×CLIP定义的连续身份区域。

## 技术方案

### 1. 固定宿主与单变量

- 宿主固定为sealed exp411 zero-owner：
  - Swin-Tiny；
  - batch64，`P×K=16×4`；
  - seed1234；
  - learned CE、D0 pose loss；
  - 三图leave-one-position-out；
  - 原all-identity zero-owner set ranking；
  - 原训练recipe与eval global descriptor。
- 复用只读exp411 region-CLIP cache与exp412 frozen text-axis asset，不重建cache、不改prompt。
- PSGC与PSCCR关闭；不修改backbone forward、classifier、descriptor、score、batch、loss scale或测试路径。
- 新增默认关闭的`PSCIR_ENABLED`与`PSCIR_CONTROL_MODE`。
- PSCIR开启时，原triplet位置改为
  `0.5 * L_zero_owner + 0.5 * L_continuous_region`；两个已冻结目标等权，不新增可调loss系数。

### 2. 先完成严格LOO，再计算support内部状态

完全复用zero-owner的`support_indices[a,p]`。对每个anchor `a`与batch身份`p`，先排除与anchor相同类内位置，
得到恰好三张support `S[a,p]`；所有pose/CLIP序数与拓扑只能读取这三张图。被排除图（正类时即anchor）任何
`visibility/q/valid`修改都必须对对应region拓扑逐元素不变。

沿用增强后五槽pose visibility `v[i,r]`与冻结identity-free文本轴：

`q[i,r] = <c[i,r], t_visible[r]> - <c[i,r], t_occluded[r]>`。

在每个三图support内部独立计算严格序数：

`rank_v[i,r] = sum_j 1[v[i,r] > v[j,r]]`；

`rank_q[i,r] = sum_{j:valid_jr} 1[q[i,r] > q[j,r]]`。

q-dependent臂中，invalid图的`rank_q`固定为0，valid图只与valid peer比较；pose-only完全忽略q/valid，
q-only完全忽略visibility。text-shuffle将五槽文本轴循环错位一槽后重算q序数。

### 3. Pose×CLIP只定义连边，不提供feature内容

三张support共有三条无向边。correct的边互补度为：

`w(i,j) = sum_r |rank_v[i,r]-rank_v[j,r]| * |rank_q[i,r]-rank_q[j,r]|`。

该乘积要求同一槽在pose与CLIP两个身份无关证据轴上都发生跨视图状态变化；它不是feature相似度、PID proxy、
连续权重或loss scale。控制臂为：

- `pose_only`：`w=sum_r |Δrank_v|`；
- `q_only`：`w=sum_r |Δrank_q|`，只移除在线pose visibility轴；因q来自pose-defined region-isolated
  cache，不表述为完全pose-free；
- `text_shuffle`：错位文本轴后使用correct公式；
- `all_edges`：不做pose/CLIP拓扑选择，三条边全部进入连续region。

correct/pose-only/q-only/text-shuffle均选择权重最大的两条边，即三节点maximum-spanning tree。并列时按边两端
绝对batch index排序，取字典序最小者，保证无随机性。两条边必须覆盖三张support，禁止丢图、重复或跨PID。

CLIP与pose至此停止：它们不能进入student feature、线段插值系数、距离、梯度幅值或测试路径。region的端点内容
只来自三张support当前未detach的student global descriptor。

### 4. 连续身份region距离

对anchor descriptor `x`与一条support边端点`u,v`，计算到闭线段的欧氏距离：

`t = clamp(((x-u)·(v-u)) / max(||v-u||², eps), 0, 1)`；

`d_seg(x;u,v) = ||x - (u + t(v-u))||_2`。

两条合法边构成连续region。为避免hard-min只让一条边获得梯度，固定使用无温度参数的log-mean-exp soft union：

`d_region = -logmeanexp(-d_seg(edge_1), -d_seg(edge_2))`。

`all_edges`控制使用相同公式但包含三条边。禁止新增temperature、margin、projection、adapter或radius head。
线段端点与投影均在student空间、未detach；pose/CLIP拓扑index本身离散且无梯度。

### 5. 全身份region ranking与宿主保底

每个anchor得到16个身份region距离。沿用zero-owner的all-identity目标：

`L_region(a) = softplus(logmeanexp_{p != y_a}(d_region(a,y_a)-d_region(a,p)))`。

最终metric loss为：

`L_PSCIR = 0.5 * L_zero_owner + 0.5 * mean_a L_region(a)`。

其中`L_zero_owner`必须直接调用sealed原support/mean/listwise路径，不得重写。CE、D0 pose loss及其权重不变。
因此新变量只是在已经有效的离散身份集合旁加入一个由真实support端点形成、由pose×CLIP限定拓扑的连续身份region。

## 与旧路线的不可约差异

| 旧对象 | 已有做法 | PSCIR差异 |
|---|---|---|
| PCVT | 同图互补mask后均值一致性 | 不造伪视图，不要求两个partial均值等于full |
| SKC/PSC-JEPA/PC-MSC | 预测一个缺失feature或teacher point | 不预测缺失内容，只保留真实端点之间的可行区域 |
| CASD/CAVT | 跨图teacher、donor聚合或feature运输 | 不蒸馏、不搬运、不写回backbone |
| PCHM | pose×CLIP选择一条hard pair | 不替换单边；全部身份都由三support连续region参与listwise |
| PCMPSR owner | 给support multiplicity/权重 | 三support均为一个顶点，权重恒定 |
| PSCCR | 按语义顺序优化1/2/3离散prefix均值 | 不做prefix、不改support到达顺序；只改变允许插值的图拓扑 |

## 对照与裁决门

性能宿主：

- sealed zero-owner e120=`58.9/70.3/81.9/86.2`；
- sealed clean D0 e120=`57.6/67.7/80.8/84.6`。

首轮只运行`correct`。correct自然e120必须在mAP与R1同时严格胜zero-owner，且同时严格胜clean D0，才记
`PERFORMANCE GO`并依次运行matched controls。任一核心不严格胜即`PERFORMANCE NO-GO / CONTROLS NO-START`。

性能GO后，correct必须在mAP/R1同时严格胜：

1. `pose_only`；
2. `q_only`；
3. `text_shuffle`；
4. `all_edges`。

全部成立才记`POSE+CLIP CONTINUOUS-REGION SCIENTIFIC GO`。其中all-edges用于排除“任何同PID线段插值都有效”；
前三臂分别检验pose、CLIP与正确文本槽绑定。

## 唯一真实PK64合同

正式训练前只允许一次固定MMPOSE-ABU真实`16×4` CUDA/AMP合同；同一runner前半执行手算micro-oracle，后半执行
真实PK64，不另加preflight：

1. 默认关闭时模型state、forward、loss及Python/NumPy/Torch CPU/all-CUDA RNG与zero-owner exact；
2. LOO后再排名；修改被排除图的`v/q/valid`不改变对应三support的rank、边权与MST；
3. 手算覆盖负q、tie、部分/全部invalid、三种单轴control、text-shuffle与batch-index并列；
4. 每个MST恰有两条不同无向边、覆盖三support、无anchor self、无跨PID；
5. 线段距离手算覆盖内部投影、两个端点clamp、零长度边与finite backward；
6. `L_zero_owner`与sealed宿主bit-exact；
7. correct相对pose-only/q-only/text-shuffle/all-edges的拓扑与region distance改变率均非零；
8. 固定正确MST索引，只篡改未选候选边记录或其边权并保证仍未入选时，correct region distance不变；不把共享
   端点伪装成可独立修改的第三边几何；
9. isolated PSCIR loss finite，并使Stage-3/backbone梯度相对zero-owner发生非零改变；
10. production combined loss在原生GradScaler下取得一次真实Stage-3参数update。

任一门失败只允许修致命bug或变量混淆；合同PASS后不追加测试，立即冻结fresh formal并启动e120。

## 创新边界

连续/凸包式set metric、image-set recognition、metric learning、pose part与CLIP ReID原子均已有大量近邻，不能
分别声称新颖。当前arXiv精确查询未发现`person ReID + set-valued/box/cone embedding`直接同构结果，GitHub窄检索
也未发现pose×CLIP连续身份region实现；但网络查询不完整，且PFE/P2LR等概率不确定性、MVI²P多视图support和经典
convex-hull set classifier构成明确近邻。

因此只按`C-CLASS CONDITIONAL`推进：可争对象限于“严格LOO三support的student-space连续identity region，
pose×identity-free CLIP只确定maximum-spanning topology，并与sealed离散set ranking联合”。不作绝对首次声明，
投稿前必须补正式查新。

## 风险与失败解释

1. 三图只有三条边，correct与单轴MST可能频繁相同；合同若改变率为0则直接NO-START。
2. 负身份线段也可能插值靠近anchor，导致训练变难；若mAP下降，说明连续region扩大了错误身份支持。
3. 正类anchor可能不落在两段polyline附近；这正是连续region假设的可否证部分，不允许改成full convex hull救臂。
4. all-edges若不低于correct，收益只能归因于普通同ID插值，pose×CLIP拓扑主张失败。
5. pose-only或q-only若不低于correct，联合语义主张失败；禁止调prompt、temperature、edge公式或loss比例。
6. correct只胜D0但不胜zero-owner时，只能说明宿主保留了旧收益，PSCIR新增结构无价值。
