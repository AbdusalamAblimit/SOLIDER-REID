# 实验 exp412：PSGC（Pose-Semantic Gradient Completion）

## 动机

exp411 correct 相对 clean D0 自然上涨，但 zero-owner 的 e120 mAP/R1 反而更高，说明性能来自等支持身份集合
排序，而不是五槽 owner multiplicity；wrong-RGB 到 e90 也没有显示正确 RGB/PID 绑定优势。因此 exp412 不再
修改 owner 公式，也不把 CLIP feature 当 identity prototype、局部蒸馏 target 或 hard-pair scorer。

zero-owner 仍暴露一个训练对象缺口：PK batch 中同一 PID 的四张图共享身份监督，但某个身体部位在部分图中缺失、
低置信或被背景污染时，四张图仍平均承担该部位的反向路由系数。PSGC 将“单图补 feature”改写为“同 PID 内补监督
机会”：pose 定位当前增强视图的五个身体槽，CLIP 只给每槽一个身份无关的 visible-vs-occluded 语义标量，二者共同
决定每个 PID×槽的固定路由系数预算应由哪些图承担。

## 核心假设

保持 zero-owner 的 forward、global descriptor、CE、集合排序 loss 和 D0 pose loss 数值完全不变，只把身体槽的
身份梯度从同 PID 内被 `(pose visibility, CLIP semantic reliability)` 同时支配的视图转给 Pareto front 视图，可以
让共享 backbone 更稳定地学习跨遮挡身份证据。CLIP 不提供身份坐标，pose 不直接重写 feature，测试仍为原单 RGB
descriptor；若 correct 严格胜 zero-owner、pose-only 与文本错位，才说明 pose 与 CLIP 的联合监督路由具有独立价值。

## 技术方案

### 1. 固定宿主与单变量

- 宿主固定为 sealed exp411 zero-owner：Swin-Tiny、batch64、`P×K=16×4`、seed1234、learned classifier、
  三 support 全身份集合排序、D0 pose loss、训练 recipe 与 eval 全部不变；
- `PCMPSR_CONTROL_MODE=zero_owner`，永久不恢复五槽 owner multiplicity；
- PSGC 只新增训练期 backward-only gradient router，默认关闭时 state、forward、loss 与 RNG 保持宿主 exact；
- exp412 使用 fresh config、fresh output 和独立 text-axis asset，不续训 sealed checkpoint。

### 2. 身份无关 CLIP 语义轴

复用 frozen exp411 region-isolated CLIP 五槽 visual cache `c[i,r]`，但不使用其 768 维身份外观方向。固定沿用
exp405 已冻结的五槽短语与四组 visible/occluded 模板，经同一 OpenCLIP ViT-L/14 text encoder 得到归一化原型
`t_vis[r]` 与 `t_occ[r]`。每图每槽只保留无温度、无阈值的标量：

`q[i,r] = <c[i,r], t_vis[r]> - <c[i,r], t_occ[r]>`。

该差值只表达“对应身体槽更像可见还是遮挡”，不进入 student feature 空间，不参与分类 logits，也不携带 PID
prototype。文本、模板、checkpoint 与 asset SHA 在首个正式臂前冻结，禁止结果后调 prompt、temperature 或 scale。

### 3. 同 PID×槽的 Pareto 路由系数预算

增强后 COCO-17 scores/valid 生成五槽 visibility `v[i,r]`。对 batch 内每个 PID 的四图和每个槽，候选图若不存在
另一图同时满足 `v[j,r] >= v[i,r]`、`q[j,r] >= q[i,r]` 且至少一项严格更高，则属于 Pareto front `F[p,r]`。

槽路由权重冻结为：

- `w[i,r] = 4 / |F[p,r]|`，若图 `i` 位于 front；
- dominated 或 CLIP-invalid 图为 `0`；
- 若该 PID×槽没有任何有效 CLIP 候选，则四图都回退为 `1`，即该槽严格保持宿主梯度。

因此每个有效 PID×槽始终满足 `sum_i w[i,r] = 4`，不引入 loss weight、阈值、top-k、margin、temperature 或
连续可调 scale。并列点全部保留，比较与回退确定性执行。

### 4. forward-exact 的身体槽梯度路由

用当前增强后的 pose 在最终 Stage-3 feature map 上渲染冻结五槽 hard-owner soft field `M[i,r,h,w]`，满足
`sum_r M <= 1`。空间梯度倍率为：

`G[i,h,w] = 1 - sum_r M[i,r,h,w] + sum_r M[i,r,h,w] * w[i,r]`。

接点固定为 `norm3` 后的 `outs[-1]`、`avgpool` 前，并且只作用 descriptor 分支；Stage-2 TAPF pose-loss 分支不
经过 router。最终池化前仅在训练期执行：

`G_cast = G.detach().to(device=X.device, dtype=X.dtype)`，

`X_route = X.detach() + G_cast * (X - X.detach())`。

其 forward 值与 `X` exact 相同，但 backward 对身体 token 乘 `G`；pose field 之外保持倍率 1。router 不新增参数、
buffer、随机数或 loss。`sum_i w=4`只表示同 PID×槽的路由系数预算守恒；不同图的 pose field 面积和上游梯度
本来就不同，因此不声称真实梯度向量或范数守恒。eval 强制禁用且不读取 CLIP、text asset 或 external pose。

### 5. 强反事实

所有臂共享 zero-owner 宿主、CLIP cache、pose field、预算守恒和训练 recipe，只改变 `q`：

- `correct`：对应槽的 visible-vs-occluded CLIP 文本差值；
- `pose-only`：所有 `q` 置常量，只由 visibility 决定 front；
- `q-only`：front 比较中的 visibility 置常量，只由正确 CLIP 标量决定 front；空间路由仍使用相同 pose field；
- `text-shuffle`：五组文本轴固定循环错位一槽，保留 visual cache、文本集合、计算量和标量分布来源；
- `zero-owner`：sealed exp411 zero-owner，无 PSGC；
- `clean D0`：sealed 原 batch-hard 基线。

首轮只训练 correct。只有 correct 自然 e120 同时严格胜 sealed zero-owner 与 clean D0 的 raw mAP/R1，才串行训练
pose-only、q-only 与 text-shuffle；correct 还须同时严格胜三者，才记为 `POSE+CLIP SCIENTIFIC GO`。

## 对照组

- sealed clean D0 e120：`57.6/67.7/80.8/84.6`；
- sealed exp411 correct e120：`58.8/70.1/82.1/85.8`；
- sealed exp411 zero-owner e120：`58.9/70.3/81.9/86.2`；
- exp411 wrong-RGB：当前自然运行中，只用于关闭旧 owner 归因，不作为 PSGC 的 identity-free text-axis 对照。

## 预期结果与裁决

正式训练前只做一次最小合同：

1. text asset checkpoint/template/shape/norm/SHA 完整，五槽 `q` finite、非恒定，correct 与循环错位至少改变一个
   真实 PK64 front membership；
2. 每个 PID×槽的权重和严格为 4，无有效候选时 exact 回退全 1；
3. correct 的 router dtype不变且`torch.equal(X_route,X)`，descriptor、score、CE、set loss、pose loss与
   PSGC-off exact，梯度非 exact 且 finite；
4. 默认关闭不增加 state，不改变四类 RNG；固定 MMPOSE-ABU 的真实 PK64 CUDA/AMP 取得一次真实 update；
5. GPU 空闲、fresh output 不存在、唯一正式 config 与 asset SHA 冻结。

合同 PASS 后立即启动唯一 correct seed1234/e120，不重复 preflight。e10/20/.../120 与 sealed zero-owner、clean D0
同 epoch 记录 mAP/R1/R5/R10；不按中间点早停。e120 raw mAP 或 R1 任一不严格胜 sealed zero-owner，则
`EXP412 PERFORMANCE NO-GO`，不调 prompt、front、budget、loss、batch 或 scale 救臂。性能 GO 后再跑两条 matched
control；任一 control 的 mAP/R1 不被 correct 同时严格超过，则只保留整体性能事实，不宣称 pose+CLIP 归因。

## 风险与失败解释

1. region-isolated CLIP 的 visible/occluded 差值可能仍主要是 generic/slot prior；text-shuffle 会直接暴露该问题。
2. Pareto front 可能几乎完全由 pose visibility 决定；pose-only 若不低于 correct，则 CLIP 无独立贡献。
3. 将梯度集中到容易视图可能削弱 hard-view 鲁棒性；若 correct 不胜 zero-owner，说明“补监督机会”方向不成立。
4. backward-only routing 不改变任一单步 loss 数值，收益只能来自长期优化轨迹；因此不得用训练前 loss 差或单点
   diagnostic 冒充性能证据。
5. gradient surgery、sample reweighting、Pareto selection 各自并非新原子；C 类创新只限“同 PID×解剖槽固定预算、
   pose×identity-free CLIP 二维 front、forward-exact 身体 token 梯度补全”的整体及其强反事实证据。
