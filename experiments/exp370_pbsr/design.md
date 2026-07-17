# 实验 exp370：PBSR 共享路由分解—重组

## 当前状态

**PRE-TRAIN / DESIGN FREEZE 前**。完成代码级查新后允许进入隔离实现与无训练审计；尚未授权正式训练启动。

用户明确禁止 Claude，本实验以代码级相关工作核对、Codex 审查、确定性单元测试和 smoke test 代替旧流程中的 Claude 审查。该覆盖仅适用于本实验。

## 动机

现有 LGPA 的有效信息主要来自姿态空间监督，而不是 CLIP 文本语义；但它把部位特征作为终端分支，再通过 concat/MaxSim 参与检索。这带来三个问题：

1. 方法价值与 part matching、描述子扩维纠缠，难以说明训练端到底改进了什么；
2. 部位分支若 detach，只能读 backbone，不能改善主表征；若不 detach，`exp320` 又显示辅助损失会严重污染 backbone；
3. PAFormer、TSD、ProFD 已覆盖 pose-supervised part query 或 part decoder，继续修改 query/prototype 不足以形成新方法。

本实验改写研究对象：**人体结构不再是最终拿去匹配的一组局部描述子，而是重组标准全局表征的内部路由基。**

## 核心假设

若结构槽只从空间特征读取信息，它仍是一个附加 part branch；若把槽内聚合的结构信息沿同一分配关系写回空间 token，并且只用 identity loss 训练写回结果，那么姿态监督可以在不污染 backbone 的前提下改善主 global descriptor。

该假设必须由以下因果链支持：

```text
正确 pose 路由监督
  > uniform / shuffled 路由监督
  且 coupled write-back
  > read-only / independent write
  且最终 global
  > 相同 baseline global
```

任一关键不等式不成立，都不能把 PBSR 写成有效机制。

## 技术方案

### 1. 输入与结构路由

设 backbone 最后一层空间特征为 `X ∈ R^(B×N×C)`，使用 `K=6` 个身体结构槽和一个 background/reject 槽。槽为 learnable embeddings，不使用 CLIP 文本初始化。

经低维投影后计算路由：

```text
A = softmax(Q K(X)^T / sqrt(d), dim=token)       # B×(K+1)×N
```

`A` 是模型从图像特征自行预测的路由矩阵。pose heatmap **不加到 logits 中**，只在训练时构造监督目标 `T(P)`。

### 2. 结构分解（read）

```text
S = A V(X)                                       # B×(K+1)×d
S_body = SlotMixer(S_body)                       # 轻量 self-attn + FFN
```

SlotMixer 只表示通用的结构槽间信息交换，不以 GCN 或显式骨架图作为创新。

首验中 SlotMixer 和所有 PBSR 投影均不使用 dropout，避免额外随机采样改变与 baseline 配对的训练随机流。

### 3. 耦合结构重组（write）

写回不得重新学习一套独立 cross-attention。对 `A` 在 slot 维归一化得到：

```text
B[j,k] = A[k,j] / sum_l A[l,j]
U = W_message(S_body)
delta_X[j] = sum_k B[j,k] U[k]                   # background message 固定为 0
G[j] = 2 * sigmoid(MLP(LN(X[j])))
X_refined[j] = X[j] + tanh(alpha) * G[j] * W_up(delta_X[j])
```

`alpha=0` 初始化，从而 `X_refined == X`。`W_up` 使用正常小初始化；零初始化门保证 baseline 精确退化，同时让第一步 identity gradient 先学习是否打开写回。

“共享 A”是机制门禁：read 和 write 必须使用同一对应关系。若 write 使用独立 attention，则只能作为 `independent-write` 对照，不能作为最终方法。

`G[j]` 使路由消息必须与写入位置的原始 token 发生局部非线性交互。若省略该项，写回后立即线性 GAP 会在代数上接近 slot message 的加权求和，容易退化为普通分支融合，不能充分支持“空间重组”的主张。

### 4. 标准 global 输出

Swin/ResNet 直接对 `X_refined` 做与 baseline 相同的 global average pooling。普通 ViT 的主描述子仍保留 CLS，并增加由 `mean(X_refined-X)` 产生的同维零初始化 residual；不在最终描述子中 concat slots。

主实验 Swin-Tiny 中：

```text
g_refined = GAP(X_refined)
BN(g_refined) → ID classifier
triplet(g_refined)
```

最终评测只返回 `g_refined`，维度、距离函数和 evaluator 与 baseline 完全相同。禁止用 equal-concat、MaxSim、re-ranking 或 matching 增益作为 PBSR 结果。

### 5. 姿态监督与梯度防火墙

将 COCO-17 heatmaps 合并为六个 body groups 和一个 background target，归一化得到 `T(P)`。assignment loss 为：

```text
L_route = KL(T(P) || A_stopbackbone)
A_stopbackbone = Router(Q, X.detach())
```

同一 router 参数用于主前向和监督前向，但监督前向的 `X` 必须 detach。因此：

- `L_route` 更新 router 的 query/key 投影；
- `L_route` 对 backbone 梯度严格为 0；
- `L_id + L_tri` 通过 `X_refined` 更新 backbone、router 和 write-back；
- 不增加 per-part ID/triplet loss，避免复现 `exp320` 的梯度冲突；
- eval 完全不计算 `T(P)` 或读取 heatmap。

### 6. train/test 一致性

训练和推理的表征前向均为 `A(X) → S → X_refined → global`。训练期 pose 只产生额外 loss target，不进入表征前向。因此不存在“训练用真 pose bias、测试无 pose bias”的分布切换。

## 配置开关草案

所有开关默认关闭：

```yaml
MODEL:
  POSE_PBSR: False
  POSE_PBSR_NUM_SLOTS: 6
  POSE_PBSR_DIM: 256
  POSE_PBSR_NUM_HEADS: 4
  POSE_PBSR_ROUTE_WEIGHT: 0.5
  POSE_PBSR_SLOT_MIXER: True
  POSE_PBSR_WRITEBACK: True
  POSE_PBSR_COUPLED_WRITE: True
  POSE_PBSR_SUPERVISION: correct   # correct | uniform | shuffled | none
SOLVER:
  AMP_INIT_SCALE: 1024.0           # B0/P0 公共设置；避免首批 AMP 溢出
```

`POSE_PBSR=False` 时不得实例化模块，不得改变 RNG 流、返回结构、loss 路径或 baseline 行为。

3090 真实单批次审计发现，历史默认 AMP 初始 scale `65536` 在纯 global baseline 和 PBSR 上都会使首批 backbone 梯度溢出；这不是 PBSR 特有问题。exp370 将 B0/P0 的公共初始 scale 冻结为 `1024`，默认配置值仍保留 `65536`，因此不改变其他实验。两臂必须使用同一个 scale，禁止只给 P0 降 scale。

PBSR 开启时，模块构造必须保存并恢复全局 CPU RNG 状态，确保新增参数初始化不推进后续训练随机流。shuffled 对照使用固定的 batch roll 或独立局部 generator，不得消耗训练主 RNG。正式 manifest 还需在模型构造后重设一次 seed，使 B0/P0 的 sampler、drop-path 和数据增强尽可能成对。

## 与对照组的单变量关系

### Phase A：无训练机制审计

1. 开关关闭时与 baseline 前向逐元素相同；
2. `alpha=0` 时 PBSR 开启也与 baseline global 逐元素相同；
3. `L_route.backward()` 后 backbone grad 为 0，router grad 非 0；
4. `L_id.backward()` 后 write gate、router、backbone 均在门打开后有有限梯度；
5. eval 在 `pose=None`、correct pose、random pose 三种输入下输出逐元素相同；
6. `A`、`B` 行列归一性质、shape、AMP 和无 NaN 检查通过。

### Phase B：Swin-Tiny 严格 kill-switch

所有训练保持相同 seed、batch size、backbone、输入尺寸、数据增强、optimizer、scheduler、epoch 和 global loss。

| Arm | 唯一变化 | 回答的问题 |
|---|---|---|
| B0 global baseline | PBSR off | 主对照 |
| L0 单向 LGPA | 现有 LGPA，报告 global/equal-concat 但不把 matching 算入 PBSR | 是否超过历史单向方法 |
| P0 PBSR full | correct supervision + coupled write | 主方法 |
| P1 read-only | writeback off | 槽读取本身是否足够 |
| P2 independent-write | 用 `K(X)·S^T` 重新计算 token→slot 权重，但复用现有投影、参数量不变 | 收益是否来自共享耦合路由而非加参数 |
| P3 uniform | route target 改 uniform | 人体结构监督是否必要 |
| P4 shuffled | batch 内错配 pose target | 正确 image-pose 对应是否必要 |

优先顺序：先跑 B0/P0/P1/P4；若 P0 未明显超过 B0，则直接 NO-GO，不补小变体。只有 P0 有明确正向后再补 L0/P2/P3。

### Phase C：扩展验证

只有 Phase B 通过后才执行：

1. Swin-Tiny 三 seed；
2. ResNet-50、普通 ViT、Swin-Tiny 各至少同 seed A/B；
3. Occluded-Duke 为主，Market1501/其他 PSG 原文数据集用于通用性和 SOTA 表；
4. 报告参数量、训练期开销、推理期开销；推理不含 pose estimator；
5. 机制分析：route 对齐、write residual norm、正确/错误 pose 差值、遮挡程度分层。

## 成功门槛

Phase B 的最低 GO 条件：

1. P0 相对独立训练 B0 有明确正向，而非只在同 checkpoint 内切 descriptor；
2. P0 优于 P1，证明 write-back 有独立价值；
3. P0 优于 P4，证明正确 pose-image 对应有因果价值；
4. evaluator 只使用同维 global descriptor；
5. 机制审计全部通过，无姿态推理依赖。

最终论文门槛：

- Swin-Tiny 三 seed mean 约 `+0.8～1.0 mAP` 或以上；
- correct pose 稳定优于 uniform/shuffled；
- 至少两类不同架构显示同方向；
- 不依赖 matching、GCN、CLIP、re-ranking；
- 若不达门槛，如实记录负结果并停止继续堆 PBSR 小变体。

## 风险与失败解释

1. **组合显而易见风险**：PAFormer + PAT + PGFL-KD 的组合可能被认为显然。必须由共享路由耦合、梯度防火墙和只输出 global 的机制/证据共同支撑，而不能只讲模块列表。
2. **零门导致慢启动**：初期 identity gradient 主要更新 `alpha`；route loss 先训练 router。需监控 alpha、route KL、write residual norm，不能凭早期 mAP 误杀。
3. **低分辨率路由**：最终 Swin feature map可能过粗。首验不改 stage，若负不得立即通过换层堆变体救场。
4. **background 主导**：background/reject slot 可能吸收大量 token。需记录每槽质量、entropy、dead-slot ratio，但不先加复杂平衡 loss。
5. **监督噪声**：scene heatmap 会混入旁人。主实验优先 target-person heatmap；若训练数据没有 target-only 标注，必须在 monitor 明示。
6. **加参数假阳性**：P2 参数匹配和 P1 write-off 是必要对照。
7. **历史梯度冲突复现**：任何 pose/part auxiliary gradient 进入 backbone 都视为阻塞 bug，不允许通过调低权重掩盖。

## 论文叙事边界

论文不再叙述“在 backbone 的什么位置注入 pose”，而叙述：

> 姿态不是推理时的第二输入，也不是最终用于匹配的局部描述子。它只在训练时监督一个内部结构路由，使模型先把空间证据分解到人体结构槽，再沿同一对应关系将结构化证据重组回标准全局表征。

GCN 可保留在代码中作为历史资产，但不进入方法主图和贡献。Matching 明确为既有检索策略，不作为创新。CLIP 文本语义从 PBSR 主机制中删除。
