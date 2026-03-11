# 创新点头脑风暴 — Phase 2: Pure Pose Heatmap

## 前轮教训 (Phase 1, 33 experiments)
- ViTPose visibility 向量不够可靠（AP 相关性仅 0.237）
- 中间层 visibility modulation 有害（破坏预训练空间结构），但这是 visibility 特有问题
- PCFC alpha suppression 是该框架特有的脆弱平衡点
- NFC test-time 方法有效但不算训练端创新
- **关键结论**: 不要用 visibility 向量，用原始 pose 热图

---

## Phase 2 实验总结 (exp001-exp021, 21 个实验)

### 已证实的核心发现

**1. Backbone Injection > Post-hoc Pooling**
- PSG (特征形成阶段注入 pose) +1.7% mAP，仅 102K params
- Part Pooling (特征形成后用 pose 选择) +0.9% mAP，~2.6M params
- PFM (后置调制) 中性效果
- **结论**: 让 backbone 在特征提取过程中知道人体结构，比事后选择更有效

**2. PSG 对空间级干扰敏感，但通道级正交不干扰**
- PSG + PAB Combo: ❌ (-0.7% vs PSG-only)
- PSG + Part Pooling: ❌ (-0.6% vs PSG-only)
- PSG + Part Supervision (global test): ❌ (-0.7% mAP, -2.1% R1)
- PSG + PCG (通道级 gate): 🟡 持平 (mAP 58.0%, -0.3% vs PSG-only)
- **核心瓶颈**: 在同一个 Stage 3 上叠加任何模块都会梯度干扰

**3. 复杂度越高，效果越差**
- 有效方法排序: PSG(58.3%) > PCG-only(57.8%) > PRA(57.8%) > Part Pooling(57.5%) > PAB(57.4%) > PXA(57.3%) > CAPSG(57.2%)
- PXA/CAPSG 最复杂，效果最差
- **PSG 的极简性就是它的优势**

**4. PSG 跨数据集/backbone 均有效（4090 验证）**

| 数据集 | Backbone | PSG mAP 提升 |
|--------|----------|-------------|
| Occluded-Duke | Swin-Tiny | +1.7% |
| Occluded-Duke | Swin-Small (lr4) | +2.0% |
| Market-1501 | Swin-Tiny | +0.8% |
| Market-1501 | Swin-Small (lr4) | +0.6% |

### 关键教训
- **梯度干扰是核心瓶颈**: 21 个实验中所有在 PSG 基础上"加东西"的尝试都失败了，原因是同一个 Stage 3 内的模块共享梯度流
- **要突破 PSG，必须用独立的处理路径**，避免梯度干扰
- **可以添加大模块**: 用户确认不限于轻量模块，可以加 ResNet 分支、Decoder、GCN 等

---

## Phase 3: 新方向候选 (2025.03.11 Web 搜索 + 文献调研)

### ★ 方向 K: 双分支架构 — Pose-Guided Dual-Stream (PDS)
**优先级: ⭐⭐⭐⭐⭐**

```
Input → Swin Stage 1-2 (共享)
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
  Stage 3-A           Stage 3-B (独立权重)
  + PSG               + Pose Part Processing
    ↓                   ↓
   GAP              Part Pooling (基于 heatmap)
    ↓                   ↓
 Global Feat        Part Feats
    ↓                   ↓
 ID + Triplet      Part ID + Part Triplet
    ↓                   ↓
    └── concat (test) ──┘
```

- **核心想法**: 解决 exp008/014 暴露的梯度干扰问题。复制独立的 Stage 3 给 Part 分支。
- **为什么与前轮不同**: exp008 在同一 Stage 3 上做 PSG+Part，梯度干扰。PDS 用独立 Stage 3，各自优化。
- **代价**: ~6M 额外 params（Stage 3 ≈ 2 SwinBlocks × 768ch）
- **优势**: Part 分支可以集成更多 pose 操作（GCN、cross-attention），而不影响 Global 分支
- **参考**: PGFL-KD (ACM MM 2021) 三分支架构, FCFormer (TPAMI 2024) 双流设计
- **论文定位**: 主贡献之一 — "dual-stream pose-guided architecture with gradient-isolated part learning"

### ★ 方向 L: 关键点相对位置编码 (KP-RPE)
**优先级: ⭐⭐⭐⭐**

- **来源**: CVPR 2024 人脸识别 (Kim et al., "KeyPoint Relative Position Encoding for Face Recognition")
- **核心想法**: 将 Swin 的 attention bias 从像素相对位置改为关键点相对位置
  - 标准 RPE: bias(i,j) = table[xi-xj, yi-yj]
  - KP-RPE: bias(i,j) = MLP(dist(i,kp1),...,dist(j,kp1),...)
- **与 PSG 完全正交**: PSG 调制 feature 幅度（乘法），KP-RPE 调制 attention 路由（加法 bias）
- **与 exp012 PAB 的区别**: PAB 是单像素 spatial map，KP-RPE 编码 token 对之间的**结构关系**
- **参数量**: ~5-10K，几乎零开销
- **风险**: Swin 用 window attention (7×7 window)，12×4 feature map 上 window 划分可能限制效果
- **论文定位**: 与 PSG 叠加使用 — "spatial gating + structural attention routing"

### ★ 方向 M: 骨架图卷积特征传播 (Skeleton GCN)
**优先级: ⭐⭐⭐⭐**

```
PSG-enhanced Stage 3 features (12×4, 768ch)
              ↓
Bilinear sample at 17 keypoint locations → (17, 768)
              ↓
Skeleton GCN (2-3 layers, COCO 19 bone edges)
              ↓
可见部位特征 沿骨骼边传播到遮挡部位
              ↓
Part Feat Pool → concat with Global
```

- **核心价值**: 遮挡补全 — 当下半身被遮挡时，GCN 沿"髋→膝→踝"传播上半身特征
- **参考**: Tran-GCN (IET 2025), skeleton action recognition 领域成熟技术
- **参数量**: ~3-4M (17 nodes × 768 features × 2-3 GCN layers)
- **与 PDS (方向 K) 的结合**: 可以作为 Part 分支的核心模块
- **风险**: 前 21 个实验显示 post-backbone 方法收益有限，但 GCN 提供了全新的结构推理能力
- **论文定位**: "skeleton-topology-aware feature propagation for occlusion recovery"

### 方向 N: ControlNet-Style 加法注入
**优先级: ⭐⭐⭐**

- **来源**: ControlNet (ICCV 2023), LLaMA-Adapter
- **核心想法**: 复制 Stage 3 为 "pose encoder branch"，处理 pose heatmaps，通过 zero-conv 加法注入主干
- **与 PSG 的区别**: PSG 是乘法 x*(1+gate)，ControlNet 是加法 x + zero_conv(pose_feat)
- **可以和 PDS 整合**: 就是 PDS 的 Part 分支不做独立 pooling，而是加法 inject 回 Global 分支
- **参数量**: ~6M（完整 Stage 3 clone）或 ~100K（轻量 conv encoder）

### 方向 O: Pose Attention Supervision (PAS)
**优先级: ⭐⭐⭐**

- **来源**: PAFormer (arXiv 2024)
- **核心想法**: 不把 pose 热图作为输入，而是作为 attention 的**监督信号**。训练时让 attention map 匹配 pose heatmap 分布，推理时不需要 pose。
- **优势**: 零推理开销，改变 backbone 的内在表征
- **风险**: Swin window attention 使监督复杂化

### 方向 P: Pose-Guided Token Pruning
**优先级: ⭐⭐⭐**

- **来源**: PrATo (2025), HeatViT, Zero-TPrune (CVPR 2024)
- **核心想法**: 用 pose 热图计算 token 重要性，剪掉背景/遮挡物 token，只保留人体区域
- **效果**: 不仅加速推理，还从根本上消除遮挡物对特征的污染
- **与 PSG 的区别**: PSG 给所有 token 不同权重，token pruning 直接删除无关 token
- **挑战**: Swin window attention 对 token 数量有要求

### 方向 Q: 特征补全 (Feature Completion)
**优先级: ⭐⭐⭐**

- **来源**: FCFormer (TPAMI 2024)
- **核心想法**: 用 pose 热图识别遮挡区域（热图响应低），用 learnable tokens + decoder 重建遮挡区域特征
- **与 Skeleton GCN 的区别**: GCN 沿骨架边传播，Feature Completion 用 decoder 直接预测
- **可以结合 PDS**: 作为 Part 分支的特征补全模块

---

## 推荐实验路线

### Round 1: 大架构实验
1. **exp022: PDS (双分支)** — 最有潜力，解决核心梯度干扰问题
2. **exp023: KP-RPE** — 最轻量，与 PSG 正交，快速验证

### Round 2: 基于 Round 1 结果深化
- 如果 PDS 有效 → 在 Part 分支内集成 Skeleton GCN (方向 M)
- 如果 KP-RPE 有效 → PSG + KP-RPE 组合
- 如果都有效 → PDS + KP-RPE 组合

### Round 3: 论文补充实验
- 消融实验（PDS 各组件贡献）
- 效率分析（参数量、FLOPs、推理速度）
- 可视化（attention map、t-SNE、检索结果）

---

## exp022 PDS 结果反馈 (2026-03-11)

### PDS 实验结论
- **global-only 57.9%**: 接近但未超过 PSG-only 58.3% (-0.4%)
- **concat_scaled 57.5%**: Part 有微弱贡献
- **equal_concat 56.1%**: 5:1 维度比稀释 Global，不可用
- **part-only 55.2%**: Part 分支独立效果差

### 关键洞察
1. **Stage 3 权重解耦确实有效**: PDS global (57.9%) > exp008 PSG+Part same Stage3 (57.7%)，证明独立 Stage 3 保护了 PSG
2. **但共享 Stage 0-2 仍有轻微干扰**: 57.9% vs 58.3% 的 -0.4% gap 来自 Part 分支经共享层的反向传播
3. **Part 分支学习太慢**: 120 epoch 后 Part ID loss 仍高达 2.02（Global 为 0.17）。5 个独立分类器需要更多训练容量
4. **fusion 策略需要优化**: Part 维度是 Global 的 5 倍，等权 concat 本质上是给 Part 5 倍的投票权

### 方向修正

PDS 实验证明了 **"梯度干扰是可以通过架构解耦缓解的"** 这一核心假设。但也暴露了新问题：

**问题 1: Part 分支需要更好的训练策略**
- 可以尝试：stop_gradient 阻断 Part→共享层梯度，或 Part 分支延迟启动
- 但考虑到复杂度增加和收益不确定，这个方向的性价比可能不高

**问题 2: 当前方法组合天花板**
- PSG-only 58.3% 已经是非常好的单模块结果
- 所有组合实验都未能在此基础上叠加增益
- 也许应该接受 PSG 作为核心贡献，转向其他维度（如 test-time fusion、NFC 定制化）

**修正后的优先级**:
1. **exp023: 先尝试 stop_gradient 隔离** — 最简单的 fix，验证是否能消除 -0.4% gap
2. **如果 stop_gradient 有效** → PDS + stop_gradient 作为论文的 full model
3. **如果 stop_gradient 无效** → 放弃 dual-stream，PSG 作为核心贡献 + 其他正交方向 (KP-RPE, Skeleton GCN)

---

## exp023 PDS+StopGrad 结果反馈 (2026-03-11) 🎉

### 突破性结果！
- **global-only 59.5%**: 超越 PSG-only 58.3% (+1.2%)，超越 baseline +2.9%
- **concat_scaled 59.1%**: Part 特征确实提供补充信息
- **part-only 56.7%**: 超过 baseline 和 exp022 part-only (55.2%)

### 为什么 stop_gradient 反而提升了性能？

**表面解释**: stop_grad 消除了 Part→shared 的梯度干扰，恢复了 Global 分支性能。

**更深的解释**: stop_grad 改变了优化景观：
1. **共享 Stage 0-2 只被 Global loss 优化** → 特征更适合全局 ID 任务 → PSG 获得更好的输入
2. **Part 分支使用 frozen 共享特征** → 被迫学习更好的局部特征适配 → Part Stage 3 学到的是真正的"局部化"转换
3. **良性循环**: 更好的共享特征 → 更好的 Part 输入 → Part 独立效果也提升 (56.7% > 55.2%)

### 这对论文意味着什么

**核心 story 明确了**:
1. **PSG**: 在 backbone 内部注入 pose spatial prior → +1.7% (简洁有效)
2. **PDS**: 双分支解耦 Stage 3 权重 → 让 PSG 和 Part 不冲突
3. **StopGrad**: 完全隔离 Part 梯度 → 消除共享层干扰 → 额外 +1.2%
4. **三者组合**: baseline +2.9% mAP, +3.0% R1

**消融证据链**:
- Baseline 56.6% → +PSG 58.3% (+1.7%) → +PDS 57.9% (-0.4%, 有 Part 干扰) → +StopGrad 59.5% (+1.6%, 消除干扰)
- 证明每个组件都是必要的

### 下一步方向

1. **完善消融实验**: 需要 PDS+StopGrad 但无 PSG 的结果来分离 Part 分支独立贡献
2. **跨数据集验证**: Market-1501 上跑 PDS+StopGrad
3. **test-time fusion 优化**: global-only (59.5%) > concat_scaled (59.1%)，说明 Part 特征融合可以更好
4. **NFC 测试**: 在 PDS+StopGrad 特征上测试 NFC 后处理效果
