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

### Phase 2.5 新实验 (exp022-026)

**exp022-025 PDS 系列**: PDS+StopGrad 达到 59.5%，但 exp024 (无 PSG 版) 达到 59.2%（仅 -0.3%），暗示提升可能来自训练随机性。多 seed 实验待验证。

**exp026 SPD (Stochastic Pose Dropout)**:
- mAP 57.9% vs PSG 58.3% (-0.4%)
- **关键发现**: Pose 信号在 Occluded-Duke 上一致有用，30% dropout 率只是移除了有用信息
- **推论**: PSG 不存在过度依赖问题 → 正则化方向价值有限 → 应该探索"让 loss 函数也感知 pose"的方向

### 下一步方向（更新于 2026-03-11）

1. ~~完善消融实验~~: exp024 已完成 (PDS+StopGrad 无 PSG = 59.2%)
2. **多 seed 验证**: 脚本已准备 (scripts/run_multiseed_4090.sh)，等待 4090 运行
3. ~~PCRA (Pose-Contrastive Representation Alignment)~~: exp027 验证 mAP 57.8% (-0.5% vs PSG)。17 维 pose signature 不够精确区分姿态差异，引入训练不稳定性。
4. **PVR (Pose-Guided Variance Regularization)**: 辅助 loss 鼓励同部位内特征一致、跨部位特征分散
5. ~~SPD 调参 (p=0.1/0.5)~~: SPD 方向整体不如预期，优先级降低
6. **跨数据集验证**: Market-1501 上跑 PSG (需准备 pose 数据)

### Phase 2.6 新实验 (exp026-027)

**exp027 PCRA**: mAP 57.8%, R1 66.8% (-0.5%/-1.1% vs PSG)
- **关键发现**: loss 函数维度的 pose 信号利用也不奏效
- 17 维 pose signature 的余弦相似度不够区分"相同姿态"和"不同姿态"
- 训练过程呈现锯齿形 mAP 波动（奇数十 epoch 高、偶数十 epoch 低）
- **推论**: 在 PSG 基础上的所有单点改进（forward/loss/regularization）均已失败。应转向 PDS+StopGrad 的改进或全新范式

### 阶段性总结（27 个实验后）

**已穷尽的方向**:
1. PSG + forward path 添加: exp008-021 全部失败
2. PSG + 正则化: exp026 SPD 中性

---

## 2026-03-23: MaxSim — Set-to-Set Metric Learning 范式确立

### 核心范式转变
```
旧: image → single vector → pairwise cosine → triplet loss
新: image → body-part token set → MaxSim (set matching) → triplet loss
```

### 已验证的 test-time 增益（跨 checkpoint 稳定）
| Checkpoint | equal_concat | maxsim_hybrid 1:2 | Δ mAP | Δ R1 |
|-----------|-------------|-------------------|-------|------|
| exp030a | 61.1 / 73.7 | 62.2 / 74.5 | +1.1 | +0.8 |
| PAA (exp066) | 61.6 / 74.2 | 62.6 / 75.2 | +1.0 | +1.0 |
| PAA+ROA (exp067) | 62.0 / 73.7 | 63.5 / 75.4 | +1.5 | +1.7 |
| PAA+ROA seed42 | 62.1 / 73.6 | 63.3 / 75.2 | +1.2 | +1.6 |

### 文献空白确认
- **MoS (AAAI 2021)**: Jaccard set matching, 不是 MaxSim
- **BPBreID/KPR (WACV23/ECCV24)**: per-part average distance, 不是 max-based
- **ColPali (2024)**: MaxSim training for document retrieval, 不是 ReID
- **结论: MaxSim training in person ReID = 文献空白**

### 当前进行中
- exp152 (soft MaxSim, tau=0.05) → 远程
- exp152b (hard MaxSim, tau=0.005) → 本地
- 核心假设: MaxSim training + MaxSim test > MaxSim test-only

### 论文 story 候选
1. **Problem**: Occluded ReID 是 partial-set matching，不是 vector matching
2. **Insight**: Train-test metric mismatch 是结构性缺陷
3. **Method**: ColBERT-style Soft-MaxSim for both training and retrieval
4. **Evidence**: test-time 已验证; training alignment 待 exp152

---

## 2026-03-24: STD-PR + PLBOA Synergy — 候选主创新

### 核心发现
STD-PR 单独 -2.4% mAP（弱于 GCN），但 STD-PR+PLBOA = 63.4%（超过 GCN+PLBOA = 62.7%，+0.7%）。

| 配置 | 无 PLBOA | 有 PLBOA | PLBOA 增益 |
|------|---------|---------|-----------|
| GCN (keypoint sampling) | 61.1% | 62.7% | +1.6 |
| STD-PR (cross-attention) | 58.7% | 63.4% | **+4.7** |

**STD-PR 从 PLBOA 获得的增益是 GCN 的 3 倍（+4.7 vs +1.6）！**

### 解释
- GCN 用 bilinear sampling 在固定位置采样 keypoint features → 对输入分布变化不敏感
- STD-PR 用 cross-attention 动态聚合信息 → 天然适应 PLBOA 带来的 lower-body 遮挡多样性
- Cross-attention 的 attention weights 自动重分配到可见区域，而 bilinear sampling 仍然在原位置采样（即使该位置被遮挡了）

### 论文 Story 候选
1. **Problem**: Occluded ReID 的 train-test distribution gap（95.8% vs 83.8% visible）
2. **Insight**: Static keypoint sampling (GCN) 无法充分利用 occlusion augmentation 带来的多样性
3. **Method**: Structural Token Decomposition (STD-PR) — 用 pose-guided cross-attention 替代 keypoint sampling
4. **Key Finding**: STD-PR+PLBOA synergy (+4.7%) 远超 GCN+PLBOA (+1.6%)
5. **Supporting**: MaxSim test-time matching (+1.5%), SGCFR (+2.5%)

---

## 2026-03-22: 关键数据约束发现 — 训练集 visibility 几乎无遮挡

### 训练集 person-0 visibility 统计
- 均值 visible ratio: **95.8% ± 9.3%**
- 中位数: **100%**
- 95.6% 的训练图 visibility > 80%
- 仅 1.4% 的训练图 visibility < 60%
- 每个关键点的可见率都在 88-99% 之间

### 对已有实验的解释力
1. **exp148 PCVT 早期加速但后期无效**: complementary masking 在前期提供多样性，但 backbone 最终收敛到的表示已经隐式假设"几乎全可见"
2. **exp151 PVAT pvat_acc 不下降**: visibility GT 几乎全 1，predictor 只需猜 "全可见" 就有 83% accuracy
3. **所有 visibility-based 训练方法失败**: 训练集缺乏 visibility 多样性，任何依赖训练时 visibility 变化的机制都没有足够信号

### 对后续方向的约束
1. **不要再在训练侧做 visibility-dependent 机制**: 训练数据几乎全可见，学不到有意义的 visibility-conditioned 行为
2. **真正的 occlusion gap 在 test-time**: gallery/query 有严重遮挡，但训练集没有
3. **数据增强（ROA 等）是缩小 train-test visibility gap 的唯一训练侧途径**
4. **更有前途的方向**: 改善 test-time matching（SGCFR、CVK 等），或设计不依赖 visibility 的训练目标

---

## 2026-03-20: `exp109` 主线的下一阶段机制收束

### 当前新的判断

经过 `exp110-126`，当前最重要的收束不是“support-complete 有没有价值”，而是：

1. `exp109` 的 oracle 上界仍然极强，问题定义成立
2. `SCKD` 说明 loss-only 的间接蒸馏太弱
3. `SCFR` 说明 hard replace 太硬，直接 feature replacement 并没有自动优于蒸馏
4. `CSRD` / pair routing 说明 relational teacher 有价值，但仍然偏间接

因此，`exp109` 这条线最自然的下一跳不是再扫权重，而是：

**让 support-complete prototype 以“可学习残差 prior”的形式进入 keypoint branch。**

### 新候选方向: SCRC（Support-Conditioned Residual Completion）

- 核心想法:
  - 对 low-vis keypoint，不做 hard replace
  - 也不只加 distillation loss
  - 而是学习：
    `kp_completed = kp + gate(kp, proto, score, proto_conf) * (proto - kp)`

- 它相对已有路线的定位:
  1. 比 `SCKD` 更直接
  2. 比 `SCFR` 更柔和
  3. 比 `CSRD` 更靠近 feature formation，而不是只作用于 embedding geometry

- 若成功，它对论文 story 的价值:
  1. 问题层面仍锚定 `single-image support incomplete`
  2. 机制层面从 “memory bank / routing trick” 升级成了真正的 **support-conditioned completion**
  3. 更接近一个可支撑主方法的训练机制，而不是附属 loss

---

## 2026-03-13 训练端方向收敛：从 retrieval-time CVK 转向 CSGT

### 新上下文
- `exp040` 与 `exp045` 已在两个 checkpoint 上复核出：
  - `cvk_hybrid` 均能稳定提升 mAP
- 这说明 common-support 不是噪声，而是真实的 pairwise 证据
- 但当前主收益仍停留在 test-time，论文主线不够完整

### 为什么不是继续调 test-time 权重
1. `exp041` 已说明 `1:1` 基本是当前 mAP sweet spot
2. 再细扫权重会迅速退化成 test-time trick 调参
3. 文献也说明：
   - KPR / BPBreID / QPM 都已把 pair-specific visible matching 讲得很清楚
   - 如果我们只停在 retrieval-time 距离定义，很难把训练端创新讲强

### 当前最有价值的训练端候选
**CSGT: Common-Support-Guided Triplet**

#### 核心问题
- 遮挡 ReID 下，不同正负 pair 的共同可见支撑并不相同
- 但标准 triplet 仍默认所有 pair 的可比性相同

#### 核心机制
1. 用 `kp_weights` 构造 batch 内 pairwise common-support overlap
2. 在 global branch 上增加一条 support-aware triplet
3. 优先在 overlap 足够高的 pair 上做 mining
4. 若找不到可用 pair，则回退到标准 mining，避免训练崩掉

#### 为什么它比“再加一个模块”更像主线
1. 问题层面更清楚：partial observation 下 pair comparability mismatch
2. 机制层面不同：不是 feature fusion，而是 pair mining
3. 证据层面可讲：
   - 对照 `exp030a`
   - 对照 `exp036`
   - 再看是否削弱 `cvk_hybrid` 的必要性

### exp047 结论：CSGT 失败

**实验结果**: Epoch 60 中断（无 checkpoint），但已有充分失败证据。

**根本失败原因**: `csgt_pos_overlap ≈ csgt_neg_overlap ≈ 0.65`，差异始终 < 0.02。keypoint visibility 是 image-level 属性（由相机角度和遮挡模式决定），不携带 identity 信息，因此 overlap 无法区分正负 pair。`pos_fallback ≈ 0.7-0.8` 说明 70-80% 的正样本退化为标准 mining。

**核心教训**: 把 retrieval-time 的 common-support 信号迁到训练端，不能简单用 overlap 做 mining filter。retrieval-time CVK 有效是因为它改变了距离计算方式（只在共同可见关键点上计算距离），而不是因为它筛选了更好的 pair。

**对后续方向的影响**:
1. **overlap-based mining filter 这个方向彻底否决**
2. 如果要做训练端 common-support，必须改变 loss 本身的距离计算（如只在共同可见区域上计算 triplet 距离）
3. 但这会进一步接近 KPR/BPBreID 已做的 pairwise visible matching，创新空间更窄

### 当前结论（更新）
- CSGT 失败。简单的 overlap mining filter 不可行。
3. PSG + loss 调制: exp027 PCRA 中性

**仍然有效的方向**:
1. **PDS+StopGrad**: 唯一超越 PSG 的方法 (+2.9% mAP)，但 PSG 在其中贡献很小
2. **多 seed/跨数据集验证**: 确认方法稳定性

### Phase 2.7 新实验 (exp028)

**exp028 PDS+StopGrad + Part LR 3x**: mAP 59.3%, R1 68.9% (equal_concat)
- **关键发现**: Part 收敛显著改善但测试性能未提升
  - Part ID loss: 0.4 (exp028) vs 2.0 (exp023) — 5x 改善
  - Part tri loss: 完全对齐甚至低于 Global tri
  - 但 equal_concat mAP 59.3% vs exp023 global-only 59.5% — 本质持平
- **重要结论**: **Part 分支收敛不是 PDS 性能瓶颈**
  - Part 特征即使在高 loss 状态下也已提供了其最大有用信息
  - 进一步收敛只增加了训练集拟合，没有提高泛化
  - 说明 Part 和 Global 从共享 Stage 0-2 提取的信息本质上高度冗余

### 阶段性总结更新（28 个实验后）

**已穷尽的方向**:
1. PSG + forward path 添加: exp008-021 全部失败
2. PSG + 正则化: exp026 SPD 中性
3. PSG + loss 调制: exp027 PCRA 中性
4. PDS Part 收敛改善: exp028 Part LR 中性

**未触及的创新空间**:
1. **Token-level pose 操作**: Transformer 中 token 是最自然的信息单元，但目前所有方法都在 feature map level 操作。用 pose 热图做 token 选择/路由/加权是完全未探索的方向
2. **Pose-guided attention routing**: 不是用 pose 做 gate（已验证无法叠加），而是在 self-attention 中用 pose 信息引导 token 之间的注意力分配
3. **Occlusion-aware feature extraction**: 直接用 pose 热图判断 token 是否对应可见区域，跳过遮挡 token 的贡献

### 候选创新点 C: Pose-Guided Token Selection (PGTS)

**核心想法**: 用 pose 热图的空间响应强度作为 token 重要性分数。高响应区域（可见身体部位）的 token 获得更高权重，低响应区域（遮挡/背景）的 token 被抑制或裁剪。

**与 PSG 的区别**:
- PSG: 在 feature map 上做乘法调制（soft gate），所有 token 都保留
- PGTS: 在 pooling 阶段做 token 选择（hard/soft selection），显式排除噪声 token

**与 Part Pooling 的区别**:
- Part Pooling: 按部位分组后各自 pooling，产生多个 part feature
- PGTS: 不分组，只根据"是否为人体"加权 pooling，产生一个更纯净的 global feature

**实现方案**:
1. 将 pose 热图 (17, H, W) 在 channel 维度 max，得到 body mask (1, H, W)
2. 将 body mask resize 到 Stage 3 feature map 大小 (12, 4)
3. 用 body mask 作为 token 权重进行 weighted average pooling
4. 可选: 将低于阈值的 token 直接 mask 掉（hard selection）

**预期**: 在遮挡场景中，PGTS 直接排除遮挡/背景 token 的干扰，比 PSG 的 soft gate 更直接有效

**风险**:
- 热图的 max pooling 可能丢失细粒度信息
- threshold 选择可能需要调参
- 非遮挡图片中 PGTS 退化为标准 GAP（可能中性）

### Phase 2.8 新实验 (exp029)

**exp029 PSG + Pose-Weighted Pooling (PWP)**: mAP 57.9%, R1 67.5% (-0.4%/-0.4% vs PSG)
- **关键发现**: PWP 本质上就是 PGTS 的 soft 版本（用 body mask 加权 pooling），结果中性偏负
- **重要启示**: Post-backbone 的 weighted pooling 在 PSG 已做空间调制后是冗余操作
- PSG 在 Stage 3 内部已完成空间选择 → pooling 阶段再加权只是重复工作
- **对 PGTS 方向的修正**: 如果 soft weighted pooling 无效，hard token pruning（直接删除 token）也很难在 pooling 阶段奏效。要做 token-level 操作，必须在 **Stage 3 内部**，让 token selection 影响 self-attention 计算

### 阶段性总结更新（29 个实验后）

**已穷尽的方向**:
1. PSG + forward path 添加: exp008-021 全部失败
2. PSG + 正则化: exp026 SPD 中性
3. PSG + loss 调制: exp027 PCRA 中性
4. PDS Part 收敛改善: exp028 Part LR 中性
5. **PSG + post-hoc pooling 改进: exp029 PWP 中性** ← NEW

**关键反思 — 29 个实验的根本教训**:
- PSG 在 Stage 3 内做的空间门控已经是 pose heatmap 利用的最优方式之一
- 所有"在 PSG 之上/之后加东西"的尝试都失败了（21 个 PSG 改进实验 + 1 个 PWP）
- PDS+StopGrad 是唯一成功的方向，核心不是 pose 利用，而是**梯度隔离**的训练策略
- **下一步需要全新的框架思路**，而不是继续在 PSG/PDS 上微调

### Phase 2.9 新实验 (exp030)

**exp030 PDS+StopGrad + Skeleton GCN**: mAP **60.0%**, R1 **71.0%** (equal_concat, E110 peak)
- **全实验最佳！** 超越 baseline +3.4% mAP / +4.5% R1
- vs exp023-g (59.5%): +0.5% mAP, +1.5% R1
- vs exp023-eq (57.5%): +2.5% mAP, +4.8% R1

**关键发现**:
1. **GCN 骨架特征远优于 Part Pooling 特征**: exp023 equal_concat 57.5% → exp030 equal_concat 60.0% (+2.5%)
2. **R1 提升尤为显著 (+4.5%)**: 骨架拓扑特征在 top-1 检索上有很强互补性
3. **Part 分支训练收敛快**: id_part (0.085) << id_global (0.180)，GCN 768-d 向量比 5 个 Part 分类器更高效
4. **第一个让 equal_concat 超越 global-only 的方案**: 之前所有 Part 方案的 concat 都 <= global-only

**新问题 — Loss Weighting 混淆因素**:
- Codex 分析发现 PDS 的 list-loss 路径隐式将 global loss 乘 0.5
- exp007a (PSG + 0.5x global loss) 将验证 loss weighting 是否是 PDS 增益的主因
- 如果 exp007a ≈ 59%，则 PDS/StopGrad 的架构贡献需要重新评估
- exp030 的 60.0% 仍然有效，因为 GCN concat 特征提供了 Part Pooling 无法提供的互补信息

### Phase 2.10: exp007a 结果 — 决定性发现！

**exp007a (PSG + 0.5x Global Loss Scale)**: mAP **59.5%**, R1 **69.8%**

| 方法 | mAP | R1 | 额外 Params | 说明 |
|------|------|------|-------------|------|
| exp007 (PSG, 1.0x loss) | 58.3% | 67.9% | +102K | PSG 基线 |
| exp023 (PDS+StopGrad) | 59.5% | 69.5% | +6.3M | 双流+梯度隔离 |
| **exp007a (PSG, 0.5x loss)** | **59.5%** | **69.8%** | **+102K** | **仅改 loss scale** |

**核心发现**: PDS+StopGrad 的 +2.9% 增益 **100% 来自 loss weighting 正则化**。双流架构(+6.3M params)和 StopGrad 策略都是不必要的。

**对论文 story 的重大修正**:
1. ~~PDS+StopGrad 作为核心创新~~ → PDS 的增益被 loss weighting 完全解释
2. PSG + loss scaling 是更简洁、更本质的方法
3. **exp030 GCN 的增益仍然有效**: global-only 59.5% + GCN concat → 60.5% (+1.0%)，GCN 提供了真正的互补特征

**更新后的 story**:
- PSG: backbone 内 pose 注入 (+1.7%)
- Loss scaling: 全局 loss 正则化 (+1.2%)
- Skeleton GCN: 骨架拓扑特征传播 (额外 +1.0% mAP vs global-only)
- 总计: baseline 56.6% → PSG+LS+GCN 60.5% (+3.9% mAP, +4.0% R1)

### Phase 2.11: exp030a 结果 — GCN 不需要 PDS！

**exp030a (PSG + Skeleton GCN, 无 PDS)**:

| 模式 | exp030a (PSG+GCN, ~500K) | exp030 (PDS+SG+GCN, ~6.3M) | Δ |
|------|--------------------------|-------------------------------|---|
| global | 59.8% / 69.5% | 59.5% / 69.5% | +0.3% / 0.0% |
| concat_scaled | 60.5% / 73.7% | 60.5% / 70.5% | 0.0% / +3.2% |
| **equal_concat** | **61.1% / 73.7%** | 60.0% / 70.9% | **+1.1% / +2.8%** |
| gcn_only | 58.2% / 72.9% | 56.7% / 67.3% | +1.5% / +5.6% |

**核心发现**: 独立 Stage 3 完全不必要！共享 Stage 3 的 PSG 特征对 GCN 更好（因为 PSG 已做了 pose-aware modulation），且参数减少 92%。

**R1 大幅提升的可能原因**: exp030a 的 GCN 特征质量更高（来自 PSG 增强的特征），导致 concat 后的组合特征在 Rank-1 检索上大幅优于 PDS 方案。

**最终方法 (3 个正交轻量组件)**:
1. PSG: +102K params, +1.7% mAP
2. Loss Scaling (0.5x): +0 params, +1.2% mAP (通过 GCN 列表损失隐式实现)
3. Skeleton GCN: +~400K params, +1.3% mAP (equal_concat 61.1% vs global 59.8%)

**总计: baseline 56.6% → 61.1% (+4.5% mAP, +7.2% R1)，仅 ~500K 额外参数**

### Phase 2.12: exp030b 训练方差发现

**exp030b (PSG+GCN, w_p=0.01, ≈1.0x loss)**:
- GCN 几乎未训练 (ID_part loss 5.1 vs 0.17)
- 但 global mAP = 60.6%，远超 exp007 (58.3%)

**四实验 global mAP 对比**:
| 实验 | 模型类 | Loss Scale | Global mAP |
|------|--------|-----------|-----------|
| exp007 | PSG only | 1.0x | 58.3% |
| exp007a | PSG only | 0.5x | 59.5% |
| exp030a | PSG+GCN | ~0.5x | 59.8% |
| exp030b | PSG+GCN | ~1.0x | **60.6%** |

**关键洞察**: 没有一致规律！exp030b 应该最低（1.0x loss, GCN 无贡献）却最高。这 2.3% (58.3%→60.6%) 的波动很可能主要来自训练方差。

**对创新 story 的影响**:
1. **PSG 增益 (+1.7%) 的确认需要多种子数据**
2. **Loss scaling 增益 (+1.2%) 可能被高估** — exp030b 用 1.0x loss 也达到 60.6%
3. **GCN concat 增益 (+1.3%) 在 exp030b 中消失** (equal_concat 60.5% ≈ global 60.6%)
4. **多种子实验 (exp031) 是论文可信度的基石**

**修正后的保守估计**:
- PSG: +1.0~2.0% (需 multi-seed 确认)
- Loss Scaling: +0~1.5% (可能被方差覆盖)
- GCN: +0.5~1.5% (exp030a 中有效, exp030b 中因 GCN 未训练而无效)

> 注：以下 Phase 2.13 / 2.14 保留的是当时的中间判断；最终校正以 **Phase 2.15** 为准。

### Phase 2.13: Loss Scale 敏感性分析完成 (exp007b/c)

| Loss Scale | mAP | R1 |
|-----------|-----|-----|
| 0.25x | 58.3% | 67.6% |
| 0.5x | 59.5% | 69.8% |
| 0.75x | 58.6% | 67.6% |
| 1.0x | 58.3% | 67.9% |

**单种子结论**: 0.25x/0.75x/1.0x 在 58.3-58.6% 范围内，仅 0.5x=59.5% 异常高。
**⚠️ 但这是单种子数据！** 0.5x 是否真的是 sweet spot，还是训练方差，**需要 exp007a 多种子验证**。
PDS+SG 多种子 (59.20% mean) 暗示 0.5x 效果可能是真实的。

### Phase 2.14: ✅ 多种子验证完成 (4090, 3 configs × 3 seeds)

**这是整个 Phase 2 最关键的里程碑。**

| 方法 | Seed 1234 | Seed 42 | Seed 2024 | Mean±Std |
|------|-----------|---------|-----------|----------|
| Baseline | 56.7% | 55.9% | 56.9% | 56.50±0.53% |
| PSG | 58.3% | 57.9% | 57.3% | 57.83±0.50% |
| PDS+SG (global) | 59.7% | 59.2% | 58.7% | 59.20±0.50% |

**PSG 增益 +1.33% 确认** — 所有 3 seeds 正向 (paired: +1.6/+2.0/+0.4), p≈0.054
**PDS+SG 增益 +2.70% 确认** — 统计显著 p<0.02
**PDS+SG vs PSG +1.37% 极一致** — paired diffs (1.4/1.3/1.4), p<0.001

**🔑 核心问题: PDS+StopGrad 的增益来源**

**已有证据 (单种子)**:
1. exp023 (PDS+SG, global) = 59.5% ≈ exp007a (PSG, 0.5x loss) = 59.5% → 增益来自 loss*0.5
2. Part 分支有 detach()，梯度不回传 → Part 分支本身不影响 global
3. 所以 PDS+SG 中真正起作用的是 global loss weight 从 1.0 变成 0.5

**多种子确认 PDS+SG 效果一致** (4090):
- PDS+SG mean = 59.20% vs PSG mean = 57.83% → +1.37% 极一致
- 这暗示 0.5x loss scaling 可能是一个真实的 sweet spot

**待确认 (关键缺失)**:
- ⚠️ **exp007a (0.5x loss) 的多种子验证尚未完成**
- 如果 exp007a 多种子也给出 ~59%，则**最终确认** loss*0.5 是核心增益来源
- 这将是论文的重要贡献：发现 global loss scaling 的 sweet spot

**对论文 story 的影响**:
- **PSG 是核心贡献** — +1.33% 确认
- **0.5x Loss Scaling 可能是第二贡献** — 需 exp007a 多种子最终确认
- **PDS+StopGrad 架构不是必需的** — 简单的 loss*0.5 即可复现其效果
- **GCN 仍需多种子** — 安排 4090 验证

**修正后的方法估计**:
- PSG: **+1.33%** (3-seed confirmed)
- 0.5x Loss Scaling: **+1.37%** over PSG (PDS+SG multi-seed 暗示, 需 exp007a multi-seed 直接确认)
- GCN: **+1.3%** (single-seed, 待确认)

### Phase 2.15: ✅ 终版修正（exp007a / exp030a 多种子补齐）

后续 4090 又补齐了：

- `exp007a` 3 seeds
- `exp030a` 4 个测试模式的 3 seeds

#### 新增核心结果

| 方法 | 模式 | Mean±Std (mAP) | Mean±Std (R1) |
|------|------|----------------|---------------|
| exp007a | global | **59.37±0.32%** | **69.43±0.12%** |
| exp023 | global | **59.20±0.50%** | **68.63±0.47%** |
| exp030a | global | **59.33±0.40%** | **68.87±1.00%** |
| exp030a | concat_scaled | **60.20±0.44%** | **73.13±0.29%** |
| exp030a | equal_concat | **60.73±0.47%** | **72.57±0.58%** |

#### 现在可以最终定下来的判断

1. **`0.5x global loss` 不是方差**
   `exp007a vs exp007` 的 paired diffs = `(1.3, 1.6, 1.7)`，`p=0.0061`。

2. **PDS+StopGrad 的 global-only 收益基本被 exp007a 复现**
   `exp007a = 59.37%` vs `exp023-g = 59.20%`，差异不显著。
   所以 PDS+StopGrad 更像一个“揭示了 loss-weighting 机制”的中间 scaffold，而不是最终主创新。

3. **GCN/KPP branch 的增益现在也不再只是单 seed 现象**
   `exp030a-eq = 60.73%`，对 `exp030a-global = 59.33%` 的 paired diffs 为 `(1.3, 1.1, 1.8)`，`p=0.0214`。

4. **`equal_concat` 明显优于 `concat_scaled`**
   三个 seed 全部成立，均值差 `+0.53%`，`p=0.0039`。
   因此后续所有主表都应以 `equal_concat` 为主，不再以 `concat_scaled` 为主模式。

5. **更准确的 branch 解释**
   `exp032` 说明 keypoint pooling 本身就很强；
   `exp030a` multi-seed 说明训练好的 graph branch 还能继续提高 fusion。
   所以更合理的 framing 是：
   **KPP 提供 branch 主体信息量，GCN 负责 relation refinement。**

### Phase 2.16: 2026-03-13 文献/代码复盘后的方向修正

#### 这轮复盘看了什么
- KPR (ECCV 2024) 论文 + 官方代码
- BPBreID (WACV 2023) 论文 + 官方代码
- FRT (TIP) 摘要 + 官方仓库状态
- QPM 摘要

#### 共同结论
1. **真正强的 occluded ReID 工作，核心不是“再加一个小模块”**
   - KPR: target ambiguity / promptable target selection
   - BPBreID: partial observation 下 global embedding 的理论局限
   - FRT: retrieval-time feature recovery
   - QPM: quality-aware common non-occluded reasoning

2. **visibility / quality 的主要落点在 pairwise distance，而不是 train-time pooling 小改动**
   - BPBreID / KPR 都把 visibility 用在 query-gallery 距离计算
   - 我们最近的 `exp035b / exp036 / exp037` 则主要在 branch 内部调权重、调 loss，问题定义偏弱

3. **当前代码线的真实 gap 已经更清楚了**
   - `exp030a` 证明 branch 的价值主要发生在 fusion
   - 但当前测试仍用 `equal_concat`
   - 也就是说：**结构化 keypoint branch 被训练出来了，但在检索时被过早压缩**

#### 因此不再推荐的主线
- AFF / learnable fusion gate
- 继续做 branch 内部 learnable weight
- 继续做额外局部 triplet / auxiliary loss

这些方向的问题不够新，且已有文献早已覆盖“quality-aware / adaptive weighting”叙事。

#### 新的主线候选
**共同可见关键点检索（Common-Visible Keypoint Retrieval）**

核心想法：
- 保留 `GCN` 增强后的关键点级表征到测试阶段
- 只在 query-gallery 共同可靠的关键点上计算局部距离
- 再与 global feature 做距离级融合，而不是特征级拼接

#### 为什么这个方向更像主线
1. **问题层面**: 直接针对 partial observation 下“谁和谁的哪一部分可比”
2. **机制层面**: 从 feature concat 切到 pair-specific keypoint reasoning
3. **证据层面**: 可以设计清晰对照
   - global
   - equal_concat
   - keypoint-only pairwise distance
   - global + common-visible keypoint distance

#### 当前优先级更新
1. 先收紧 `exp035 / exp036 / exp037` 文档表述
2. 让 `exp037` 自然结束
3. 下一实验优先做 **共同可见关键点检索诊断**
4. `AFF` 降为备选，不再当主线默认项

### Phase 2.17: 2026-03-13 `exp040` 复核后的创新判断更新

#### 新增证据
- `040a` (`exp030a` checkpoint, `equal_concat`) = `61.1 / 73.7`
- `040b` (`exp030a` checkpoint, `cvk_hybrid`) = `61.9 / 73.2`
- 相对差值 = `+0.8% mAP / -0.5% R1`

#### 与 `exp039` 合并看后的判断
1. **这已经不是单次偶然波动**
   - `exp039b`（`exp035a` checkpoint）= `61.9 / 73.2`
   - `exp040b`（`exp030a` checkpoint）= `61.9 / 73.2`
   两次在不同来源 checkpoint 上给出几乎一致的输出。

2. **这条线已经开始满足“问题 + 机制 + 初步证据”三件事中的前两件半**
   - 问题层面：partial observation 下的共同可见支撑
   - 机制层面：pair-specific keypoint reasoning，而不是再堆 train-time module
   - 证据层面：已有重复单 checkpoint 正信号，但还缺更强统计支撑

3. **它比 AFF / LKA 更接近可投稿主线**
   因为 AFF / LKA 本质上仍是 branch 内部或 feature-level weighting，而 `cvk_hybrid` 已经把问题重新落在 retrieval-time reasoning 上。

#### 还不能夸大的地方
1. 现在只能说“可复核正信号成立”，还不能说“最终主方法已经确认”。
2. 收益主要体现在 mAP，不是 R1；因此后续必须解释它改善的是哪类 pair 的排序。
3. 这仍是 test-time reasoning，不能伪装成训练端创新。

#### 下一步最值得做的不是新模块，而是证据加固
1. 权重敏感性：`global : cvk`
2. 多 checkpoint / 多 seed 复核
3. 失败样例分析：它究竟修正了哪些遮挡 pair

### Phase 2.18: 2026-03-13 `exp041` 后的机制判断

#### 新增证据
- `2:1` = `61.6 / 72.6`
- `1:1` = `61.9 / 73.2`
- `1:2` = `61.6 / 73.6`

#### 能新增确认的不是“更高点”，而是机制形状
1. **`1:1` 是当前小范围内的 mAP 最优点**
   两侧偏移都会掉到 `61.6%`，因此正向收益不是随便混一点就有。

2. **这条线更像 balanced correction，而不是 single-side domination**
   - 偏 global：mAP / R1 一起掉
   - 偏 CVK：mAP 掉，但 R1 更接近 baseline

3. **因此 story 可以进一步收紧**
   不是“CVK 越强越好”，也不是“少量 CVK 即可”；
   而是：
   **global identity space + balanced common-support correction**

#### 研究策略更新
1. 不再继续做细粒度 test-time 参数扫点
2. 优先寻找更多 checkpoint 复核这条形状是否成立
3. 若多 checkpoint 仍成立，再考虑做 pair-case 可视化 / 错误类型分析

### Phase 2.19: 2026-03-13 `exp042` 后的证据层更新

#### 新增关键统计
- `positive_delta_ap = 1129`
- `negative_delta_ap = 822`
- `zero_delta_ap = 259`
- `top1_fixed = 47`
- `top1_degraded = 58`

#### 这一步真正补上了什么
1. **回答了“为什么 mAP 会涨而 R1 会跌”**
   因为它改善了更多 query 的整体 AP，但没有把这些改善都转成 top-1 修复。

2. **说明 gain 不是 few-case cherry-pick**
   如果只是少数 query 偶然翻转，不会出现 `1129 > 822` 这样的整体分布。

3. **把机制表述进一步收紧**
   当前最准确的表达不是：
   - “CVK 提升检索”
   而是：
   - **CVK 主要做 deeper-rank common-support correction**

#### 对创新判断的意义
现在这条线已经不只是：
- 有问题定义
- 有机制
- 有 aggregate 指标

而是开始具备：
- **对有效性的可解释证据**

这比继续刷一个小模块或再调一个 loss，更接近能支撑论文主叙事的证据链。

---

## Phase 2.20: 2026-03-13 新文献补充（ProFD / DPEFormer / SSSC-TransReID）

### 文献学习来源

本轮研究了 3 个近期仓库/论文（详见 paper_notes 16-18）：
1. **ProFD** (ACM MM 2024, 代码已开源)：CLIP ViT + 体部位 text prompt + PartFeatureDecoder + SemiAttentionDecoder
2. **DPEFormer** (arXiv 2402.10435, 代码未开源)：ViT + DPSM 动态 token 选择 + FBM 特征混合 + ROA 真实遮挡增强
3. **SSSC-TransReID** (arXiv 2410.15613, 代码未找到)：双分支 Transformer + 随机矩形遮挡增强 + SimSiam 风格自监督

### 三篇论文对当前研究的关键观察

#### 1. DPSM（DPEFormer）与我们 PGTS 方向的对比

DPSM 用 CLS token 相似度动态选择"干净" token，本质上是无监督版的 token 重要性打分。
我们有 ViTPose 热图，直接提供空间重要性 — 但 exp029（PWP）已证明 post-backbone 的 token 加权是冗余操作（PSG 已在 Stage 3 内部做了空间调制）。
**因此 DPSM 思路对我们而言价值有限**。即使我们用热图代替 CLS 相似度做 token 选择，exp029 的负结论已经说明 post-backbone pooling 阶段做任何 pose-weighted 操作都不如 PSG 内部调制有效。

#### 2. PartFeatureDecoder（ProFD）与当前方向的关系

ProFD 的 PartFeatureDecoder 用文本 prompt 做 Query、spatial tokens 做 K/V，通过 cross-attention 解码出 part 特征。
这与 exp030a 的 GCN branch 思路截然不同：GCN 是沿骨架传播 keypoint 特征，PartFeatureDecoder 是让部位 query 从 spatial tokens 中聚合信息。
**潜在创新**：可以把 PartFeatureDecoder 的 text prompt 替换为 **pose-heatmap-guided learnable queries**（K 个由热图初始化的 query vectors），让 cross-attention 从 Swin 特征中聚焦到每个关键点对应的 spatial tokens 上。这比 GCN 的 bilinear sampling 更灵活，也比 ProFD 的 text prompt 更 spatially precise。

但这需要独立的 cross-attention decoder，参数量 ~200-400K，且引入新的 collapse 问题（需要 Dissimilar Loss）。

**结论**：这个方向与 exp047 失败路线在本质上不同（exp047 是 training-end CSGT，而这是 feature extraction decoder），但当前 exp030a 的 GCN 方案已足够，不优先推进 PartFeatureDecoder 路线。

#### 3. SSSC-TransReID 的两个可借鉴技巧

**3a. Random Rectangle Mask 数据增强**
- 比 random erasing 更模拟真实遮挡（多个不重叠矩形 vs 单个矩形）
- SSSC 报告 +0.6% Rank-1 vs Hide-and-Seek
- 移植成本极低：只需在 dataloader 中加一个增强函数
- **可以与 exp030a 基线结合，作为一个正交的性能提升点**
- 不需要任何模型修改，不依赖 pose 数据

**3b. 自监督对比分支（SimSiam 风格）**
- 用强增强图像做 stop-gradient 对比，鼓励遮挡不变特征
- 计算代价：双前向传播，在 Swin-Tiny + 3090 上可行（约增加 30-40% 训练时间）
- 但其核心信号（遮挡一致性）与我们的 PSG 目标高度重叠
- **创新空间**：如果将随机矩形遮挡替换为**热图引导的 body-aware masking**（只遮挡低置信度关键点区域），则自监督对比信号更有 pose 语义，与 SSSC 形成明确差异化

### 新候选方向：Pose-Aware Masking Consistency (PAMC)

**核心想法**：结合 SSSC 的自监督对比框架 + 热图引导的遮挡模拟

1. 用 ViTPose 热图识别每张训练图的"低置信度关键点区域"（热图响应 < threshold）
2. 用这些区域的 union mask 作为遮挡 mask，贴在原图上生成"进一步遮挡的版本"
3. 双分支对比：原图特征 vs 进一步遮挡图特征应当一致（stop-gradient SimSiam）
4. 直觉：这让模型学会"即使关键点被遮挡，也应该从可见区域重建出一致的特征"

**与 SSSC 的差异**：
- SSSC 用随机矩形（不知道遮挡了什么）
- PAMC 用热图引导的 body-aware masking（知道遮挡了哪个身体部位）
- 这是一个真正的"姿态引导遮挡一致性"信号，而非 generic 遮挡鲁棒性

**与 exp047（CSGT）失败的区别**：
- exp047 失败原因：overlap 无法区分正负 pair（keypoint visibility 是 image-level 属性）
- PAMC 不做 pair mining，而是单图层面的增强 + 单图层面的一致性监督
- 不会遇到 CSGT 的"无法区分正负 pair"问题

**初步可行性评估**：
- 需要对现有 pose_data 的 scores 做 threshold 筛选（已有数据）
- 实现较简单：数据增强 + 双前向传播 + SimSiam loss
- 参数增加：Projector MLP ~1M，可接受
- 显存：双前向传播在 Swin-Tiny batch=64 上约 18-20GB，3090 可行

**创新门槛检查**：
1. **问题层面有新意**？是 — 不是随机遮挡，而是"把已遮挡图的遮挡模式进一步强化，测试模型能否保持身份一致性"
2. **机制层面有新意**？是 — 热图引导的 body-aware masking + stop-gradient consistency，而非随机遮挡增强
3. **证据层面能讲清楚**？是 — 对照 Random Rectangle Mask 版本；消融热图引导 vs 随机 mask

**结论**：PAMC 满足创新门槛（问题 + 机制 + 证据），且实现成本较低，值得作为下一个主线候选。

---

## Phase 2.21: 2026-03-13 exp048 SGMKC 失败后的方向修正

### exp048 结论

**实验**: SGMKC（Skeleton-Guided Masked Keypoint Completion）— 在 GCN 前随机 mask 30% keypoints，GCN 输出计算 MSE reconstruction loss。
**最终结果**: mAP 58.9%, R1 72.1%（vs exp030a: 60.5%, 73.7%），稳定负 -1.6% mAP。

**失败原因分析**:
1. **双任务梯度冲突**: SGMKC loss 在总 loss 中占比 30-50%（早期更高），分散了 GCN 的 ID 分类梯度
2. **GCN 容量瓶颈**: 2 层 GCN 无法同时完成 ID 分类和特征重建——重建需要保留低级特征信息，ID 分类需要丢弃低级信息
3. **重建目标与 ID 目标根本矛盾**: MSE 要求输出接近原始特征（保持不变），ID loss 要求输出更具区分性（改变特征）
4. **Loss 量级问题**: 即使 weight=1.0，SGMKC loss 的绝对值（~1.7）仍接近甚至大于 triplet loss（~0.1-1.0），主导了优化方向

### 连续失败总结（exp047 + exp048）

两个连续的训练端 GCN 改进实验均失败：
- **exp047 CSGT**: 尝试用 keypoint visibility overlap 做 triplet mining → 失败（overlap 无法区分正负 pair）
- **exp048 SGMKC**: 尝试用自监督重建任务增强 GCN → 失败（梯度冲突 + 容量不足）

**共性教训**: 在 exp030a 的 GCN branch 上做额外训练端改进，都遇到了"小型 GCN 容量不足以承载第二个任务"的瓶颈。GCN 的最优角色就是其当前角色——纯 ID 分类的关键点特征传播器。

### 已穷尽的方向（更新）

1. PSG + forward path 添加: exp008-021 全部失败
2. PSG + 正则化: exp026 SPD 中性
3. PSG + loss 调制: exp027 PCRA 中性
4. PDS Part 收敛改善: exp028 Part LR 中性
5. PSG + post-hoc pooling 改进: exp029 PWP 中性
6. **GCN branch 训练端自监督: exp048 SGMKC 负面** ← NEW
7. **Overlap-based mining: exp047 CSGT 失败** ← NEW

### 当前方向决策

根据 CLAUDE.md 止损规则："如果某条路线已经连续出现多个负结果...应记录负结论后立即转入文献与代码学习、gap analysis、新问题定义或新机制设计"。

**需要做的**:
1. 深入学习尚未研究的论文/代码仓库，寻找真正新的 gap
2. 重新审视 PAMC（Pose-Aware Masking Consistency）方向的可行性
3. 或者发现全新的方向

**PAMC 仍是当前最有希望的候选**，因为：
- 它不依赖 GCN branch 改进（避开已证伪的方向）
- 它在 backbone 层面操作（PSG 成功的关键洞察）
- 它有清晰的创新门槛：问题（pose-aware occlusion simulation）+ 机制（body-aware masking + consistency）+ 证据（vs random masking 消融）
- 但实现前需要更深入的文献调研（SimSiam 在 ReID 中的已有工作、遮挡增强的 SOTA）

---

## Phase 2.22: 2026-03-14 exp050 PAMC 结果 — 中性/无效

### exp050 最终结果
- mAP: 60.7%, R1: 72.2%（vs exp030a 3-seed mean: 60.73% / 72.57%）
- **完全中性**：mAP 差 0.03%，R1 差 0.37%，均在训练方差范围内

### 失败原因分析

1. **PSG 已提供充分的遮挡鲁棒性**：PSG 在 Stage 3 内部做 pose-aware spatial gating，模型已经"知道"人体结构。额外的 consistency loss 没有提供新的训练信号。

2. **Consistency loss 的双重效应相互抵消**：
   - 正面：特征对遮挡更不变，提高 mAP（检索覆盖面广）
   - 负面：特征对细粒度差异的区分度降低，降低 R1（top-1 精度）
   - 最终两个效应大致抵消

3. **Masking 不够激进**：PAMC cosine similarity 从 epoch 11 的 0.90 到 epoch 120 的 0.82，说明 masked 图和原图的特征已经很接近，consistency 目标太"容易"。单个身体部位的遮挡不足以创造有挑战性的训练信号。

4. **负总损失的潜在问题**：后期 PAMC 负贡献主导总损失变为负值，可能导致优化器行为不理想。

### 已穷尽的方向（更新至 50 个实验）

1. PSG + forward path 添加: exp008-021 全部失败
2. PSG + 正则化: exp026 SPD 中性
3. PSG + loss 调制: exp027 PCRA 中性
4. PDS Part 收敛改善: exp028 Part LR 中性
5. PSG + post-hoc pooling 改进: exp029 PWP 中性
6. GCN branch 训练端自监督: exp048 SGMKC 负面
7. Overlap-based mining: exp047 CSGT 失败
8. **Consistency loss (self-supervised): exp050 PAMC 中性** ← NEW

### 重大战略判断

**"在 PSG+GCN 基础上添加训练端辅助 loss"这个方向已被连续 3 次实验否定**：
- exp047 (CSGT): ❌ overlap mining 失败
- exp048 (SGMKC): ❌ 自监督重建 梯度冲突
- exp050 (PAMC): 🟡 self-supervised consistency 中性

**核心教训**：PSG+GCN 的训练已经高度优化。任何辅助 loss 都无法在不干扰主 ID+triplet 目标的情况下提供额外增益。训练端改进这条路线应当彻底关闭。

### 下一步方向

需要从根本上跳出"在 exp030a 上加东西"的思路。以下方向值得探索：

1. **全新的特征提取机制**（而非辅助 loss）：
   - Pose-guided token pruning（直接删除噪声 token，而非 gate/mask）
   - Cross-attention decoder for part features（类似 ProFD 但用 pose query 代替 text prompt）

2. **Test-time 策略的训练端配套**：
   - 已知 CVK 在 test-time 有效 → 设计训练端方案让 keypoint 特征更适合 pairwise distance
   - 不是加 loss，而是改变特征的使用方式

3. **完全不同的问题定义**：
   - 从 "单图特征提取" 转向 "图对关系推理"
   - 从 "遮挡鲁棒特征" 转向 "不确定性感知匹配"

4. **深入文献学习**：
   - 优先研究最近的遮挡 ReID 方法（2024-2025），寻找尚未尝试的思路
   - 重点关注不是 "加模块" 而是 "改范式" 的工作

---

## Phase 2.23: 2026-03-14 深度文献调研后的方向更新

### 本轮研究的论文/代码

1. **PADE** (ICASSP 2024): 三视图并行增强训练 + 双增强策略
2. **ProFD** (ACM MM 2024): CLIP ViT + 文本 prompt 引导 part 解耦 + SemiAttention decoder
3. **PersonViT**: 大规模自监督 ViT 预训练
4. **Pose2ID** (CVPR 2025): 训练无关 NFC + 身份引导行人生成
5. **CION** (NeurIPS 2024): 跨视频身份相关预训练，提供 Swin-T 预训练权重
6. **SEAS** (CVPR 2024): 3D 体形作为监督信号（而非输入模块）
7. **Camera Bias Debiasing** (ICLR 2025 Spotlight): 特征维度级别的相机偏差分析
8. **P3E**: 概率性部位嵌入（高斯分布建模不确定性）
9. **OGFR**: 强化学习引导的 token 选择

### 核心发现

#### 1. 领域范式转移
近年的强工作正在从"更好的特征提取"转向"更智能的匹配/检索"：
- Pose2ID: test-time 特征中心化
- P3E: 分布式匹配（Wasserstein/KL 距离）
- Camera Bias: 维度级特征分析
- CVK（我们自己的）: 逐关键点 pairwise 匹配

这意味着：**进一步改进特征提取的边际收益递减，改进距离度量/匹配策略的空间更大**。

#### 2. 训练端辅助 loss 确认为死胡同
3 个失败（CSGT/SGMKC/PAMC）+ 文献中的趋势共同说明：在已高度优化的特征上叠加辅助训练信号，预期收益极低。

#### 3. 新的有价值方向
- **Train-test metric alignment**（PAML 方向）：改变已有 loss 的距离计算方式，而非添加新 loss
- **概率嵌入**（PKE 方向）：如果 PAML 有效，可扩展为概率性关键点嵌入
- **CION 预训练权重**：drop-in 实验，验证 backbone 质量是否是瓶颈
- **Camera dimension bias**：分析性方向，理解特征维度的信息分布

### exp051 PAML 的定位

PAML 是"不添加新机制，只对齐训练-测试距离度量"的最小化实验：
- 将 GCN branch 的 part triplet 距离从聚合特征距离改为逐关键点 pairwise 距离
- 对齐了训练目标与 CVK 测试时逻辑
- 如果有效：证明"距离度量对齐"比"添加模块"更重要
- 如果中性/失败：证明距离计算方式不是瓶颈，需要更根本的范式转变

---

## Phase 2.24: 2026-03-14 exp051 PAML 结果 — 训练端辅助 loss 方向彻底关闭

### exp051 结果

| 模式 | exp030a | exp051 PAML | Δ |
|------|---------|-------------|---|
| equal_concat | 60.73% / 72.57% (3-seed) | 60.7% / 72.7% | ≈0 |
| cvk_hybrid | 61.9% / 73.2% | 62.0% / 73.6% | +0.1% / +0.4% |

**结论：完全中性。训练-测试 metric alignment 假设未得到验证。**

### 训练端辅助 loss 连续失败汇总（5 次）

| 实验 | 方法 | 类型 | 结果 |
|------|------|------|------|
| exp047 | CSGT (Common-Support-Guided Triplet) | 新 loss | ❌ 中止，pos/neg 无法区分 |
| exp048 | SGMKC (Skeleton-Guided Masked Keypoint Completion) | 自监督辅助 loss | ❌ 负面 (-1.6% mAP) |
| exp050 | PAMC (Pose-Aware Masking Consistency) | 一致性辅助 loss | 🟡 中性 |
| exp051 | PAML (Pose-Aware Metric Learning) | 替换已有 loss 距离 | 🟡 中性 |
| (exp036) | Per-Keypoint Triplet | 新 loss | ❌ 负面 (-0.5% mAP) |

### 结论与方向转移

**训练端 loss 修改方向已彻底关闭。** 不论是：
- 添加新 loss（CSGT, SGMKC, Per-KP Triplet）
- 添加一致性 loss（PAMC）
- 替换已有 loss 的距离计算（PAML）

都未能超越 exp030a 基线。这说明：

1. 当前 GCN branch 的 ID loss + Triplet loss 已经足够——增量训练信号无法带来额外增益
2. 问题不在 loss 函数本身，而在更深层的架构或数据表示
3. 需要转向全新的机制方向

### 下一步方向候选

1. ~~**KP-RPE (Keypoint Relative Position Encoding)**~~: 已在 exp052 中验证 → 中性结果
2. **深度文献学习**: 寻找完全不同的问题定义或机制
3. **概率嵌入方向**: 不确定性感知的关键点特征表示

---

## Phase 2.25: exp052 KP-RPE 最终评估 (2026-03-14)

### 实验结果

| 测试模式 | exp052 | exp030a single-seed | Δ | exp030a 3-seed mean | Δ vs 3-seed |
|----------|--------|-------------------|---|--------------------|----|
| equal_concat | 61.0% / 72.7% | 60.5% / 73.7% | +0.5% / -1.0% | 60.73% / 72.57% | +0.27% / +0.13% |
| global | 59.5% / 68.4% | 59.8% / 69.5% | -0.3% / -1.1% | 59.33% / 68.87% | +0.17% / -0.47% |
| cvk_hybrid | 61.7% / 72.6% | 61.9% / 73.2% | -0.2% / -0.6% | — | — |

### 关键发现

1. **KP-RPE 最终结果在 3-seed 方差范围内**：与 exp030a 3-seed mean 的差距仅 +0.27% mAP / +0.13% R1，不具统计显著性
2. **训练过程 mAP 稳定正向（10/12 checkpoint 为正，均值 +0.76%）但最终收敛到基线水平**：这说明 KP-RPE 改善了中期训练动态但未能保持到最终收敛
3. **Global 模式 KP-RPE 微负**：说明注意力偏置未改善 backbone 特征本身，"正向"仅在 fusion 模式中短暂出现
4. **CVK 模式 KP-RPE 也微负**：KP-RPE 与 CVK 不正交，可能因为两者都试图编码结构信息

### 方向评估

**注意力偏置方向（PAB + KP-RPE）已可关闭：**
- PAB（exp012，unary）：+0.8% mAP（单 seed，但当时 baseline 不同）
- KP-RPE（exp052，pairwise）：+0.5% mAP vs 单 seed / +0.27% vs 3-seed mean
- 两者都在方差范围内
- **结论：无论是 unary 还是 pairwise 注意力偏置，都不是 pose 信息注入的有效方式**

这与我们的核心发现一致：**backbone 中有效的 pose 注入方式是 PSG（乘性门控），不是注意力偏置（加性偏移）**。PSG 直接抑制/增强特征值，而注意力偏置只能微调注意力权重——后者的影响力更小。

### 当前有效模块清单（更新）

| 模块 | 效果 | 3-seed 确认 | 类型 |
|------|------|------------|------|
| PSG | +1.33% mAP | ✅ | backbone 乘性门控 |
| 0.5x global loss | +1.53% mAP | ✅ | 训练技巧 |
| GCN (fusion) | +1.40% mAP | ✅ | branch fusion |
| CVK | +0.8% mAP | 2-ckpt 复核 | test-time |

| 方向 | 效果 | 状态 |
|------|------|------|
| Auxiliary losses (5种) | 全部中性/负面 | ❌ 关闭 |
| Attention bias (PAB, KP-RPE) | 方差内 | ❌ 关闭 |
| Visibility | 负面 | ❌ 关闭 |
| Part pooling | 弱正面(+0.9%) | 被 PSG 取代 |

### 关键洞察：到此为止的实验揭示了什么？

**我们已经耗尽了"在现有 PSG+GCN 框架上做增量修改"的空间。** 包括：
- 训练 loss 层面：5 种辅助 loss 全部失败
- 注意力层面：2 种注意力偏置在方差内
- Visibility 层面：多次尝试均负面
- 后处理层面：CVK 有效但不算训练端创新

**能继续的方向必须是架构级别的改变**，而不是在当前框架上"加一点东西"。

### 下一步方向（更新后优先级）

1. **全新架构方向**：探索完全不同的 pose 信息利用范式，而非在现有框架上微调
2. **文献与代码学习**：深入阅读最新 ReID/pose 论文，寻找未被探索的 gap
3. **PPE (概率嵌入)**：备选方向，如果能找到清晰的问题定义和论文 story

---

## Phase 2.26: exp053 XCAD 最终结果 — Cross-Attention Decoder 劣于 GCN

### 实验结果

| 模式 | exp053 XCAD | exp030a 3-seed mean | Δ |
|------|-------------|---------------------|---|
| equal_concat | 59.7% / 70.8% | 60.73% / 72.57% | **-1.03% / -1.77%** |
| global | 59.2% / 68.6% | 59.33% / 68.87% | -0.13% / -0.27% |
| cvk_hybrid | 60.7% / 71.8% | ~61.9% / 73.2% | -1.2% / -1.4% |

### 关键发现

1. **GCN 的骨架拓扑先验不可替代**：XCAD 用自由 cross-attention 替换固定拓扑 GCN，结果更差。COCO skeleton 的 19 条边编码了真实的人体解剖结构关系，这在 15K 训练图上是 cross-attention 学不会的强归纳偏置。

2. **Global 模式几乎持平**：说明 XCAD 没有损害 backbone 特征（PSG 工作正常），问题完全出在 keypoint branch 的特征质量上。

3. **R1 是结构性弱点**：整个训练过程中 XCAD 的 R1 始终落后 GCN 2-3%。cross-attention 的 soft attention pattern 产出"更平滑"的特征，缺乏 GCN 骨架传播的离散结构性。

### 已关闭的方向（更新）

| 方向 | 效果 | 状态 |
|------|------|------|
| Auxiliary losses (5种) | 全部中性/负面 | ❌ 关闭 |
| Attention bias (PAB, KP-RPE) | 方差内 | ❌ 关闭 |
| Visibility | 负面 | ❌ 关闭 |
| **Cross-attention decoder 替换 GCN** | **负面 (-1% mAP)** | **❌ 关闭** |

### 当前确认有效的唯一模块组合

PSG + 0.5x loss + Skeleton GCN（exp030a）仍然是最强方法，无一竞争者能超越。

### 战略判断

**我们已经耗尽了"在现有框架上做架构级修改"的尝试：**
- 增量修改（5 种 loss、2 种 attention bias）：全部失败
- 架构级修改（XCAD 替换 GCN）：失败
- 总计 8 种不同类型的训练端改进尝试，0 个成功

**下一步必须是完全不同的思路**，不能再在 PSG+GCN 框架上做任何修改。

---

## Phase 2.27: exp054-059 PGAM/KDL/ROA 系列结果 (2026-03-14)

### 新发现

1. **PGAM（注意力 masking）**: 微弱正向（+0.4% mAP / +1.1% R1），零参数，阈值/Stage 不敏感，但在 3-seed 方差边缘
2. **KDL（Dissimilar Loss）**: 中性，第 6 个失败的 auxiliary loss
3. **ROA（真实遮挡增强）**: **+1.07% mAP**，超出方差！61.8% 历史最高 mAP。但 ROA 已在多篇论文中使用（FCFormer/DPEFormer/ProFD），不能作为独立创新点
4. **ROA+PGAM 组合**: 与 ROA-only 完全相同，说明 PGAM 与 ROA 不正交（都解决遮挡鲁棒性）

### 当前格局

**有效的工程手段**（可用于提升基线但不构成创新）:
- PSG: +1.33% mAP（已 3-seed 确认）
- 0.5x global loss: +1.53% mAP（已确认）
- GCN branch: +1.40% mAP（已确认）
- ROA: +1.07% mAP（需多 seed）
- CVK: +0.8% mAP（test-time）

**不够格作为创新的方向**:
- ROA: 已有先例（FCFormer OIA, DPEFormer ROA, synthetic-occlusion）
- PGAM: 效果在方差边缘，且与 ROA 冗余
- 所有 auxiliary loss: 6 次全部失败

### 需要的创新方向

用户明确指出：ROA 不能作为创新点，除非融入新东西或改进增强方式。需要思考：

1. **Pose-Guided ROA**: 用 pose heatmap 指导遮挡物的放置位置——只在身体可见区域粘贴（模拟真实遮挡模式）。与现有 ROA 的区别：现有方法随机放置，我们根据 pose 智能放置
2. **Adaptive ROA**: 根据图像已有的遮挡程度动态调整 ROA 的概率和覆盖面积——已经高度遮挡的图像少加、干净图像多加
3. **完全不同的方向**: 放弃在数据增强上做文章，转向特征提取/匹配机制的创新

---

## Phase 2.28: 62 个实验后的终极总结 (2026-03-15)

### 已穷尽的方向清单（10 类，全部中性/负面）

| 类别 | 实验 | 结果 | 教训 |
|------|------|------|------|
| Auxiliary Loss | 7 个 (CSGT/SGMKC/PAMC/PAML/Per-KP Tri/KDL/LKU) | 全部中性或负面 | PSG+GCN 的训练已充分，增量训练信号无效 |
| Attention Bias | 2 个 (PAB/KP-RPE) | 方差内 | Swin 12×4 特征图上 attention 修改影响太小 |
| Attention Masking | 3 个 (PGAM 3变体) | 微弱正向但方差内 | 与 ROA 冗余 |
| 架构替换 | 1 个 (XCAD) | 负面 | GCN skeleton topology 不可替代 |
| Dropout/正则化 | 3 个 (SPD/GKD/PWP) | 中性 | 现有正则化已足够 |
| 数据增强 | 2 个 (ROA/PA-ROA) | ROA +1.07%，但不新颖 | 数据增强有效但不构成创新 |
| Loss Weighting | 已在 Phase 2 验证 | 0.5x 有效 | 已确认 |
| Backbone 注入 | PSG 已验证 | +1.33% | 唯一成功的训练端创新 |
| GCN Branch | 已验证 | +1.40% fusion | 架构级成功 |
| Learned Uncertainty | 1 个 (LKU) | R1 -1.37% | 额外 head 干扰 keypoint weighting |

### 核心教训

**1. PSG+GCN 框架已达到优化天花板**
62 个实验中，只有 PSG（backbone injection）、GCN（independent branch）、0.5x loss 和 ROA 是有效的。所有在此基础上的增量修改（loss/attention/weighting/dropout/uncertainty）都失败了。

**2. 有效创新的共性**
成功的方向都满足一个条件：**在全新的层面引入信息，而不是在已有层面做微调**。
- PSG：在 backbone feature 形成阶段引入 pose（之前没有）
- GCN：引入了 skeleton topology 这个新信息源
- 0.5x loss：改变了优化景观
- ROA：改变了训练数据分布

**3. 下一步创新必须跳出当前框架**
不能再在 PSG/GCN 上做任何修改。需要的是一个全新的信息维度或全新的问题定义。

### 当前可行的下一步

1. **深度文献调研**（正在进行）：寻找 2025 年最新的 ReID 范式
2. **全新问题定义**：如 target ambiguity（多人）、domain adaptation、test-time reasoning
3. **全新信息源**：如 3D body shape（SEAS）、text descriptions（CLIP-ReID）、video temporal cues

---

## Phase 2.29: exp063-065 PTD/PKE 系列 (2026-03-15)

### exp063 PTD (Pose-Token Distillation): ❌❌ 严重负面 (-4.03% mAP)
- Learned part tokens 无法替代 GCN 的 bilinear sampling + skeleton topology
- Heatmap KL distillation 教了一些定位但精度远不如关键点坐标
- **教训**: GCN 的空间定位精度和骨架拓扑先验是不可替代的

### exp064 PKE (Probabilistic Keypoint Embeddings): 🟡 微弱正向 (+0.27% mAP)
- Precision-weighted feature (mu/sigma) 在 test-time 改变了特征空间
- Sigma 从 0.135 增长到 0.577（有意义的 uncertainty learning）
- 不损害 R1（vs LKU -1.37%），但改进在方差内
- **这是目前唯一成功改变 GCN 特征表示方式而不造成负面影响的实验**

### exp065 PKE+ROA: 进行中，预计 ≈ ROA alone（不正交）

### 当前最佳方法栈
1. PSG (backbone injection) — +1.33% mAP ✅
2. 0.5x global loss — +1.53% mAP ✅
3. GCN (skeleton branch fusion) — +1.40% mAP ✅
4. ROA (realistic occlusion augmentation) — +1.07% mAP ✅ (不够新颖)
5. PKE (precision-weighted features) — +0.27% mAP 🟡 (需多 seed 确认)
6. PGAM (attention masking) — +0.37% mAP 🟡 (与 ROA 冗余)

### 总计 vs baseline: ~60.73% → 61.8% (ROA) 或 61.0% (PKE)
加 NFC test-time: 64.0%

---

## Phase 2.30: PAA 突破与后续方向 (2026-03-15)

### 🎉 PAA (Pose Additive Adapter) — 66 个实验中最重要的发现

**exp066**: PSG + GCN + PAA = mAP 61.6% / **R1 74.2%** (+1.63% R1 vs 3-seed!)
**exp067**: PSG + GCN + PAA + ROA = **mAP 62.0%** / R1 73.7% (+1.27% mAP vs 3-seed!)

**PAA 的核心创新**：在 PSG 乘性门控之后，用加法 adapter 注入 pose-derived content
- PSG: `x = x * (1 + gate)` — 调制幅值（哪里重要/哪里抑制）
- PAA: `x = x + adapter(heatmap)` — 添加 pose 语义内容
- 两者互补形成 **dual-channel pose injection**
- 仅 51.8K 额外参数，零推理开销

### 用户建议的后续方向（按优先级）

1. **Suppress-and-Complete**：PSG 抑制非目标人区域，PAA 只补全目标人的缺失区域
   - 需要区分 target person 和 non-target person 的 heatmap
   - 与 KPR 的 target ambiguity 问题对标
   - 最强创新潜力——从 "pose injection" 升级为 "target-aware dual-path mechanism"

2. **Reliability-routed PAA** (exp068 正在测试)：只对低置信区域加 adapter
   - exp068 训练中，目前与 PAA 接近

3. **Part-Structured PAA**：按身体部位结构化 adapter
   - generic conv → part prototypes / graph prototypes
   - PAA 变成"按身体结构补语义"

4. **Pose-ControlNet/LoRA**：把 PAA 从 block 后 conv 升级到 Q/K/V 或 FFN 内部的低秩条件分支

---

## Phase 2.31: PAA 消融系列完成 (exp069-074) — 原始设计最优

### PAA 变体消融总结

| 变体 | 实验 | vs PAA | 结论 |
|------|------|--------|------|
| PAA b128 (增大容量) | exp069 | -0.3% mAP, +0.4% R1 | 容量不是瓶颈 |
| S&C target-only (分离热图源) | exp070 | -0.2% mAP, -0.8% R1 | PAA 需要 scene 上下文 |
| PCL LoRA (特征依赖) | exp071 | -0.9% mAP, -2.2% R1 | Feature-independent 更好 |
| PS-PAA (身体部位分组) | exp072 | -0.5% mAP, -0.4% R1 | Generic mixing 更好 |
| Multi-stage (Stage 2+3) | exp073 | -0.5% mAP, 0.0% R1 | Stage 3 already 足够 |
| +PGAM (attention mask) | exp074 | ≈0 (no-op) | PGAM 实际无效 |

### PAA 设计选择的消融证据

**所有 6 个变体都不如原始 PAA (exp066: 61.6%/74.2%)**。PAA 的最优设计是：
1. **Generic Conv2d encoder** (17→32→768) — 不需要 body-part 分组
2. **Scene-level heatmap** — 多人上下文比 target-specific 更好
3. **Feature-independent** — 不需要依赖当前特征
4. **Stage 3 only** — 更多 stage 不提供额外增益
5. **Uniform injection** (无 routing) — 全空间均匀注入效果最好

### 跨硬件验证
- Remote 5060 Ti (PyTorch 2.9): PAA = 61.2%/74.3% — 与本地 3090 一致 (Δ<0.4%)

### 下一步需要的方向

PAA 消融系列已完成。原始 PAA 是最优设计，不需要进一步调优。

**当前需要的不是更多 PAA 变体，而是**：
1. PAA 的多 seed 验证（exp075 正在进行）
2. 全新的创新方向——**不是在 PAA 上改，而是在 PAA 之外找新的贡献点**
3. 回到文献学习，寻找 gap

**当前最强方法栈**:
- PSG (乘性门控) + PAA (加性适配器) + GCN (骨架分支) + 0.5x loss
- 无后处理: 61.6% mAP / 74.2% R1
- +ROA: 62.0% mAP / 73.7% R1 (不是创新)
- +NFC: 64.0% mAP (test-time, 不是创新)

---

## Phase 2.32: 2026-03-16 周度方向校正

### 当前最重要的判断
1. **PAA 是重要发现，但还不够成为 B 类主线**
   - 它证明“加性 pose adapter”有效
   - 但如果没有更强的问题定义，仍容易被归类为模块级改进

2. **ROA 只能保留为 recipe，不能继续往主创新上抬**
   - DPEFormer / FCFormer 已经有真实遮挡增强的明确先例

3. **下一步不能再做 generic PAA 变体**
   - `exp069-074` 已经系统说明：
     - 更大容量没用
     - target-only hard switch 不行
     - feature-dependent LoRA 不行
     - part-structured 分组不行
     - multi-stage 不行

### 重新对齐后的问题定义
- 不是“怎样把 pose 再注入一次”
- 而是：
  **scene-level pose prior 会不会把 target 与 distractor 混在一起，从而在多人遮挡图里损伤目标表征？**

### 新的推荐主线：TDPC（Target-Distractor Pose Conditioning）

#### 为什么它比 exp070 更合理
- `exp070` 用的是 `scene -> target-only` 的硬切换
- TDPC 要做的是：
  - `PSG` 仍保留 `scene` 路径
  - 新增 `target / distractor` differential conditioning
  - 只在高歧义样本上增强 target-specific 注入

#### 这条线满足的创新门槛
1. **问题层面有新意**
   - 对齐 KPR 的 `target ambiguity`
   - 对齐 TTPM 的 `non-target pedestrian occlusion`
2. **机制层面有新意**
   - 不是再叠一个 generic adapter
   - 而是显式区分 target pose 和 distractor pose
3. **证据层面能讲清楚**
   - 全量指标
   - 多人 subset
   - ambiguous cases 可视化

### 一周内的现实执行建议
1. 先补完 `exp075` 的 PAA 多 seed，确认当前 strongest baseline
2. 然后只开一个 `TDPC` 单 seed 主实验
3. 同时准备：
   - multi-person subset 评测
   - target/distractor case study
4. 若首轮无正信号，立即止损，回退到 retrieval-time `common-support recovery`

---

## 2026-03-16 实验更新：exp075-083

### 重大发现 1: PAA 是 multi-person occlusion specialist

exp066 subset analysis 显示：
- 多人图 (n>=2): PAA vs baseline **+1.69% mAP / +2.02% R1**
- 单人图 (n=1): PAA vs baseline **+0.47% mAP / -1.61% R1**
- PAA 在单人图上损害 R1！

**结论**: PAA 不是通用 feature enhancer，而是专门针对多人遮挡场景的改进。

### 重大发现 2: ROA 和 PAA 的 mAP 增益完全重叠

| 方法 | mAP (均值) | R1 (均值) |
|------|-----------|----------|
| exp030a 3-seed mean | 60.73% | 72.57% |
| exp066 PAA | 61.6% | 74.2% |
| exp079 ROA (无 PAA) | ~61.9% | ~73.2% |
| exp067 PAA+ROA | ~61.9% | ~73.9% |

**ROA alone ≈ PAA+ROA**！PAA 的 mAP 贡献被 ROA 完全覆盖。PAA 的独特贡献仅在 R1 上约 +0.7%。

### 已证伪的方向（exp076-078）

三个 target-aware PAA 变体全部失败：
1. **exp076 TDPC** (differential adapter): -0.3% mAP / -1.5% R1
2. **exp077 ST-PAA** (34ch scene+target concat): -0.6% mAP / -0.6% R1
3. **exp078 APG** (adaptive gate): -1.1% mAP / -1.7% R1

**结论**: PAA 的 generic scene heatmap 已是最优输入。所有 target-aware 修改都不如原始设计。原因可能是：74% 训练数据是单人图，target-aware 机制在这些图上只增加噪声。

### 已证伪的方向（exp081 PQTD）

3-layer Transformer Decoder 替代 GCN: -4.7% mAP / -7.0% R1

**结论**: Transformer decoder 在 15K 训练图 + 120ep 下严重不够收敛。GCN (400K params) 在当前数据规模上远优于 Decoder (2.5M params)。

### 当前进行中：exp083 PGFI

Pose-Guided Feature Inpainting — 在 feature map 空间恢复遮挡区域特征。
这是一个不同于"suppress/inject/select"的新范式："recover"。
两台服务器同时跑中。

### 修正后的论文 story 优先级

1. **PSG**: 已确认，稳定 +1.33%
2. **0.5x global loss**: 已确认，稳定 +1.53%
3. **GCN fusion**: 已确认，稳定 +1.40%
4. **ROA**: 已确认，+1.27%（独立有效，与 PAA 重叠）
5. **PAA**: 已确认，+0.87% mAP 但主要贡献在 R1；是 multi-person specialist
6. **创新点仍在探索中**: PGFI 或后续方向


## 2026-03-19: exp107 DACHM 负结果后的方向收敛

### 已排除的实现形式
- **Duplicate-Aware Counterfactual Hypothesis Matching（coarse pooled 版）** 已被 `exp107` 否定：
  - `base_equal_concat = 61.14 / 73.71`
  - `dachm_penalty = 60.72 / 73.17`
- 该负结果在 `clean multi` 和 `duplicate-suspect multi` 上都成立，不是简单的 detector artifact 问题。

### 这个负结果真正告诉我们的事
1. `target/distractor ambiguity` 不能再用“每人一个 pooled embedding”来粗暴建模。
2. 现有正信号都来自更细粒度的 pair-specific 机制：
   - `cvk_hybrid`
   - `SGCFR`
3. 因此如果还要继续 ambiguity 方向，真正值得做的是：
   **在 per-keypoint / common-visible support 层面做 duplicate-aware confuser reasoning**。

### 当前更可信的新主线约束
- 不再做新的 pooled person rerank trick
- 不再把 raw `num_persons` 当作 ambiguity proxy
- 新机制必须同时满足：
  1. 去重后再推理
  2. pair-specific
  3. per-keypoint / common-support 粒度


## 2026-03-19: exp108 DACCM 后的进一步收敛

### 新增负证据
- `exp108 DACCM` 已经把 `ambiguity/confuser` 线推进到更合理的粒度：
  - 主基线 `base_cvk_hybrid = 61.88 / 73.26`
  - `raw_daccm_penalty = 61.35 / 72.85`
  - `daccm_penalty = 61.39 / 72.94`
- 这意味着：
  - 不是只有 `pooled person embedding` 太粗的问题
  - 就连 `per-keypoint / common-support` 层面的 test-time confuser penalty 也不稳定

### 现在可以更强地排除什么
1. 不能再把“设计一个更好的 retrieval-time confuser penalty”当作主创新。
2. 不能再期待通过 `topk / alpha / dedup threshold` 的小调参把这条线救回来。
3. `duplicate-aware` 本身可以作为分析维度，但不足以独立构成方法主干。

### 对新主线的约束进一步收紧
下一条主线最好满足下面至少两条：
1. 不是 test-time rerank trick，而是训练端或表征端机制
2. 能解释为什么 `cvk_hybrid` 有效而 `DACHM/DACCM` 无效
3. 能把“多人遮挡中的有效信息”表述成可学习的结构，而不是基于 hand-crafted penalty 的后处理


## 2026-03-19: exp109 Oracle Support Bank 后的新主线收敛

### 新增强阳性证据
- `exp109` 用 GT same-ID per-keypoint prototype 做 leave-one-out oracle recovery：
  - `base_cvk_hybrid = 61.88 / 73.26`
  - `oracle_feat_only_cvk = 66.15 / 77.87`
  - `oracle_feat_weight_cvk = 70.40 / 81.36`
- 这说明：
  - “support-complete latent representation” 的 headroom 非常大
  - 关键问题不只是比较公式，而是**单图支持本身不完整**

### 这个结果真正改变了什么
1. `SGCFR` 的成功不再只像一个 test-time trick，它更像是在暴露一个真实的训练缺口：
   **模型没有学会从单图中逼近完整 identity support。**
2. 先前 batch-local recovery 失败，不应再解释为“recover 路线整体错误”，更可能是：
   - support 来源太弱
   - 蒸馏目标太局部
   - 没有稳定的 identity-level prototype
3. 因而下一条主线不应再围绕 confuser penalty，而应围绕：
   **same-ID support bank → single-image support-complete distillation**

### 当前最值得赌的具体机制
- `Support-Complete Prototype Distillation`
- 最小版应包含：
  1. per-identity, per-keypoint prototype bank
  2. 仅对低可见 keypoint 做 distillation
  3. 不新增 test-time rerank
  4. 先验证单 seed 是否能在 `equal_concat` 或 `cvk_hybrid` 上转正


## 2026-03-19: exp110 SCKD 后的主线收紧

### 新增强阳性证据
- `exp110` 作为训练端最小原型，已经在单 seed 上转正：
  - 对照 `exp030a-eq seed1234 = 61.1 / 72.9`
  - `exp110_sckd = 61.2 / 73.7`
- 虽然幅度不大，但这和 `exp109` 的 oracle headroom 连起来后，意义很明确：
  - `support-complete` 不是只存在于上界分析里的幻觉
  - 它已经能以非常轻量的训练机制落到真实增益

### 这个结果真正告诉我们的事
1. 当前最值得继续赌的，不再是“有没有必要做 support-complete”，而是：
   **怎样让 prototype teacher 更可靠、更接近真正的 multi-view support。**
2. 第一版增益小，最自然的解释不是“方向错了”，而是：
   - bank teacher 还太 noisy
   - `MIN_COUNT=1` 过于宽松
   - low-visibility 蒸馏目标里混入了不够可信的 prototype
3. 因此下一步不应直接堆 decoder / completion block，而应先做：
   **reliable-support bank / teacher reliability gating**

### 对主创新表述的进一步约束
- 主创新可以开始往下面的叙事收拢：
  1. 问题不是简单 occlusion comparison，而是 single-image support incomplete
  2. 方法不是通用补全 decoder，而是 identity-level support-complete distillation
  3. 第一性瓶颈是 teacher reliability，而不是 feature fusion trick


## 2026-03-19: exp111 后对 teacher reliability 的进一步收紧

### 新增负中性证据
- `exp111` 把 `POSE_SCKD_MIN_COUNT` 从 `1` 提到 `4`：
  - `exp110 = 61.2 / 73.7`
  - `exp111 = 61.1 / 73.8`
- 结果几乎等价，说明“要求多个 support 样本共同支撑 teacher”这件事本身，并没有把当前增益显著放大。

### 这个结果真正告诉我们的事
1. 当前 `support-complete` 主线并没有被否定，因为结果仍保持正向区间。
2. 但 teacher reliability 的关键点大概率不是 `count gating`。
3. 更值得继续赌的是：
   **teacher purity / write quality / support cleanliness**

### 因而下一步应优先关注
- 更高的 bank update visibility threshold
- 基于 support 置信度的 soft reliability weighting
- 让 prototype “更干净”，而不是只是“更晚生效”


## 2026-03-19: exp112/113 后对核心创新的再次收紧

### 新增关键信号
- `exp112` 说明更干净的 support 写入有用，但当前只形成弱正向：
  - `ep80 = 59.7 / 71.6`
- `exp113` 则更关键，它说明：
  - coverage 没明显扩大
  - proto confidence 没坏
  - 但 `proto_count` 一路增长时，`sckd_cos` 会持续下降

### 这意味着什么
1. 当前最值得讲的主创新，已经不只是 “support-complete distillation”。
2. 更精确的核心问题应该是：
   **如何在 pose-aligned support-complete learning 中控制 teacher hardening / non-stationary target。**
3. 这比继续扫 `MIN_COUNT / UPDATE_THR` 更接近论文级机制，因为它在回答：
   - 为什么最小原型只弱正向
   - 为什么 purity 改进不够
   - 为什么 raw `sckd` 不降

### 因而下一步更像论文主方法的方向
- Freeze-after-warmup teacher
- Lagged / stale support bank
- Reliability-aware soft teacher weighting

其中最先该验证的是：
**freeze-after-warmup**，因为它最直接、最干净，最能说明问题是不是出在 teacher non-stationarity。

---

## 2026-03-20: SCKD 系列完全结案（exp110-116, 7 个变体）

### 最终结论

| 变体 | 核心改动 | mAP | R1 |
|------|----------|-----|-----|
| exp110 | 基础 SCKD (online, thr=0.5) | 61.2% | 73.7% |
| exp111 | MIN_COUNT=4 | 61.1% | 73.8% |
| exp112 | UPDATE_THR=0.7 | 59.7%* | 71.6%* |
| exp114 | freeze epoch 20 | 61.3% | 73.6% |
| exp115 | freeze epoch 30 | 61.3% | 73.6% |
| exp116 | SCFR 直接替换 | 61.1% | 74.1% |

(*exp112 在 ep84 提前停表)

**所有变体收敛到 61.1-61.3% mAP**。无论调整：
- count 门槛（exp111）→ 无效
- purity 门槛（exp112）→ 弱正向但不够
- teacher freeze 时机（exp114/115）→ 中性
- 替换 vs 蒸馏（exp116）→ 中性

**EMA prototype bank 方向已穷尽。** 其增益上限约 +0.1% mAP / +0.7% R1，不足以支撑论文主创新。

### 为什么 EMA prototype 不工作

oracle experiment (exp109) 给出 +8.5% mAP 的 headroom，但 SCKD 只能捕获 1%。核心原因：

1. **EMA prototype 是 lossy compression**：把多个观测平均成一个方向向量，丢失了 instance-specific 的判别细节
2. **15K 训练集太小**：每个 identity 只有约 15-20 张图，EMA 平均后的 prototype 缺乏足够的统计支撑
3. **keypoint-level distillation 信号太弱**：只对 ~14.5% 的 keypoints 施加蒸馏，这些 keypoints 本来就是遮挡位置，对最终 pooled feature 的影响有限

### 下一步方向

SCKD 系列的结案意味着必须转向全新方向。当前最值得探索的不再是"如何让 prototype bank 更好"，而是：

1. **文献/代码学习**：寻找 2024-2025 年的新机制，特别是：
   - 跨实例特征增强（不依赖 memory bank）
   - 结构化对比学习（利用 skeleton topology）
   - 自监督预训练任务（针对遮挡场景）
2. **接受当前配置**：PSG+GCN+PAA+ROA 作为训练端最强配置（~62.7% 单 seed），SGCFR 作为测试端独特创新（+2.6%）
3. **gap analysis**：重新审视 oracle headroom 的分布，找到真正可操作的改进点

---

## 2026-03-20: 文献搜索新发现（2024-2025 最新方向）

### 最有潜力的新方向（按优先级排序）

#### 方向 R: Late Interaction / MaxSim 匹配（ColBERT 风格）
**优先级: ⭐⭐⭐⭐⭐**
- 来源: ColBERT (SIGIR 2020), Video-ColBERT (CVPR 2025)
- 核心: 每个 body part 产生一个独立 embedding，匹配时用 MaxSim（每个 query part 找 gallery 中最佳匹配的 part，取 max cosine similarity 后求和）
- 为什么新颖: **从未在 person ReID 中使用过**。现有方法要么 concat 所有 part 为一个向量，要么做 pair-wise visible matching。MaxSim 天然处理遮挡（缺失 part 贡献低）且不需要显式的 visibility 估计
- 与我们框架的关系: 我们的 GCN branch 已经产出 17 个 per-keypoint embedding。当前用 weighted pooling 聚合为一个向量。改为 MaxSim 匹配可能释放 per-keypoint 信息的全部价值
- 风险: 检索效率降低（需要 17×N 次相似度计算），但可以用 inverted index 优化

#### 方向 S: 遮挡驱动对比学习（Occlusion-Driven Contrastive）
**优先级: ⭐⭐⭐⭐**
- 来源: POFR (Neurocomputing 2025), SSSC-TransReID (Multimedia Systems 2025)
- 核心: 生成真实遮挡并创建精确遮挡 mask，然后对同一身份的遮挡和完整视图做对比学习
- 与 SCKD 的区别: SCKD 是对 prototype 蒸馏；对比学习是直接拉近同 ID 的遮挡/完整特征
- 可行性: 我们已有 ROA 基础设施（生成遮挡），只需添加一个对比 loss 分支
- 风险: 可能与标准 triplet loss 冗余

#### 方向 T: 实例感知对比（target assignment）
**优先级: ⭐⭐⭐⭐**
- 来源: InstanceHMR (CVPR 2024)
- 核心: 对比 loss 把 target person 的 keypoint 特征拉向 target center，推开 non-target person 的特征
- 直接解决我们的 "多人图中 target assignment" 问题
- 与我们框架高度匹配: 我们的 pose data 有 person 0-N 的检测，但目前只用 person 0

#### 方向 U: OGFR 风格的特征净化
**优先级: ⭐⭐⭐**
- 来源: OGFR (arXiv 2025) — 76.6% R1, 64.7% mAP on Occluded-Duke (SOTA!)
- 核心: 用 RL agent 动态识别并替换低质量 patch tokens
- 与 PSG 的区别: PSG 是 soft gating，OGFR 是 hard replacement with learned tokens
- 风险: 复杂度高，需要 RL training

### 当前最推荐的下一步

**方向 R（MaxSim Late Interaction）** 最值得优先尝试：
1. 不需要重新训练（可在现有 checkpoint 上测试）
2. 机制层面真正新颖（ReID 领域首次）
3. 与我们现有的 per-keypoint GCN branch 完美对接
4. 如果有效，可以设计训练端的 MaxSim triplet loss 作为后续创新


## 2026-03-20: 从 prototype 压缩转向 pairwise teacher 几何

### 复核后的新收束

- `exp117/118` 已确认为偏题旁路线，不再作为主线参考。
- 重新看当前最扎实的证据链：
  1. `cvk_hybrid` 说明 common-support 的 pairwise 几何是真实的
  2. `exp047` 只否定了 overlap mining，不是否定 pair comparability
  3. `exp051` 只否定了“把距离改在 part triplet 上”这一种弱实现
  4. `exp109-116` 说明 `support-complete` 若被压成 `per-ID prototype`，会损失 pair-specific 细节

### 当前更值得赌的新机制

**CSRD: Common-Support Relational Distillation**

核心想法：
- 不再把 support 压成 prototype
- 也不只在 test-time 用 `cvk_hybrid`
- 而是在训练期，把 skeleton branch 计算出的 `CVK-style` pairwise 距离当作 **privileged relational teacher**
- 直接约束 global embedding 的 batch-wise 几何关系

### 为什么这条线比 `exp047 / exp051` 更值得继续

1. `exp047` 只看 overlap，teacher 太弱；`CSRD` 直接看 feature distance，teacher 更接近真正有效的检索信号。
2. `exp051` 只改了 part triplet，自身没有把 pairwise 几何迁到 global；`CSRD` 的目标正是 global branch。
3. `SCKD/SCFR` 的问题是 prototype 压缩丢失细节；`CSRD` 则显式保留 pair-specific 关系。

### 如果成功，论文主叙事会怎么变

1. 问题层面：partial observation 下存在 **pair comparability mismatch**
2. 机制层面：pose/keypoint branch 作为 **common-support relational teacher**
3. 训练目标：把 global embedding 蒸馏成更符合 common-support 几何的空间
4. 证据层面：可直接和 `exp047 / exp051 / exp109-116` 构成一条非常清晰的对照链

### exp119 结果后的进一步收束

- `exp119-eq = 61.1 / 73.2`：相对 `exp030a-eq seed1234 = 61.1 / 72.9`，表现为 `+0.0 / +0.3`
- `exp119-g = 60.4 / 70.3`：相对 `exp030a-g seed1234 = 59.8 / 69.9`，表现为 `+0.6 / +0.4`
- `exp119-cvk = 62.0 / 73.2`：相对 `exp040b = 61.9 / 73.2`，表现为 `+0.1 / +0.0`

这说明：
1. `CSRD` 已经不是“纯概念验证”，而是**明确弱正向**
2. 增益主要落在 `global`，符合“把 pairwise comparability 蒸回 backbone”的预期
3. 但 `equal_concat` 仍没有被明显拉起，暴露出第一版 teacher 的真正瓶颈：
   **teacher 自身还是单图 `kp_feats`，并不 support-complete**

### 当前最值得赌的下一跳

**Support-Complete Relational Teacher**

核心想法：
1. 保留 `exp119` 的 relational distillation 形式
2. 不再把 bank 当作 pointwise 蒸馏目标
3. 而是先用 `exp109` 方向的 support bank 补全 low-vis keypoint teacher，再用补全后的 teacher 去做 `CSRD`

为什么这比回到 `SCKD` 更合理：
1. `exp109` 已证明 support-complete teacher 有巨大 headroom
2. `exp110-116` 只否定了“prototype 直接拉 student”这件事
3. `exp119` 则证明了 relational teacher 形式是对的
4. 因此更合理的合体不是 `prototype pointwise distillation`，而是：
   **support-complete teacher + relational distillation**

### exp120 后的新收束：问题不再是 “teacher 强不强”，而是 “监督打给谁”

- `exp120` 到 `ep90 = 59.9 / 73.2`，仍略弱于 `exp119 ep90 = 60.1 / 73.7`
- 但这次不能简单说 `support-complete teacher` 失败，因为机制统计很清楚：
  - `csrd_sr ≈ 0.145`
  - `csrd_sn ≈ 157~159`
  - `teacher_gap` 明显更强

这说明：
1. `teacher completion` 已经真实发生
2. 但 `teacher 更完整` 并没有自动转成更好的监督
3. 因而当前更合理的解释不是“teacher 还不够完整”，而是：
   **support-complete 监督的收益主要属于 support-incomplete 样本，被 clean 样本等权平均后稀释掉了**

### 当前最值得赌的下一跳

**SGW-SCRD: Support-Gap Weighted SCRD**

核心想法：
1. 保持 `exp120` 的 support-complete relational teacher 完全不变
2. 不再对所有 anchor 等权施加 `CSRD`
3. 让每个样本的 distillation 强度正比于：
   - 它有多少 keypoint 真正被 support-complete teacher 补全
   - 即 sample-level `replace_ratio`

为什么这条线比继续增强 teacher 更合理：
1. `exp109` 的 headroom 本来就主要集中在低可见样本
2. `exp120` 已说明“teacher 更强”不是充分条件
3. 所以下一步更像是 **selective supervision**，而不是继续做更硬的 teacher

### exp122 后的新收束：问题不是 “该打给哪些样本”，而是 “该打给哪些 pair”

- `exp122` 到 `ep40 = 55.4 / 68.2`
- 对照：
  - `exp119 ep40 = 55.9 / 68.7`
  - `exp120 ep40 = 55.5 / 67.8`
- 同时机制统计显示：
  - `csrd_ar ≈ 0.56`
  - `csrd_aw ≈ 0.145`

这说明：
1. sample-level `replace_ratio` selective weighting **已经正确工作**
2. 但它没有转成收益，反而更像把原本有用的监督整体削弱了
3. 因而 `exp120` 暴露出的真正问题不是 “监督应该打给哪些样本”
4. 而是：
   **support-complete teacher 实际只改变了一部分 pairwise 关系，distillation 应聚焦这些 pair-change relations**

### 当前最值得赌的下一跳

**Pair-Delta Focused SCRD**

核心想法：
1. 保持 `exp120` 的 support-complete teacher 完全不变
2. 不再用 sample-level `replace_ratio` 去统一缩放整条 anchor loss
3. 而是直接比较：
   - 原始单图 teacher 几何
   - support-complete teacher 几何
4. 对那些 **被 support completion 真正改变过的 pair** 赋予更高 distillation focus

为什么它比 `exp122` 更合理：
1. `exp122` 已否定 sample-level routing 太粗
2. `exp109` 的 headroom 本质上是 pairwise comparability 被修正
3. `exp119` 的有效性也本来就是 relational，而不是 sample classification
4. 所以下一步应把 selective supervision 从 **sample 级** 收紧到 **pair 级**

### exp121/123 之后的进一步收束

- `exp121` 最终到 `ep120 = 60.6 / 74.0`，说明 `stable teacher` 确实能把 `SCRD` 从中性偏负拉回弱正向
- 但它的量级仍明显不够主方法，因此更合理的定位是：
  **teacher stability = supporting mechanism**

- `exp123` 到 `ep60 = 57.8 / 70.9` 首次同时超过 `exp119/120` 同阶段，说明：
  1. pair-level `teacher-change focusing` 方向本身成立
  2. 当前更像是“兑现偏慢”，而不是“机制不对”

- 同时它把下一步收得更具体：
  - `csrd_pd` 长期只有 `0.002~0.003`
  - `csrd_pf` 长期只有 `1.06~1.08`
  - 这意味着当前第一版 pair focus **放大力度过弱**

### 当前最值得赌的下一跳

**Stronger Pair-Delta SCRD**

核心想法：
1. 保持 `exp123` 的 pair-level delta focusing 完全不变
2. 不改 teacher、不改 bank、不改主 loss
3. 只提高 `POSE_CSRD_PAIR_WEIGHT_ALPHA`
4. 验证当前 delayed weak-positive 是否只是因为放大不够

为什么它比再扫 freeze / sample weighting 更合理：
1. `exp121` 已说明 freeze 只是 supporting，不值得再扩成一条线
2. `exp122` 已否定 sample-level routing
3. `exp123` 已提供“pair focus 方向对”的最重要证据
4. 因而现在最有信息量的，不是换问题，而是测试 **pair focus strength 是否就是当前瓶颈**

### exp123 正式评估后的新收束

- `exp123` 的正式结果是：
  - `equal_concat = 61.1 / 73.4`
  - `global = 60.2 / 70.3`
  - `cvk_hybrid = 61.9 / 73.2`
- 对照 `exp119`：
  - `61.1 / 73.2`
  - `60.4 / 70.3`
  - `62.0 / 73.2`

这说明：
1. pair-level `teacher-change focusing` 没有被否定，但 `alpha=1.0` 版本并没有把 `CSRD` 稳定推到更强正式结果
2. 训练监控的 delayed gain 没有清晰地转成最终 eval gain，说明当前第一版 focus 仍偏弱、偏散
3. 与此同时，远程 `exp124` 到 `ep40` 已经说明：
   - `alpha=4.0` 确实能把 `csrd_pf` 从 `1.06~1.08` 放大到 `1.24~1.29`
   - 但中期指标仍只是近乎持平
4. 因而当前最值得赌的新下一跳，不是继续平滑放大，而是：
   **Sparse / Top-Delta Pair Routing**

### 当前最值得赌的下一跳

**Sparse Pair-Delta SCRD**

核心想法：
1. 保持 `exp123/124` 的 support-complete relational teacher 完全不变
2. 不再对所有 pair 做连续平滑加权
3. 而是只保留每个 anchor 下被 teacher 真正显著改变的那部分 pair 进入 `CSRD`
4. 把“pair focus”从**弱连续加权**升级成**稀疏 pair 选择**

为什么这比继续扫 `alpha` 更合理：
1. `exp123` 已说明“有 pair focus”不够
2. `exp124` 到 `ep40` 已说明“更大 alpha”也未必足够
3. 当前更像是 teacher-change pairs 本来就稀疏，连续加权仍然被大量近零变化 pair 稀释
4. 所以下一步应测试 **更结构化的 sparse pair routing**，而不是继续做平滑强度微调

### 2026-03-20 晚间再收紧：问题可能不在 routing，而在 target dilution

- `exp127 SCRC` 到 `ep100 = 60.5 / 73.1`，并未优于 `SCFR/SCKD`
- 同时其 gate 几乎塌到 `1.0`，说明 per-ID prototype 的 direct feature completion 这条兑现线暂时可以收住
- 用户也已明确否定继续试 `freeze`；结合 `exp121` 的既有结果，更合理的定位是：
  **stable teacher 只是 supporting mechanism，不再值得单独扩线**

这会把当前主问题再收紧一步：
1. `support-complete teacher` 的新增信息是真实存在的
2. `pair routing` 也是真实有效的
3. 但 `exp120/123/125` 仍然主要是在让 student 拟合 **完整 teacher 几何**
4. 于是 support-complete 带来的那部分新增 correction，极可能被 base teacher 的主体结构稀释掉

### 当前最值得赌的下一跳

**Residual-Correction SCRD**

核心想法：
1. 保留 `exp125` 当前最强的在线 relational 主线
2. 不改 teacher，不改 `delta_top`，不改 backbone
3. 只把 distillation target 从完整 `dist_sc` 改成：
   - `dist_sc - dist_base`
4. 让 global embedding 学习的不是“再复刻一遍 skeleton teacher”，而是只学 **support completion 真正带来的关系修正**

为什么它比继续扫 sparse 强度更合理：
1. `exp125` 已说明 routing 方向不是负的
2. `exp126` 正在远程回答“真稀疏本身是否更优”
3. 本地最该补的因果问题已经变成：
   **当前瓶颈究竟是 routing 不够稀疏，还是 target 没把新增 correction 单独抽出来**

### 2026-03-20 夜间更新：`residual_kl` 没有推翻 `exp125`

- `exp130` 最终到：
  - `ep110 = 60.1 / 73.4`
  - `ep120 = 60.1 / 73.1`
- 对照 `exp125`：
  - `ep110 = 60.4 / 73.8`
  - `ep120 = 60.5 / 73.5`

这说明：
1. `residual_kl` 不是“没接上”，因为它的 `csrd` 后期始终稳定在 `0.011~0.013`
2. 但它也没有把 `exp125` 推得更强
3. 所以当前可以把假设进一步收紧为：
   - **target dilution 不是主瓶颈**
   - 真正更值得继续赌的是 **changed-pair coverage**

### 当前最值得赌的下一跳

**Cross-Batch Changed-Pair SCRD**

核心想法：
1. 保留 `exp125` 当前最强的 online support teacher 与 `delta_top` routing
2. 不再改 teacher target，不再改 pair weight 公式
3. 只扩大每个 anchor 可见的 candidate relations：
   - 从 batch 内 pairs
   - 扩大到 batch + cross-batch queue
4. 让 sparse changed-pair supervision 不再受单个 batch 覆盖率限制

为什么它比继续改 target 更合理：
1. `exp125` 已说明 pair routing 有效
2. `exp130` 已说明 target 改写不是主突破口
3. 如果 changed pairs 本来就稀疏，那么 batch-only teacher-change matching 仍可能遗漏大量有信息量的 relations
4. 因而更强的下一跳不该是“再换一个 target”，而应是：
   **让 student 在更大的 relation support 上学习 support-complete comparability correction**

### 2026-03-21 凌晨更新：`queue coverage` 也不是主瓶颈

- `exp131` 最终到：
  - `ep110 = 60.4 / 73.7`
  - `ep120 = 60.5 / 73.7`
- 直接对照 `exp125`：
  - `ep110 = 60.4 / 73.8`
  - `ep120 = 60.5 / 73.5`
- 更关键的是，queue 机制不是没接上：
  - `csrd_qn = 256`
  - `csrd_qr = 0.427~0.441`

这说明：
1. cross-batch queue 真实参与了约四成 candidate relations
2. 但它并没有把 `exp125` 推成更强 mAP
3. 所以当前可以把假设进一步收紧为：
   - **relation coverage 不是主瓶颈**
   - 真正卡住的更像是 **pair-specific correction 的表示形式**

这里还要补一个边界：
1. 仓库里虽然有 `exp089 PAMN` 设计稿
2. 但它从未真正接入 checkpoint 与测试检索流程
3. 因而“learned pair module”这条线 **还没有被真正做过，更没有被证伪**

### 当前最值得赌的下一跳

**LTCS / Learn-to-Trust Common Support**

核心想法：
1. 不再强迫单个 global embedding 吃下 support-complete correction
2. 而是训练一个真正进入检索流程的 pair-adaptive fusion head
3. 让它根据当前 pair 的 `global / CVK / overlap / visibility` 描述，自适应预测：
   - 该在多大程度上相信 global distance
   - 该在多大程度上相信 common-support distance
4. 训练监督不再是 pair label 直接打分，而是：
   - 用 `support-complete teacher` 提供更理想的 pairwise target
   - 学习一个 **learned correction rule**

为什么它比继续扩 `CSRD` 更合理：
1. `exp125` 已说明 pair correction 值得学
2. `exp130/131` 又说明：
   - 不是 target form
   - 也不是 coverage
3. 这就把主矛盾收紧成：
   **当前 correction 不适合继续被压进单向量 embedding**
4. 因而下一步应从“embedding distillation”转到：
   **proper learned matching / adaptive fusion**

### 2026-03-21 早间更新：`LTCS alpha-fusion` 作为第一版实现判负

- `exp132` 已完成同 checkpoint 正式对照：
  - `cvk_adaptive = 62.1 / 72.8`
  - `cvk_hybrid  = 62.1 / 72.8`

这说明：
1. 检索期 learned pair module 这个大方向还没有被否定
2. 但第一版实现已经被较干净地否定：
   - 单个 `alpha`
   - 只在两种标量距离之间做凸组合
   - 用 teacher distance 做回归监督
3. 当前最合理的解释不是“learned pair module 没用”，而是：
   - **表示能力太弱**
   - **监督不够 ranking-aligned**

### 当前最值得赌的下一跳

**Ranking-Aligned Pair Scorer / Pair Residual Correction**

核心想法：
1. 不再让 head 只预测 `alpha in [0,1]`
2. 而是直接预测一个 pair-specific residual score / correction score
3. 输入不再只限于两种总距离，而应包含更细粒度的 pair descriptors：
   - `d_global`
   - `d_cvk`
   - `|d_global-d_cvk|`
   - overlap / visibility
   - 必要时再加入更细的 keypoint-wise common-support statistics
4. 监督也不再只是“逼近 teacher distance”，而要更明确地对齐排序目标：
   - pair label
   - pairwise margin
   - 或 teacher-induced relative order

为什么这比继续调 `LTCS` 更合理：
1. `exp132` 已说明 scalar fusion 不足以改变最终排序
2. `exp125` 又说明 pair-specific correction 确实真实存在
3. 因而下一步最自然的升级不是：
   - 更大的 `alpha` 头
   - 更多 bank trick
   - 更复杂的凸组合
4. 而是：
   **从“学信谁”升级成“学该修正多少”**

### 2026-03-21 上午补记：`exp133/134` 当前不能用于创新判断

- `exp133 LPCS` 和 `exp134 Sparse LPCS` 在运行中暴露了共享接线 bug：
  - `kp_aux_data` 的构建条件漏掉了 `ltcs_enabled / lpcs_enabled`
  - 结果是 `LPCS` loss 从未真正被加入训练
  - 日志里也完全没有 `lpcs_*` 统计

- 这意味着：
  1. 当前不能把 `exp133/134` 的数值拿来支持 “pair-specific correction scoring 成立”
  2. 也不能拿它们来判负
  3. 它们的价值只剩下：
     - 暴露了一个共享接线 bug
     - 提醒我们之后所有新主线都必须先看机制统计是否真的激活

- 因而当前正确动作不是切题，而是：
  1. 修复 bug
  2. 用新编号重跑 corrected `LPCS`
  3. 再继续判断 sparse routing 是否真能把 `LPCS` 推成论文主线

### 2026-03-21 晚间更新：`LPCS` 已经真正成立，但 sparse routing 最终只是 supporting 机制

- `exp135 corrected LPCS` 已收敛到：
  - `ep120 = 61.1 / 72.3`
- `exp136 corrected sparse LPCS` 已收敛到：
  - `ep120 = 60.9 / 72.1`
- 更关键的是机制层面：
  - `exp135`: `lpcs_psr / lpcs_pf = 1.000 / 1.000`
  - `exp136`: `lpcs_psr = 0.254`、`lpcs_pf ≈ 3.0`

这批证据把当前创新判断收得更紧了：

1. `LPCS` 本身不是伪命题
   - 修复共享接线 bug 后，full-pair 版本已经明确显示 learned pair correction 能改变排序
   - `lpcs_fg` 长期显著高于 `lpcs_bg`，说明 head 不是空转

2. 但 sparse routing 不是当前最像论文主突破的部分
   - `exp136` 已经第一次把真稀疏 routing 跑成设计语义
   - 可是更干净的机制到收敛也没有超过 full-pair `LPCS`

3. 这意味着：
   - supervision dilution 可能是次要问题
   - 当前更像是 `LPCS` 的 **ranking objective / pair aggregation** 不够贴近最终检索目标

因此当前最合理的主线升级不是：
- 继续扫 `top_ratio`
- 继续做 routing 小变体

而是：
- **Ranking-aligned LPCS**
- 让 pair correction head 更直接对 hardest / top-k hard 的排序错误负责

如果这条线转正，论文主创新会更像：
- `pose-defined common support`
- `learned pair correction`
- `ranking-aligned pair supervision`

### 2026-03-21 深夜更新：`exp137` 给出新的负边界，hard selection 不是答案

- `exp137 Hard-Rank LPCS` 到停表点是：
  - `ep80 = 60.1 / 70.4`
- 对照：
  - `exp135 ep80 = 60.8 / 71.9`
  - `exp125 ep80 = 59.4 / 72.0`
- 但机制一直是对的：
  - `lpcs_rsr = 0.254`
  - `lpcs_psr / lpcs_pf = 1.000 / 1.000`

这条线把判断又收紧了一步：

1. 不是“ranking-aligned”这个方向错了
   - 而是当前 **hard-top 25%** 这种离散 hard selection 太激进

2. 这意味着：
   - full-pair `LPCS` 可能保留了有价值的上下文
   - 直接砍掉 75% pairs 会伤害 top-rank 稳定性

3. 因而下一步更值得做的，不是更硬，而是更平滑：
   - rank-decayed weighting
   - top-sensitive continuous weighting
   - 或直接学习 top-rank residual scorer

### 2026-03-21 转向后的双线候选：`exp138` vs `exp139`

基于 `exp136/137` 的双重负边界，当前最值得并行验证的不是同一机制的两个系数，而是两个不同的问题解释：

1. `exp138 Rank-Decayed LPCS`
   - 假设：`hard-top` 失败不是因为 top-rank 不重要，而是因为选择太离散、太激进
   - 方案：保留 full-pair，上 hardest pairs 用连续 rank-decay 增权

2. `exp139 Query-Context LPCS`
   - 假设：`exp135` 的 `R1` 弱，不是因为权重不对，而是因为 scorer 只看 pair 本身，缺少 query-level context
   - 方案：给每个 pair descriptor 追加 query 的正负均值距离、margin、support 完整度与 teacher change 统计

如果这两条里有一条明显转正，论文主创新会更靠近：
- `pose-defined common support`
- `learned pair correction`
- `top-rank sensitive / context-aware pair correction`

这比“学一个更好的 sparse router”更像 B 类主方法。

### 2026-03-21 审查后修正：`exp138` 可直接验证，`exp139` 必须先改成无标签 context

- `exp138 Rank-Decayed LPCS` 已通过 Claude 全面审查，可以直接启动。
- `exp139 Query-Context LPCS` 当前版本被 Claude 明确驳回，原因不是“想法没新意”，而是：
  1. test-time descriptor 仍是 6 维，而训练版 scorer 已改成 11 维
  2. 当前 query context 用到了 `row_pos_mean / row_neg_mean / row_margin` 等 label-dependent 统计，测试时天然不可得

这说明一个很关键的创新边界：

1. **query-level context 这条问题线仍然值得保留**
   - 因为它仍在回答一个和 `exp138` 不同的问题：
     - 当前 scorer 是否太短视，缺少 query 级语境
2. **但上下文设计必须是 retrieval-time 可获得的**
   - 否则就会退化成“训练时 oracle context，测试时无 context”的不可信对照
3. 因而这条线下一步的正确形态，不是当前版 `exp139`，而是：
   - 无标签
   - train/test 一致
   - 可直接在 evaluator 中构造的 query context

### 2026-03-22 当前收敛：`rank-decay` 退为 supporting，`query-context correction` 升为主候选

- `exp138 Rank-Decayed LPCS` 到停表窗口的结论已经够清楚：
  - `ep80 = 60.7 / 71.7`
  - 对照 `exp135 ep80 = 60.8 / 71.9`
  - 它证明了“平滑 top-sensitive”比 `hard-rank` 合理，但最终只形成 supporting 级别的改进

- `exp139 Query-Context LPCS` 则在当前阶段首次同时给出：
  - `ep20 = 47.6 / 60.0`
  - `ep40 = 57.0 / 68.8`
  - `lpcs_ctxm ≈ 0.46`
  - `lpcs_fg > lpcs_bg`

这让主创新候选开始进一步收紧成：

1. 不是“更聪明地选择哪些 pairs”
2. 而是“给 pair correction 一个更完整的 query-level语境”
3. 从而让同样的 common support 在不同 query 上被不同地解释

如果这条线继续转正，它会比 `rank_decay` 更像论文级机制，因为它在回答一个更有新意的问题：
- **共同可见的身体证据并不是孤立解释的，它依赖 query 当前的整体难度与上下文。**

### 2026-03-22 本地新增候选：`exp140` 测试 correction confidence 是否才是 `R1` 瓶颈

在远程继续跑 `exp139 query-context` 的同时，本地不再沿 `rank` 线修小补小，而是新开一个不同问题解释：

1. 当前 `LPCS` 也许不是不会修正
2. 而是不会判断：
   - 哪些修正应该强信
   - 哪些修正应该被抑制

因此 `exp140` 的核心创新点候选是：

- **Confidence-Calibrated Pair Correction**

它的机制不是再给更多 context，而是让同一个 scorer 同时学：
- `raw_delta`
- `conf`

如果这条线成立，它会支撑另一种论文式表达：

1. pose 定义 common support
2. support-complete teacher 提供 correction 方向
3. confidence calibration 决定 correction 是否该真正落到检索距离上

### 2026-03-22 当前最强创新候选已收紧到两条互补线

到 `exp139 ep50` 这一步，当前创新探索已经开始明显分层：

1. **主候选：Query-Context Pair Correction**
   - 代表实验：`exp139`
   - 当前信号：
     - `ep50 = 58.7 / 70.4`
     - 同时高于 `exp135/138`
     - `lpcs_ctxm ≈ 0.47`
   - 它在回答：
     - 同一份 common support，是否需要放在 query-level 语境里解释

2. **并行候选：Confidence-Calibrated Pair Correction**
   - 代表实验：`exp140`
   - 当前状态：
     - clean rerun 已证明确实接上
     - 但当前版本的 gate 很快塌成接近常数 1，已判当前实现形态不成立
   - 它在回答：
     - pair correction 不仅要会修，还要知道这次修正该不该被信任

### 2026-03-22 本地新候选：Competition-Context Pair Correction

在 `query-context` 与 `confidence-calibration` 之后，当前最新的本地候选不再问：

1. query 整体难不难
2. correction 该不该被信任

而是问一个真正不同的问题：

**当前这个 candidate，在本 query 的所有候选竞争里到底处于什么位置？**

因此 `exp141` 的核心创新点候选是：

- **Competition-Context LPCS**

它给每个 pair 新增的不是 query 级常数摘要，而是 pair-specific 的竞争特征：

1. `base_rank`
2. `kp_rank`
3. `support_rank`
4. `gain_rank`
5. `gain_zscore`

如果这条线成立，它会支撑一种比 `query_ctx` 更 retrieval-native 的 story：

- pose 定义 common support
- pair correction 不仅看 query 全局语境
- 还要看当前 pair 在整个候选竞争中的相对位置

这两条线的共同点是：
- 都还紧扣 `exp109` 的核心发现：单图 support 不完整
- 都没有退回去做 generic backbone/module 堆叠

区别在于：
- `exp139` 强调 **如何解释 common support**
- `exp140` 强调 **如何应用 correction**

### 2026-03-21 本地大转向：从 pair correction 切回 feature-space support completion

`exp141` 虽然二审通过，但它本质上仍是 `LPCS` 家族里的 context 变体。用户明确要求不要继续围绕同一个小点试小改动，因此本地主线需要真正跳出：

- `query context`
- `competition context`
- `confidence gate`
- `rank weighting`

这些 retrieval-side scorer 变体。

当前新的大方向是：

- **SKC（Support-Conditioned Keypoint Completion）**

它的核心问题定义不是：

- “这个 pair 该怎么修正距离”

而是：

- **这张图的哪些 pose-aligned 证据本来就缺失，能不能在特征层被条件补全出来？**

这条线仍然强依赖 pose，但方式完全不同：

1. pose 不再只是用来构造 `common support distance`
2. pose 现在要定义：
   - 哪些关键点低置信
   - 哪些高置信关键点可作为自证据
   - 哪些 support prototype 可作为跨图补全证据
   - 哪些 skeleton 邻接关系约束 completion

如果这条线成立，它比 `LPCS` 系列更像真正的方法级创新，因为它回答的是：

- `exp109` 暴露出的单图 support incomplete，能否在编码阶段被修复

而不是继续在检索距离上打补丁。

---

## 2026-03-22: feature-level completion 方向彻底证伪，转入注意力 inductive bias

### exp142 SKC 最终结论

exp142 (Support-Supervised Keypoint Completion) 最终 mAP 60.3%（equal_concat），相对 exp030a -0.8% mAP / -1.9% R1。

这是 feature-level completion 方向的第 5+ 次尝试，全部失败：
- exp048 SGMKC: -1.6%
- exp084 CIPGFR: -0.2%
- exp091 TTSFR: -0.2%
- exp092 LSRM: -0.7%
- exp142 SKC: -0.8%

**根本原因分析**:
1. 15K 数据集无法学习复杂的 completion 函数
2. support bank / EMA prototype 的质量本身受限于数据量
3. completion module 的 gate 无法学会"该修改多少"——要么太保守（不起作用），要么太激进（破坏特征）

### 新方向: SASA (Skeleton-Aware Self-Attention)

从"修改特征值"转向"修改注意力路由"。核心区别:
- PSG/PAA/SKC: 修改 **what** (特征本身)
- SASA: 修改 **how** (token 之间的注意力分配)

SASA 使用骨架测地距离作为 Swin window attention 的固定偏置，零参数。

如果 SASA 也中性，说明 15K 数据集上注意力偏置也不够有效。下一步应考虑:
1. **PGCO** (课程式遮挡增强): 修改数据分布而非模型
2. **SCFA** (对称特征对齐): 利用人体双侧对称性
3. 或接受训练端已到天花板，转向 SGCFR 优化

### 已确认无效的完整方向列表 (截至 exp142)
1. PSG + forward path 添加 (21 实验)
2. PSG + 正则化/dropout
3. PSG + loss 调制 (PCRA, CSGT, GKD, DPF, etc.)
4. PDS Part 收敛改善
5. Post-hoc pooling 改进
6. Feature-level completion (SGMKC, CIPGFR, TTSFR, LSRM, SKC)
7. Token merging (PGTM)
8. Transformer decoder (PQTD)
9. Attention supervision (PCQA)
10. OT matching (exp099)

## 2026-03-22 上午方向重置：从“小修 scorer / completion”转向两条更大的新机制

昨晚的收口结果把几件事讲得更清楚了：

1. retrieval-side pair correction 不是完全无效，但 `query_ctx / comp_ctx / confidence gate` 都还没有长成主方法
2. feature-level completion 也不是“没接上”，而是反复接上后依然不成立
3. skeleton attention bias 这类纯 inductive bias 在 Swin-Tiny 上基本中性

因此接下来不能再继续：
- scorer 小修补
- gate 小调参
- attention bias 小改造

而应转向两条更大的候选：

### 候选 A: PCVT（Pose-Complementary View Training）

核心问题：
- 单图 support incomplete 能否通过 **pose-defined complementary pseudo-views** 改写成“伪多 support 学习”？

核心机制：
- 不是随机遮挡
- 而是基于姿态热图，把可见 body groups 分成互补两组
- 构造 `full / view_a / view_b`
- 让 `view_a` 与 `view_b` 的联合表示逼近 `full`

为什么比 PAMC/ROA/PADE 更大：
- PAMC 只是 body-aware masking consistency
- ROA/PADE 只是增强 recipe
- PCVT 直接改写训练对象，把单图变成“互补 support 组合体”

### 候选 B: SCFA（Symmetry-Conditioned Feature Aggregation）

核心问题：
- 当前 keypoint branch 是否浪费了 **单图内部的双侧同源冗余**？

核心机制：
- 左右同源关节不再完全独立
- 用同源聚合 token 表示“体部证据”
- 用非对称残差 token 保留左右差异

为什么比 direct completion 更合理：
- 不依赖 same-ID teacher
- 不依赖 memory bank
- 只利用单图内部更稳定的结构先验

### 当前判断

这两条线都比继续做：
- retrieval scorer 小修补
- feature completion 小变体
- attention bias 小变体

更像真正有机会支撑主故事的新机制。

## 2026-03-22 下午更新：`PCVT` 与 `SCFA` 已出现第一轮明显分化

### `PCVT` 的当前价值

`PCVT` 现在是少数真正跑出连续验证正信号的新方向：
- `ep10: 40.2 / 51.4`
- `ep20: 49.1 / 60.7`
- `ep30: 54.6 / 65.8`

相对 `exp030a`，它已经连续三次表现为：
- `mAP` 明显更高
- `R1` 基本持平或仅轻微波动

这意味着：
1. “把单图改写成 pose-defined complementary pseudo-views” 至少不是空想
2. `single-image support incomplete` 可能确实更适合被改写成“伪多 support 学习对象”，而不是继续做 scorer / completion 小修补
3. 当前真正要验证的核心问题已收紧为：
   - 它最终能否同时兑现到 `R1`
   - 还是会收敛成 `mAP` 导向 recipe

### `SCFA` 的当前结论

`SCFA` 已足够判负，不值得继续当主线：
1. 它不是完全没接上
2. 但 `scfa_pg ≈ 0.086~0.093` 说明真正有用的 bilateral gap case 太少
3. 这意味着“单图内部双侧冗余”在当前 benchmark 上不是足够强的主问题

### 对创新点判断的更新

这轮结果把创新排序进一步收紧成：

1. **更值得继续追的方向**:
   - `PCVT` 这类直接改写训练对象的方向
2. **已被快速排除的方向**:
   - `SCFA` 这类依赖单图双侧冗余前提的新结构
3. **仍需警惕的风险**:
   - 即便 `PCVT` 成立，也必须证明它不是“更复杂的数据增广/一致性 recipe”

## 2026-03-24: 文献调研确认 — 我们的独特位置

### SOTA 参考 (Occluded-Duke)
- OGFR (2025.7): 76.6/64.7 (ViT, teacher-student distillation)
- KPR+SOLIDER (ECCV24): ~82/73 (Swin-L, keypoint prompt at test time)
- PAB-ReID (2024): 72.6/63.5 (ViT, human parsing)

### 我们的位置 (Swin-Tiny)
- STD-PR+PLBOA: 63.4/73.4 eq, 63.6/73.6 maxsim
- PLBOA+GCN+MaxSim: 64.1/75.0
- PLBOA+GCN+SGCFR: 65.2/75.3

### 确认的独特组合
没有论文同时做过：
1. Pose-aware backbone injection (PSG)
2. Structural body-part tokens via cross-attention (STD-PR)
3. Pose-guided lower-body occlusion augmentation (PLBOA)
4. ColBERT-style late interaction for matching (MaxSim)

### 论文可行定位
- STD-PR 的 PLBOA synergy (+4.7 vs +1.6) 是核心发现
- MaxSim 是 test-time 的独立贡献
- PLBOA 是 data-side 的独立贡献
- 三者组合形成完整的 pose-guided pipeline

---

## 2026-03-24: 关键发现 — per-token 是训练技巧，test 用 pooled 更好

### 实验证据
- exp166 per-token training + pooled test: **63.1/73.9** ← 实际运行结果
- exp166 per-token training + per-token concat test: **61.8/72.5** ← 修复后 test.py 评估
- 差异: **-1.3 mAP / -1.4 R1**

### 解释
1. Per-token CE 强制每个 token 独立判别 → training diversity（更好的特征学习）
2. 但 test 时 6 个 L2-normalized tokens concat (5376-d) 稀释了信号
3. Confidence-weighted pooling (1536-d) 保留了最重要的信息
4. 类比: Dropout 在训练时 mask neurons，但 test 时用全部 neurons

### 对后续方向的影响
1. Per-token 的价值在于 TRAINING 正则化，不在于 test 特征结构
2. Test 特征应该是 compact pooled 形式，不是 long concat
3. DPTL (exp167) 仍然合理：self-attention refined tokens → better pooled feature
4. 17-token (exp168) 的 test 也应该用 pooled，不用 per-token concat

### exp166 完整结果表

| 配置 | mAP | R1 | R5 | R10 | 备注 |
|------|-----|----|----|-----|------|
| exp166 (per-token+PLBOA) ep120 | 63.1% | 73.9% | 86.1% | 89.5% | pooled test (best) |
| exp166 (per-token+PLBOA) ep100 | 62.9% | 74.5% | 86.2% | 89.3% | peak R1 |
| exp166 bugfix (concat test) ep120 | 61.8% | 72.5% | 85.0% | 89.1% | per-token concat test |
| exp166r (per-token, 无PLBOA) ep120 | 60.3% | 72.8% | 83.6% | 87.0% | PLBOA 贡献: +2.8/+1.1 |

---

## 2026-03-25: 创新性调研结论 — Per-Token SupCon + PLBOA Synergy

### 文献调研确认的独创性

| 组件 | 先例程度 | 我们的差异化 |
|------|---------|------------|
| PSG (backbone pose gate) | 低 — 无直接先例 | Swin block 内乘法 gate |
| STD-PR (cross-attn tokens) | 中 — PAFormer 类似 | pose bias init vs MSE supervision |
| **Per-token SupCon** | **无先例** | **首次在 ReID 中对 body-part token 用 SupCon** |
| **PLBOA (解剖定位增强)** | **低** | **基于 train-test gap 分析定位下半身** |
| **SupCon × PLBOA synergy** | **无先例** | **loss-augmentation 超线性交互** |

### 核心创新点（论文主 contribution）

**Part-Level Supervised Contrastive Learning with Occlusion-Augmentation Synergy**

1. 首次在 occluded ReID 中对 structural body-part tokens 应用 supervised contrastive loss
2. 发现 SupCon × PLBOA 超线性 synergy（SupCon 单独无效，必须与 PLBOA 配合）
3. 机制解释：PLBOA 创造部分 token 缺失的训练条件，恰好是 per-token contrastive 最擅长的 regime
4. Cross-attention (STD-PR) 比 keypoint sampling (GCN) 更善于利用 augmentation (+4.7 vs +1.6)

### 消融证据链

| 配置 | mAP | R1 | 论文作用 |
|------|------|------|---------|
| Baseline (CE) | 56.6% | 66.5% | 出发点 |
| +PSG | 58.3% | 67.9% | 贡献 1: backbone injection |
| +STD-PR+per-token+PLBOA (CE) | 63.1% | 73.9% | 贡献 2+3: structural tokens + augmentation |
| +SupCon (replace CE) | **64.1%** | **75.5%** | **贡献 4: per-token SupCon (核心)** |
| SupCon only (no PLBOA) | 59.8% | 70.4% | 证明 SupCon 依赖 PLBOA |
| SupCon on base (no arch enhance) | 64.2% | 74.9% | 证明 SupCon > 所有架构增强 |

### 详细文献调研结果（3 个 Opus Agent 并行搜索）

#### 完全没有先例的方向
1. **Per-token SupCon for ReID** — 搜索了 MCLNet, SSSC-TransReID, PCL-Former, ABC-Learning, CION, TokenMatcher, BPBreID, PAB-ReID, SORN, SDCL, POFR — 全部用 global/prototype contrastive 或 CE+triplet
2. **SupCon × PLBOA synergy** — 没有论文报告过 contrastive loss 和 occlusion augmentation 的超线性交互
3. **PLBOA 解剖定位** — 现有 ROA (PADE, POFR) 全部用随机矩形，没有 pose-guided hip 定位

#### 最接近的竞争者
- **PAFormer (2024)**: pose tokens + cross-attention，但用 MSE 监督 attention maps（不是 SupCon），且无 PLBOA
- **BPBreID/GiLt (WACV 2023)**: per-part training，但用 parsing masks（不是 pose heatmaps），用 CE+triplet（不是 SupCon）
- **KPR (ECCV 2024)**: keypoint prompts，但 test-time only，不是 training 方法

#### 推荐论文 Story
**"Training Objective Matters More Than Architecture for Occluded Person ReID"**

核心 contrarian claim: SupCon 在 base 架构上 (+3.9/+2.1) 超过所有架构增强的总和 (+2.8/+1.1)。

三个论文贡献：
1. Per-token SupCon on structural body-part tokens — 首次在 ReID 中
2. SupCon × PLBOA 超线性 synergy — 新发现
3. STD-PR 的 augmentation adaptivity — cross-attention 比 keypoint sampling 更善于利用 PLBOA (3× gain)

---

## 2026-03-26: OA-SD 系列实验总结 + Global-Only 新发现

### OA-SD 核心特性（exp191-194 消融）

1. **OA-SD + CE = 强正向**: +2.9/+2.6 vs CE base (exp191)
2. **OA-SD + SupCon (all-token) = 负向**: -0.7/-0.4 (exp188) — 梯度冲突
3. **EMA decay 不敏感**: 0.99 vs 0.999 最终差异 <1% (exp192)
4. **Loss weight 不敏感**: 1.0 vs 2.0 最终差异 <1% (exp194)
5. **OA-SD + 3-view 是 additive**: exp193 = 64.4/76.5 vs exp190 = 64.2/75.6 (+0.2/+0.9)
6. **OA-SD late-stage boost**: ep40 前拖累 → ep40 crossover → ep60+ 大幅正向

### 关键新发现：OA-SD Global-Only 解决 SupCon 梯度冲突

**问题**: OA-SD all-token distillation 与 SupCon 在 per-token features 上产生梯度冲突
**机制**: 
- SupCon 鼓励同 ID tokens 拉近、异 ID 推远
- OA-SD distillation 鼓励 student tokens 逼近 teacher tokens（不管 ID）
- 两者在 token 级别方向矛盾

**解决方案**: OA-SD GLOBAL_ONLY — 只在 global (GAP后) feature 上做 distillation
- Global feature: CE + triplet + OA-SD distill（三者协同）
- Per-token features: CE + triplet + SupCon（三者协同，无 OA-SD 干扰）

**验证** (exp195): SupCon + OA-SD global-only ep70=60.2/73.4
- 没有出现 exp188 的负向效应
- R1 稳定领先 CE+OA-SD（SupCon 的 R1 优势保持）

**论文定位**: 
- 这是一个新的 **"职责分离" (role separation)** 机制
- 全局遮挡不变性 (OA-SD) 和局部判别力 (SupCon) 在不同特征级别独立优化
- 消融链: exp188 (冲突) → exp195 (分离) 是清晰的证据

### 当前最强结果总表

| 排名 | 实验 | 方法 | mAP | R1 |
|------|------|------|------|------|
| 1 | exp187 | 3-view + SupCon | 64.9% | 76.6% |
| 2 | exp193 | 3-view + OA-SD + CE | 64.4% | 76.5% |
| 3 | exp190 | 3-view + CE | 64.2% | 75.6% |
| 4 | exp176 | SupCon (1-view) | 64.1% | 75.5% |
| 5 | exp194 | OA-SD + CE (w=2.0) | 63.4% | 74.8% |
| 6 | exp191 | OA-SD + CE | 63.2% | 75.4% |
| 7 | exp166 | CE baseline (full) | 63.1% | 73.9% |

### 待验证: exp196 终极配置

3-view + SupCon + OA-SD global-only — 预计 65.0-65.5/77.0-77.5

---

## 2026-03-27: 多 Agent 并行调研 — 新方向候选

### 调研范围（5 个 Explore Agent 并行）
1. Token Pruning for ViT/ReID
2. Contrastive Distillation (RKD/CRD)
3. Cross-Attention Innovation
4. Part-Aware Losses
5. Performance Ceiling Analysis

### 关键发现

**1. Relational Knowledge Distillation (RKD) — 最有前途 ⭐⭐⭐⭐⭐**
- RKD (CVPR 2019): distill pairwise distance/angle structure
- CRD (ICLR 2020): distillation AS contrastive learning (InfoNCE-based)
- **直接解决 OA-SD vs SupCon 冲突**: 不 match 个体特征(会冲突) → match 关系结构(不冲突)
- 实现: ~100 行 (pairwise cosine sim + KL divergence)

**2. Performance Ceiling Analysis**
- 我们 (Swin-Tiny): 64.9/76.6 — 与 OGFR (64.7/76.6) 持平!
- Gap vs FRT (66.2/78.2): mAP -1.3, R1 -1.6
- **mAP 是瓶颈**，R1 已接近 SOTA

**3. Batch-Mate Keypoint Cross-Attention (BMKCA)**
- 同 batch 同 ID 图像间 cross-attention 补全被遮挡部位
- ~300 行实现
- 需要仔细处理 batch 内 ID 分组

**4. 已证伪的方向更新**
- STM (Token Mixup): 只加速不改善天花板 (exp197/198)
- OA-SD + SupCon: 互斥，即使 global-only 也无法叠加 (exp195/196)

### 选定方向: OA-RD (exp199)
Occlusion-Asymmetric Relational Distillation
- Teacher: clean image → global feat → pairwise similarity matrix
- Student: occluded image → global feat → pairwise similarity matrix  
- Loss: KL(teacher_sim_softmax || student_sim_softmax)
- 与 SupCon 不冲突（操作对象不同）

---

## 2026-03-30: EMA Distillation 与 SupCon 互斥性确认

### 完整实验链

| 实验 | 方法 | 路线 | vs 最佳 | 结论 |
|------|------|------|---------|------|
| exp188 | OA-SD all-token + SupCon | SupCon | -0.7/-0.4 | ❌ 梯度冲突 |
| exp195 | OA-SD global-only + SupCon | SupCon | ~-2.8 mAP | ❌ 信号太弱 |
| exp196 | OA-SD global-only + SupCon + 3v | SupCon | -2.5/-1.4 | ❌ 同上 |
| exp199 | OA-RD relational + SupCon + 3v | SupCon | ~-1.5/-3.4 | ❌ 关系级也冲突 |
| exp191 | OA-SD all-token + CE | CE | +2.9/+2.6 | ✅ CE 兼容 |
| exp193 | OA-SD all-token + CE + 3v | CE | +0.2/+0.9 | ✅ CE 兼容 |
| exp200 | OA-RD relational + CE | CE | ~-1.0/-3.4 | ❌ OA-RD 不如 OA-SD |

### 核心结论

1. **EMA self-distillation 与 SupCon 本质互斥** — 无论 distill 什么（feature/relation），都不行
2. **OA-SD (feature distillation) 在 CE 路线有效**，但 OA-RD (relational) 在 CE 路线不如 OA-SD
3. **最佳配置已确立**：
   - SupCon 路线: exp187 = 64.9/76.6 (不加任何 distillation)
   - OA-SD 路线: exp193 = 64.4/76.5 (不加 SupCon)
4. **OA-RD 是负结果** — relational distillation 的 KL divergence 信号太弱且不稳定

### 下一步方向（非 distillation）

从 5 个研究 agent 的建议中剩余可行方向：
1. **BMKCA** — batch-mate cross-attention part completion
2. **Multi-Granularity Contrastive** — 多粒度 SupCon (不同 temperature)
3. **Deformable Part Tokens** — 动态 part query
4. **Mixture of Visibility Experts (MoVE)** — 按 visibility routing 到不同 expert

但 ceiling analysis 指出 Swin-Tiny 的天花板约 65-66% mAP，我们已在 64.9%。
剩余空间 ~1%，需要权衡是否值得大改。

---

## Phase 4: Swin-Small/Base Scaling + MaxSim (2026-03-31)

### 重大发现: MaxSim Hybrid Test-Time Matching

**MaxSim Hybrid 在 exp206 checkpoint 上无需重训即可获得 +1.8% mAP！**

| Test Mode | mAP | R1 | delta vs equal_concat |
|-----------|------|------|----|
| equal_concat (baseline) | 70.3% | 81.8% | — |
| **maxsim_hybrid (gw=1.0)** | **72.1%** | **82.9%** | **+1.8/+1.1** |
| maxsim (pure) | 69.3% | 81.1% | -1.0/-0.7 |
| cvk_hybrid | 70.7% | 81.9% | +0.4/+0.1 |
| maxsim_hybrid (gw=0.5) | 71.5% | 82.7% | +1.2/+0.9 |

**机制**: MaxSim 是 ColBERT 风格的 late interaction matching:
- 对 query 的每个 keypoint feature，找 gallery 中最相似的 keypoint
- 用 keypoint confidence 加权平均
- 与 global distance 混合 (hybrid mode)

**为什么有效**: 
- pure MaxSim (69.3) < global (70.3) → part-only matching 不够可靠
- MaxSim hybrid (72.1) > equal_concat (70.3) → 但 part matching 提供了 global 没有的 part alignment 信号
- 这证明 GCN-enhanced per-keypoint features 有显著的 matching 信号，只是之前被 equal_concat 的简单拼接浪费了

**论文价值**: 这不是 reranking，是 model-integrated matching mechanism。可以作为方法论的一部分。

### Backbone Scaling 发现

| Backbone | mAP (equal_concat) | 预期 + maxsim_hybrid |
|----------|------|------|
| Swin-Tiny | 64.9% | ~67% |
| Swin-Small | 70.5% | **72.1%** (已确认) |
| Swin-Base (ep40) | 66.6% (3-view, 进行中) | ~68%+ |

Base 在 ep40 trailing Small — 可能因为 LR 过低 (0.0002)。但后期增长更强，预期 final 73-75%。

### 到达 76% mAP 的路径

1. **Swin-Small + GCN+PAA+OA-SD + maxsim_hybrid = 72.4%** (exp210b with PKC=0.05)
2. 训练端改进: 目前所有尝试均未超过 OA-SD-only ceiling
3. 目标: **74-76% mAP on Small, 无 NFC/reranking**

---

## Phase 5: Per-Keypoint Training Loss 全面探索 (2026-04-02)

### 核心发现: detached GCN 是架构瓶颈

**所有 per-keypoint training innovations 失败:**

| 实验 | 方法 | detach? | vs OA-SD-only |
|------|------|---------|------|
| exp210 | PKC w=0.5 | detached | 灾难 3.6% |
| exp210b | PKC w=0.05 | detached | 无效 (=baseline) |
| exp211 | MST w=0.5 | detached | 完全无效 (所有 loss 一致) |
| exp213 | PKC+MST 组合 | detached | 灾难 40.6% |
| exp215 | BA-PKC non-detach | non-det | 灾难 0.5% |
| exp217 | OERL non-detach cosine | non-det | `62.2/75.2`，相对 `exp191 63.2/75.4` 为 `-1.0/-0.2` |
| exp218 | PACI prototype bank | detached | `61.9/74.2`，相对 `exp191 63.2/75.4` 为 `-1.3/-1.2` |
| exp219 | PACI without OA-SD | detached | 远程日志当前只确认到 `ep30=51.9/64.9`，早期即落后 baseline `52.2/65.2` |
| exp220 | GSPB gradient scale 5% | 5% scale | `62.9/74.3`，相对 `exp191 63.2/75.4` 为 `-0.3/-1.1` |

**根本原因:**
1. detached: 梯度不到 backbone (50M) → 只更新 GCN (200K) → 无效
2. non-detached: 与 CE 冲突 → 灾难
3. gradient scaling: 加速早期 (+5.8% ep10!) 但 final 持平

### MaxSim 行为修正

早期只看 `OA-SD / OERL / PACI` 三条 Tiny 线时，`maxsim_hybrid` 确实都落在 `64.1~64.3`；
但这个“~64.2 ceiling”后来被新日志推翻了。

| 方法 | equal_concat | maxsim_hybrid |
|------|------|------|
| OA-SD-only | 63.2 | 64.2 |
| OERL+OA-SD | 62.2 | 64.3 |
| PACI+OA-SD | 61.9 | 64.1 |
| GSPB+OA-SD | 62.9 | 64.6 |
| PADPQ+OA-SD | 63.7 | 63.9 |

更准确的结论是：
1. `MaxSim` 对 OA-SD 本身仍然有效（`63.2 -> 64.2`）
2. `GSPB` 改善了 per-keypoint consistency，因此 `MaxSim` gain 更大（`+1.7`）
3. `PADPQ` 的 deformable sampling 破坏 cross-image keypoint consistency，因此 `MaxSim` gain 很小（`+0.2`）

### GSPB (Gradient-Scaled Part Branch) — 有价值的发现

exp220 (scale=0.05) 完整对照:

| Epoch | GSPB mAP | OA-SD mAP | delta |
|-------|------|------|------|
| 10 | 40.1 | 34.3 | **+5.8** |
| 20 | 49.1 | 46.0 | **+3.1** |
| 30 | 54.5 | 50.6 | **+3.9** |
| 60 | 59.8 | 60.6 | -0.8 |
| 120 | 62.9 | 63.2 | -0.3 |

**3x 早期收敛加速！** 但 final 持平。训练效率提升有实用价值。

### 结论与方向

**GCN+PAA+OA-SD 在 Tiny 上已达极限 ~63%。** 需要:
1. 完全不同的 Part 架构（不在禁止列表内的方向）
2. 重新定义训练范式（不是 loss 增改）
3. 利用 Small/Base scaling 的已有成果 (72.4% maxsim on Small)
