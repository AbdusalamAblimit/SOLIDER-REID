# 创新点头脑风暴 — Phase 2: Pure Pose Heatmap

## 2026-04-15 PRCV 重审：回到 PSG 主线，重做 multi-stage 消融

### 这轮重审后的核心判断

1. `PSG` 仍然是当前最稳的主创新点  
   - `exp007` 单次 `58.3 / 67.9`
   - 3-seed mean `57.83 / 67.13`
   - backbone injection 明确优于 post-hoc pooling

2. `2-stage PSG` 有希望作为最终版本，但现有证据还不够干净  
   - `exp009 / exp251 / exp253` 说明 multi-stage **不是普遍自动更优**
   - `exp255 vs exp255b` 又强烈说明：在 `GCN512` 结构分支下，`2-stage PSG` 是关键条件

3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
   - `LGPA-D` 像 detached semantic part asset
   - `MaxSim/POT/flip` 主要是 test-time
   - `exp257-259` 已说明 recipe 空间基本耗尽

### 当前真正该补的不是新故事，而是干净消融

用户已明确说明所有实验都可以重跑，因此下一步最该做的是：

1. 不再把旧结果当最终消融闭环
2. 重新设计 `PSG` / `2-stage PSG` / `3-stage PSG` 的干净对照
3. 把“multi-stage PSG 什么时候有用”这件事说清楚

### 当前推荐验证顺序

1. **基础 PSG 消融**
   - no PSG
   - 1-stage PSG
   - 2-stage PSG
   - 3-stage PSG

2. **结构分支依赖性消融**
   - GCN256 + 1-stage
   - GCN256 + 2-stage
   - GCN512 + 1-stage
   - GCN512 + 2-stage

3. **必要时再补 semantic 分支依赖性**
   - LGPA-only + 1-stage / 2-stage
   - LGPA+GCN + 1-stage / 2-stage

### 当前主线口径

从现在开始，PRCV 方向优先写成：

- `PSG` = 主创新
- `2-stage PSG` = scalable extension / 当前最终版本
- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets

### 参考

详细文献压缩与路线说明见：
`experiments/paper_notes/2026-04-15_prcv_reset.md`

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

---

## 2026-04-03: BT-PKD 系列 — Non-Detached Gradient 全面证伪

### 实验
- exp229: BT-PKD constant (Tiny) → -1.0/-0.4
- exp230: BT-PKD constant (Small, no PAUG) → ~0/-0.7 (ep110)
- exp231: BT-PKD cosine decay (Tiny) → -1.5/-1.1
- exp232: BT-PKD cosine decay (Small) → terminated ep37

### 核心发现
1. **Cosine distillation 是唯一在 Small 上存活的 non-detached 梯度**
   - BA-PKC (SupCon): 0.5% → 灾难
   - GSPB ≥0.01 (CE/triplet): 2.3-15.1% → 灾难
   - BT-PKD (cosine dist): 48.4% at ep10 → 存活
   
2. **所有 non-detached 方法都展现相同模式**:
   - ep10-30: +3~5% 早期加速
   - ep60-90: -1~-2% 后期干扰
   - Final: ~-1% vs baseline
   
3. **Cosine decay 不解决问题**: 干扰在 active 阶段已造成

### 根本限制
**任何额外的 backbone 梯度 — 无论类型 (CE/SupCon/cosine distillation)、
无论 schedule (constant/decay) — 都会在后期干扰 backbone 的 CE+triplet 收敛。**

这不是"找到正确的梯度类型"的问题，而是"backbone 在后期需要完全由主 loss 驱动"的根本性约束。

### 对后续方向的影响
1. **Non-detached gradient 方向完全关闭** — 不再尝试任何变体
2. **Detached Part branch 是正确的** — Part 必须在 detached 特征上工作
3. **创新必须在 detached 范围内寻找** — 或者完全跳出 Part branch 框架
4. **BT-PKD 的实用价值**: 如果只需要 60 epoch 训练，BT-PKD 可以用作训练加速器

### 下一步方向思考
1. **不再试图让 Part 梯度到达 backbone** — 这条路已彻底证伪
2. **改善 Global branch** 而不是 Part branch — PSG 已证明有效，还有什么可以在 Global 上做？
3. **改善 test-time matching** — MaxSim 有 +1.7%, 还能更好吗？
4. **跨领域迁移新机制** — 需要读新论文寻找灵感

---

## 2026-04-03: Paper Search Update — KPR (ECCV 2024) = 75.1% SOTA

### KPR vs Our System

| Aspect | KPR (ECCV 2024) | Our System |
|--------|------|------|
| Backbone | ViT + SOLIDER pre-training | Swin + SOLIDER pre-training |
| Pose injection | Additive prompt tokens | PSG multiplicative gating |
| Part features | Per-body-part independent | GCN pooled into one |
| Part training | Per-part ID + triplet (GiLt) | Pooled Part ID + triplet |
| Test matching | Part-to-part visibility-weighted | MaxSim hybrid |
| Occluded-Duke mAP | **75.1%** | **72.4%** (Small maxsim) |
| Gap | — | **-2.7%** |

### 最大差异分析

1. **ViT vs Swin**: KPR 用 ViT backbone (token-based), 更适合 per-part token 操作
2. **Per-part independent training**: KPR 每个 body part 有独立 classifier + triplet → 更好的 part 判别力
3. **Visibility-aware matching**: KPR 用 sqrt(v_q * v_g) 作为 pair-specific 权重

### 我们能做什么

1. **Per-part GCN loss**: 不是把 17 keypoints pool 成 1 feature，而是分成 6 body parts，每个独立训练
   - 这与 STD-PR per-token 类似，但在 GCN features 上
   - 需要 6 个独立 classifier + 6 个 triplet loss
   - 操作在 detached features 上，无 backbone 干扰

2. **更好的 test-time matching**: 用 visibility-weighted part-to-part distance 而不是 MaxSim
   - sqrt(v_q * v_g) weighting (KPR 的做法)
   - 这可以直接测试，无需训练

3. **Accept that 72.4% is near our ceiling**: Swin-Small + our pipeline may not reach 76% without ViT backbone

---

## 2026-04-04: PPA — Pose-Prompted Part-Assignment Head (范式级创新候选)

### 核心洞察

236 个实验后的根本发现：**所有在 detached features 上操作的方法都无法改善 final 结果。**
- BT-PKD: detached 上 cosine distillation → 无效
- FSDC: detached 上 feature completion → 无效
- Per-part: detached 上独立训练 → 无效
- Per-keypoint losses: detached 上任何 loss → 无效

**唯一有效的方向是改变 backbone 本身**: PSG (+1.7%), OA-SD (+2-3%), PLBOA (+1.5%)

### PPA 方案

**从 "detached GCN sampling" 到 "learnable part assignment, end-to-end"**

1. 在 Stage 3 输出后，添加 `(768 → K+1)` 线性层
2. 对每个 spatial token 预测 part assignment probabilities (softmax)
3. 用 pose heatmaps (resized to 12x4) 作为 GT 监督
4. Per-part weighted pool → K part embeddings
5. 所有梯度端到端流过 backbone

### 为什么不会像 GSPB/BT-PKD 那样干扰

- GSPB: 17 个独立 per-keypoint losses → 高方差梯度冲突
- BT-PKD: cosine distillation → 温和但仍是额外梯度
- **PPA: 单个 softmax cross-entropy → clean segmentation gradient，与 CE ID loss 同类型**

KPR 证明这个 exact mechanism 在 SOLIDER+Swin 上有效 (75.1%)。

### 与现有 pipeline 的关系

PPA 替换 GCN Part branch，但保留 PSG + OA-SD + PLBOA。
这是我们缺少的最后一块拼图。

---

## 2026-04-08: CCF-B 创新方向深度分析

### 250 实验的根本发现

1. **只有 backbone 修改有效**: PSG (+1.7%), OA-SD (+2-3%), LGPA-D (+2.1%)
2. **Detached 分支无法超越 backbone 质量上限**: 任何在 detached features 上操作的方法都无法改善 final
3. **Non-detached 梯度干扰**: 无论梯度类型/schedule, 额外 backbone 梯度都干扰后期收敛
4. **训练集 95.8% 可见**: 所有依赖训练端 visibility 信息的方法失败

### 候选方向评估 (深度 agent 分析)

| 方向 | Novelty | 风险 | 评估 |
|------|---------|------|------|
| POT (Partial OT) | 5/10 | 5×5 太小, 可能 ≈ MaxSim | 值得快速测试, 不作主线 |
| CPRE (Cross-Part Relations) | 7/10 | 关系编码 pose 而非 identity, HOReID 等已有 | 可能 ≤ MaxSim |
| AQGP (非对称 Query-Gallery) | 8/10 | 退化为 visibility-weighted pooling | 概念好但实现退化 |
| Pose-Conditioned Masking | 6/10 | 类似 PLBOA 但在 feature 层 | 低风险, 可能有 marginal 增益 |

### 核心瓶颈诊断

**"The real gap is training data diversity, not model architecture."** — 深度 agent 分析

当前系统 (73.0% mAP Small MaxSim) 距离 SOTA (KPR 75.1% ViT) 仅差 2.1%。
考虑到 Swin vs ViT 的 backbone 差异, 73% 可能接近 Swin+SOLIDER 的天花板。

### 当前最佳策略

1. **短期**: 完成 exp249 (Small LGPA-D+GCN) → 可能 73-74% mAP
2. **快速测试**: POT test-time eval → 判断是否作为 secondary contribution
3. **论文策略**: LGPA-D (CLIP 语义 part assignment) 作为核心贡献, 配合完整 pipeline + 详细消融
4. **如需更强创新**: 需要跳出当前 Swin-Tiny/Small + detach 框架
   - 换 ViT backbone (用户 4090 已在跑)
   - 或找到全新的问题定义

---

## 2026-07-13：exp370 PBSR 严格因果门禁后的创新判断

### 原假设

将 LGPA/PAFormer 式单向部位读取改为共享路由双向结构重组：空间特征读取结构槽，槽间推理后沿同一路由写回标准 global；pose 只监督 detached routing target，推理不依赖姿态。

### 证据

- novelty/实现/梯度/无姿态推理/CUDA smoke 均通过，机制没有工程性失效。
- 同机同运行时 epoch 60：PBSR-off B0 `54.5/63.8`，PBSR P0 `54.4/63.7`，差 `-0.1/-0.1`。
- route loss 明显下降，alpha、entropy、background share 与 delta norm 均健康，但 identity global 不受益。
- 完整 mAP 差序列 `+0.8/-4.7/-0.7/-1.4/+0.2/-0.1` 说明孤立中间正点不稳定。

### 收缩后的判断

1. “姿态监督的内部结构路由学会了”不等于“身份表征更好”；监督可辨认性与检索效用必须分开证明。
2. 共享 read/write 与无姿态推理仍有概念差异，但没有效果证据就不能成为论文贡献。
3. 主 B0/P0 门禁已失败，uniform/shuffled/write-off 不再值得投入；因而不能宣称正确 pose 对应优于伪控制。
4. 不再沿结构槽数量、write scale、mixer 深度或 loss 权重堆小变体，也不扩跨 backbone。
5. 历史 LGPA 的价值边界保持为“pose/part 分支在特定系统与融合中有信号”；PBSR 负结果不能上升为 pose 普遍无效，也不能挽救 LGPA 与 PAFormer 的新颖性重合问题。

**结论**：PBSR 从主创新候选降为有完整机制审计的负结果；论文故事不得围绕它重写。

---

## 2026-07-13：exp371 CASD——从“单图姿态部位模块”转向“跨图解剖 support”

### 新问题定义

LGPA 的最新证据不支持“CLIP 语义定位了人体部位”这一旧解释：

- pose-aware local descriptor 三 seed 稳定约 `+0.9 mAP`；
- fixed canonical prior 已取得约 `+0.7 mAP`；
- cross-image pose 只比 correct 低约 `0.7 mAP`；
- 解剖通道身份打乱只低约 `0.3 mAP`；
- PBSR route 学会但 standard global 为 `-0.1 mAP`。

因此真正的问题不是继续设计 part token，而是：

> 如何用训练期姿态组织同 ID 多图中的互补可见证据，构成当前单图缺失的 identity support，再让无姿态 student 学会吸收它？

### 唯一主候选

CASD 将 LGPA 降为 detached、训练期 pose-aware extractor。对每个 anchor，只聚合同 ID 其他图像中可见的对应 part，构成 leave-one-view-out anatomical support；image-only student 只学习 support 相对 same-image teacher 真正新增的 identity relation，而不是模仿当前图自己的 pose map。

它与 PAFormer/TSD/PGFL-KD 的差异不在 pose-free、part query 或蒸馏，而在：

1. 训练对象从单图 teacher 改为同 ID 跨图 support；
2. support 由互补可见部位组织；
3. 严格 leave-one-view-out，禁止 current-view copy；
4. student 学跨实例 identity relation，而不是普通 feature KD。

二次查新发现 AAAI 2020 UMTS 已覆盖 multi-shot comprehensive teacher → single-shot student，所以 CASD 不能以“多图教单图”为新意。查新边界进一步收紧为：pose-organized part support、硬 leave-one-view-out、support-vs-self advantage、伪 pose controls 四项必须同时成立。

IPER/姿态干预仍保留为 correct/uniform/shuffled/wrong-person support 的因果门禁。由于 2022 年已有题名完全相撞的 Pose-guided Counterfactual Inference，它不再作为方法 headline。

### 门禁

先做缓存特征 support oracle 与冻结 student 六臂：无 support、same-image KD、correct CASD、伪 pose CASD、允许 current-view 的泄漏对照、UMTS 式完整 multi-shot feature KD。correct CASD 必须相对 baseline 至少 `+0.8 mAP`、领先最强有效 control 至少 `0.5 mAP`，且 pose-free student 至少恢复原 LGPA 增益的 80%；否则停止 LGPA 改造，不转 OT/MoE/slot/权重扫描。

## 2026-07-14：exp371 Gate C 后的最终创新裁决

CASD 的必要前提已经被正式 frozen oracle 否定。target POSE-RESP 为 `94.2355` episodic mAP，低于最强 `PART-EQUAL=94.3121`（`-0.0766` pp），也低于 `POSE-SCALAR=94.2517` 与 `RESP-PERM=94.2727`。相对每折最强 control 五折全部为负；PID-grouped bootstrap 对 PART-EQUAL 的 95% CI 为 `[-0.1561,+0.0022]` pp。scene-merged 协议同样五折全负。

这轮审计保留下来的事实只有：

1. same-ID support 很强，wrong-ID 会灾难性失败；
2. 固定 slot correspondence 有效，`PART-EQUAL−SLOT-PERM=+1.2347` pp；
3. 但逐图 pose response 没有超越 equal、scalar 或 permuted routing，因此不能解释前两项收益。

创新判断随之收缩：

- **CASD 不能成为主创新。** generic multi-view identity support 已有 UMTS/MVI²P 邻居，part-aligned support 本身也不足以建立新的 pose 机制；
- “过去 `exp123` 失败、现在成功”本来可以形成创新证据，但前提是现在的新机制成功越过强 controls。本次没有越过，因此不能靠历史失败反向包装；
- 不进入 student，不转 AERC/OT/MoE/slot/temperature/queue/权重小变体；
- 不把 matching、GCN、CLIP 文本、same-ID support 或 fixed part correspondence 改名为贡献；
- 历史 LGPA 的 honest 定位仍是**有用的结构化局部性能资产**，而不是当前已经归属我们的新方法。

因此 LGPA 自有化主线在 IPER、PBSR、CASD 三个正交门禁后停止。这个结论不证明所有未来 pose 方法都无效，但足以禁止在本仓库当前证据上继续以小变体消耗算力或重写论文故事。

## 2026-07-15：PCAR 查新后封板——canonical residual 不是新的 attention 机制

CASD 后额外审计了一个真正换 backbone/作用位置的候选：在 official CLIP-ReID 的 CLIP ViT 内，仅修改少量 self-attention heads，用 `B(Pinstance)-B(Pcanonical)` 表示相对 canonical layout 的实例姿态残差，保持 untouched semantic heads 与标准 global descriptor。

这条路线仍不能恢复 LGPA 自有化资格：

1. 数学上，固定 canonical subtraction 只是普通 additive pose bias 的中心化，不扩大函数族；
2. PeVL 已覆盖 pose mask调制 CLIP visual attention，PAAB 已覆盖 pose-pair mask进入 ViT logits，MUVA 又已在 ReID 中逐层向 CLIP ViT 注入动态 body-part mask；
3. zero-init、少量 heads、语义锚和 global-only 输出是干净实现属性，不足以单独形成方法贡献；
4. exp371 correct≈shuffled/canonical 的内部证据也不支持 instance-relative pose 是强燃料。

因此 PCAR 在新颖性 Gate 直接 NO-GO，不用性能实验为一个已可归约的机制“争取创新”。允许保留的只有实验纪律：matched derangement、affine-fitted canonical、correct/shuffled 2×2 train/eval，以及 frozen Gate 只审 parity/attention尺度而不乱用 mAP 杀 adapter。

未来若还要从 pose attention 重新出发，最低要求不再是“换到 CLIP”“减去 canonical”或“只改少量 head”，而是提出一个无法分解成实例 pose bias加静态 bias、且有独立问题对象和强控制可验证的机制。否则继续写 PSG/已有论文资产，比硬包装 LGPA 更诚实。

## 2026-07-15：SA 正交耦合查新封板——多层 PSG+PAA 与正交补都不能承担主创新

用户提出一个合理质疑：既然 PSG 可能从更深的层次调制中受益，PAA 是否应在
相同 block 同时工作，并把 scale 与 shift 合成统一 SA。仓库复盘先纠正了两个前提：

1. 当前实现本来就是每个启用 block 后 PSG→PAA，不是最终层才做一次 PAA；
2. “PSG 层数越多越好”不是普遍规律。Tiny 的 clean mAP 为
   `59.2→60.2→60.5→60.5`，Small 为 `68.1→68.8→68.3→68.3`，主要收益来自
   Stage3，后续 stage 边际接近零且依赖 backbone。

普通 SA 可直接写成 `gamma(H)*x+beta(H)`，与 FiLM/SPADE 重合。唯一稍强的
候选是把 PAA residual 投影到 PSG actual displacement 的正交补，使 shift
显式不重复 scale 更新。但专项查新表明：hard orthogonal residual update 已有
arXiv 2025 直接先例；Shape-Erased VI-ReID 和 Ortho-ReID 又覆盖了 ReID 中
结构/外观子空间与正交补身份表征。将投影参考从输入换成 PSG displacement，
不足以形成新问题对象或新 operator。

因此 exp373 在新颖性 Gate 直接 NO-GO。保留下来的有用认识是：

- `POSE_PAA_STAGES` 独立开关和“中层 scale、深层 shift”是合理工程消融，
  但不是贡献；
- `[0,1]` heatmap 重复 sigmoid 会令 zero-input 变成 0.5，未来任何 pose 机制
  都必须保证 true no-pose/identity 可解释；
- overlap energy、经验 null、virtual projection 是好的诊断协议，但证据设计
  不能为已有 operator 创造新颖性；
- PAA 若保留，只能是 PSG 系统中的辅助组件，不再承担第二主创新。

## 2026-07-15：PSG 本体查新封板——有效组件，不再作为主创新

对当前 PSG 公式 `Y=X*(1+E(H))` 做了原文级 prior-art 审计。决定性先例是
Bhuiyan et al. 的 WACV 2020 *Pose Guided Gated Fusion for Person
Re-identification*：它已经把 pose confidence/affinity maps 与 backbone 中层
appearance feature 融合，生成同空间、同通道 gate，再用 Hadamard product 调制
feature 并继续向后传播，还系统比较了 C2-C5 注入位置。

通用机制边界同样明确：SFT 的 `gamma(Psi)*X+beta(Psi)` 令
`Psi=H, gamma=1+E(H), beta=0` 即严格得到 PSG；FiLM 补充材料也明确使用
`gamma=1+delta_gamma`，ControlNet/adaLN-Zero 则覆盖新增条件分支从 identity
启动。因此 pose-only、Swin、zero-init、`1+g`、多 block/stage 和小参数量都只是
有价值的实现差异，不能把 PSG 变成新的函数类。

本轮结论不推翻 PSG 的实证价值：旧协议三 seed mAP 从 `56.50` 到 `57.83`
且方向全正；新 clean sweep 中 Tiny `59.2->60.2->60.5->60.5`，Small
`68.1->68.8->68.3->68.3`。但这只支持“PSG 有用、主要收益来自 Stage3、更多
stage 边际依赖 backbone”，不支持“PSG 本体是首创”或“层数越多越好”。

论文与后续实验边界更新为：

1. PSG 可保留为轻量、有效的 SFT/FiLM 式 pose-conditioned recalibration、强基线
   或新机制的载体；
2. 不再写“现有 pose-ReID 都在 feature extraction 后使用 pose”或“首次在
   backbone 中层做 pose gating”；
3. 移植到 ResNet/普通 ViT 只能补通用性，不会创造新颖性，ResNet 反而更贴近
   WACV 2020；
4. 下一候选必须离开逐位置对角仿射 `gamma*X+beta`，并先过独立查新；
5. 若声称恢复/移动视觉证据，必须含真正的跨位置 off-diagonal interaction，并用
   correct/shuffled/canonical/bypass 和 parameter-matched PSG/SFT 做强控制。

完整来源、公式和 claim 表见
`experiments/paper_notes/psg_novelty_audit_20260715.md`。

## 2026-07-15：exp374 因果 Gate——PSG 的涨点成立，但不是实例姿态对应收益

三 seed 冻结 checkpoint 的 primary intervention 给出一个非常清楚的分离：correct 相对
true bypass 为 `+3.8577 mAP / +5.1433 R1`，但 correct 相对 matched 错姿态只有
`+0.0012 mAP / 0.0000 R1`，且 mAP 区间跨过零、三 seed 中两 seed 非正。

所以不能把“PSG 本体有效”和“正确实例姿态有效”混成同一个结论。前者得到更强确认；
后者在当前机制上被否定。最合理解释是 PSG 学到通用人体空间先验、条件容量或训练正则，
并不真正使用 image-pose correspondence。继续改变 gate 的层数、非线性、温度、
canonical 或 shuffle 方式，只是在缺乏因果燃料的函数族内微调。

这反而给下一条路线划出了更好的机制边界：pose-controlled Mamba 不能再做
`gamma(H)*X+beta(H)`，而要让 pose 进入状态动力学，例如控制 selective update 的
`Delta/B/C`、保留/遗忘系数、解剖扫描顺序或跨身体区域状态传递。是否值得推进取决于两个
问题：它是否超出已有 pose-Mamba 的普通输入融合；correct pose 是否相对 shuffle 和
parameter-matched image-only SSM 产生可重复的额外价值。

## 2026-07-15：exp375 PRSM 封板——状态动力学也没有恢复实例姿态因果性

exp375 真正实现并跑满了上一节提出的非仿射候选：姿态只控制 6 个身体状态槽的
recurrent write/retain，RGB 内容独立产生 candidate 和 read query，并使用 pre-write read
避免退化成当前 token 的 memoryless pose gate。它在机制上已经离开 PSG 的
`gamma(H)*X+beta(H)`；M0/P0 参数完全匹配，B0/M0/P0 的训练配方匹配。

但结果给出两层一致否证：

1. 同运行时 e120，P0=`57.1/66.3`，低于 B0=`58.4/67.1` 和 M0=`58.8/67.5`；
2. 同一 P0 checkpoint 下，correct 与 matched-shuffle、foreground-uniform、zero-bypass
   的 mAP 差仅 `-0.000338/-0.000690/+0.001795` 百分点，所有 R1/R5/R10 完全相同。

这里不能再用“模块没有工作”解释：residual/retention 已离开初始化，各反事实 descriptor
确实不同，matched nuisance gate 通过，zero 精确 identity，训练又完整跑到 e120。最克制的
结论是：**当前 heatmap 所提供的实例姿态，在状态写入可信度、部位槽归属和推理时 memory
写回上都没有可测身份排序价值。** canonical M0 与 image-only B0 同样强，进一步说明通用
状态容量或人体先验足以解释这条机制可能提供的作用。

创新边界据此更新：

- 不把 PRSM 写成主创新，不转 graph、动态 scan、更多槽、额外 loss 或 Mamba block 小变体；
- 不把“用了 recurrent state”本身当成新颖性，也不把负结果包装为成功方法；
- 不外推为所有 pose×state-space 都不可能。未来若重开，必须换新的可观测信号或问题对象，
  并在最小 frozen/parameter-matched gate 上先证明 correct input 明确优于 matched controls；
- PSG/LGPA 仍是已验证的性能资产，但当前仍没有一个可归属的新 pose-state 第二贡献。

## 2026-07-16：exp376/377 封板——提高调制复杂度与真实 selective SSM 都未恢复 pose 燃料

exp376 与 exp377 分别验证了用户提出的两种更强机制，而不是只换 MLP 宽度：

1. Pose Hyper-LoRA 在 stage 2/3 逐 block 生成低秩动态变换；
2. Pose Selective SSM 让 RGB 产生基础 `Δ/B/C`，实例姿态直接修正状态离散步长、写入和读取。

两者都通过了全分支梯度、参数更新、数值稳定和 target-person heatmap 审计，但 e60 分别低于
同机 clean B0 `1.0` 和 `0.7 mAP`。exp377 跑到 e120 后也只有 `+0.2 mAP`，且 RGB-only
SSM 的跨机趋势不弱于 P0。结合 exp374 correct≈matched、exp375 correct≈matched/zero，
当前最一致的解释已经不是“函数还不够复杂”，而是这份实例 heatmap 在这些控制位置上没有
足够的额外身份信息。

因此创新搜索边界进一步收紧：

- 不再沿当前 heatmap 做 LoRA rank、层数、MLP、`Δ/B/C` 拆分、state 数量、scan 或 graph
  小变体；这些都缺少先验燃料；
- 不能把 ordinary RGB-selective SSM 的容量收益归给 pose；
- 若未来重开 pose×Mamba，必须先更换可观测信号或问题对象，例如可靠的遮挡/可见性状态、
  视频时序、3D/SMPL 或跨帧结构，而不是继续喂同一单图 2D heatmap；
- 当前论文仍应把 PSG/LGPA 视为性能资产或强基线，不把 exp376/377 包装成成功贡献。

## 2026-07-16：exp378 重新打开姿态场主线——从单一 PSG gate 转向可迁移的内生状态模块

exp378 的 hard/relax × residual OFF/ON 同机2×2给出了与exp374–377不同的信号。关闭
geometry residual时，hard F0与MR-F0分别为`55.9/56.0 mAP`，相对同机B0为
`+0.8/+0.9`；显式relaxation只有描述性`+0.1 mAP`，当前`17×4` residual在两种transition
下都为`-0.3 mAP`。这仍是单seed，且R5/R10没有同步提升，但足以说明不能因residual失败而
关闭TAPF：真正有燃料的是“由视觉特征内生预测、经姿态监督bootstrap、推理时不读外部pose、
再控制后续视觉层”的姿态场配置。

这比原始PSG的`x*(1+heatmap)`更接近完整方法对象，但创新定位必须克制。单锚点TAPF仍混合
了bootstrap课程、LiteHR-style anchor、Gaussian renderer与PSG；在D0/J0/R0、RG0、N0及
置换/错误姿态审计完成前，不能声称收益来自正确关节语义，也不能把`+0.8/+0.9`写成稳定或
显著提升。当前geometry residual与SGD relaxation都不承担主创新。

若单锚点归因闭合后仍保留燃料，下一主机制候选是Progressive Anatomical Field，而不是简单
复制多个PSG：维护一套跨视觉层级传递的17关节姿态状态，每层只预测受confidence release和
位移/尺度上限约束的关节级残差，重渲染后控制下一视觉层；共享姿态解码器，仅保留
stage-specific轻量投影。这把闭环写成“视觉特征→姿态状态细化→下一层视觉更新”，可自然映射到
Swin stages、ResNet layer1–4、ConvNeXt stages和分组ViT blocks。

未来证据链分三层推进：

1. **单图机制**：Swin-T同条件单seed，加入residual-off、stage permutation、joint
   permutation、错误姿态退化、常量场与stage关闭；成功后多seed；
2. **backbone-agnostic**：迁移ResNet-50，验证共享状态递进而非Swin结构特例；轻量HRNet只作
   anchor容量对照，不作为创新headline；
3. **时序扩展**：Video ReID中把跨stage姿态状态扩展为跨帧可靠性状态，利用轨迹内关节置信度、
   遮挡恢复与运动连续性做有界temporal correction。必须与逐帧TAPF、普通temporal pooling、
   外部pose smoothing和RGB-only video backbone对照，避免把普通时序建模误归给姿态。

论文潜力来自统一问题对象——**可靠性有界、跨层/跨帧传递并逐级修正的内生姿态状态**——而不是
“多层加PSG”。在针对hierarchical pose estimation、multi-stage pose-guided ReID、recurrent
pose refinement、feature modulation及video pose-ReID完成专项查新前，只称B类潜力候选，
不提前宣称新颖性或可投稿性。

## 2026-07-17：冻结语义审计否定当前PSG消费路径，Hierarchical候选必须先解决“可空分离”

D0 e90的同checkpoint干预给出了比训练final差值更直接的机制边界。内生anchor本身并非完全
无结构：teacher posterior cosine=`0.8276`、pseudo-PCK@0.05=`0.5539`、flip posterior
cosine=`0.9467`，17通道全部占用；但检索指标对matched wrong field、joint/confidence
permutation、spatial constant和zero field均近似不变。只有把Stage-3 PSG模块整体旁路才下降
`2.68 mAP`。因此“场看起来像姿态”成立，而“检索使用了姿态语义”不成立。

这暴露了现有consumer的关键混淆：PSG先对raw field做`sigmoid`，所以raw zero仍变成常量`0.5`，
再经过具有bias和已学习权重的encoder；zero field不是identity，PSG容量可以在几乎不使用空间场
的情况下学成静态重标定。把同一个PSG复制到多层只会复制混淆，不能形成Progressive Anatomical
Field。

Hierarchical TAPF若继续，机制定义必须先满足以下不可退让条件：

1. **null-separable consumer**：显式null field在数学和实现上都严格返回identity，不能再让
   常量输入触发可学习bias；field-dependent分支可用中心化posterior、零均值空间残差或显式
   `gate(field)-gate(null)`，但默认行为仍需config隔离；
2. **容量匹配**：每个pose consumer都有parameter-matched RGB-only/static consumer，对照相同
   参数量、optimizer group、初始化与RNG；正确field必须同时优于二者，模块整体收益不能再冒充pose；
3. **逐层因果门禁**：训练后对每个stage分别做matched field、spatial constant、stage order和
   bypass干预；只有correct相对破坏臂达到预注册`>=0.3 mAP`，才允许进入multi-seed、ResNet和Video；
4. **状态而非门控堆叠**：跨层传递的是归一化关节posterior/confidence及其有界更新，consumer
   只读取状态的可验证非恒定分量；必须能报告每层状态变化与下游descriptor变化的配对关系；
5. **迁移顺序后移**：当前实现不做ResNet/Video迁移。Video ReID的时序姿态可靠性仍有方法潜力，
   但必须等新的单图consumer先通过correct-vs-static/matched因果门禁，否则时序扩展只会把普通
   temporal capacity重新包装成姿态贡献。

因此B类潜力没有被逻辑上永久关闭，但候选方法已经从“多层TAPF/PSG”收紧为**可空分离、容量匹配、
逐层可干预的内生姿态状态消费者**。这是一项新的机制设计任务，不是H0旧方案的训练触发条件。

## 2026-07-17：用户重定论文对象——从子部件因果转为完整的pose-supervised、pose-free模块

冻结语义审计回答了“训练后的PSG是否依赖精确field语义”，但没有否定另一个更贴近产品和论文
方法定义的问题：能否用训练期姿态监督，训练出一个测试期不再运行姿态模型、同时保留原PSG收益的
完整ReID模块。fresh D0相对B0为`+1.1 mAP`，并基本匹配external ViTPose R0；在这一问题定义下，
anchor和PSG不必被拆成两个各自显著的贡献，正如许多teacher-student方法也不要求每个中间张量在
冻结干预下独立贡献指标。

据此，最有价值的升级不是继续geometry residual，而是把单点内部场推广为**Progressive
Hierarchical TAPF**：浅层生成初始结构状态并调制中层，中层结合上一状态与更强视觉语义进行修正，
再调制深层；各stage只保留轻量投影，decoder共享。它与旧multi-stage PSG的关键区别是，后者把
同一个外部heatmap复制到多层，前者让内部状态随视觉层级递进。

论文创新边界调整为：

1. 可主张完整模块的训练期pose supervision、推理期RGB-only与整体收益；
2. 不主张当前17通道名称、confidence或空间field在冻结推理时各自具有独立因果贡献；
3. 逐层版本必须直接超过或至少稳定匹配单点D0，不能只对B0报总增益；
4. Swin-T强预训练可能造成饱和，后续用同backbone内的B0/D0/HT0三臂迁移ResNet-50和ViT，检验
   方法排序而非跨backbone绝对数值；
5. Video ReID的真正新空间在跨帧关节可靠性、运动连续性和遮挡恢复，但应晚于单图逐层模块与
   backbone迁移，避免一次引入层级和时序两个核心变量。

因此B类潜力现在主要来自一个更完整的系统问题：**如何把昂贵的测试期姿态依赖蒸馏为可跨层、
可跨backbone、最终可跨帧扩展的内部结构状态。** 这比“再给PSG加一个gate”更像方法级工作，
但仍需逐层增益、multi-seed、迁移和效率证据共同支撑。

## 2026-07-17：exp379首轮结果——逐层可行但Swin-T上没有额外燃料

HT0把候选机制真正落到了两级：Stage-1内部anchor/state驱动Stage-2 PSG，Stage-2在上一state上
refine后驱动Stage-3 PSG；两级projection和PSG独立，decoder严格共享。全程参数轨迹、梯度隔离、
field路由和final RGB-only parity均通过，因此“每个anchor对应一个PSG”的结构不是概念草图。

但性能证据必须克制：HT0=`56.1/67.6/79.9/83.4`，D0=`56.2/67.6/79.8/83.4`，四项差值
`-0.1/+0.0/+0.1/+0.0`。这说明在SOLIDER Swin-T上，增加浅层anchor→PSG并让pose state递进，
没有形成可分辨的额外检索收益。不能把“结构更完整”直接等价为“机制已优于单层”，也不值得继续
在Swin上调stage数、decoder共享方式或loss比例。

仍值得保留的创新对象是完整的pose-supervised、pose-free模块：D0相对B0=`+1.1 mAP`，HT0相对
B0=`+1.0 mAP`，二者都在推理期完全不读外部pose。下一步ResNet-50迁移要回答两个不同问题：

1. 单点`anchor+PSG`收益是否跨backbone保留；
2. 较弱或不同归纳偏置的backbone是否释放逐层HT0相对D0的空间。

只有同backbone B0/D0/HT0排序能回答这两个问题。若ResNet与后续ViT仍是HT0≈D0，则论文应以
单点progressive pose distillation为核心，把hierarchical当可选扩展；若HT0在至少两个backbone上
稳定优于D0，才把跨层递进状态升级为headline。Video ReID仍放在backbone证据之后，届时新增的
核心变量必须是跨帧关节可靠性、运动连续性和遮挡恢复，而不是普通temporal pooling。

## 2026-07-17：exp380结果——完整方法跨骨干成立，逐层收益呈backbone条件性

ResNet-50三臂给出了比Swin更清楚的层级排序：B0=`35.0/45.3/61.3/68.2`，D0=
`38.1/49.4/64.6/71.1`，HT0=`38.9/50.5/65.9/72.0`。因此完整单anchor+PSG相对B0为
`+3.1 mAP`，每anchor一组后继PSG的逐层版本再相对D0增加`+0.8 mAP`。这不是跨backbone绝对
指标比较，而是两个backbone各自内部matched排序。

创新判断因此分成两层：

1. **较稳的主对象**：训练期用姿态监督学习内部结构状态，和PSG组成完整模块；推理期只读RGB。
   它在Swin-T为`+1.1 mAP`、ResNet-50为`+3.1 mAP`，跨骨干迁移证据已经形成。
2. **仍待判别的层级扩展**：HT0−D0在Swin-T为`-0.1 mAP`、ResNet为`+0.8 mAP`。这支持
   “强人体预训练可能吸收部分层级结构先验”的工作假设，但一个正、一个中性不能写成规律。

下一步ViT不是简单再换一个backbone刷表，而是判别层级机制的必要实验：固定ViT内部matched
B0/D0/HT0，若HT0再次优于D0，才把progressive hierarchical refinement升为headline；若中性，
则承认ResNet增量具有架构条件性，把论文中心收敛到完整pose-supervised、pose-free模块。无论哪种
结果，都不在ResNet继续调层数或loss。Video阶段仍聚焦单图不存在的新信息——跨帧可靠性、运动
连续性和遮挡恢复——并晚于ViT判别。

## 2026-07-17：exp381完成第三骨干判别——原子方法保留，逐层headline关闭

ViT-B内部matched结果为B0=`52.9/59.5/77.1/82.0`、D0=
`54.9/61.4/78.9/84.0`、HT0=`54.6/60.6/78.4/84.1`。D0−B0=`+2.0 mAP`，再次
支持训练期pose监督与后继PSG作为完整原子方法；HT0−D0=`-0.3 mAP`，没有复现ResNet的`+0.8`。
三骨干HT0−D0为Swin `-0.1`、ResNet `+0.8`、ViT `-0.3`，逐层refinement不能作为架构无关
headline。最合理的主对象收敛为**pose-privileged training for pose-free ReID**，而不是
“更多视觉层必然从递进pose state获益”。

ViT还暴露了结构放置边界：post-block11 PSG发生在最后CLS–patch交互之后，无法影响最终CLS；
任何Transformer consumer都必须位于仍有descriptor下游路径的位置。该发现不推翻D0的`+2.0`，
因为post-block9/10有效，也不混淆HT0−D0，因为两臂共享terminal冗余，但它禁止把“配置里列出的
consumer数”直接写成“有效调制层数”。

下一阶段不再继续单图层数/宽度/decoder组合，而转向Video ReID中单图不存在的真实信息：

1. **跨帧可靠性传播**：高置信可见帧为相邻遮挡帧提供joint/state teacher，而不是平均热图；
2. **运动连续性约束**：内部pose posterior沿时间保持速度/加速度可解释的连续性，并允许镜头切换
   或检测断裂时重置；
3. **遮挡恢复**：用时序state补全当前帧不可见身体区域，但必须与RGB temporal memory参数匹配；
4. **三臂归因**：RGB temporal B0、逐帧原子D0、时序pose T0。只有T0−D0稳定为正，才能把收益
   归给时序姿态，而不是视频容量；
5. **部署边界**：pose仍只作训练期privileged supervision，测试期不运行外部姿态模型。

这条路线的创新机会不在“把单图PSG复制到每帧”，而在**可靠性有界的跨帧内部结构状态如何恢复
单帧遮挡信息**。正式训练前需先确认仓库可用video数据、tracklet采样与同backbone RGB temporal
强基线，再写独立设计与门禁。

## 2026-07-17：exp382查新关闭Video headline，创新对象进一步收窄

专项查新推翻了“视频天然提供新的pose-privileged故事”这一预期。GAE-Net已经把训练期gait+RGB
视频教师蒸馏为RGB-only视频学生，PAFormer已经覆盖pose supervision→pose-free inference，
KPRTrack又覆盖tracklet同部位聚合。再叠加PSTA、STMN、TF-CLIP的时序关系、遮挡/干扰memory与
sequence memory，跨帧pose reliability、motion continuity或occlusion recovery若没有新的可测
中间变量，很容易只是这些机制的重命名组合。

因此Video TAPF不再作为下一主创新。它未来最多回答“单图原子方法能否迁移到视频”，不能重新把
训练期姿态、测试期RGB-only包装成首创。当前真正需要保护和强化的对象是更窄的机制差分：

**训练期结构target不是只蒸馏logit/feature，而是在backbone内部形成anchor/state，并通过后继PSG
直接改变仍有descriptor下游路径的视觉特征。**

这个差分仍面临PAFormer/PGFL-KD/TSD/KPR强近邻，不能只靠命名成立。下一证据优先级改为：

1. Market fresh B0/D0验证第二训练域；
2. 同一checkpoint在Occluded-ReID做严格pose-free跨域遮挡评测；
3. 同时补TAPF专属参数、FLOPs、训练成本和推理成本；
4. 方向成立后再补主骨干multi-seed，而不是为失败的HT0补seed。

exp383由此不是新模块探索，而是论文主张的高信息量证据实验。若Market域内与Occluded-ReID跨域
均不支持D0，三骨干同一数据集正差只能保留为Occluded-Duke特定证据；若两域方向支持，才有理由
把原子方法从“跨backbone”推进到“跨训练域/遮挡target”的描述性结论。

## 2026-07-18：官方干净重启把创新判断从“跨骨干强信号”收紧为“小效应待稳健性验证”

官方最后代码上的 fresh 复现改变了效应量判断。clean Occluded-Duke D0−B0 只有
`+0.2/+0.3/+0.2/−0.6`，clean Market D0−B0 为 `+0.4/+0.2/+0.1/+0.1`。这两个结果方向上
支持训练期 pose target→内部 anchor→后继 PSG→测试期 RGB-only 的完整链路可迁移到第二训练域，
但幅度不足以继续沿用“跨三骨干明确增益”的强语气。旧 runtime 的结果仍是探索资产，不能替代
当前 clean implementation 的稳健性证据。

计算开销不是当前主要问题：D0 参数约 `+0.375%`、supported-op FLOPs `+0.242%`、train/eval
step 约 `+1.96%/+1.64%`。真正缺口是效应是否超过 seed 方差，以及该小增益是否足以在
PAFormer、PGFL-KD、TSD、KPR 等强先例旁形成方法级贡献。不能用轻量本身补偿证据强度不足。

clean hierarchical 已给出明确反证：新增早期 anchor 和六个真实可达 PSG 后，HT0−D0=
`−0.7/−1.8/−0.8/−0.5`。所有模块都更新、所有 consumer 都能改变 final descriptor，所以失败
不能归因于 terminal dead path。由此继续增加 stage、共享/独立 decoder、loss 权重或 gate 数量，
都只会回到模块堆叠，不再满足创新门槛。

当前仍可争取的窄机制差分是：**结构 target 不是仅蒸馏最终 feature/logit，而是训练一个位于
backbone 内部、其输出直接控制后继视觉更新的 RGB 内生 anchor；部署时外部 pose 路径完全删除。**
但这只能在 matched 多 seed 确认小正效应后写成论文中心。下一阶段的研究问题不再是“怎样让
hierarchical 涨点”，而是“clean 原子增益是否可重复”。若答案为否，应主动更换问题对象，不能
再用 Video、更多层或旧跨骨干数字包装当前方法。

## 2026-07-18：exp390后创新判断——TAPF有小mAP燃料，但“多stage”必须重构为可归因问题

exp390把原子TAPF的clean证据从单seed推进到paired三seed：mAP差为`+0.2/+0.8/+0.4`，
mean±sample std=`+0.47±0.31`，方向`3/3`为正。这说明TAPF不是完全无效；训练期pose target经
RGB内生anchor和后继PSG形成了一个小而可重复的mAP regularization/modulation效应。与此同时，
R1/R5/R10的paired均值为`−0.10/+0.00/−0.20`，所以现版本没有“整体检索更强”的燃料，也仍未
解决correct/shuffle/wrong field近似不可辨识的问题。

因此创新任务不能只是“把TAPF放到每个stage，看起来更大”。真正可争的机制问题应拆成三层：

1. 多anchor是否因为pose-loss总预算或AMP skip耦合而改变优化轨迹；
2. early field是否被6个consumer过度重复消费，而late field只有2个consumer，造成结构不平衡；
3. 在前两项闭合后，新增Stage-0 direct anchor的route本身是否相对参数完全matched的route-off
   control提供增益。

这正是exp391的A→B→C顺序。最终三anchor定义采用用户提出的“所有anchor都直接算”：A0/A1/A2
分别从自身stage feature独立预测同一套17-joint absolute field，不共享decoder、不读取上一field、
不预测offset；每层只写入紧邻的下一stage，consumer为`2/2/2`。三层native grid不同，必须把
Gaussian sigma设为`6.0/3.0/1.5`以匹配约24px物理尺度，否则“stage差异”会被target sharpness
混淆。只有H3-ON相对matched H3-OFF有增益，且至少两个stage冻结旁路各有非零贡献，才能把
“全stage可执行解剖状态”升级为方法贡献。

CLIP语义校准仍保留为后续PRIMARY DESIGN CANDIDATE，而不是现在并入exp391。它要解决的是joint
channel语义不可辨识；exp391先回答多stage route本身是否有燃料。两者同时加入会混淆“层级结构”
与“语义teacher”，也违反单变量原则。

## 2026-07-18：exp391后创新判断——多阶段route可执行，但topology本身没有性能燃料

H2-M把exp389两层pose loss从sum改为mean后，final从HT0的`56.9/65.9/80.0/84.1`恢复到
`57.2/67.3/80.2/84.5`。冻结early-bypass给出`+0.141 mAP`独立贡献，八个consumer也都能改变
final descriptor，因此“浅层完全是dead consumer”不成立。这里得到的是一个有价值的负归因：
loss budget确实影响多层优化，但即使修正预算，完整H2-M仍比单层D0低`0.4 mAP`。

这关闭的是按`loss mean→consumer balance→更多direct anchor`继续堆纯结构的理由。exp391 Phase
B/C不再执行，也不能用early的局部正贡献包装“多阶段有效”；但不永久否定语义校准后的多阶段
TAPF。更强的方法改造必须先改变当前真正未解决的问题对象：17个joint channel对检索近似可置换，
anchor可能只学到一般条件扰动，而不是可辨识的解剖语义。

后续CLIP候选只有在以下定义下才值得继续只读审查：frozen CLIP image encoder先从与学生几何对齐的
RGB patch token中，经训练期pose heatmap池化得到实例级joint/part视觉teacher；frozen text encoder
只提供全部body-part原型，视觉局部特征与全部原型相似度形成sample-specific分布；各stage anchor
池化本层ReID feature并蒸馏该分布，且必须通过PSG真实影响final descriptor。CLIP不能直接蒸馏最终
ReID descriptor，推理期必须删除双encoder、文本与external pose。核心强对照应是joint-channel
shuffle、wrong field、text-only prototype、image-only local teacher和matched non-semantic teacher。

该方向的潜在新意不在“CLIP+pose+multi-stage”组合，而在**以实例级视觉—文本双编码器teacher
打破内部解剖状态的通道置换对称性，再验证这种可辨识状态是否成为检索因果变量**。在近邻查新、
代码路径审计、单变量设计和上述语义门禁完成前，不实现、不占用4090，也不承诺它会带来更大涨点。
若这些门禁通过，后续应先建立semantic single-stage，再以其为直接对照测试semantic multi-stage；
这将是新的机制实验，而不是对已封板exp391的重启。

## 2026-07-18：Phase 0A把“语义不可辨识”从怀疑变成可执行证据

clean D0的冻结反事实给出了一个比继续堆stage更有价值的事实：两个PSG都对final descriptor有显著
作用，但它们几乎不在乎17个通道究竟代表哪个关节。channel-cycle和matched-wrong field的mAP变化
分别只有`+0.024`与`−0.005`，all-bypass却下降`1.359`；更关键的是，删除所有空间结构、只保留
每通道均值的spatial-constant控制反而提升`0.346 mAP`。

这把创新问题重新写清楚了：当前TAPF不是“pose没有用”，而是**pose field被自由PSG吸收成了有效的
低频条件调制，anatomical channel identity与精确geometry没有成为可辨识的执行变量**。因此真正的
方法空间不是再加pose loss、复制anchor或加普通CLIP KD，而是让`geometry × semantic identity ×
local visual evidence`必须共同决定一个可干预的router update。

由此新增三条不可删除的强控制：

1. `spatial-constant/static-state`：因为它已在旧D0上超过correct，任何新方法必须胜过这个低频解释；
2. `generic low-rank adapter/expert-mean`：排除收益只是更强feature transform；
3. `correct vs channel-shuffle vs matched-wrong binding`：只有correct至少拉开`0.3 mAP`，才称为
   anatomical mediator。

CLIP的角色也更精确：它不是给final descriptor提供第二套表征，而是训练期为内部coarse anatomical
state提供sample-specific visual support。Phase 0B若证明双编码teacher退化为固定region one-hot，
则CLIP对当前问题没有新增信息；若通过，再由semantic expert gather-transform-scatter router把该状态
变成不可被自由channel mixing消解的执行对象。只有semantic single-stage同时超过D0、static和generic
controls，多阶段才值得重新进入。

## 2026-07-18：Phase 0B揭示“CLIP局部语义”必须经过受监督readout，不能直接命名patch token

全量审计给出一个反直觉但清楚的分解：last-block patch feature不是常量，也确实含人体结构——同一
feature做无文本K-means可达到`52.8–60.0%`五region best-permutation accuracy；但直接与body-part
text prototype相似度分类时，correct只有`2.7–4.6%`，shuffle/wrong-text反而更高。OpenCLIP官方
token parity、pose坐标和label顺序均已排除，说明失败不在“没取到token”，而在**把未经局部
contrastive calibration的token方向误当成text-aligned semantic axis**。

同一region改成tight crop后重新走完整CLIP global CLS，macro top-1立刻升到`44.7%`；只在最后block
加入pose-conditioned CLS attention则恢复到`32.5%`。这给出比“换prompt再试”更强的下一机制：

1. 共享frozen CLIP早期trunk，避免每region五次完整encoder；
2. 在多个后段block引入只作用于teacher CLS query的pose mask readout，让region state沿CLIP真正受
   text监督的CLS→projection路径形成；
3. anatomy slot identity仍由固定pose ontology给出，CLIP负责slot内sample-specific appearance/support，
   不再让CLIP重复猜“这个mask本来就是哪一类body part”；
4. student只蒸馏可执行state，router与NULL/counterfactual门禁不变，推理删除teacher。

这里更大的方法机会是把“anatomical identity”和“appearance evidence”解耦：pose提供不可置换的
slot identity，CLIP提供每个slot的实例属性/支持度，而不是用一个五类body-part softmax同时承担两者。
这有望避免当前teacher的语义自指（mask已知region，却再让CLIP猜region）并增加真正与ReID相关的
衣着属性信息。但它与ALADIN/ProFD/RegionCLIP的局部attribute teacher存在novelty近邻，论文差分仍
必须落在executable mediator、NULL identity和反事实可辨识证据，不能把attribute KD本身当贡献。

## 2026-07-19：Semantic C0负结果把瓶颈收紧到“窄动态support无法成为可执行语义变量”

PC-MBCLS小样本反事实证明五slot readout会随局部遮挡单调变化，因此它不是完全错误的CLIP接口；但
正式Semantic C0 final仍比clean D0低`0.7 mAP`。终审给出关键内部量：support均值约`0.512`、混合
五slot的pooled std=`0.0169`，q loss停在`0.692`，两个router的gate-delta abs-mean只有
`3.6e-06/1.0e-05`。与此同时
mask/presence确实学会、两个consumer也真实到达final descriptor。负结果因此不是“路径没跑”或
“CLIP完全无局部信息”，而是**sample-specific CLIP信息在窄动态q标量中被压成近常量先验，随后又
被小幅router稀释，难以相对D0提供新增可执行变量**。

这否定了一个隐含假设：只要teacher q对遮挡方向敏感，逐slot BCE就会自动把它变成有检索价值的
内部state。当前teacher target集中在0.5附近，BCE最容易学到slot-level均值；presence又几乎全为1，
于是router主要看到coarse mask乘近常量support。CLIP在计算图中“深度耦合”不等于CLIP信息在统计上
拥有足够动态范围，也不等于它对final descriptor有可辨识增量。

下一创新候选应从“标量可见度蒸馏”升级为**相对化的局部视觉证据中介**，但先以单变量证据决定：

1. 在封板模型上做learned-q/static-q/pose-only/router-bypass冻结拆因，量化q本身的边际贡献；
2. 若q边际近零，teacher端不再输出集中于0.5的绝对概率，而应保留slot内相对证据，例如以同图
   non-target、同slot跨图基线或遮挡前后差形成centered residual/ranking target；
3. student端必须报告每slot target variance、跨图rank保持与wrong-RGB/wrong-mask敏感性，不能只看
   BCE下降；router端则以counterfactual descriptor差证明该动态被执行；
4. generic low-rank、static support和pose-only仍是强对照。只有correct CLIP residual同时超过这些
   控制并提高clean D0，才能称为semantic mediator，再谈balanced multi-stage。

因此CLIP–TAPF仍值得继续，但研究对象已从“给TAPF加CLIP teacher”进一步收紧为：**怎样让冻结
视觉—语言模型的局部相对证据以非退化动态进入可反事实验证的ReID路由**。这比换prompt、调温度或
复制更多stage更接近真正的机制贡献。

## 2026-07-19：Phase 0D发现更深的耦合断点——CLIP监督停在anchor，router只靠ReID梯度近零启动

全验证集拆因修正了前述统计解释：五slot各自跨图std只有`0.00009–0.00029`，`0.0169`几乎全是
固定slot均值差。更关键的是，即便把q设为1、删除mask geometry、合并expert或直接旁路全部router，
四项检索几乎完全不变；all-bypass只有`−0.000077 mAP`。所以当前问题不只是“q动态窄”，而是整条
semantic route在final retrieval上近似identity。

回看实现，原因具有结构性：CLIP的mask/presence/q loss只监督anchor；state在进入router前detach，
这是为了避免ID loss把anchor改写成身份码。但router本身没有任何CLIP/局部语义objective，只能从
global ReID loss间接得到梯度。与此同时每个router的slot expert从exact zero初始化；初始时
token/context projection因乘到zero expert而拿不到有效梯度，只有expert先缓慢离零。最终虽然三组参数
都“changed”，expert L2仍只有`0.012/0.019`，路由残差停在`10^-6–10^-5`。这是一种**拓扑上接入、
梯度所有权上断开的伪深耦合**：CLIP决定输入state，却不直接约束执行该state的更新。

因此下一版不能只扩大q动态。更合理的机制是把rich、centered的CLIP局部视觉证据直接监督router的
内部slot latent/delta，而不是直接蒸馏final ReID descriptor：

1. teacher输出每slot相对局部视觉residual（减去同图global或slot prior），保留方向和幅度，不压成
   0.5附近单标量；
2. student anchor预测对应低维evidence code，router的gathered context与该code共同生成slot delta；
3. 对router latent/delta增加训练期internal alignment或ranking objective，让CLIP梯度真正拥有
   executable mediator，同时仍对backbone/ID descriptor保持隔离；
4. 用ReZero式非零branch+零标量，或极小非零residual scale替代zero-expert冷启动，并在preflight验证
   token/context/expert从首个finite step都有梯度；
5. final必须同时通过correct-vs-wrong evidence和all-router-bypass贡献，防止再次出现“参数更新但
   retrieval-inert”。

这个修复仍然是CLIP与TAPF深耦合，但耦合对象从弱q门控改成**CLIP-owned executable local residual**。
它不需要恢复多stage；先在single-stage证明route有燃料和语义反事实，再讨论balanced multi-stage。

## 2026-07-19：COER把“CLIP-owned”从接线关系改成可审计的梯度与执行所有权

exp393的核心不是把768维CLIP feature换个名字塞进adapter，而是把两个互相独立的问题拆开：

1. **执行通路能否离开identity**：nonzero branch加zero ReZero scalar保持初始descriptor exact，却让
   alpha首步从ReID loss得到梯度；final all-bypass差决定route是否真正参与排序。
2. **CLIP证据是否拥有执行变量**：region CLS减同图global、再按slot中心化和共享PCA，保留局部视觉
   方向；student evidence既被CLIP relation监督，又成为生产router hidden的必经输入。

代码seam审查暴露了一个重要方法学边界：若alignment只写在`context+evidence`这个pre-expert latent，
它无法更新expert或ReZero branch，仍会重演“拓扑深耦合、优化浅耦合”。因此COER改为直接对生产
expert生成的pre-alpha branch proposal做关系蒸馏，并用相同权重在detached tokens上重算，让CLIP梯度
到达真实执行参数但不回流backbone。ReZero alpha仍只接受ReID loss，避免alignment自己放大残差后
伪装成retrieval contribution。

这也给“一个FAIL不否定全部”建立了形式化边界：Phase 0E只检验teacher code，Phase A只检验route
parameterization；两者逻辑独立而算力串行，只有Phase B需要二者同时通过。若最终correct evidence
不能同时拉开wrong/static控制和all-bypass，就只能称为local CLIP KD或generic adapter，不能成为论文
主贡献。

### 0E-128带来的创新边界修正

0E-128支持的是一个更窄但重要的事实：经过同图global与slot prior中心化后，CLIP局部视觉残差在
held-out PID上同时保留较高维变化和可反事实定位的局部binding。五slot macro effective rank达到
`11.050/16`，wrong RGB与same-RGB wrong mask的逐slot置信区间均为正；这比exp392的scalar-q只保留
between-slot prior前进了一步。

但fixed random orthogonal同样保留强margin，说明PCA本身既非必要语义轴，也不能作为贡献。真正可争的
对象仍是：rich residual如何通过生产expert成为counterfactually executable mediator，并在final
all-bypass中留下检索贡献。0E-128只是证明“有燃料”，没有证明router会使用它；因此下一证据必须保持
teacher richness与route activation两门分立，不能再次把teacher审计成功包装成ReID方法成功。

### 0E-FULL把“teacher有燃料”从小样本证据升级为全量证据

full审计在341个held-out PID、7,758图上把macro effective rank推到`12.335/16`，并使wrong RGB与
wrong mask的五slot置信区间全部稳定为正。这排除了0E-128只是抽样偶然的解释，也确认exp392失败并
非CLIP局部视觉完全无信号，而是scalar化和执行接口丢失了信号。

创新边界仍不变：random orthogonal比PCA拥有更高rank且保留强margin，所以不能叙述“PCA发现语义
方向”。可争贡献只能是rich residual对生产branch的梯度/执行所有权，以及final可反事实检索贡献。
下一步Phase A必须先证明route parameterization本身能离开identity，防止把teacher侧GO误当方法GO。

## 2026-07-19：RZ-C0证明“梯度可达”仍不等于“执行幅度被任务拥有”

Phase A把zero-expert冷启动修成random nonzero expert加zero ReZero scalar，CUDA preflight也确实观察到
alpha首步梯度以及后续token/context/expert连续更新。但e120两个alpha只到
`-1.843e-4/-1.363e-4`，synthetic descriptor gap仅`2.861e-6`，full与all-bypass的raw mAP差为
`-0.000249709 point`。所以ReZero解决的是局部Jacobian和梯度拓扑，不会自动让global ReID objective
为一个缺乏语义燃料的branch分配足够执行预算。

这使“CLIP-owned executable mediator”的机制边界进一步收紧：下一接口不能再让一个只受ReID loss的
自由scalar决定整条语义路径是否存在。候选应让rich evidence控制production branch方向，并用预注册、
与D0残差能量匹配的有界执行预算避免静默塌零；但固定/强制预算本身绝不能算贡献，必须由correct相对
wrong/static/generic和all-bypass的检索差共同证明CLIP ownership。CLIP仍不得直接蒸馏final descriptor，
推理仍删除teacher/text/pose。

Phase 0E全量PASS继续说明“有燃料”；Phase A FAIL说明“当前发动机拒绝点火”。下一研究对象是执行预算
与证据方向的因果绑定，而不是重跑RZ、调一个更大的alpha初值或回到普通local CLIP KD。

## 2026-07-19：exp394 CPU实现把“所有权”落实为四条互斥梯度路径

static/CPU contract带来的价值不是又得到一个能forward的adapter，而是把“CLIP-owned”拆成可否证的
四条所有权：evidence loss只拥有evidence head；mask/presence拥有anchor；`L_exec`拥有evidence head与
真实生产T/C/E/Expert；ReID拥有backbone、生产router与ID head。任何一条越界，都会把rich code改写成
identity code、把teacher梯度泄到final descriptor，或让expert再次只受弱global信号。

relation helper最初把768维proposal与16维code的维度差当错误，CPU FAIL反而澄清了关键机制：需要
对齐的是跨样本/slot的关系矩阵，不是直接特征坐标。这保留了CLIP对production direction的所有权，
又避免训练后删除一个768↔16 projector。固定rho与RMS仍只是防塌缩基础设施，不能写成创新；只有后续
真实checkpoint中correct相对wrong/static/generic和all-bypass同时改变检索，才有资格称为
counterfactually executable mediator。

## 2026-07-19：exp394揭示CPU梯度所有权正确仍不等于AMP数值可执行

exp394的CPU exact contract证明了四条梯度路径在拓扑上互斥，但actual batch64的首个scaled backward
在unscale后产生non-finite gradient。这增加了一条不能被CPU contract替代的创新约束：所谓
“executable mediator”不仅要让CLIP梯度到达production expert，还必须在正式AMP尺度下首步finite；
否则梯度所有权只是符号图正确，尚未成为可训练机制。

当前result未记录具体爆溢parameter组，不能事后声称relation loss、evidence normalization或某个router
就是根因，也不能通过降低GradScaler初值把同一臂救活。若未来另立机制，必须在协议冻结前把逐loss、
逐parameter-group的scaled/unscaled gradient range作为首步诊断输出，并把数值稳定性处理写成机制的
必要实现，而不是观察FAIL后调出来的超参。Phase0E仍说明rich evidence“有信息”，本次FAIL说明当前
production图还没有把它变成AMP-stable executable signal。

## 2026-07-19：exp395把AMP稳定性从“一个finite布尔值”改成可否证的归属矩阵

exp394只知道total backward后某处non-finite；这不足以指导机制创新，因为同一现象可能来自shared
ReID图、某个rich auxiliary子图，或多个有限loss聚合后的尺度交互。exp395先把证据单位冻结为
`loss × parameter-group × scaled/unscaled`矩阵，并保留D0 baseline、rich ReID-only、逐auxiliary、
pose和total四层比较。这样未来的数值稳定设计必须回应具体支持子图，而不是盲目降低scale或loss。

Phase0S只在CPU synthetic图上证明reporter本身可信：11个loss、15组、NaN/±Inf分类、动态范围和
zero-update均exact。它不产生新的方法贡献，也不暗示根因已经定位。真正可争的机制要求仍是：rich
evidence拥有production方向、固定预算可执行、actual AMP首步有限，并最终由wrong/static/generic与
all-bypass检索反事实证明语义因果。归因工具只是把后续设计从事后猜测变成先验可否证。

CUDA归因实现的静态封板进一步消除了一个执行偏差：两个consumer不能再被均值掩盖，D0 baseline与
rich ReID-only也不能被省略；任何actual结果都必须在同一first batch上同时呈现scaled和unscaled支持
集合。此处仍没有科学结果，只有可复核测量仪器。若未来actual matrix定位到具体子图，后续稳定机制仍需
另立实验并解释为何数值处理是机制必要条件，而不是为通过AMP门临时调出来的工程参数。

### exp395 actual暴露“synthetic-correct”之外的测量规模契约

exp395的CPU synthetic reporter在小张量上13/13 PASS，但真实backbone组的元素规模超过canonical
`torch.quantile`支持上限，导致actual在第一行scaled capture中止。这个失败不是模型机制证据，却修正了
诊断方法的创新边界：逐loss归因器不仅要在数学上正确，还必须把统计复杂度、内存上界和超大组的exact
percentile算法写进先验契约。否则“完整梯度矩阵”仍可能只是小样本仪器幻觉。

下一版只能把reporter作为独立测量对象：count、NaN/±Inf、abs-max与L2应单遍chunk归约；P50/P95/P99
必须采用可证明与线性插值定义一致的chunk-safe exact selection/sort方案，并在小张量上逐字节对齐
`torch.quantile`、在超过原限制的synthetic规模上完成。即使exp396最终得到矩阵，这仍是诊断基础设施，
不能包装成CLIP–TAPF贡献；方法主张仍需后续AMP-stable机制与retrieval反事实支持。

### exp396把“AMP-stable”从绝对首步finite改成matched baseline-relative轨迹

完整矩阵显示D0与rich的ReID loss、backbone non-finite支持和NaN/±Inf计数逐项相同，而所有rich
auxiliary均finite。这意味着exp394的首步失败不是CLIP-owned mediator新增损失特有的数值问题；更可能
是default GradScaler初始scale下shared ReID backbone的正常overflow候选。GradScaler本来就通过skip与
动态降scale处理这种事件，因此“第一步必须绝对finite”不是一个经过D0校准的可训练性定义。

下一门的创新价值不在调小scale，而在把数值可执行性重新定义为matched dynamics：D0与rich从同一默认
scale出发，记录自然skip/update与scale轨迹，要求rich不增加skip、不延迟首个成功update，并证明
rich-specific head/router在成功step中finite。若两者轨迹相同，exp394只能说明旧门设计过严；若rich
后续额外失败，才能把问题重新归到production graph。这个定义仍是机制开发纪律，不是论文贡献本身。

### exp397说明“relative parity”与“绝对生产力门”必须拆开

exp397中D0与rich的12步scale/skip轨迹逐项相同，rich-specific 11组也从未产生non-finite；但两臂都要
经过e1五次native backoff才首次更新，e6切换又出现一次shared-backbone skip，所以按预注册的
`>=10/12`和首个成功`<=3`仍必须FAIL。这个结果不能被改写成rich PASS，却清楚说明失败来自把matched
relative hypothesis与一个未经canonical D0校准的绝对吞吐阈值绑在同一门里。

后续若继续，新的测量对象应是production-shaped阶段内的baseline-relative适应与稳态，而不是事后把
exp397阈值降到刚好能过。必须保持default scaler自然backoff、同batch/RNG、rich-specific finite和
零手工scale；再以更长的预注册窗口分别检验e1收敛、e6迁移和稳态连续更新。该诊断仍不是论文贡献，
最终贡献门依然是formal checkpoint的retrieval与semantic counterfactual。

### exp398再次暴露“synthetic row正确”不等于真实parameter容器正确

exp398的轨迹裁决器在synthetic row上覆盖了extra skip、尾窗、shared subset和group inactive，但新增state
hasher没有用真实reporter返回的`(name, parameter)`容器做contract，actual因此在首个forward前失败。
这不是AMP或方法证据，却补充了诊断器纪律：涉及parameter ownership/state的工具必须对真实容器类型、
顺序、重名和空组做exact测试，不能只对标量化的synthetic结果做AST审计。

下一编号若修复，只能把这一点视为测量器必要条件；32步baseline-relative问题、rich production稳定性和
论文贡献都仍未被回答。不能因为update为0就把失败解释成模型安全，也不能把reporter修复包装成方法创新。

### exp399把shared scale适应与rich生产稳定性真正分开

32步actual显示两臂前期六次skip完全同源于shared backbone，随后在scale=`1,024`连续25步成功；rich既
没有额外skip，也没有任何独有non-finite，同时11个新增组在e6有非零梯度并真实改state。这是首个真实
CUDA证据说明production rich graph在native scaler稳态上不劣于D0，而不是靠手工降低scale过门。

这仍只是可训练性基础设施，不是方法贡献。下一final preflight必须证明内存更新后的production state可
strict reload、teacher-free、RGB-only，并让两个consumer在nonzero rho下产生有限且非零的bypass差；
之后仍需e120 retrieval和semantic counterfactual决定论文故事。

### exp400把“能更新”收紧为可部署的四重接口契约

exp399只回答native scaler稳态，仍可能存在四类未闭合风险：训练state携带teacher资产、strict reload
不完整、eval意外读取pose、或两个consumer中只有一个真正进入descriptor。exp400因此把生产门定义为
state接口、RGB-only接口、rho schedule接口和双consumer反事实接口四者同时成立，而不是再增加loss或调参。

这个门仍属于方法开发纪律，但它避免把“参数更新过”误写成“可训练且可部署”。尤其是单独bypass0/
bypass1必须各自产生严格非零差，能提前排除一条router名义存在却没有执行影响的假阳性；真正的方法贡献
仍只能由后续e120 retrieval和final all-router-bypass差决定。

### exp400 actual闭合了接口证据，但仍不替代检索反事实

actual中两个consumer的单独bypass差分别达到max-abs `0.07276`与`0.08659`，all-bypass mean L2为
`0.42050`；这比只看极小GateAbs更直接地证明两个执行接口都进入descriptor计算。与此同时RGB-only和
teacher-free strict state全过，说明teacher仅是训练监督源，不是部署依赖。

不过这些差仍来自32-step内存state，不能推断e120后的mAP贡献。下一步的创新证据单位必须回到正式
checkpoint上的`full - all-router-bypass`检索差，并与Semantic C0性能底线共同裁决route alive。

### exp401证明fixed-budget route存活，但信号仍处在接口级而非贡献级

exp401首次在自然e120 checkpoint上同时满足full mAP floor与all-router-bypass差：full raw mAP=
`57.1230075595`，bypass=`57.0035860757`，差=`+0.1194214838 point`。这意味着owned fixed budget没有像
free ReZero scalar那样静默塌成retrieval identity，Phase0E rich evidence、AMP稳态和正式执行route终于
在同一链条上闭合。

但这个PASS只比预注册门高`0.0194214838 point`，而R1差为`−0.0904977322 point`；更准确的解释是当前
route对整体排序有小而可测的重分配，不是已经形成强top-1优势。下一创新对象因此不能是继续调rho、loss、
batch或route scale，而应是Phase-B语义接口：在保持exp401执行预算与production graph冻结的前提下，
验证correct evidence是否稳定优于wrong RGB、wrong mask、slot-static/random-orthogonal与generic expert-mean，
同时保留all-bypass。只有correct-specific差与route差共同扩大，才能把“活的执行接口”升级为
“CLIP-owned semantic mediator”；否则exp401仍只是一个严谨但弱效的接口证据。

### exp402把“route alive”与“semantic mediator”正式拆开

exp402给出了关键的反事实分离：关闭两个router时mAP下降`0.1194214838 point`，所以exp401的执行route
确实存活；但把每张图的evidence换成同camera不同PID donor或全零，并不会降低检索，wrong-RGB甚至高
`0.0066900358 point`。因此“route有贡献”不能再被当成“route使用了图像特定CLIP语义”的替代证据。

所有干预都产生大于零的descriptor差，排除了patch没有触达的解释。generic expert mean下降
`0.1240184555 point`，router0 bypass下降`0.1307568556 point`，而router1 bypass反而略升；这更像是
slot-specific expert参数先验与两个router的非对称组合在重排检索，而16维sample-specific evidence没有
形成正确身份条件。下一创新对象若仍保留CLIP–TAPF，必须在训练目标中直接建立
`correct > wrong RGB > zero/generic`的可辨识顺序，或重新定义能被错误身份证据破坏的结构接口；仅扩大
rho、增加stage、调loss权重或继续装饰现有evidence head都不再具备机制依据。

这也收紧论文边界：exp401可作为fixed-budget route非identity的弱接口证据，exp402则是有效负消融，
明确阻止把该route宣传为CLIP-owned semantic mediator。Phase0E证明teacher evidence存在、Phase0R证明
某些readout可分，并不等于当前C0 student接口在检索中使用了这些信息；后续主张必须重新闭合这一断点。

### exp403：Evidence-owned Low-rank Operator + Counterfactual Utility Ranking

exp402之后，值得保留的不是static expert前继续增加evidence loss，而是把evidence从activation bias升级为
operator coefficient。ELO使用跨五个slot共享的`U/V/C/H/Q/K`：`H(e)`生成逐rank系数，local
feature/evidence compatibility直接乘生产delta；没有slot-specific expert prior，`e=0`严格identity。

CUR在同一Stage-3 input上重放wrong-RGB、generic和NULL完整执行，但reference全部stop-gradient；loss只能
提高correct execution相对同ID positive prototype的utility，不能靠主动破坏control制造margin。同时兼容性
预注册`correct>wrong>generic>NULL`，把“有效人类evidence”和“属于当前RGB的evidence”拆开。

它满足三项创新门槛，但风险仍为`6/10`：审稿人可把它概括成dynamic adapter + CAL loss。只有以下证据
同时成立才有方法级价值：

1. NULL exact identity与shared-operator source contract；
2. reference无梯度、correct生产组有梯度；
3. full不低于sealed clean D0；
4. correct相对wrong/generic/NULL至少`+0.1 mAP`；
5. correct相对all-bypass至少`+0.1 mAP`。

standalone contract已两遍`26/26 PASS`，只授权生产实现。compatibility proxy若成功而final retrieval失败，
仍必须判mechanism NO-GO，不能继续调margin/rho/loss。

生产合同与真实batch64 CUDA/AMP门现已分别`34/34`、`16/16 PASS`，证明ELO-CUR在完整生产图中具备预期的
参数所有权、reference隔离和AMP可更新性；formal已经启动。这里不新增创新主张：这些仍只是机制可执行证据，
最终是否形成“图像特定evidence ownership”只能由e120后correct相对wrong/generic/NULL与all-bypass的全量检索
反事实决定，不能用训练期compatibility或CUR曲线替代。

### exp403封板：operator可执行不等于retrieval ownership

唯一fresh e120及七臂正式反事实已经给出明确否证。correct raw mAP=`0.569929559315091`，低于clean D0；
wrong-RGB、generic、NULL和all-bypass分别为`0.569934358329593/0.569937131506918/
0.569937304669369/0.569937304669369`。correct相对semantic-control最大值和all-bypass的margin均为
`−7.745354277944e-06` raw mAP，四个性能/所有权门全部失败。七臂R1/R5/R10逐项相同。

这个负结果不是“算子没动”：所有干预臂`exact_equal_rows=0`，其中NULL/all-bypass descriptor mean L2均为
`0.03582507`，generic为`0.03506619`。因此可排除patch未触达或route数值死亡；真正失败的是这些明显的
descriptor变化没有形成identity ranking ownership。训练期compatibility约`0.92`且CUR持续提供有限信号，反而
暴露了一个重要shortcut：局部feature/evidence兼容性与batch prototype utility可以被优化，却不保证最终
全局检索按正确图像evidence排序。

由此关闭当前`dynamic low-rank operator + stop-gradient CUR`组合，不能用更大rho、loss重权、额外stage、
batch变化或删control继续包装。下一轮机制学习应优先寻找一种**最终检索对象本身不可绕过的所有权约束**，
或重新定义比“evidence调制descriptor”更强的结构对象；候选仍须同时回答问题/机制/证据至少两项，先保留
wrong/generic/NULL/all-bypass强反事实，再决定是否值得进入新编号。exp401 route-alive仍是边界事实，但不能
为任何后继机制代替semantic ownership证明。

### exp403后查新：终端概念瓶颈不是现成的新机制

CHAIR已经把人工修正concept经线性edit加到最终embedding，并在归一化后直接做retrieval；IntCEM已经用
多步intervention trajectory、干预后task loss和干预策略imitation训练可干预性；MCBM则精确指出“能预测
concept不等于latent只含concept”，并用每概念minimality/IB约束降低信息泄漏。三者分别封住了additive
concept edit、intervention-aware training和普通minimality loss的创新空间。

更强的结构替代也已有近期直接先例。SupCBM说明hard/binary concept仍会泄漏，并用层级concept与稀疏
intervention matrix约束决策；MM-CBM让image/text双分支只在非负concept response空间计算最终相似度；
2026-07的Caption Bottleneck Models让下游分类器严格只读冻结LMM生成的文本，以stage isolation声称
leakage-free by construction。PDiscoNet的无监督part slot也只是结构化part discovery，不提供外部evidence
对检索排序的所有权。

因此`terminal concept-only subspace + minimality + intervention-aware loss`不能作为exp404：它是公开CBM
原子的组合，而且fixed direct-sum/norm quota仍可能只是在数值上强迫扰动，不证明语义所有权。目前仅有
**问题/证据缺口**可能成立：open-set instance retrieval尚缺matched donor、generic、NULL和all-bypass共同
构成的path-complete ownership检验。机制缺口尚未找到，继续查source attribution、interventional path
completeness与conditional metric；在至少满足问题/机制/证据两项之前不编号、不占GPU。

### exp403后第二轮查新：modality laziness与evidence-use已高度拥挤

**状态**：`AUDIT ACTIVE / MECHANISM GATE STILL EMPTY / NO EXP404`

本轮把exp403的“route active但ranking不拥有correct evidence”映射到modality laziness，代码/公式级审计了
UniCat、MCR、Data Remixing、ResTacVLA、SCOPE、RCL、VIGIL、MiMIC与VLM2Rec。最重要的新边界不是一个可开跑
模块，而是若干已被占用的机制空间：

1. UniCat已经在multimodal ReID中证明独立单模态loss、测试concat可胜过joint fusion；
2. MCR已经用latent permutation、预测JSD/MI surrogate和game梯度平衡模态贡献；
3. ResTacVLA已经把弱模态写成“实际触觉减视觉可预测触觉”的residual，再经VQ与uncertainty gate注入；
4. SCOPE/VLM2Rec已经用matched-mismatched关系和跨模态rank/topology一致性保护弱模态几何；
5. RCL已经把counterfactual channel suppression变成instance reliance profile并做teacher/student matching；
6. VIGIL已经直接优化seeing相对attention-masked blind的output likelihood gap；
7. Data Remixing/MiMIC已经覆盖样本拆分、单模态mixin/dropout和额外hard-negative stage。

这些工作与当前缺口仍有一个本质差异：它们的wrong/masked modality要么只被删除、置零、置换或推远，要么仅
作为弱模态统计量；matched donor没有保留“对donor identity必须正确”的独立正目标。于是模型仍可通过破坏
counterfactual branch或维持强模态路径取得margin，不能建立source-specific ownership。

本轮末曾把下一机制暂定为三方合同：同一构图同时使correct execution服务current ID、matched wrong
execution服务donor ID，并让二者经过相同最终欧氏descriptor路径。第三轮已确认其中`wrong -> donor ID`只对
身份充分组件成立，不能无条件用于当前semantic evidence，修正见下。generic/NULL/all-bypass仍保留为终审；
`full > blind`、MI/topology、固定concat、predictive residual、dropout/remix或loss重权均不得单独建立exp404。

### exp403后第三轮查新：身份标签必须跟随身份充分组件

**状态**：`AUDIT ACTIVE / PROVISIONAL DONOR-ID CONTRACT REJECTED / NO EXP404`

DG-Net代码中的`x_ba=decode(s_b,f_a)`之所以按A身份监督，是因为`f_a`本身来自完整图像的ReID appearance/ID
encoder；其标准测试descriptor直接由原图ID encoder产生，交换生成路径并不进入部署。Hi-CMD提供了更关键的
反例：`(c_b_recon,s_a_recon)`的标签为B，`(c_a_recon,s_b_recon)`的标签为A，身份跟随prototype/content，
而不跟随被交换的style donor。CIFT则只以高斯反事实输入替换affinity，最大化当前身份下正确/反事实topology
的输出差，没有donor identity，且正式测试是query-gallery graph。

这使上一轮“三方合同”必须修正。当前16维evidence表达support/appearance语义，不是identity-sufficient code；
要求`feature_A + evidence_B -> ID_B`会迫使evidence泄漏身份，或给现实中不存在的组合任意贴B标签。若只要求
重建B的semantic state，目标虽可定义，却回到DG-Net/Hi-CMD已经覆盖的swap/re-encode/cycle辅助目标，并不能
保证最终欧氏身份排序拥有该语义。

因此`wrong -> donor ID`不再作为普适准入门。新的必要条件是：正目标随信息承载者分配；semantic donor只对
semantic state负责，但该正目标还必须与最终identity descriptor共享不可绕过结构，并在correct/wrong/
generic/NULL/all-bypass终审中产生因果顺序。当前没有公开或自洽机制同时闭合这些条件，创新门仍只有问题/证据
缺口，exp404继续NO-START、GPU继续NO-START。

### exp403后第四轮查新：semantic ownership缺的是realized target，不是composition loss

**状态**：`AUDIT ACTIVE / IDENTIFIABILITY GATE FAIL / NO EXP404`

Composed Person Retrieval把问题写成`(reference image, relative caption, same-ID target image)`，FAFA公开代码
直接用融合query对齐真实target的多token视觉表示。这是semantic modification拥有检索正目标的合法形式，因为
target已经在数据中实现该修改。其代价也很明确：115万SynCPR triplet依赖LLM、Flux成对生成与MLLM过滤；
ITCPR需要人工relative caption；测试继续输入caption并使用query-conditioned top-k token similarity。

DiCE-CIR虽以LLM生成target caption代替真实target image，仍保留显式edit与target-semantic proxy，并在测试时
输入edit text。两者共同说明，单纯写`Phi(image,evidence)`并不能创造正目标；必须观测到修改、target或其可信
semantic proxy。

当前wrong-RGB来自不同PID，其16维evidence不是对host A的relative edit，也不存在已知的同身份
`target(A,e_B)`。把same-ID多图硬配成target仍可由identity trunk绕过evidence；把target合成、标注或改成测试时
conditional query则分别落入已有CPR/DG-Net或违反单RGB固定descriptor边界。

因此下一问题被收紧为“如何从official RGB内部得到可验证的realized semantic target”，而不是再加composition/
contrastive loss。当前只增加了可识别性负边界，没有机制创新，exp404与GPU继续NO-START。

### exp403后第五轮查新：解析equivariance只适用于已知action

**状态**：`AUDIT ACTIVE / IDENTIFIABILITY AND MECHANISM FAIL / NO EXP404`

DiP给出一个可比较的RGB内部target：对原图施加已知affine矩阵`K`，其implicit part position可解析变换为`Kp`，
所以position-equivariance有真实监督。但该位置在测试时被丢弃，最终仍是pair-specific part-weight distance；
它既不是固定descriptor ownership，也不处理外部sample evidence。

当前support/appearance code没有已知semantic action `R(K)`：translation/flip/occlusion对这16维量应保持、置换
还是改变并无解析定义，different-PID donor也不是由已知变换生成。因此仿射一致性只能添加已有几何proxy，不能
得到wrong/generic/NULL的合法排序。

可逆网络同样不解决归属。经典不可识别性结果表明，bijective latent仍可做无穷多因子混合重参数化；这里虽有
teacher code监督，但它只定义student复现code，不定义code对final ranking的唯一作用，恰好与exp402/403观察一致。

故`equivariance + invertibility`没有通过机制门，不建立exp404。下一对象必须显式给出当前evidence的可验证
semantic action，并让该action本身成为最终固定identity metric的一部分。

### exp403后第六轮：封板资产不足，canonicalization不是现成ownership机制

**状态**：`AUDIT ACTIVE / ARTIFACT AND MECHANISM FAIL / NO EXP404`

只读检查发现exp402/403的19,871行evidence与各臂descriptor仅存在于formal audit内存。最终JSON最长数组为2，
远端目录没有逐样本tensor；Phase0E codebook只保留`5x768` slot mean、`16x768` PCA basis与计数，generic asset
只是`5x16`常量。因此无法在不补跑封板编号的前提下计算identity separability或camera confounding，后续叙事
不得把这两点当已验证事实。

针对“让evidence直接执行canonical action”的查新也给出三条边界：

1. 3D-VAN已经使用3D人体重建和四视角canonical rendering，测试仍拼接原RGB feature；
2. CSCL用人工密集2D–3D correspondence监督pixel-to-SMPL-vertex，依赖当前official资产不存在的SMPL/DP3D变量；
3. VPFA用同图LR/HR exact pair训练冻结descriptor后的MSE residual，测试以文件名resolution suffix选择MLP。

这说明可识别canonical target来自额外观测action，不是结构名称本身。把wrong donor的pose/code用于host warp只会
构成damage control，不能定义wrong source的正目标；保留RGB旁路又复现modality laziness。故本轮只增加了
artifact与近邻边界，没有满足机制创新门，不建立exp404。

### exp403后第七轮：已知patch source仍缺global identity target

**状态**：`SOURCE-PROVENANCE TRILEMMA / INNOVATION GATE FAIL / NO EXP404`

SPT公开代码表明，official RGB内部确实能构造source-exact token composition：saliency mask保留target identity
token，并从另一身份搬入真实背景/遮挡。SPT最终仍训练标准global descriptor，但论文必须忽略candidate class，
而不是把candidate残余身体区域当第二个identity target。

Token Labeling和TokenMix已分别覆盖dense token class supervision、teacher activation加权的mixed target，所以
“给每个source patch贴标签”本身也不是新机制。对当前目标有三种可能：

- context-only donor：host target合法但无donor ownership；
- full-identity donor：donor target合法但evidence退化为完整身份像素，变成已知foreground transfer/swap；
- partial semantic donor：局部source label合法但global chimera没有单一PID正样本。

因此source mask只能解决“patch从哪来”，不能回答“组合后的固定descriptor应检索谁”。继续加part CE、mix ratio、
mask规则或global soft label只会落入现有augmentation原子，不建立exp404。

### exp403后第八轮：uncertainty-owned metric不是新的可识别对象

**状态**：`UNCERTAINTY PRIOR SATURATION / ORDER UNIDENTIFIABLE / NO EXP404`

PFE已经冻结mean embedding、为每张图预测逐维variance，并用pairwise mutual likelihood作为正式metric；QPM、
part uncertainty、spatial/channel uncertainty、local uncertainty和probabilistic ReID又把质量/方差对象带进了
遮挡ReID。Bayesian Metric Learning则说明posterior uncertainty也可只服务校准而不改变均值ranking。

让16维evidence生成fixed-rank covariance或orthogonal deletion projector只能改写这些已有原子。它还缺少正确
subspace标签：same-ID positive可由RGB mean绕过，固定谱预算只保证“删了东西”；wrong donor可能与host具有相同
nuisance，generic prior也可能优于sample estimate，NULL语义取决于人为定义。故无法从概率结构推出
`correct > wrong > generic/NULL`，不建立exp404。

### exp403后第九轮：set-valued identity只把缺口推给target selector

**状态**：`SET TARGET SELECTION GAP / NO EXP404`

source-separated multi-vector可在合成A+B图中为两组token提供各自PID，看似绕开partial chimera的global label。
但KPR已经覆盖SOLIDER/Swin的一人多part embedding、visibility与multi-person ambiguity；遇到多人时，它需要正负
keypoint prompt指出目标。现有part/multi-token方法的多个vector也仍共享一个sample identity。

真实official query没有多PID token ownership标注。保留所有component会改变为multi-label任务，选择host需要
额外prompt/instance assignment或heuristic gate，聚合则恢复无单一PID的chimera。因此“输出一个set”不是闭合
ownership的新机制，不建立exp404。

### exp403后第十轮：强反事实仍可能是random-key假阳性

**状态**：`COUNTERFACTUAL NECESSARY-BUT-NOT-SUFFICIENT / NO EXP404`

learned canonicalizer看似能让correct evidence解锁canonical identity、wrong evidence只对donor有效，但没有
已知group action时，这等价于把evidence当密钥。NeurIPS 2024 canonicalization与TARGET-VAE分别依赖已知graph
symmetry和translation/rotation group，不能为任意16维code提供语义许可。

纯CPU正反合同进一步给出existential counterexample：随机、PID无关key也形成correct/wrong/generic/NULL
mAP=`1.000000/0.608134/0.039243/0.030195`，随机置换key后仍通过两级margin，mutant则失败。因此
`correct > wrong > generic/NULL`只能保留为必要门。未来机制还必须证明semantic evidence优于random-key/
null-semantic source checksum；否则只证明authentication，不证明CLIP语义所有权。

### exp403后第十一轮：跨样本重复不等于语义

**状态**：`SEMANTIC REPLICATION PRIOR SATURATION / DIAGNOSTIC INCONCLUSIVE / NO EXP404`

用identity-level共享类别替代unique sample key，看似能切断逐样本checksum；但MVI²P已覆盖同ID多视图综合、
CAM可靠性加权与full-feature传播，AG-ReID已覆盖identity-majority属性伪标签、属性token和噪声屏蔽。普通
prototype、跨样本状态重复或多视图语义复制不是新的机制对象。

频率保持random-cluster诊断也不能提供正面许可：原始与label-permutation观察都出现
`correct > wrong > generic/NULL`，mutant被抓，但原始一簇只有38个PID，未过冻结`>=40`门，正式裁决只能是
`DIAGNOSTIC_INCONCLUSIVE`。不得把这次观察升级成“随机共享类别已经证明可伪造语义”，也不得重跑救门。

下一候选必须同时击败unique random-key与frequency-matched semantic-blind null，并保持sample-specific正确
语义的可验证target；把当前region-global code做identity-majority聚合会改变原问题，不能建立exp404。

### exp404 C-track：Semantic Product Kernel final descriptor

**状态**：`C-CLASS CANDIDATE / PRODUCTION CPU PASS / CUDA PREFLIGHT AUTHORIZED`

目标降为C类后，可接受“已有基础原子 + 新问题对象 + 强证据协议”的适度组合创新。SPK不再把evidence送入可被
global trunk绕过的hidden residual，而把`16 x 48`分组后的final feature与无参数
`16*softmax(aggregate(evidence))`逐组相乘。NULL产生全1 factor，故bypass exact；训练/测试始终读取同一个
768维固定descriptor。

亮点不在张量积首创，而在：标准欧氏单descriptor中把semantic path变成结构必经项，并同时用matched wrong、
generic、NULL、unique random-key与frequency-matched random-cluster终审。若只过wrong/NULL而不过random
controls，则仍判source authentication，不写semantic ownership。

production实现进一步确认该适度机制不是auxiliary-only包装：真实`build_transformer.forward`在BNNeck、分类器、
triplet返回和两种eval neck路径之前执行同一SPK，CPU动态合同中的global/evidence梯度均finite/nonzero；state中
没有旧C0/ELO router。v2连续两次`41/41 PASS`。

这只把exp404从概念候选推进为可执行C类结构候选，不增加理论新颖性。贡献边界仍冻结为：final descriptor
必经绑定 + random semantic null证据协议。CUDA preflight可以开始，但在formal强反事实前不能写semantic
ownership或性能贡献。

CUDA preflight静态合同连续两次`33/33 PASS`只说明该机制能按冻结recipe进入真实硬件验证，不提高创新主张。
C类候选边界继续保持“适度结构 + 强证据”，formal前状态仍不是有效方法。

actual CUDA v1暴露的5/17通道错误进一步说明SPK的适度机制必须与D0 pose对象严格分离：5-slot rich region state
服务semantic supervision/SPK，17-joint field服务原D0 spatial gate。修复属于设计一致性，不是新创新点；v3合同
通过也不提高论文claim。

CUDA v2的默认scaler backoff是数值执行边界，不是机制创新或性能证据。v3沿用既有native-scaler稳态合同，不改变
SPK主张；无论v3结果如何，都不能把AMP稳定性写成论文贡献。

v3 actual与formal prelaunch均已通过，说明SPK可以进入唯一正式验证；它仍不增加创新性。C类贡献是否成立只看
最终strong-control retrieval证据，而不是preflight数值。

唯一formal已启动不改变创新判断。C类门槛保持为“固定final-descriptor语义绑定 + random null强证据”；在e120
与全量反事实结束前，训练健康、factor active或中间精度都不能升级为贡献结论。

exp404 e120未超过clean D0的mAP/R1，因此它不具备“更强主干性能”贡献。C类候选只剩机制证据路径：若correct
能同时击败wrong/generic/NULL、unique random-key、frequency-matched random-cluster与all-bypass，仍可形成
适度结构加严格ownership证据；任一主门失败则不再包装为正面方法。

终审v1 static两次`32/32 PASS`只确认强证据协议能在真实`student_evidence/student_presence -> SPK`输入缝执行，
且wrong/random/bypass/validity失败不会被reporter漏掉。它不增加机制新颖性，也不恢复已失败的D0性能主张；
创新判断继续只等待全验证集correct相对五个主null和product bypass的冻结mAP门。

CUDA preflight还给出一个负向结构事实：actual student hard presence在抽样中五槽全1，使presence循环无作用；
slot-cycle又因SPK均值聚合而置换不变。这不影响wrong/random等主ownership null，但说明不能把槽位顺序或mask
敏感性写进贡献。v2修正的是reporter作用域，不是增强机制，创新上限反而进一步收紧。

formal终审最终否定SPK：correct与wrong几乎相同，generic略优，NULL/bypass更高`0.18094 mAP point`。两类random
更低只说明任意强扰动会伤害descriptor，不能证明正确语义拥有排序。由此关闭“把单图CLIP/student evidence作为
final feature乘法因子”的创新解释；下一pose+CLIP候选必须改变训练或跨图结构对象，并且目标应直接带来D0增益，
不能再把sample evidence注入位置换名重试。

### exp405候选：CAVT二维反事实解剖视图运输

**状态**：`BROAD NOVELTY NO-GO / NARROW PROBLEM+EVIDENCE CONDITIONAL GO / STUDENT NO-START`

“CLIP选槽 + 同ID donor + token搬运”本身分别落入pose-aware part ReID、multi-view LUPI/KD、masked feature
recovery与TokenMix/SPT近邻，不能直接包装创新。当前可争差分被收紧为一个完整对象：在单RGB固定descriptor
ReID中，以original/deleted/donor提供可观察target，让同一pose-defined anchor同时负责slot定位、局部读取与
中间stage回写，并用identity轴 x slot轴二维反事实证明CLIP双编码状态拥有可执行转移，而非普通条件扰动。

最低成立门不是涨点，而是teacher端和donor-free student端都满足same-ID/same-slot分别高于
same-ID/wrong-slot与wrong-ID/same-slot，并击败pose-only、image-only、text-only、generic、NULL、random-key
和frequency-matched random-cluster。CLIP若只提高donor筛选、不控制residual地址/预算/内容，降级为普通
multi-view completion；多stage若不超过semantic single-stage，只能判层级增益不成立。

第二轮近邻审计确认，机制项目前仍不成立。最小Phase 0还必须加入MVI²P-full、pose-part、
attribute-relation与generic-transport对照，并证明teacher residual在held-out PID上可由单图`not-k`状态预测。
只有这些门全部成立后才允许重新设计production operator；不能把oracle闭式公式直接包装成方法。

static v14的`56/56 PASS`及`0B/0H/0M/0L`盲审只说明二维干预和donor-free门可以被无歧义实现，不增加
CAVT的新颖性分数，也不是teacher有效证据。创新判断的下一信息增益只来自真实train-only数据：correct是否在
identity轴与slot轴上同时拥有顺序、CLIP是否独立胜过pose-only，以及held-out PID transition是否可预测。
因此不再投资静态启动器变体。

真实teacher static v8进一步关闭了“测量器本身可能制造顺序”的主要出口：correct/wrong-mask使用全局隔离的
recipient/donor、caliper内完整一对一匹配，non-torso CI以跨槽共同PID抽样，preflight与formal科学裁决彻底
分开；紧凑匹配实现也避免在得到科学结果前因metadata OOM消耗once-only。两次`8/8 PASS`与三路无B/H盲审
只说明下一次512图CUDA preflight值得运行，不提高机制创新分数。

创新状态仍是`BROAD NOVELTY NO-GO / NARROW PROBLEM+EVIDENCE CONDITIONAL GO`。只有full-train P0B未来证明
真实双编码slot state具有正确语义、删除单调性和non-torso稳定性，才继续transport；若P0B NO-GO，则关闭当前
readout对象，不通过调prompt、temperature、mask、loss或batch补救。

v9的MMPOSE-ABU兼容只属于复现边界，不构成问题、机制或证据创新；它没有改变CAVT创新评分与kill-switch。

v10只修复MMPOSE-ABU Python 3.8下的AST合同兼容，三路盲审无B/H。该修复既不是新机制，也不增加科学证据；
真实信息增益仍只能来自region-isolated teacher的二维反事实与后继donor-free可预测性。

远端MMPOSE-ABU v10 static通过只关闭运行时歧义，不提高CAVT创新评分。512图preflight仍只测机械接线；只有
后继full-train P0B的真实二维反事实才可能产生科学信息增益。

exp405 preflight v1只暴露512图子集内wrong-mask候选池不足，未计算任何科学指标，因此不改变CAVT的创新评分，
也不能被当作机制负证据。后继若修正，只能扩大/重构机械候选池合同，不能放宽正式caliper或删减强反事实。

exp406采用固定尺度的单调donor扩池，严格属于测量合同修复，不计入问题/机制/证据创新。它的价值只是避免512
机械子池用覆盖不足提前烧掉正式科学问题；CAVT创新状态仍等待full P0B二维反事实，未获得任何加分。

exp406 static两次PASS只证明“不降门的full donor universe”可被无歧义实现；它仍是测量基础设施，不增加
CAVT机制新颖性或论文证据。

v1盲审否决与v2合同修复仍只属于测量基础设施，不计创新。为避免合同工程吞噬主线，v2只闭合已知B/H并做一次
最终复审；之后若CAVT teacher formal NO-GO，立即停止该对象，回到新的pose+CLIP训练机制设计。目标为可支撑
C类会议的清晰机制与实际涨点，但仍保留clean D0、同epoch mAP/R1和关键pose/CLIP反事实。

exp406的cache自检runtime失败不改变CAVT创新判断；它发生在结果发布层，不是teacher机制证据。exp407只做受信任
cache roundtrip兼容修复，同样不计创新。若exp407机械PASS，应立即推进formal科学裁决，避免继续在基础设施停留。

exp407 targeted roundtrip与盲审0B/0H仅恢复可测性，不满足创新门槛。后续创新判断只看formal是否建立pose-region
CLIP evidence的强反事实顺序，以及student相对clean D0是否自然e120涨点；任一失败即换下一训练对象。

exp407 preflight八项validity PASS仍不提高CAVT创新评价，只说明formal可被可信执行。下一步不再增加基础设施实验，
直接让formal回答correct region是否稳定胜wrong-mask/deletion/generic controls。

exp407 formal最终被wrong-mask caliper有效性阻断，未回答CAVT科学问题。连续三次测量器失败说明这条路线的证据成本
已经超过当前C类会议目标可接受范围。CAVT不作科学否定，但从活动主线移除；下一机制必须直接改变训练对象并通过
clean D0、wrong-pose/zero-pose反事实和ReID mAP/R1形成证据，禁止再设计donor匹配测量器。

## 2026-07-21：exp408 PICRD——从语义路由改为pose-indexed关系训练

### 新对象

旧路线问“CLIP evidence如何驱动router”；PICRD改问“pose定义的局部CLIP关系能否直接改变形成global descriptor
的中层表示”。每槽分别在batch维构造关系矩阵，避免head-vs-leg固定语义主导跨槽Gram；correct target为唯一
正绑定，wrong-RGB、generic和zero进入训练内排序。学生没有投影头，loss直接回传未detach Stage-2。

### 创新门

- 问题门：PASS——标准单图ReID中的实例pose-CLIP binding可辨识性；
- 机制门：CONDITIONAL PASS——只有“逐槽relation+强反事实排序+Stage-2直传”整体成立；
- 证据门：PASS——四臂距离、backbone梯度、clean D0同epoch和自然e120双门可以直接裁决。

该组合未发现公开同构实现，但π-VL、ProFD、PAFormer、KPR、MUVA和CVPR26 Composite-Attribute ReID覆盖了
大部分原子，因此定位为C类候选。若退化为per-slot cosine/KL或feature add，立即失去机制新意。

### exp408封板后的创新判断

PICRD的证据门只通过一半：冻结diagnostic证明实例级正确binding可学，排除了“CLIP局部target完全无信息”和
“实现未反传”两种解释；但自然e120 mAP低于clean D0，说明仅让Stage-2复现逐槽跨图关系，并不会自动增强
identity-discriminative retrieval geometry。因而“relation KD本身就是贡献”的叙事关闭。

下一候选不能再换一种relation/cosine/KL或在同一路径上调权。它必须让pose+CLIP直接参与身份判别对象的构造，
例如改变正负样本、身份原型或可见证据一致性，而非只约束中层相似矩阵；同时仍需wrong-RGB/generic/zero等强
反事实证明收益来自正确绑定。问题/机制/证据创新门重新评估，不继承PICRD的C类资格。

## 2026-07-21：exp409 PCHM——从语义关系监督改为身份训练边

### 新对象

PCHM不再要求student复现CLIP feature或relation。它把增强后五槽pose coverage与fresh五槽region-isolated CLIP
appearance变成无权ordinal rank，在真实PK batch内直接选择final global descriptor原soft-margin triplet的边：
同ID选择pose互补且CLIP一致的正对，异ID选择pose匹配且CLIP相似的负对。pose/CLIP不参与梯度尺度，eval完全
回到D0 RGB descriptor。

### 创新门

- 问题门：PASS——batch-hard没有显式区分跨遮挡互补正支持与同姿态同外观异ID混淆；
- 机制门：CONDITIONAL PASS——必须离散替换pair index；一旦变成loss/distance加权、margin或top-k调参即FAIL；
- 证据门：PASS——D0、pose-shuffle、CLIP-only及wrong-RGB/generic/zero可在共享候选支持上直接归因。

普通hard mining、pose-aware sampling和CLIP negative本身不是新意；PCHM只按C类窄候选推进。PCOIR暂不选，因为
其foreign-part copy容易被归入pose-CutMix并制造partial-chimera标签噪声。若PCHM自然e120不过clean D0 mAP/R1
双门，立即关闭该对象，不调rank fusion或cache。

PCHM现已通过fresh cache与真实batch门：正确联合miner相对D0更换绝大多数negative edge并更换过半positive
edge，pose-shuffle与CLIP-only也都显著改变选择；梯度直接进入Stage-3/final backbone。这只证明机制active，尚不
证明涨点或创新成立。下一份有效证据只能来自自然e120相对clean D0的mAP/R1及GO后的matched controls。

### exp409 封板后的创新结论

PCHM自然e120=`57.0 mAP/68.6 R1`，相对D0形成“mAP下降、R1上升”。这否定了把联合pose×CLIP hard mining
作为主创新的条件资格：它能改变真实训练边并优化最难首位混淆，却没有改善AP所要求的整条正样本排序。下一对象
必须让pose与CLIP作用于多正样本分布、身份表示结构或全排序一致性，而不是继续选择单个hard positive/negative；
仍须保留D0与错误绑定反事实，且不能退化为loss/temperature/margin调参。

## 2026-07-21：exp410 PC²P——pose-complete visual identity classifier

PC²P不给单图加局部teacher loss，也不选pair；它把同PID多图的五槽CLIP支持先逐槽聚合、归一化，
再等槽合成冻结identity proxy，直接替换learned classifier。无Q/projection使监督不能被auxiliary
head吸收；原global feature同时被proxy CE、triplet与eval共用。

创新只能窄定位：CLIP-ReID已有frozen identity text logits，ProFD已有part prompt/memory，因此“固定
CLIP classifier”不新。可争的是pose逐槽跨图补全visual identity set并无adapter接管全分类几何。
问题/证据门PASS，机制门CONDITIONAL PASS；必须后续胜wrong-RGB、generic和random-code才能排除任意
source-key与普通CLIP prototype解释。

### exp410封板后的创新结论

PC²P唯一fresh e120=`45.0 mAP/56.4 R1`，相对clean D0大幅下降`12.6/11.3 point`。训练前合同已证明冻结proxy
真实替换classifier且梯度进入Stage-3/backbone，因此该结果否定的不是接线，而是“用冻结CLIP visual identity axes
接管ReID全部分类几何”这一机制对象。跨图pose补全没有消除CLIP与student空间错配，反而使分类目标长期约束在
不适配的外部坐标系中。

下一候选不得用projection、adapter、temperature或scale把PC²P软化后换名重试，也不得退回PICRD局部relation或
PCHM单边hard mining。更合理的对象是：保留learned classifier与student度量空间，只让pose×CLIP定义同PID多正
支持的集合结构、跨视图补全关系或listwise排序约束；这样外部语义只决定“哪些支持应互补/一致”，不规定student
最终坐标轴。新机制仍需wrong-RGB/pose-shuffle/CLIP-only强反事实与自然e120双门，至少满足问题和证据创新门。

## 2026-07-21：exp411 PCMPSR——pose-complete多正身份集合排序

PCMPSR保留student learned classifier与global坐标，只改变triplet所假设的训练对象：每个anchor面对16个等大小
leave-one-position-out身份支持集，而不是一个hard positive/negative。每个集合的三张支持图全部进入距离，五槽
pose coverage×同PID CLIP共识再离散选择owner并以multiplicity强调相应视图；所有距离仍在student空间计算。

- 问题门PASS：遮挡ReID的AP要求完整正支持相对所有负身份集合排序，单pair与固定proxy都不是同一问题；
- 机制门CONDITIONAL PASS：单独的listwise、set loss、pose owner或CLIP共识均不新，只保留三者加等支持排除的整体；
- 证据门PASS：zero-owner、generic、wrong-RGB、pose-only与D0共享支持，可区分集合收益和pose×CLIP归因。

该方向的关键科学边界是“外部证据只组织support multiplicity，不规定student轴”。若owner坍缩或controls不改变集合
距离，机械门直接NO-START；若自然e120双门FAIL，说明全身份集合对象仍不足，不得改owner公式或loss聚合救臂。

fresh cache PASS只确认PCMPSR所需的五槽CLIP support可被完整、有限且可追溯地提供，不增加问题/机制创新评分，也
不是pose×CLIP有效证据。创新状态仍为C类CONDITIONAL；下一份有效信息只能来自真实batch中correct-vs-controls的
owner活动性、isolated set-loss梯度以及自然e120相对clean D0的双门。

real-batch v1缺文件退出发生在科学路径前，不改变PCMPSR创新评分，也不能支持正负机制解释。补齐sealed D0 config
仅恢复default-off可测性；创新状态继续为C类CONDITIONAL，等待fresh v2真实合同与后续性能。

fresh v2将“机制未接通”从后续失败解释中排除：correct对三种强control的owner改变率均非零，五槽owner平均覆盖
2.3125张支持图，isolated set loss独立更新Stage-3/backbone。它只支持PCMPSR机制active与证据门可执行，尚不支持
性能或论文主张；C类CONDITIONAL资格下一步只由自然e120双门与GO后的matched controls决定。

correct自然e120=`58.8/70.1`，相对clean D0=`+1.2/+2.4`，首次使“保留student空间、改成pose-complete身份集合
排序”通过性能门。这把PCMPSR从纯条件想法推进为`PERFORMANCE GO / ATTRIBUTION PENDING`：问题对象与整体训练
机制已有正证据，但创新不能提前归因给pose×CLIP owner。zero-owner若同样上涨，贡献更接近普通all-identity set
ranking；wrong-RGB若不降，则CLIP语义组织不成立。只有correct严格胜两个matched control，C类pose+CLIP候选才
升级为科学GO。

最终zero-owner与wrong-RGB的e120 mAP/R1均高于correct，且wrong-RGB三臂最高。因此PCMPSR的可保留创新信息只剩
“等支持的全身份集合排序能改善遮挡ReID”，不能把pose×CLIP owner写成贡献。这个失败进一步表明：外部CLIP不应通过
PID绑定、prototype或support multiplicity决定身份几何；新机制必须把CLIP约束收缩为identity-free语义，并让pose
控制它作用于哪些身体token或监督机会。

## 2026-07-22：exp412 PSGC——pose语义梯度补全

PSGC把single-image support incomplete改写为“同PID内身体槽监督机会分配不完整”：pose给增强后五槽visibility，
CLIP只提供visible-vs-occluded文本轴上的身份无关标量；二维Pareto front决定同PID×槽固定预算由哪些视图承担。
forward、loss与测试descriptor不变，仅改变最终身体token的反向路由。

- 问题门PASS：对象从补feature改为补监督机会，直接对应单图部位缺失；
- 机制门CONDITIONAL PASS：Pareto/sample reweighting并非新原子，但同PID×槽预算、pose×identity-free CLIP front与
  forward-exact token router的整体可形成C类差分；
- 证据门PASS：zero-owner、pose-only、q-only、text-shuffle共享宿主与计算量，可分别检验pose、CLIP及正确文本绑定。

首轮只训练correct。它必须自然e120严格胜sealed zero-owner与clean D0的mAP/R1才保留性能资格；随后还须严格胜三条
matched control才能宣称pose+CLIP联合路由有效。失败即封板并转向下一结构对象，不做prompt或scale微调。

### exp412封板后的创新结论

PSGC自然e120=`56.9/69.7/82.5/86.1`，核心mAP/R1相对zero-owner低`2.0/0.6`，相对clean D0的mAP也低`0.7`。
合同已证明forward exact且28/28 Stage-3梯度tensor被路由改变，所以失败不是机制未接通，而是“把监督机会集中给
当前二维可靠视图”本身没有改善完整检索排序。它可能过度训练易见区域，并让被遮挡图失去学习恢复线索。

下一候选不得重调front、budget、文本或scale，也不能继续做梯度/样本权重变体。应重新定义结构对象：让pose确定
单图中哪些身体槽缺证据，让identity-free CLIP只确定缺失的语义类型，但由student自己的同PID多视图token提供
补全内容；训练时应保留被遮挡图作为接收者，而不是把它的梯度转走。必须用无跨图补全、pose错位与文本错位反事实
区分普通多视图一致性、pose定位和CLIP语义的作用。

## 2026-07-22：exp413 PSCCR——无丢弃互补support prefix

exp413不继续exp412的梯度路由，也不回到CAVT/PC-MSC/PSC-JEPA的feature completion。它把exp411已证有效的
zero-owner集合对象拆成嵌套训练条件：先完成leave-one-position-out，再让pose visibility与identity-free CLIP
遮挡margin只在剩余三support内形成严格序数；`min(rank_v,rank_q)`定义五槽可靠度，贪心最大覆盖增益给原三support
一个确定性顺序；三图最终全部使用。student
分别优化长度1/2/3 prefix相对全部身份集合的排序，第三项与zero-owner exact。

- 问题门PASS：从“完整三图平均后排序”推进到“部分但互补支持到达时也能排序”，且不牺牲难图query梯度；
- 机制门CONDITIONAL PASS：ordinal、greedy coverage、prefix curriculum和listwise都不新，只保留四者与宿主exact
  的整体窄差分；
- 证据门PASS：zero-owner、pose-only、q-only、text-shuffle共享support、loss、recipe和计算规模。

当前只能标记`C-CLASS CONDITIONAL`。在线检索超时使本轮不能做绝对新颖性声明；correct若不过zero-owner的e120
mAP/R1双门，直接证明新增prefix对象不值得继续，禁止调prefix权重、序数或coverage公式救臂。
