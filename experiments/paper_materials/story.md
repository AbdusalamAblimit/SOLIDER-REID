# 论文故事线（持续更新）

> **⚠️ Phase 1 内容保留在下方（PCFC/GiLt）。Phase 2 更新如下。**

## Phase 2 Story Update (2026-03-13)

### 暂定标题
Pose Spatial Gate and Skeleton Complement for Occluded Person Re-Identification

### 当前最可靠的核心发现
1. **PSG (Pose Spatial Gate)**
   在 Swin Stage 3 blocks 内部注入 pose heatmap，通过轻量门控 `x * (1 + gate)` 调制特征。
   4090 三 seed 均值：`56.50% -> 57.83%`，仅 `+102K` 参数。

2. **`0.5x global loss` 是真实有效的训练 recipe**
   `exp007a` 三 seed 均值 `59.37%`，相对 PSG 稳定 `+1.53% mAP`。
   这不是单 seed 偶然值，而是 paired diffs `(1.3, 1.6, 1.7)` 的稳定改善。

3. **PDS+StopGrad 不再是主故事，更多是“揭示机制”的中间实验**
   `exp023-g = 59.20%` 与 `exp007a = 59.37%` 无显著差异。
   因此 PDS+StopGrad 在 global-only 指标上的增益，基本可由它隐式带来的 `0.5x global loss` 解释。

4. **PSG + KPP/GCN branch 的贡献应写成 fusion 增益**
   `exp030a-global = 59.33%` 与 `exp007a = 59.37%` 几乎相同；
   但 `exp030a-equal_concat = 60.73%`，对自身 global 稳定 `+1.40% mAP`。
   说明 branch 的价值主要体现在检索时提供互补信息，而不是抬高 global 主干。

5. **KPP 是 branch 的强基线，GCN 是 refinement**
   `exp032` 说明 keypoint pooling 本身已经很强；
   `exp030a` multi-seed 又说明训练好的 branch 确实还能继续提高 fusion。
   因此更准确的 framing 是：**sparse keypoint pooling 提供主体信息，GCN 负责关系建模与 branch refinement。**

### 当前主结果表（Occluded-Duke, Swin-Tiny, 4090）

| 方法 | 测试模式 | Mean±Std (mAP) | Mean±Std (R1) | 备注 |
|------|----------|----------------|---------------|------|
| Baseline | global | 56.50±0.53% | 66.33±0.67% | 3-seed |
| PSG | global | 57.83±0.50% | 67.13±0.84% | 3-seed |
| PSG + 0.5x loss | global | 59.37±0.32% | 69.43±0.12% | 3-seed |
| PDS+StopGrad | global | 59.20±0.50% | 68.63±0.47% | 3-seed |
| PSG + GCN | global | 59.33±0.40% | 68.87±1.00% | 3-seed |
| PSG + GCN | concat_scaled | 60.20±0.44% | 73.13±0.29% | 3-seed |
| **PSG + GCN** | **equal_concat** | **60.73±0.47%** | **72.57±0.58%** | **当前最强且已确认的无后处理模式** |

### 关键统计结论

| 对比 | Mean Δ | Paired Diffs | p-value | 解读 |
|------|--------|--------------|---------|------|
| PSG vs Baseline | +1.33% | (1.6, 2.0, 0.4) | 0.1091 | 3 个 seed 全正，样本数仍小 |
| exp007a vs PSG | +1.53% | (1.3, 1.6, 1.7) | 0.0061 | ✅ `0.5x loss` 是稳定增益 |
| exp007a vs exp023-g | +0.17% | (-0.1, 0.3, 0.3) | 0.3377 | 两者无显著差异 |
| exp030a-eq vs exp030a-global | +1.40% | (1.3, 1.1, 1.8) | 0.0214 | ✅ fusion 增益成立 |
| exp030a-eq vs exp030a-cs | +0.53% | (0.6, 0.5, 0.5) | 0.0039 | ✅ `equal_concat` 优于 `concat_scaled` |

### 修正后的证据链
1. Baseline `56.50%` → PSG `57.83%`
   说明 backbone 内部 pose gate 稳定有效。

2. PSG `57.83%` → PSG + `0.5x loss` `59.37%`
   说明更弱的 global 梯度是一个真实有效的训练配方。

3. PSG + `0.5x loss` `59.37%` ≈ PDS+StopGrad `59.20%`
   说明 PDS global 增益的主因不是双流结构本身。

4. PSG + GCN(global) `59.33%` ≈ PSG + `0.5x loss` `59.37%`
   说明 branch 训练并不抬高 global 主干。

5. PSG + GCN(equal_concat) `60.73%` > PSG + GCN(global) `59.33%`
   说明训练好的 branch 在测试时提供了稳定互补信息。

### 当前可 claim 的贡献
1. **PSG**: 极简的 backbone 内 pose 注入，稳定提升 Occluded-Duke，并在 Market / Swin-Small 上可复现。
2. **`0.5x global loss` 机制发现**: PDS+StopGrad 的 global-only 收益可以被更简单的训练 recipe 复现。
3. **Skeleton branch as complement**: 基于 sparse keypoint pooling 的 skeleton branch 在 `equal_concat` 下带来稳定 fusion 增益；GCN 负责 refinement，而不是单独承担全部提升。

### 当前不应再主张的结论
1. **PDS + Gradient Isolation** 不应继续作为主创新点。
   它更像一个帮助暴露 loss-weighting 机制的中间 scaffold。

2. **“0.5x loss 只是训练方差”** 已被推翻。
   `exp007a` multi-seed 已直接否定这一说法。

3. **“GCN 是否有效仍完全未知”** 也不成立。
   更准确的说法是：GCN 的收益主要发生在 fusion，而不是 global；其增益规模需要和 KPP 基线一起解释。

### 2026-03-13 文献/代码复盘后的 story 修正
1. **现阶段不应把主线继续写成“再优化 branch 内部权重”**
   `exp035b / exp036 / exp037` 这一轮都在 branch 内部调权重或调 loss，且收益弱；同时 KPR/BPBreID/QPM/FRT 也表明，这类工作很难构成新的主叙事。

2. **更合理的新叙事是：branch 的结构信息应该服务于检索时的共同可见推理**
   我们已有证据是：
   - `exp030a-global` 不涨
   - `exp030a-eq` 稳定涨
   这说明 branch 真正提供的是“可补充的局部支撑”，而不是更强的单向量 global embedding。

3. **因此下一阶段的候选主线应转向 retrieval-time common-support reasoning**
   候选表达可以是：
   - PSG 提升 backbone 全局表征
   - Skeleton branch 提供语义对齐的局部关键点表征
   - 检索阶段基于 query-gallery 共同可见关键点进行距离推理

4. **这一段 story 目前还是候选，不是已验证结论**
   需要后续实验先回答：
   - 共同可见关键点距离是否真的优于 `equal_concat`
   - 它是否能解释 branch 的 fusion 增益来源

### 2026-03-13 exp039 诊断更新
- `cvk_only` = `59.3 / 72.9`
- `cvk_hybrid` = `61.9 / 73.2`
- `exp035a equal_concat` = `61.1 / 73.8`

当前可得出的更细判断是：
1. 共同可见关键点支撑 **确实存在**，否则 `cvk_only` 不会有接近 baseline 的 R1。
2. 但它 **不适合单独替代** 当前主距离，因为 `cvk_only` mAP 明显下降。
3. 把它作为 global 的 pair-specific 补充后，mAP 出现 `+0.8%` 正信号。

这使得新的候选 story 变得更具体：
- PSG 负责 backbone 级 pose prior
- Skeleton branch 提供结构化局部证据
- 检索阶段通过共同可见关键点支撑提升整体排序质量

### 2026-03-13 exp040 原始基线复核更新
- `040a exp030a-eq recheck` = `61.1 / 73.7`
- `040b exp030a-cvk_hybrid` = `61.9 / 73.2`

相对 `040a`，`040b` 给出：
- mAP `+0.8%`
- R1 `-0.5%`

这一步的重要性在于：
1. 它把 `exp039` 的结果从“bundled checkpoint 上的单次信号”推进成了“原始主基线 checkpoint 上可复核的信号”。
2. 两次结果几乎一致，说明新的 story 不是偶然波动：
   - `exp039b` = `61.9 / 73.2`
   - `exp040b` = `61.9 / 73.2`
3. 因而当前更有把握的表述是：
   **Skeleton branch 的价值不只是在 embedding-level concat，更可能在 retrieval-time 提供 pair-specific common-support correction。**

但这条 story 仍需补两层证据：
- `global : cvk` 权重敏感性
- 多 checkpoint / 多 seed 复核

### 2026-03-13 exp041 权重敏感性更新
- `2:1` = `61.6 / 72.6`
- `1:1` = `61.9 / 73.2`
- `1:2` = `61.6 / 73.6`

这一步把 story 又往前推了一点：
1. `1:1` 不是随手设的偶然比例，因为两侧偏移都会把 mAP 从 `61.9` 拉回 `61.6`。
2. 这说明共同可见关键点 reasoning 的作用方式不是“global 为主，CVK 轻微修正”或“CVK 主导，global 陪衬”，而是 **两种证据的平衡补充**。
3. 偏向 CVK 会把收益更多转向 R1，偏向 global 则两项都掉，这也符合“CVK 在困难 pair 上做判别校正”的直觉。

因此当前更稳的叙事可以写成：
- global feature 提供主体身份空间
- CVK reasoning 提供 pair-specific common-support correction
- 两者需要保持平衡，而不是由一侧完全主导

### 2026-03-13 exp042 pair-case 分析更新
`exp042` 给 story 补上了“为什么有效”的证据层：

- `positive_delta_ap = 1129`
- `negative_delta_ap = 822`
- `top1_fixed = 47`
- `top1_degraded = 58`

这几组数字合起来说明：
1. `cvk_hybrid` 的收益不是只靠少数 query 暴涨，而是来自 **更多 query 的 AP 小幅转正**。
2. 但它修复的 top-1 数量少于新引入的 top-1 退化，因此整体呈现：
   - `mAP +0.8`
   - `R1 -0.5`
3. 这与前面的假设非常一致：
   **CVK 不是 top-1 booster，而是 deeper-rank common-support correction。**

因此论文表述应继续避免写成：
- “显著提升 top-1”

更适合写成：
- “在不完全观测下，通过共同可见关键点支撑修正整体排序”

### 2026-03-13 exp043 qualitative 素材更新
已经生成并同步两张候选图：
- `experiments/paper_materials/figures/qualitative/cvk_top_improved.png`
- `experiments/paper_materials/figures/qualitative/cvk_top_degraded.png`

这两张图的价值在于：
1. 它们不是只挑“最好看”的成功样例，而是同时保留改进与退化。
2. 可以直接和 `exp042` 的统计结论配套使用：
   - 为什么 mAP 上升
   - 为什么 R1 小幅下降
3. 这样 qualitative 部分就能和当前 story 保持一致：
   **CVK 是 pair-specific ranking correction，而不是无代价增强。**

### 跨数据集 / Backbone 验证 (4090)

| 数据集 | Backbone | Baseline mAP | PSG mAP | Δ |
|--------|----------|-------------|---------|-----|
| OccDuke | Swin-T (3-seed mean) | 56.50% | 57.83% | **+1.33%** |
| OccDuke | Swin-S (lr4) | 65.8% | 67.8% | **+2.0%** |
| Market | Swin-T | 91.6% | 92.4% | **+0.8%** |
| Market | Swin-S (lr4) | 93.3% | 93.9% | **+0.6%** |

→ PSG 在所有组合上均有效，且在遮挡数据集上的增益更大。

---

> **以下为 Phase 1 原始 Story（保留参考）**

## 暂定标题（Phase 1，待更新）
Pose-Calibrated Part Learning with Visibility-Weighted Matching for Occluded Person Re-Identification

## Motivation（为什么做这个）
- **现有问题**: 遮挡行人重识别中，被遮挡的身体部位产生噪声特征，严重干扰检索。现有方法要么忽视遮挡（global-only），要么简单拼接 part features（稀释全局信号）
- **现有方法的不足**:
  1. Part features 通常通过简单的特征拼接加入匹配，但 part 维度远大于 global 维度时会稀释全局信号（我们 exp003 验证: -4.9% mAP）
  2. 姿态信息的利用位置不当：在 backbone 输入层（KPE: ±0%）或中间层（PVFM: -0.7%）注入 pose 信号反而有害，只有在最终特征层（GAP）利用才有效
  3. Test-time 的邻域增强（NFC/re-ranking）与 part 信息的结合未被充分探索
- **我们的洞察**:
  1. Visibility 信息应在两个阶段发挥作用：训练时指导 part feature learning（通过 vis-weighted GAP + part triplet loss），测试时指导距离度量（per-part visibility-weighted distance）
  2. NFC（邻域特征中心化）在 global 和 part feature 空间都有效，说明邻域增强是一个通用原则
  3. Part triplet loss (GiLt) 能同时改善 global 和 part 特征质量

## 核心贡献（预计 3 点）
1. 提出 PCFC (Pose-Calibrated Feature Calibration)：利用离线 ViTPose 预测的关键点可见性，在最终特征层进行可见性感知的 GAP + 部件级 ID/Triplet 联合训练
2. 提出 GiLt-style 部件三元组损失：对每个可见部件独立计算 triplet loss，显著提升部件特征的判别力（part-only mAP +1.0%）
3. 提出多层级测试时增强框架：部件级可见性加权距离 + 全局/部件双空间 NFC，与 re-ranking 互补，累计提升 +18.7% mAP

## 方法概述
### 训练阶段
- **Backbone**: SOLIDER-pretrained Swin-Tiny
- **PCFC 模块**:
  - 输入：关键点坐标 + 可见性预测（ViTPose 离线提取）
  - 功能1: Visibility-weighted GAP — 对最终特征图 (768×12×4) 使用可见性加权的 Gaussian attention pooling
  - 功能2: 5-part feature extraction — 基于关键点分组的部件特征提取
  - Alpha 参数自适应学习 attention 强度
- **损失函数**: Global ID + Global Triplet + Part ID + **Part Triplet (GiLt)**
  - Part Triplet 对每个可见部件独立计算 hardest positive/negative
  - 可见性阈值过滤不可靠的部件

### 测试阶段（三种互补方案）
1. **无后处理**: 仅用 global feature — mAP 58.0%
2. **NFC 增强**: Global NFC(k=3) + Part NFC(k=3) + vis-weighted part distance fusion — mAP 64.7%
3. **Re-ranking 增强**: k-reciprocal re-ranking + part distance as local_distmat — mAP 75.3%

## 实验证据链
- [x] **exp001**: Baseline = 56.6% mAP, 66.5% R1
- [x] **exp003**: Part concat 有害 (-4.9%) → 证明需要更好的 part 利用方式
- [x] **exp005**: PCFC = 57.5% mAP (+0.9%) → Vis-weighted GAP 有效
- [x] **exp005 消融**: Part loss 和 vis attention 都有独立贡献，且互补
- [x] **exp007**: PVFM 有害 → 证明 pose 信息只应在最后层注入
- [x] **exp010**: KPE 无效 → 证明 pose 信息在输入层无用
- [x] **exp011**: Vis-weighted part distance 有效 → 证明 per-part 距离优于 concat
- [x] **exp012**: GiLt = 58.0% mAP (+0.5% vs PCFC) → Part triplet 改善 part quality
- [x] **exp013**: ptri_w=2.0 过重 → 验证 inverted-U 曲线
- [x] **exp014**: ptri_w=0.5 = 57.3% mAP → 完善消融曲线，确认 w=1.0 最优

## 消融实验表格（论文核心表格）

### Table: 训练组件消融
| PCFC | Part ID | Part Tri | mAP | R1 | Δ mAP |
|------|---------|----------|-----|-----|-------|
| | | | 56.6 | 66.5 | - |
| ✓ | | | 57.3 | 66.9 | +0.7 |
| | ✓ | | 57.4* | 67.2* | +0.8 |
| ✓ | ✓ | | 57.5 | 67.4 | +0.9 |
| ✓ | ✓ | ✓(0.5) | 57.3 | 67.1 | +0.7 |
| ✓ | ✓ | ✓(1.0) | **58.0** | **68.0** | **+1.4** |
| ✓ | ✓ | ✓(2.0) | 56.4 | 65.8 | -0.2 |

*注: 纯 Part ID 需要不同 config (PosePart), 数据可能不完全可比

### Table: 测试时增强方法
| Method | mAP | R1 | Δ mAP |
|--------|-----|-----|-------|
| Global only | 58.0 | 68.0 | - |
| + Part Distance (vis-weighted) | 58.3 | 69.5 | +0.3 |
| + NFC query (k=2) | 61.5 | 69.2 | +3.5 |
| + NFC both (k=3) | 64.0 | 67.6 | +6.0 |
| + NFC both + Part Distance | 64.2 | 69.3 | +6.2 |
| + NFC both + Part NFC + Part Dist | **64.7** | 69.4 | **+6.7** |
| + Re-ranking | 75.0 | 73.7 | +17.0 |
| + Re-ranking + Part Distance | **75.3** | **74.4** | **+17.3** |

### Table: NFC vs Re-ranking 交互
| NFC | RR | Part Dist | mAP | R1 |
|-----|-----|-----------|-----|-----|
| | | | 58.0 | 68.0 |
| ✓ | | | 64.0 | 67.6 |
| | ✓ | | 75.0 | 73.7 |
| ✓ | ✓ | | 72.3 | 69.5 |
| | ✓ | ✓ | **75.3** | **74.4** |

结论: NFC 和 RR 不兼容（都做邻域增强），Part Distance 与 RR 兼容

## 与 SOTA 对比 (数据来源: KPR ECCV'24 Table 3)

**核心 narrative**: 我们使用最轻量的 backbone (Swin-Tiny, 28M params)，仅增加 2.7M 参数的 PCFC 模块，
就超越了使用 ViT-B (86M) 的 FED/SSGR/LDS 等方法，证明姿态信息能有效弥补 backbone 容量的不足。

- 无后处理: mAP 58.0% 超越 FED(56.4), SSGR(57.2), LDS(55.7), PAT(53.6) 等 ViT-B 方法
- 与 Swin-Base (88M, 3x 参数) 的 SOLIDER(61.9) 差距 -3.9% mAP，我们用 pose 信息缩小了 1.4%
- 含后处理时 mAP 64.7% 超越 BPBreID(62.5) 和 PFD(61.8)
- Re-ranking 75.3% 与 KPR(75.1) 持平（但 R1 低 10%）

详见 `tables/main_results.md`

### Table: N_PARTS 消融
| N_PARTS | 覆盖部位 | mAP | R1 | Δ mAP |
|---------|---------|-----|-----|-------|
| 3 | head, torso, arms | 55.0 | 66.6 | -3.0 |
| 4 | + thighs | 56.5 | 67.1 | -1.5 |
| **5** | **+ calves (全部)** | **58.0** | **68.0** | **0** |

结论: 性能随 part 数量单调递增；下半身信息至关重要

### Table: Backbone 规模影响 (4090 实验)
| Backbone | 参数量 | Baseline mAP | +PCFC mAP | Δ |
|----------|--------|-------------|-----------|-----|
| Swin-Tiny | 28M | 56.6 | 58.0 | +1.4 |
| Swin-Small | 50M | 65.7 | 62.8* | -2.9 |

*LR=0.0004 (修复 alpha collapse 后的结果)

结论: PCFC 对轻量 backbone 有效，对大 backbone 反而有害。
可能原因：大 backbone 自身已有足够的特征表达能力，额外的 pose 约束反而限制了学习

## 待补充的实验
- [x] exp014: ptri_w=0.5 消融 — 完成, w=0.5 = 57.3% mAP
- [x] exp030: N_PARTS 消融 — 完成, 3/4/5 parts
- [x] t-SNE 可视化 — 完成, `figures/tsne/tsne_comparison.png`
- [x] 检索结果可视化 — 完成, `figures/qualitative/retrieval_comparison.png`
- [x] 计算效率分析 — 完成, `tables/efficiency.md` (+9.6% params, +0% FLOPs)
- [x] SOTA 对比表 — 完成, `tables/main_results.md` (20+ 方法)
- [ ] 不同遮挡程度下的性能分析
- [ ] Attention map 可视化 (vis-weighted GAP 的 attention 热图)
