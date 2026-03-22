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

### 2026-03-16 周度评估：当前 story 仍不够支撑 B 类主线
（原评估保留，以下新增 exp076-083 实验反馈。）

### 2026-03-16 晚间重大更新

#### 发现 1: PAA 是 multi-person specialist（不是通用 enhancer）
exp066 subset analysis:
- 多人图 (n>=2, 49% of query): PAA **+1.69% mAP / +2.02% R1**
- 单人图 (n=1, 51% of query): PAA **+0.47% mAP / -1.61% R1**

**论文意义**: PAA 的 story 应写成 “pose adapter specifically addresses multi-person occlusion”，而非 “general feature enhancement”。

#### 发现 2: ROA ≈ PAA+ROA（PAA 的 mAP 被 ROA 完全覆盖）
| 方法 | mAP (跨硬件均值) | R1 (跨硬件均值) |
|------|-----------------|----------------|
| exp030a 3-seed | 60.73% | 72.57% |
| PAA only (exp066) | 61.6% | 74.2% |
| ROA only (exp079) | ~61.9% | ~73.2% |
| PAA+ROA (exp067) | ~61.9% | ~73.9% |

ROA 的 mAP 增益 (+1.27%) 完全包含了 PAA 的增益 (+0.87%)。PAA 独特贡献仅 R1 ~+0.7%。

**论文意义**:
- ROA 是当前最有效的单一改进（数据增强级）
- PAA 不应作为 mAP 主贡献来 claim
- 但 PAA + ROA 的 R1 (73.9%) > ROA alone (73.2%)，说明 PAA 在 R1 上有独特贡献

#### 发现 3: TDPC 方向全面证伪
exp076 TDPC (-0.3%), exp077 ST-PAA (-0.6%), exp078 APG (-1.1%) — target-aware PAA 全失败。
**原因**: 74% 训练数据是单人图，target-aware 机制在这些图上只增加噪声。

#### 发现 4: Transformer Decoder 在当前数据量不可行
exp081 PQTD (-4.7%) — 3-layer decoder 120ep 严重不够收敛。

#### 当前进行中
exp083 PGFI (Pose-Guided Feature Inpainting) — 在 feature map 空间恢复遮挡区域特征。
这是 “recover” 范式，不同于 “suppress”(PSG) / “inject”(PAA) / “select”(pruning)。

#### 修正后的 story 候选方向
1. **PSG + 0.5x loss + GCN**: 基础三件套，已确认
2. **ROA**: 最有效的单一改进，但本质是数据增强
3. **PAA**: multi-person specialist，R1 贡献
4. **PGFI 或后续创新**: 需要找到真正能支撑论文主贡献的机制
5. **问题层面**: 如果 PGFI 也失败，应考虑把 story 转向 “pose-guided multi-granularity representation”（PSG+GCN+equal_concat 的整体范式叙事），而非继续追求单一新模块

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

这里需要补一个边界：
- 上述 `R1 -0.5` 的形状来自 `040a/040b` 这个 checkpoint 对照
- 它是当前最完整的机制证据，但不应被写成所有 checkpoint 都必须出现的固定代价

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

### 2026-03-13 exp045 第二 checkpoint 复核更新
- `045a` = `60.2 / 72.7`
- `045b` = `61.1 / 73.2`

相对 `045a`，`045b` 给出：
- mAP `+0.9%`
- R1 `+0.5%`

这一步对 story 的推进非常关键：
1. `cvk_hybrid` 的正 mAP 信号已经不只停留在主 checkpoint，而是又在重建的 `seed42` checkpoint 上复核成功。
2. 它的增幅量级与 `exp040` 非常接近：
   - `exp040`: `+0.8% mAP`
   - `exp045`: `+0.9% mAP`
3. 但这次 R1 没有回落，说明更准确的总表述应该改成：
   - **稳定项**: mAP 跨 checkpoint 转正
   - **可变项**: R1 的具体变化方向会随 checkpoint 而变

因此当前更稳的论文叙事应写成：
- PSG 提供 backbone-level pose prior
- skeleton branch 提供结构化局部证据
- CVK reasoning 在检索阶段利用共同可见支撑修正整体排序
- 其最稳定的外显收益是 **mAP 改善**，而不是某种固定的 R1 trade-off 形状

### 2026-03-13 exp046 第三个 checkpoint 资产补齐
- `exp046` 最终结果 = `60.1 / 72.9`

这一步本身不是新方法结果，但它对 story 有两个直接作用：
1. 本地已经补齐第三个 `exp030a` 可复用 checkpoint，后续不再受“缺 seed2024 权重”阻塞。
2. 它说明当前最该推进的已经不是继续做资产恢复，而是把 common-support 机制真正推进到训练端验证。

因此从 story 角度，`exp046` 的意义应写成：
- **证据资产补齐**
- 不是新的主贡献实验
- 但为 `exp047` 或第三 checkpoint 复核提供了后续支撑

### 2026-03-13 下一跳候选：CSGT（训练端化 common-support）
基于当前两类事实：
1. `cvk_hybrid` 的正 mAP 信号已在两个 checkpoint 上复核
2. KPR / BPBreID / QPM / FRT 都说明 common-visible reasoning 的主价值在 pair-specific comparability

当前最合理的训练端候选，不是再做一个融合模块，而是：
- 用 skeleton branch 的 `kp_weights` 构造 batch 内 common-support overlap
- 在 global triplet 上加一条 support-aware hard mining 约束

这一步现在只能写成 **候选机制**，还不是结果。
但如果它成立，story 就能从：
- “检索时补一个 CVK correction”

推进成：
- “训练期先学会 pair comparability，检索期再用 CVK 做剩余修正”

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


## 2026-03-19: exp107 DACHM 的负结果给出的 story 约束

- `exp107` 否定了一个很诱人的简化故事：
  “把每张图拆成多个 person embedding，再做 duplicate-aware 的反事实 rerank，就能解决 ambiguity。”
- 实际结果不是中性，而是稳定负面；说明：
  - ambiguity 不是简单的 person-level candidate selection 问题
  - coarse pooled person embedding 丢失了真正有用的遮挡/可见性结构
- 这反过来强化了一个更清晰的论文约束：
  **如果我们要讲 ambiguity / confuser 这条主线，机制必须发生在 per-keypoint / common-support 粒度，而不是 person-level pooled feature 粒度。**


## 2026-03-19: exp108 DACCM 的负结果进一步收紧了 story

- `exp108` 继续否定了另一个更强的简化故事：
  “只要把 confuser reasoning 下沉到 per-keypoint / common-support，再做 duplicate-aware penalty，就能在 retrieval-time 把 ambiguity 解开。”
- 实际结果仍然整体负面：
  - `base_cvk_hybrid = 61.88 / 73.26`
  - `daccm_penalty = 61.39 / 72.94`
- 这给 story 一个更明确的边界：
  1. `cvk_hybrid` 的正增益不是因为“显式打压 confuser”这个公式
  2. 有效信息更像是 target-target common-support 的正向匹配，而不是额外的反事实 penalty
  3. 因此下一条可投稿主线不应继续写成 retrieval-time rerank，而应转向：
     - 训练端结构学习，或
     - 一个比 `ambiguity penalty` 更本质的问题定义


## 2026-03-19: exp109 Oracle Support Bank 把 story 从“比较”推向“补全”

- `exp109` 给出了一条非常关键的新证据：
  如果我们给每张图一个 GT same-ID 的 per-keypoint support bank，`cvk_hybrid` 会从 `61.88 / 73.26` 直接跃升到 `66.15 / 77.87`；
  若连权重一起恢复，则到 `70.40 / 81.36`。
- 这不是可报告结果，但它强烈说明：
  **主要缺口不是缺一个更聪明的比较公式，而是缺一个更完整的 latent support representation。**
- 尤其在极低可见 query 上，上界提升极大：
  - `target_vis<=8`: `29.42 / 26.92` → `78.26 / 84.62`
  - `target_vis<=5`: `16.85 / 14.29` → `78.43 / 85.71`
- 因而更有潜力的论文主叙事开始成形：
  1. PSG/PAA 负责 suppress noisy context / inject pose prior
  2. GCN/CVK 暴露出 common-support reasoning 的真实性
  3. 但真正的缺口在于：单图只包含 partial support
  4. 下一条主创新应是：
     **用 same-ID multi-view support 作为 teacher，把 single-image representation 蒸馏成更接近 support-complete 的关键点表征**


## 2026-03-19: exp110 SCKD 让 story 从“上界存在”走到“训练可行”

- `exp110` 给出的不是大幅涨点，而是一条更重要的可行性证据：
  在不改 backbone、不加 decoder、不做 test-time trick 的前提下，
  仅用一个轻量 `per-identity / per-keypoint prototype bank`，就能把 `exp030a-eq` 从 `61.1 / 72.9` 推到 `61.2 / 73.7`（单 seed）。
- 这个结果的价值不在于绝对幅度，而在于它完成了一个关键跨越：
  **support-complete 不再只是 oracle 上界，而是已经能作为真实训练信号转正。**
- 因而 story 可以更自然地写成：
  1. retrieval-time penalty 线失败，说明问题不在“再设计一个更聪明的比较公式”
  2. oracle support bank 证明缺口来自 incomplete support
  3. SCKD 最小原型进一步证明：
     训练端确实可以把 same-ID support 转成正向监督
  4. 所以下一步真正需要攻克的，不是“是否做 distillation”，而是：
     **如何让 support teacher 更可靠、更接近真正的 multi-view support-complete representation**


## 2026-03-19: exp112/113 让 story 从“teacher reliability”推进到“teacher non-stationarity”

- `exp112` 说明提高写入纯度是有帮助的，但帮助还不够大：
  - 中期曾明显领先
  - 到 `ep80` 更接近弱正向 / 近乎等价
- `exp113` 则把更深一层的问题显式暴露出来：
  1. 蒸馏覆盖率没有明显扩大
  2. teacher 置信度没有崩
  3. 但随着 prototype count 持续增长，student 对 teacher 的平均余弦持续下降

- 这给 story 一个很重要的新收束：
  **当前最值得讲的，不只是 reliable support，更是 stable / non-hardening support teacher。**

- 换句话说，主创新现在更可能写成：
  1. pose 定义 keypoint-level support units
  2. same-ID bank 提供 support-complete teacher
  3. 但 naive online bank 会不断变硬，形成 non-stationary distillation target
  4. 因而需要一个更可靠、更稳定的 support-complete learning 机制


## 2026-03-20: 复核后把 story 收回到 pairwise teacher，而不是模块组合

- `exp117/118` 已确认为偏题旁路线，不能进入主 story。
- 重新看目前最可信的事实链，真正没有被做完整的不是 generic local matching，而是：
  **如何把 `cvk_hybrid` 已验证过的 pairwise common-support 几何，迁到训练端并写进 global embedding。**

- 这使得当前的候选主叙事进一步变成：
  1. 单图遮挡带来的不是简单噪声，而是 **pair comparability mismatch**
  2. skeleton/keypoint branch 已经学到一份更贴近遮挡比较规则的 pairwise 几何（`cvk_hybrid` 证明）
  3. prototype bank 失败，说明问题不该被压成 `per-ID average support`
  4. 因而更合理的主方法应当是：
     **把 common-support pairwise 几何作为 privileged relational teacher，蒸馏到 global embedding**

- 这条线若成立，会比单纯的 `SCKD` 更适合作为论文主创新，因为它同时解释：
  - 为什么 `cvk_hybrid` 有效
  - 为什么 `exp047` 的 overlap mining 不够
  - 为什么 `exp051` 的 part-triplet 对齐不够
  - 为什么 `exp109-116` 的 prototype 压缩达不到 oracle headroom

### exp119 正式评估后的 story 收紧

- `exp119` 的正式结果是：
  - `equal_concat = 61.1 / 73.2`（对 `exp030a-eq seed1234` 为 `+0.0 / +0.3`）
  - `global = 60.4 / 70.3`（对 `exp030a-g seed1234` 为 `+0.6 / +0.4`）
  - `cvk_hybrid = 62.0 / 73.2`（对 `exp040b` 为 `+0.1 / +0.0`）

- 这给 story 一个新的正信号：
  1. `pairwise relational teacher` 这件事本身是成立的
  2. 且它最先改善的是 `global`，这非常符合“把遮挡下的比较规则蒸进 backbone”的机制预期

- 同时也把瓶颈写得更清楚：
  1. 第一版 `CSRD` 的 teacher 仍来自**单图** `kp_feats`
  2. 所以它虽然能传递 `pair comparability`，但 teacher 自身还受 `support incomplete` 限制
  3. 换句话说，`exp109` 和 `exp119` 并不冲突，而是正好可以接起来：
     **需要一个 support-complete relational teacher，而不是 prototype pointwise teacher**

- 因而当前最合理的主叙事升级为：
  1. 单图遮挡导致 `support incomplete`
  2. 这进一步表现为 `pair comparability mismatch`
  3. skeleton/keypoint branch 可以提供 relational teacher
  4. 但 teacher 还必须被 support-complete 化，才能把 `exp109` 的 headroom 真正转成训练端收益

## 2026-03-20: exp120 把 story 再收紧到 selective supervision

- `exp120` 做了一件很重要但容易误判的事：
  它没有把指标推高，却把当前瓶颈说得更清楚了。

- 到 `ep90`：
  - `exp120 = 59.9 / 73.2`
  - `exp119 = 60.1 / 73.7`

- 但与此同时，`exp120` 的机制统计清楚表明：
  1. support-complete teacher 真实在工作
  2. low-vis keypoint 的替换覆盖是稳定的
  3. teacher 几何也确实更强

- 所以这轮实验并不是在说：
  “support-complete relational teacher 不对”

- 它真正推动 story 变成：
  1. `support-complete teacher` 是必要的，但**不是充分的**
  2. oracle headroom 主要属于低可见 / support-incomplete 样本
  3. 如果对所有样本等权施加 relational distillation，clean 样本会稀释掉真正有价值的监督

- 因而 story 的下一层应写成：
  **pose-guided support-complete relational distillation 必须是 selective 的。**

- 这让当前最合理的下一跳不再是“更强 teacher”，而是：
  **按 sample-level support gap 分配 distillation 强度。**

## 2026-03-20: exp122 继续把 story 从 sample-level 收紧到 pair-level

- `exp122` 很重要，因为它否定得很具体：
  不是 `support-complete teacher` 不对，
  而是 **sample-level `replace_ratio` 太粗**。

- 到 `ep40`：
  - `exp122 = 55.4 / 68.2`
  - `exp119 = 55.9 / 68.7`
  - `exp120 = 55.5 / 67.8`

- 同时它的机制统计又是成立的：
  1. selective supervision 的确发生了
  2. 参与 `CSRD` 的 anchor 比例明显下降到 `~0.56`
  3. 说明这不是接线错误，而是路由粒度不对

- 因而 story 现在要再收紧一层：
  1. `support incomplete` 的影响不是均匀落在整张图/整个人样本上
  2. 它真正改变的是 **某些 pair 的 comparability**
  3. 所以 `support-complete relational distillation` 不能只做 sample-level selective
  4. 必须进一步变成 **pair-level teacher-change focusing**

- 这一步对论文叙事反而是好事：
  因为它把方法从“谁更难就多蒸馏一点”这种普通加权，
  收紧成了“只蒸馏那些被 support completion 真正改变过的关系”。

## 2026-03-20: exp121/123 把 story 进一步分成“supporting mechanism”与“主突破口”

- `exp121` 的最终结果是：
  - `ep120 = 60.6 / 74.0`
  - 相对 `exp119 ep120 = 60.4 / 73.4` 为 `+0.2 / +0.6`

- 这说明：
  1. `stable teacher` 不是伪命题
  2. support-complete relational teacher 的确受 teacher stability 影响
  3. 但这个量级更像 supporting mechanism，而不是足以单独支撑整篇论文的方法核心

- 与此同时，`exp123` 到 `ep60` 的形态变得更关键：
  - `exp123 ep60 = 57.8 / 70.9`
  - `exp119 ep60 = 57.7 / 70.5`
  - `exp120 ep60 = 57.5 / 69.7`

- 这给 story 一个新的正信号：
  1. pair-level `teacher-change focusing` 方向本身是成立的
  2. 也就是说，`exp122` 否定的只是 sample-level routing 太粗，不是否定 selective relational distillation

- 但 `exp123` 也同时暴露出主突破口还没被打透：
  1. `pair_delta` 长期只有 `0.002~0.003`
  2. `pair_focus` 长期只有 `1.06~1.08`
  3. 正向收益直到 `ep50/60` 才开始兑现

- 所以当前 story 最合理的下一步不是再讲：
  “teacher 还要更稳定”

- 而是更精确地讲：
  **pair-level teacher-change focusing 是对的，但当前第一版 focus 强度太弱。**

- 这让主方法进一步收紧成：
  1. pose/keypoint branch 定义 common-support relations
  2. support-complete bank 只负责增强 relational teacher
  3. stable teacher 是 supporting mechanism
  4. 真正的主突破口在于：
     **如何更强、更精确地把 teacher-change pairs 蒸进 global embedding**

### 当前最自然的下一跳

- 不换题，不回到 sample-level，不回到 generic GCN 模块
- 直接测试：
  **更强的 pair-delta focusing**

## 2026-03-20: exp123 正式评估把 story 再收紧到“稀疏 pair routing”

- `exp123` 的正式结果是：
  - `equal_concat = 61.1 / 73.4`
  - `global = 60.2 / 70.3`
  - `cvk_hybrid = 61.9 / 73.2`

- 相对 `exp119`：
  - `equal_concat` 只保留了 `R1 +0.2`
  - `global` 变成 `mAP -0.2 / R1 +0.0`
  - `cvk_hybrid` 也只是近乎等价

- 这把 story 进一步收紧成：
  1. pair-level `teacher-change focusing` 方向本身没错
  2. 但第一版 `alpha=1.0` 的连续 delta weighting 还太弱，没能把训练监控里的 delayed gain 稳定转成正式 eval gain

- 同时，远程 `exp124` 到 `ep40` 又给了一个很关键的补充：
  1. `alpha=4.0` 能把 `pair_focus` 明显放大到 `1.24~1.29`
  2. 但中期指标仍然只是在 `exp123/119` 附近轻微摆动

- 所以当前 story 最合理的新收束不再是：
  “继续把 pair focus 调得更大”

- 而是更精确地写成：
  1. teacher-change pairs 是稀疏的
  2. 如果对所有 pair 做连续平滑 weighting，真正有信息量的 changed pairs 仍会被大量近零变化 pair 淹没
  3. 因而主方法下一步应升级为：
     **sparse pair routing for support-complete relational distillation**

- 换句话说，当前主创新点已经越来越不像“再做一个 loss 权重”，而更像：
  **只把被 support completion 真正改变过的 comparability relations 蒸进 global embedding。**

## 2026-03-20: 本地主线补上一条更直接的 feature-level兑现

- 到 `exp126` 为止，story 的一半已经比较清楚：
  1. `support-complete relational teacher` 有价值
  2. `pair routing` 决定这种价值能否被 global embedding 吃进去

- 但另一半还没有被真正打透：
  1. `exp109` 的 oracle 是直接改 feature 的
  2. `SCKD/CSRD` 都偏间接
  3. `SCFR` 虽然直接，但过于硬替换，结果只和 `SCKD` 近乎等价

- 所以当前本地主线补上的不是“再一个 routing 变体”，而是：
  **SCRC: Support-Conditioned Residual Completion**

- 它在 story 里的角色是：
  1. 继续坚持 `single-image support incomplete` 这个问题定义
  2. 但把训练机制从 “prototype 蒸馏 / hard replace” 升级成
     **support prior 参与 feature formation 的可学习残差补全**

- 如果 `SCRC` 有效，论文核心就会开始从
  “更会蒸哪些 pair”
  扩展到
  “如何让 support-complete prior 真正进入遮挡部位的表征形成”

## 2026-03-20: `SCRC` 与 `freeze` 都没有成为本地主突破口

- `exp127` 到 `ep100 = 60.5 / 73.1`，没有超过 `SCFR/SCKD` 系列
- 更重要的是，它的 `gate` 几乎塌到 `1.0`，说明当前 learned residual completion 在 late-stage 实际退化成了近似 hard replace
- 这一步很重要，因为它把 story 重新收紧成：
  1. `single-image support incomplete` 的问题定义仍然成立
  2. 但 per-ID prototype 的 direct feature completion 兑现方式当前不成立

- 与此同时，`freeze20/30` 的既有结果又已经足够说明：
  - `stable teacher` 只是 supporting mechanism
  - 不值得继续扩成新的本地主线

## 2026-03-20: 当前 story 进一步收紧到 “新增 correction 如何被学到”

- 到这里，主线里已经同时有三类证据：
  1. `exp119` 证明 relational distillation 有效
  2. `exp120/121` 证明 support-complete teacher 会改变 teacher 几何，但仅靠 teacher 变强不够
  3. `exp123/125` 证明 pair-level routing 有效，但收益仍偏弱、偏慢

- 这三件事拼起来，当前更像是在说：
  **真正难学的不是完整 teacher 几何，而是 support completion 相对 base teacher 带来的那部分新增 relation correction。**

- 所以本地主线的下一步不再是：
  - `freeze`
  - direct completion
  - 或继续扫 `alpha/top_ratio`

- 而是：
  **Residual-Correction SCRD**

- 它在 story 里的意义是：
  1. 不再让 student 去复刻整份 support-complete teacher
  2. 而是只学习 `support completion` 真正引入的那部分 **pairwise correction**
  3. 如果这一步成立，论文主创新就会从“结构化 pair focus”继续升级成：
     **support-complete relation correction learning**

## 2026-03-20 夜间更新：`Residual-Correction` 没有成为主突破口

- `exp130 residual_kl` 最终是：
  - `ep110 = 60.1 / 73.4`
  - `ep120 = 60.1 / 73.1`

- 直接对照 `exp125`：
  - `ep110 = 60.4 / 73.8`
  - `ep120 = 60.5 / 73.5`

- 这一步很重要，因为它把 story 再收紧了一次：
  1. `residual target` 不是没接上，它的 `csrd` 信号全程稳定
  2. 但它到收敛都没有压过 `exp125`
  3. 所以当前不能再把 “teacher target 太完整、稀释了新增 correction” 当作主矛盾

- 因而主 story 现在更像是：
  1. `support-complete relational teacher` 有价值
  2. `pair routing` 也有价值
  3. 但真正的上限，可能卡在 **单个 batch 内 changed pairs 覆盖太少**

- 这意味着下一步最自然的方法升级，不是继续改 target，而是：
  **Cross-Batch Changed-Pair SCRD**

- 如果这一步成立，论文主创新会从
  “结构化 pair focus”
  再升级成：
  **在更大的 relation support 上学习 support-complete comparability correction**

## 2026-03-21 凌晨更新：`Cross-Batch Changed-Pair` 没把 story 推过下一道坎

- `exp131` 最终是：
  - `ep110 = 60.4 / 73.7`
  - `ep120 = 60.5 / 73.7`

- 直接对照 `exp125`：
  - `ep110 = 60.4 / 73.8`
  - `ep120 = 60.5 / 73.5`

- 更关键的是，它的 queue 不是没工作：
  - `csrd_qn = 256`
  - `csrd_qr ≈ 0.43`

- 这一步的重要性不在于点数大小，而在于它把 story 再次收紧：
  1. `target form` 不是主矛盾
  2. `relation coverage` 也不是主矛盾
  3. 也就是说，当前不是“没看见足够多的 changed pairs”
  4. 而更像是：
     **现有 student 形式不适合承载 pair-specific support-complete correction**

- 到这里，旧 story 里那条
  “继续把更多 relation 蒸进 global embedding”
  已经开始失去解释力

- 新 story 更合理的改写应是：
  1. global embedding 负责主体身份空间
  2. pose/keypoint branch 定义 common-support evidence
  3. support-complete teacher 负责告诉模型“哪些 pair 真的需要 correction”
  4. 但最终 correction 不一定应该继续被压成单个 embedding

## 2026-03-21: story 的下一跳应转成真正的 learned pair correction

- 这里还要明确一个边界：
  - 仓库里曾有 `exp089 PAMN` 设计稿
  - 但它从未真正接入 checkpoint 保存与测试检索流程
  - 所以“learned pair module”并没有被做过，更谈不上被证伪

- 这让当前最自然的方法升级变成：
  **LTCS / Learn-to-Trust Common Support**

- 它在 story 里的角色是：
  1. 不再继续逼单个 embedding 去吸收 correction
  2. 而是显式学习一个 pair-adaptive fusion rule
  3. 让模型在检索时决定：
     - 什么时候更该相信 global distance
     - 什么时候更该相信 common-support distance
  4. 这个 decision rule 由 `support-complete teacher` 监督，而不是人工固定成 `1:1`

- 如果这一步成立，论文主创新将从
  “support-complete relational distillation”
  进一步升级成：
  **support-complete guided pair-adaptive correction**

## 2026-03-21 早间更新：`LTCS alpha-fusion` 没把 story 推过下一道坎

- `exp132` 的正式结果是：
  - `cvk_adaptive = 62.1 / 72.8`
  - `cvk_hybrid  = 62.1 / 72.8`

- 这一步的重要性不在于点数高低，而在于它把 story 再收紧了一次：
  1. learned pair module 这个大方向没有死
  2. 但第一版 “学一个 `alpha` 决定信 global 还是信 CVK” 并没有真正改变最终排序
  3. 也就是说，当前不是“检索期 head 没必要”
  4. 而是：
     **当前 head 太弱，只能学到接近固定 `1:1` 融合的行为**

- 这让 story 的下一跳不该再写成：
  - `pair-adaptive fusion`

- 而应该升级成：
  **pair-specific correction scoring**

- 更具体地说，新的 story 更合理的版本是：
  1. global embedding 提供主体身份空间
  2. keypoint/common-support 分支提供 pair-specific 可比较证据
  3. support-complete prior 告诉模型“哪些 pair 的比较关系应被修正”
  4. 但这种修正不该只被压成一个混合权重，而应被表示成：
     **一个真正的 pair correction score / residual score**

- 如果下一步成立，论文主创新就会从：
  **support-complete guided pair-adaptive correction**
  进一步收紧并升级成：
  **support-complete guided pair-specific correction scoring**

## 2026-03-21 上午补记：`exp133/134` 目前不能进入 story 证据链

- `exp133 LPCS` 与 `exp134 Sparse LPCS` 当前都被判定为失效 run。

- 原因不是方法负结果，而是共享接线 bug：
  1. `kp_aux_data` 构建条件漏掉了 `ltcs_enabled / lpcs_enabled`
  2. 导致 `lpcs_teacher_feats` 永远不会生成
  3. `LPCS` loss 实际从未被加入训练
  4. 日志里也因此完全没有 `lpcs_*` 统计

- 所以 story 上必须明确：
  1. 当前我们还**没有真正测到** `LPCS`
  2. 不能把 `exp133/134` 的数值当成 learned pair correction 的证据
  3. 正确的下一步不是改写 story，而是：
     - 修 bug
     - clean rerun
     - 重新收集能证明 `LPCS` 真激活的机制统计与正式结果

## 2026-03-21 晚间更新：`LPCS` 终于被真正测到，但当前瓶颈更像 ranking 而不是 routing

- `exp135 corrected LPCS` 已跑满：
  - `ep120 = 61.1 / 72.3`
- `exp136 corrected sparse LPCS` 已跑满：
  - `ep120 = 60.9 / 72.1`

这批新证据把 story 又往前推进了一步：

1. 现在我们终于可以说：
   **learned pair correction 这个大方向是真的，不再只是设计猜想。**
   原因是 `exp135` 里 `LPCS` loss 已真实进入训练，而且：
   - `lpcs_fg` 长期显著高于 `lpcs_bg`
   - 排序确实被系统性改写

2. 但 corrected full-pair `LPCS` 的收益形态不是“全面变强”，而是：
   - `mAP` 更强
   - `R1` 偏弱
   这说明当前 head 更像在做 deeper-rank correction，而不是更直接地修 top-1 错误。

3. `exp136` 又给出一个很关键的负边界：
   - 真稀疏 routing 已经被首次干净验证
   - `lpcs_psr = 0.254`
   - `lpcs_pf ≈ 3.0`
   但它到收敛也没有压过 full-pair `LPCS`

因此，当前 story 最合理的收束不再是：
- “只要把 pair 挑得更稀疏就会更强”

而更像是：
- **pose 定义共同可见证据**
- **模型学习对 pair 做 correction**
- **真正的下一步瓶颈在于，如何让 correction 目标更 ranking-aligned**

也就是说，论文主线现在更接近：
**Pose-guided Learned Pair Correction for Common-Support ReID**

而下一步最关键的机制升级应该是：
**让 `LPCS` 直接为 hardest / top-ranked 错误负责，而不是平均地对所有 selected pairs 做回归式排序约束。**

## 2026-03-21 深夜补充：`exp137` 说明“更 ranking-aligned”不能简单做成 hard selection

- `exp137 Hard-Rank LPCS` 到 `ep80` 为止：
  - `60.1 / 70.4`
- 相对：
  - `exp135 ep80 = 60.8 / 71.9`
  - `exp125 ep80 = 59.4 / 72.0`

这条负结果很重要，因为它不是实现没接上，而是：
- `lpcs_rsr = 0.254`
- `lpcs_psr / lpcs_pf = 1.000 / 1.000`

所以 story 现在要进一步收紧：

1. `LPCS` 的确需要更贴近最终排序目标
2. 但**不能**简单粗暴地只保留 hardest 25% pairs
3. 更合理的下一步应该是：
   - 保留 full-pair 的上下文稳定性
   - 同时对 top-ranked mistakes 做更平滑、更连续的强调

## 2026-03-21 转向后的两个候选升级

基于 `exp136` 和 `exp137`，story 现在自然分成两条待筛选的升级线：

1. **平滑 top-sensitive 线**
   - 代表实验：`exp138 Rank-Decayed LPCS`
   - 核心想法：
     - 不删除大部分 pairs
     - 只用连续 rank-decay 去更重视 top-ranked mistakes

2. **上下文感知 correction 线**
   - 代表实验：`exp139 Query-Context LPCS`
   - 核心想法：
     - 不是 pair weighting 不够，而是 current scorer 太短视
     - 让每个 pair correction 同时感知 query 的整体难度、margin 与 support 完整度

这两条线都保持同一个大 story 不变：
- pose 负责定义 common support
- learned pair correction 负责改写检索距离

它们的区别在于：
- `exp138` 改的是 **如何强调 top-ranked mistakes**
- `exp139` 改的是 **scorer 是否具备足够上下文**

## 2026-03-21 审查后补充：上下文线要保留，但必须改成 test-time 可用的 context

`exp138` 的全面审查已经放行，说明“平滑 top-sensitive correction”这条线在实现上是闭环的，可以直接验证。

但 `exp139` 的审查结论很重要：当前版 query-context 不能直接进入 story，因为它的 context 依赖 label，测试阶段天然不可得，而且 evaluator 仍在构造 6 维 descriptor。也就是说，**它不是结果不好，而是实验定义本身还没闭环。**

这反而让 story 更清楚了：

1. 我们想保留的不是“oracle query context”
2. 我们真正要验证的是：
   - 检索时，是否能用 **test-time 可得的 query-level statistics** 改善 pair correction
3. 因而上下文线的下一步必须改成：
   - 无标签
   - train/test 对称
   - evaluator 可直接构造

这条修正后的上下文线仍然和 pose 主线一致，因为它不是在替换 common support，而是在问：
**给定 pose 定义出的 common support 之后，pair correction 是否还需要 query 级语境。**

## 2026-03-22 当前 story 收紧：主候选从“平滑 rank 强调”转向“query-context pair correction”

到现在为止，这两条升级线已经开始分出层级：

1. `exp138 Rank-Decayed LPCS`
   - 它说明：
     - `hard-rank` 的问题确实在于过于离散、过于激进
     - 更平滑的 rank-decay 能恢复稳定性
   - 但它到停表窗口仍只达到：
     - `ep80 = 60.7 / 71.7`
     - 与 `exp135 ep80 = 60.8 / 71.9` 基本持平
   - 因而更适合被讲成：
     - supporting evidence
     - 用来证明“纯 ranking emphasis 不是主突破口”

2. `exp139 Query-Context LPCS`
   - 它现在已经开始表现出更像主方法的特征：
     - `ep20 = 47.6 / 60.0`
     - `ep40 = 57.0 / 68.8`
     - 同时超过 `exp135` 与 `exp138`
   - 更关键的是：
     - `lpcs_ctxm ≈ 0.46`
     - `lpcs_fg > lpcs_bg`
     - 说明 query-level context 不是挂件，而是真的在参与 pair correction

所以 story 现在可以进一步收成一句话：

**pose 定义哪些身体证据是共同可比的；query context 决定这些共同证据在当前检索 pair 中应该被如何解释。**

这比“再设计一个更好的 rank weighting”更像论文级主贡献，因为它把问题从：
- 如何挑 pair

推进成了：
- 如何理解同一个 common-support signal 在不同 query 语境下的意义

## 2026-03-22 本地并行 story 候选：correction 不仅要“会修”，还要“知道何时该收手”

在 `exp139` 继续验证 query-context 的同时，本地新的 `exp140` 代表另一条不同的 story 候选：

1. `exp135` 的长期形态一直是：
   - `mAP` 能涨
   - `R1` 不够稳
2. 这不一定意味着 scorer 缺 context
3. 也可能意味着：
   - scorer 会产生 correction
   - 但不会判断该不该信这次 correction

因此 `exp140` 的故事候选是：

**pair correction 需要 confidence calibration。**

如果它成立，story 可以进一步分层：

- pose 定义 common support
- support-complete teacher 提供 correction 信号
- confidence gate 决定 correction 以多大强度写回最终检索距离

这条线与 `exp139` 的区别很清楚：

- `exp139` 强调 **context-aware interpretation**
- `exp140` 强调 **confidence-aware application**

## 2026-03-22 到 `exp139 ep50` 的最新 story 位置

当前 story 已经不再平均分散在多条线之间，而是出现了比较清楚的主次关系：

1. `exp139 Query-Context LPCS`
   - 已给出：
     - `ep40 = 57.0 / 68.8`
     - `ep50 = 58.7 / 70.4`
   - 相对 `exp135/138` 同期都更强
   - 这意味着 story 里的这句话开始有证据支撑：
     - **同一份 common-support signal，必须放在 query 的整体语境中解释**

2. `exp140 Confidence-Calibrated LPCS`
   - clean rerun 已完成关键验证
   - 当前版本的问题不是没接上，而是 gate 很快塌成接近常数 1
   - 它不是替代 `exp139`，而是在问另一个同样合理的问题：
     - correction 是否需要显式的 confidence gate

## 2026-03-22 本地新的不同创新点：从 query 语境转向 candidate competition 语境

在 `exp140` 止损后，本地新线不再沿 confidence 小修补，而是切到另一类 story：

**pair correction 可能不只是缺 query-level context，  
还可能缺“当前这个候选相对其他候选到底排在哪”的 competition context。**

这就是 `exp141 Competition-Context LPCS` 的位置。

它和 `exp139` 的区别必须讲清楚：

1. `exp139`
   - 给 pair 一个 query 级摘要语境
   - 回答“这个 query 整体难不难”
2. `exp141`
   - 给 pair 一个 candidate competition 语境
   - 回答“这个 pair 在所有候选中是不是少数真正值得被强修正的对象”

如果 `exp141` 成立，story 会进一步从：

- query-aware correction

推进成：

- **competition-aware correction**

所以当下最合理的论文主叙事排序是：

1. 第一主候选：
   - pose-defined common support + query-context pair correction
2. 第二并行候选：
   - pose-defined common support + confidence-calibrated correction

如果后续 `exp139` 继续稳定转正，它会比 `exp140` 更先具备“主方法”的资格；  
而 `exp140` 更像是在验证：当前剩余的 `R1` 缺口，究竟是解释问题，还是应用问题。

## 2026-03-21 本地主线切换：不再围绕 `LPCS` 家族继续小改动

到 `exp141` 这个节点，虽然 `competition-context` 的二次审查已经通过，但我们需要主动承认一件事：

- 它仍然属于 `LPCS` 家族的小变体
- 即使做成，也大概率只是把“pair correction”讲得更复杂
- 这不足以回应用户提出的核心要求：
  - 做一个真正大的、真正可能成为主创新的改动

因此，本地主线现在明确切回 `exp109` 提出的根问题：

**Occluded ReID 的关键缺口不是 pair scorer 还不够聪明，  
而是单张图里 pose-aligned support 本身不完整。**

这会把 story 从：

- pose-defined common support
- pair-specific correction

推进到：

- **pose-defined support incompleteness**
- **support-conditioned keypoint completion**

这条新 story 的价值在于：

1. 它仍然和 pose 强相关
   - pose 定义关键点语义与结构
2. 它比 retrieval-side correction 更“靠前”
   - 不是改分数
   - 而是改表征
3. 它更直接对应 `exp109 oracle` 的发现
   - 上界来自 support completion
   - 因此下一步就应直接尝试 completion，而不是继续在 scorer 侧挤牙膏

所以当前论文主叙事开始出现新的分工：

1. 远程主线：
   - `exp139 Query-Context LPCS`
   - 继续回答：如果坚持 retrieval-side correction，最有希望的版本是什么
2. 本地主线：
   - `exp142 SKC`
   - 直接回答：能否把 `support incomplete` 在特征层修掉

如果 `exp142` 成立，它会比 `LPCS` 家族更有资格成为论文的主方法；  
如果 `exp142` 不成立，我们也能更有底气地说：

- `exp109` 的 headroom 不是简单 feature completion 就能兑现
- 那么 retrieval-time structured reasoning 才更像最终主线

## 2026-03-22 下午：story 再次收缩，`PCVT` 成为新的主要观察对象

今天这轮两条大改动方向已经开始分化：

1. `exp148 PCVT`
   - 不是 retrieval scorer
   - 不是 feature completion 小残差
   - 而是直接把单图改写成：
     - `full`
     - `complementary view a`
     - `complementary view b`
   - 去学习“互补 support 的联合表示如何逼近完整表示”

2. `exp149 SCFA`
   - 尝试利用单图内部双侧冗余
   - 但当前 benchmark 上真正有用的 bilateral gap case 太少
   - 已在快速止损窗口内判负

这带来的 story 收缩是：

- 我们不再把“单图内部左右对称冗余”视为主要缺口
- 而开始更认真地看待另一种解释：
  - **单图的问题不是局部 token 之间还不够会交互**
  - **而是训练对象本身没有把“互补可见证据”显式组织起来**

如果 `PCVT` 后续继续成立，story 会从：
- support incomplete
- support completion / pair correction

进一步推进成：
- **pose-defined complementary pseudo-view learning**
- **把单图改写成伪多 support 训练对象**

这条叙事比继续做 scorer 小修补更大，也更有可能回应“为什么 `exp109` 的 oracle headroom 到现在一直兑现不出来”。
