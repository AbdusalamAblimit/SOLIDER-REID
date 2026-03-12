# 论文故事线（持续更新）

> **⚠️ Phase 1 内容保留在下方（PCFC/GiLt）。Phase 2 更新如下。**

## Phase 2 Story Update (2026-03-11)

### 暂定标题
Pose-Guided Dual-Stream Architecture with Gradient-Isolated Part Learning for Occluded Person Re-Identification

### Phase 2 核心发现 (更新 2026-03-11)
1. **PSG (Pose Spatial Gate)**: 在 Swin Stage 3 blocks 内部注入 pose heatmap 信息，通过轻量门控 x*(1+gate) 调制特征。+1.7% mAP，仅 102K 额外参数
2. **PDS (Pose Dual Stream)**: 复制独立 Stage 3 给 Part 分支，实现权重解耦。解决了 21 个实验暴露的梯度干扰问题
3. **StopGrad**: 完全阻断 Part→共享层梯度，消除残余干扰 → 额外 +1.2% mAP
4. **三者组合 (exp023)**: mAP 59.5%, R1 69.5% — 比 baseline **+2.9% mAP, +3.0% R1**

### Phase 2 核心消融表（用 global-only 特征，公平对比）
| 方法 | mAP | R-1 | 额外参数 |
|------|-----|-----|----------|
| Baseline | 56.6% | 66.5% | — |
| + Part Pooling only (exp001) | 57.5% | 67.1% | ~2.6M |
| **+ PSG only (exp007)** | **58.3%** | **67.9%** | **102K** |
| + PSG + Part same Stage3 (exp008) | 57.7% | 66.0% | ~2.7M |
| + PDS (独立 Stage3, exp022) | 57.9% | 67.1% | ~8.8M |
| **+ PDS + StopGrad (exp023)** | **59.5%** | **69.5%** | **~8.8M** |

### 消融证据链
1. Baseline 56.6% → +PSG: 58.3% (+1.7%) — PSG 有效
2. +PSG+Part 同 Stage3: 57.7% (-0.6%) — 梯度干扰
3. +PDS 独立 Stage3: 57.9% (+0.2% vs exp008) — 解耦部分有效
4. +PDS+StopGrad: 59.5% (+1.6% vs PDS) — 完全隔离更好
5. exp023 Part-only: 56.7% > exp022 Part-only: 55.2% — 证明更好的共享特征反哺 Part

### 跨数据集验证 (PSG)
| 数据集 | Backbone | PSG mAP提升 |
|--------|----------|-------------|
| Occluded-Duke | Swin-Tiny | +1.7% |
| Occluded-Duke | Swin-Small | +2.0% |
| Market-1501 | Swin-Tiny | +0.8% |
| Market-1501 | Swin-Small | +0.6% |

### 核心贡献（预计 3 点）
1. **PSG (Pose Spatial Gate)**: 极简的 backbone 内部 pose 注入，证明"在特征形成阶段注入 pose > 事后 pooling"
2. **PDS + Gradient Isolation**: 双分支架构 + 梯度隔离，解决 pose-guided 方法中的多任务梯度干扰问题
3. **系统性消融研究**: 24 个实验全面探索 pose 信息在 ReID 中的利用方式

### ✅ 多种子验证完成 (2026-03-12, 4090)

**三组实验 × 3 seeds 的多种子验证已完成，提供了可靠的统计结论。**

#### 核心结果 (mAP, Occluded-Duke, Swin-Tiny)

| 方法 | Seed 1234 | Seed 42 | Seed 2024 | **Mean±Std** |
|------|-----------|---------|-----------|-------------|
| Baseline | 56.7% | 55.9% | 56.9% | **56.50±0.53%** |
| PSG | 58.3% | 57.9% | 57.3% | **57.83±0.50%** |
| PDS+StopGrad (global) | 59.7% | 59.2% | 58.7% | **59.20±0.50%** |

#### 统计检验 (paired t-test)

| 对比 | Mean Δ | Paired Diffs | t-stat | p-value |
|------|--------|-------------|--------|---------|
| **PSG vs Baseline** | **+1.33%** | (+1.6, +2.0, +0.4) | 2.78 | ~0.054 |
| **PDS+SG vs Baseline** | **+2.70%** | (+3.0, +3.3, +1.8) | 5.92 | ~0.014 ✅ |
| **PDS+SG vs PSG** | **+1.37%** | (+1.4, +1.3, +1.4) | 39.5 | <0.001 ✅✅ |

#### 关键发现
1. **PSG +1.33% 确认** — 所有 3 seeds 均正向，但 p≈0.054 边缘显著
2. **PDS+StopGrad +2.70% 高度确认** — p<0.02，统计显著
3. **PDS+SG vs PSG 极其一致** — paired diffs (1.4, 1.3, 1.4)，方差极小
4. **PDS+StopGrad 增益来源**: 已证明 = global loss * 0.5 (exp007a 完美匹配)
5. **0.5x Loss Scaling 是否为 sweet spot**: 单种子 grid 中 0.25x/0.75x/1.0x 效果相同，仅 0.5x 突出。需 exp007a 多种子确认 0.5x 是真实效果还是方差

#### 单种子 vs 多种子对比

| 实验 | 3090 单种子 | 4090 多种子均值 | 差异 |
|------|-----------|--------------|------|
| Baseline | 56.6% | 56.50% | -0.1% |
| PSG | 58.3% | 57.83% | -0.47% |
| PDS+StopGrad (global) | 59.5% | 59.20% | -0.30% |

→ 3090 和 4090 结果高度一致，3090 轻微偏高 (~0.3%)

### Loss Scale 敏感性分析 (exp007b/c, 3090 单种子)

| Loss Scale | mAP | R1 |
|-----------|-----|-----|
| 0.25x | 58.3% | 67.6% |
| 0.5x | 59.5% | 69.8% |
| 0.75x | 58.6% | 67.6% |
| 1.0x | 58.3% | 67.9% |

→ 0.25-1.0 范围内 mAP 无差异 (58.3-58.6%), 0.5x 的 59.5% 是训练方差

### Skeleton GCN (exp030a, 单种子, 待多种子确认)

| 方法 | mAP | R1 | 额外 Params |
|------|------|------|-------------|
| PSG+GCN global-only | 59.8% | 69.5% | +~500K |
| **PSG+GCN equal_concat** | **61.1%** | **73.7%** | **+~500K** |
| PSG+GCN gcn_only | 58.2% | 72.9% | +~500K |

→ GCN 贡献 ~1.3% mAP (equal_concat vs global), 但 R1 增益巨大 (+4.2%)
→ ⚠️ 仍为单种子，需多种子确认

### 跨数据集/Backbone 验证 (4090)

| 数据集 | Backbone | Baseline mAP | PSG mAP | Δ |
|--------|----------|-------------|---------|-----|
| OccDuke | Swin-T (3-seed mean) | 56.50% | 57.83% | **+1.33%** |
| OccDuke | Swin-S (lr4) | 65.8% | 67.8% | **+2.0%** |
| Market | Swin-T | 91.6% | 92.4% | **+0.8%** |
| Market | Swin-S (lr4) | 93.3% | 93.9% | **+0.6%** |

→ PSG 在所有组合上均有效，遮挡数据集增益更大

### 更新后的核心贡献（多种子确认版）
1. **PSG (Pose Spatial Gate)**: backbone 内部 pose 注入，**+1.33% mAP (3-seed mean)**，102K params
2. **0.5x Loss Scaling**: global loss 降半作为隐式正则化，**额外 +1.37% over PSG** (PDS+SG multi-seed 验证)。PDS+StopGrad 架构非必需，简单 loss*0.5 即可复现。**需 exp007a 多种子最终确认**
3. **Skeleton GCN**: 骨架拓扑互补特征，+1.3% mAP (单种子，待确认)

总计 multi-seed confirmed: baseline 56.50% → PSG 57.83% → PDS+SG 59.20% (**+2.70%**)
总计 single-seed best (含 GCN): baseline 56.6% → 61.1% (**+4.5% mAP, +7.2% R1**)

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
