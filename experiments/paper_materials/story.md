# 论文故事线（持续更新）

> **⚠️ 2026-07-15 PSG 新颖性审计覆盖：下方 PRCV Reset 中“PSG 主创新”的旧判断
> 已失效。** WACV 2020 已直接覆盖 ReID backbone 中层的 pose-guided spatial x
> channel gating；SFT/FiLM 又覆盖其条件仿射与 `1+delta gamma` 形式。PSG 仍是
> 有效组件和实验资产，但不能以当前公式单独承担论文主贡献。完整 claim-by-claim
> 审计见 `experiments/paper_notes/psg_novelty_audit_20260715.md`。本文件下方历史
> 段落不删除，以保留决策轨迹，但不得再把“首次前移 pose 注入”复制到新稿。

> **⚠️ Phase 1-4 内容保留在下方。Phase 5 更新如下。**

## PRCV Reset (2026-04-15) — PSG 主线 + GCN 结构补充

### 当前一句话故事

现有 pose-guided occluded ReID 大多在特征形成之后再使用 pose 信息；我们提出 `PSG`，将 pose 先验前移到 backbone 表征学习阶段，并在最终系统中引入 `GCN` 结构分支做显式 skeleton relational reasoning，形成 semantic-structural complementary evidence。

### 当前重审结论

这轮重审后，PRCV 主故事优先回到 `PSG`，而不是继续把 `LGPA-D + MaxSim + flip` 当主创新。

当前更稳的写法是：
1. **PSG** 是主创新点
2. **2-stage PSG** 只作为 `PSG` 的最终 instantiation / final configuration，不单独抢主贡献位置
3. **GCN 必须明确写进方法**，但定位为 *structural pose branch*，不是与 `PSG` 并列的第二主创新
4. `LGPA-D / OA-SD / PLBOA` 作为完整系统资产
5. `MaxSim / POT / flip` 只作为 supporting evaluation，不当训练端主贡献

### 当前主判断

1. `exp007` 已经足够支撑 `PSG` 本体：
   - 单次 `58.3 / 67.9`
   - 3-seed mean `57.83 / 67.13`
   - backbone-level pose injection 明确优于 post-hoc part pooling

2. `GCN` 应该被强调，但应强调其**作用位置**而不是单独吹成主创新：
   - `GCN` 的价值是提供显式 skeleton structure evidence
   - 更适合作为 `PSG` 支撑下的结构分支，而不是与 `PSG` 平行的主贡献
   - `exp249` 与 `exp246` 已经说明 `LGPA-D + GCN` 双分支具备稳定互补性

3. `2-stage PSG` 可以作为最终版本，但**不必在主叙事里和 1-stage 正面对打**
   - `exp009`、`exp251`、`exp253` 都说明：multi-stage 不会在所有 scaffold 上自动更强
   - 但 `exp255 vs exp255b` 明确说明：在 `GCN512` 这类高容量结构分支上，`2-stage PSG` 是关键条件

4. 因为实验都可以重跑，接下来不把旧消融当最终版，而是重新设计干净的 `PSG` stage 消融矩阵

### 当前最强系统 scaffold

当前训练端最强实验是 `exp255`：
- `Swin-Small + 2-stage PSG + LGPA-D + GCN512 + OA-SD + PLBOA`
- `FINAL = 73.2 / 83.3`

当前最关键的结构证据是：
- `exp255`: `GCN512 + 2-stage PSG = 73.2 / 83.3`
- `exp255b`: `GCN512 + 1-stage PSG = 71.5 / 81.9`

这组对照最适合在**消融**里写成：
> 最终实现采用 `2-stage PSG`；进一步对照显示，在高容量结构分支上，它比 `1-stage` 更能稳定支撑结构证据的发挥。

### 推荐写作口径

1. **标题 / 摘要 / 引言**
   - 只讲 `PSG`
   - 可以写：我们在 backbone 中间 stage 之间注入 pose 信息
   - 最多补一句：最终实现采用 two-stage instantiation

2. **方法部分**
   - 把 `PSG` 定义成一个通用的 pose-guided spatial gating 机制
   - 再说明：实际实验中采用 `2-stage PSG` 作为最终配置

3. **消融部分**
   - 再回答为什么最终选 `2-stage`
   - 用 `1-stage / 2-stage / 3-stage` 小表说明选择依据即可

### 论文里哪些模块要重点提及

1. **第一层：主贡献**
   - `PSG`
   - 写法：backbone 内的 pose-guided spatial gating

2. **第二层：关键支撑机制**
   - `GCN`
   - 写法：`GCN` 是 explicit structural pose reasoning branch
   - `2-stage PSG` 放在最终实现与消融选择中说明，不单列为主贡献

3. **第三层：完整系统资产**
   - `LGPA-D`
   - `OA-SD`
   - `PLBOA`
   - 写法：semantic branch + training recipe，不抢主创新位

4. **第四层：附加评测资产**
   - `MaxSim / POT / flip`
   - 写法：test-time supporting evaluations

### 推荐贡献点写法

1. 提出 `PSG`，在 backbone 内进行 pose-guided spatial gating，而不是在特征形成后再做 pose-aware pooling 或 filtering
2. 构建 semantic-structural complementary occluded ReID system，其中 `GCN` 提供显式 skeleton relational evidence，`LGPA-D` 提供语义 part evidence，与 `PSG` 形成互补
3. 在 Occluded-Duke 上系统验证该框架，并采用 `2-stage PSG` 作为最终实现；实验表明该设计能够更稳定地支撑高容量结构分支，最终在 Swin-Small 上得到当前最佳训练端结果之一

### 推荐摘要骨架

可按下面 4 句展开：

1. **问题句**
   - 现有 pose-guided occluded ReID 往往在特征提取完成后才利用 pose，因而对表征学习阶段的结构先验注入不足。

2. **方法句**
   - 我们提出 `PSG`，在 backbone 中间层通过 pose-conditioned spatial gating 直接调制特征形成过程。

3. **扩展句**
   - 在此基础上，我们结合 `GCN` 结构分支，以显式建模 skeleton relational evidence，并在最终实现中采用 `2-stage PSG` 作为具体配置，从而形成 semantic-structural complementary representation。

4. **结果句**
   - 在 Occluded-Duke 等基准上，该框架取得了当前项目最优结果之一，其中 `Swin-Small` 配置达到 `73.2 / 83.3`；消融进一步表明，最终采用的 `2-stage PSG` 更适合支撑高容量结构分支。

### 执行优先级

1. 重新设计 `PSG` 的干净 stage 消融：
   - no PSG
   - 1-stage PSG
   - 2-stage PSG
   - 3-stage PSG
2. 固定 branch 容量，单独验证 `2-stage PSG` 是否是高容量 `GCN` branch 的必要条件
3. 在此基础上，再决定最终论文标题更偏 `PSG` 还是 `Hierarchical PSG`

### 说明

详细重审与文献压缩总结见：
`experiments/paper_notes/2026-04-15_prcv_reset.md`

## Phase 5 Story Update (2026-04-08) — LGPA-D 时代

### 暂定标题
Language-Grounded Part Assembly for Occluded Person Re-Identification

### 当前最佳结果

| Backbone | Method | mAP (eq) | R1 (eq) | mAP (MaxSim) | R1 (MaxSim) |
|----------|--------|------|------|------|------|
| Tiny | LGPA-D+OA-SD | 65.3% | 75.7% | 66.0% | 76.4% |
| **Tiny** | **LGPA-D+GCN+OA-SD** | **65.5%** | **77.2%** | **66.3%** | **77.7%** |
| Small | LGPA-D+OA-SD (local) | 70.2% | 80.1% | 71.9% | 82.2% |
| **Small** | **LGPA-D+OA-SD (remote)** | **71.6%** | **81.6%** | **73.0%** | **82.7%** |
| Small | GCN+PAA+OA-SD (old baseline) | 70.6% | 82.6% | 72.3% | 82.9% |
| *Small* | *LGPA-D+GCN+OA-SD (exp249, 进行中)* | *TBD* | *TBD* | *TBD* | *TBD* |

### 核心贡献

1. **LGPA-D (Language-Grounded Part Assignment, Detached)**
   - 首次将 VLM (CLIP) 语义知识用于 occluded ReID 的 part assignment
   - 5 个语义 body parts: head, torso, arms, upper_legs, lower_legs
   - CLIP frozen text prototypes + cross-attention + pose heatmap bias
   - Detached from backbone → 不干扰训练, 全程 delta 为正
   - vs GCN skeleton features: +2.1% mAP (语义 > 结构)
   - vs PPA (non-detached): +4.4% (detach 消除后期干扰)

2. **PSG (Pose Spatial Gate)**
   - Backbone 内部 pose 注入 (Stage 3 block 间)
   - 轻量 102K params, +1.7% mAP
   - 改变特征形成方式, 不只是 post-hoc pooling

3. **Dual-Branch Architecture (LGPA-D + GCN)**
   - 语义 part features (LGPA-D) + 骨架 keypoint features (GCN) 正交互补
   - Tiny: +0.2 mAP / +1.5 R1 vs LGPA-D only
   - 两个 branch 都在 detached features 上操作

4. **MaxSim Hybrid Matching**
   - ColBERT-style late interaction 首次引入 person ReID
   - +1.0~1.5% mAP across all checkpoints
   - 理论框架: partial-set-to-partial-set matching

### 关键消融发现

1. **Detach barrier 是根本性约束**: 
   - Non-detached (exp243): ep80 -1.1 mAP → 后期干扰
   - Detached (exp244): ep120 +2.1 mAP → 全程正向
   - 250 实验验证: backbone 必须完全由主 loss 驱动

2. **CLIP 语义 > GCN 结构**:
   - LGPA-D 无 OA-SD (63.6) ≈ GCN + OA-SD (63.2)
   - CLIP 的 part assignment 能力 ≈ OA-SD 的训练增强

3. **训练集 95.8% visible**: 
   - 所有 visibility-dependent 训练方法失败 (VCSR, routing)
   - PLBOA (pixel-level occlusion augmentation) 是唯一有效补充

### 论文叙事

> Occluded person ReID 的核心挑战不是"如何处理遮挡"而是"如何定义和匹配不完整的身份证据"。
> 我们提出 LGPA (Language-Grounded Part Assembly): 利用 CLIP 的语义理解能力，
> 将 backbone 空间特征分解为语义 body parts，在 detached 特征上安全操作。
> 配合 PSG (backbone 内 pose 注入) 和 MaxSim (part-level late interaction matching)，
> 形成完整的 "语义引导提取 → 部分集合匹配" 框架。

---

## Phase 4 Story Update (2026-04-02)

### 当前最佳结果 (Phase 4 时期)

| Backbone | Method | mAP (eq) | R1 (eq) | mAP (maxsim) | R1 (maxsim) |
|----------|--------|------|------|------|------|
| Tiny | GCN+PAA+OA-SD | 63.2% | 75.4% | 64.2% | 77.1% |
| Tiny | **GCN+PAA+OA-SD+GSPB** | 62.9% | 74.3% | **64.6%** | **76.0%** |
| Small | GCN+PAA+OA-SD | 70.6% | 82.6% | 72.3% | 82.9% |
| Small | GCN+PAA+OA-SD+PKC | 70.6% | 81.8% | **72.4%** | **83.1%** |

### Phase 4 发现

1. **MaxSim Behavior on Tiny**: `MaxSim` 的收益更依赖 per-keypoint consistency，而不是简单取决于 global 强弱。

2. **GSPB (Gradient-Scaled Part Branch)**: 5% Part→Backbone 梯度大幅加速早期收敛 (+5.8% at ep10!) 但不改善 final。首次发现 detach 与 non-detach 之间的中间解。

3. **OA-SD Teacher Fix**: 修复了 EMA teacher 的 Dropout/DropPath/BN 噪声问题。修复后 teacher 更稳定，但 final 结果不变（EMA 的自修正性）。

4. **per-keypoint training loss 全面证伪**: PKC, MST, PACI, OERL, BA-PKC — 10 个实验全部失败。根本原因: detached GCN 阻断梯度到 backbone，non-detached 与 CE 冲突。

---

## Phase 3 Story Update (2026-03-23)

### 暂定标题
Pose-Guided Structural Token Decomposition for Occluded Person Re-Identification

### 核心贡献（更新 2026-03-24）

1. **STD-PR (Structural Token Decomposition with Pose-guided Routing)**
   - 用 pose-biased cross-attention 将 spatial tokens 转为 structural body-part tokens
   - 替代 GCN keypoint sampling：cross-attention 比 bilinear sampling 更善于利用 data augmentation
   - 3-seed mean: 62.6%±0.87 mAP (+1.87 vs baseline)

2. **PLBOA (Pose-guided Lower-Body Occlusion Augmentation)**
   - 基于 train-test occlusion gap 分析（1.8% vs 24.4% lower-body occluded）
   - 用真实 VOC 物体贴到 hip 以下区域
   - 2-seed mean: 62.3% mAP with GCN (+1.57 vs baseline)

3. **STD-PR+PLBOA Synergy**
   - STD-PR alone: -2.4 vs GCN
   - STD-PR+PLBOA: **+0.7 vs GCN+PLBOA**
   - PLBOA 增益：GCN +1.6 vs STD-PR **+4.7** (3x 差距！)
   - 核心发现：cross-attention 比 keypoint sampling 更善于利用 augmentation

4. **MaxSim (ColBERT Late Interaction, 辅助)**
   - 零训练成本 test-time method
   - +1.0~1.5% mAP across checkpoints
   - 首次将 NLP late interaction 引入 person ReID

### 核心范式论点
Occluded person ReID 不应该是 "extract one vector, compare vectors"，而是 "extract a set of body-part tokens, match sets"。这直接类比 NLP 从 sentence embeddings (BERT) 到 token-level late interaction (ColBERT) 的范式迁移。

### 贡献（候选）
1. **问题重构**: 将 occluded ReID 形式化为 partial-set-to-partial-set matching
2. **MaxSim matching**: 首次将 ColBERT-style late interaction 引入 person ReID
3. **Set-level metric learning**: 用 Soft-MaxSim 距离替换 pooled triplet，实现 train-test metric symmetry
4. **PSG + GCN pipeline**: 提供高质量 body-part token set 的提取方法

### 实验证据链（待补）
- MaxSim test-time: +1.0~1.5% mAP across all checkpoints ✅
- MaxSim training: exp152 进行中
- Ablation: soft vs hard MaxSim (exp152 vs exp152b) 进行中

---

## Phase 2 Story Update (2026-03-13)

### 暂定标题（旧）
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

## 2026-07-13：LGPA 改造线的最终收缩——PBSR 不进入论文故事

为解决 LGPA 与 PAFormer 高度重合的问题，exp370 曾提出 PBSR：用 pose 监督共享 routing，将空间特征读入结构槽、进行槽间推理，再写回标准 global；pose 不进入表征前向，推理无姿态依赖。这个故事在问题与机制层面比“在哪里注入 pose”更自然，也避开了 GCN、CLIP 语义和 matching 作为创新点。

但严格证据不支持把它写成论文方法：

- 同机同运行时 epoch 60，PBSR-off B0 为 `54.5 mAP / 63.8 R1`，PBSR P0 为 `54.4 / 63.7`；
- P0-B0=`-0.1/-0.1`，未达到预注册 `+0.8～1.0 mAP`；
- route loss 与写回统计均健康，说明不是实现未工作，而是“结构路由学会了，身份检索没有获益”。

因此论文叙事必须明确收缩：

1. **不把 PBSR 写成主贡献，也不继续为它补结构槽小变体。**
2. **不宣称正确 pose 优于 uniform/shuffled。** 主门禁已失败，这两个控制按预注册止损没有运行。
3. **不把 matching、GCN 或 CLIP 文本语义包装成创新。** 历史结果只能说明 LGPA/pose 分支在特定融合系统中有信号。
4. LGPA 与 PAFormer 的重合风险仍然存在；PBSR 负结果没有提供新的可投稿解法。

当前没有诚实证据支持以 PBSR 重写一篇新论文。若后续恢复论文工作，应回到已经独立成立的 PSG/既有实证主线，或重新定义新的问题对象；不能把这次 NO-GO 改写成正结果。

## 2026-07-13：新叙事候选——从单图 pose token 转向跨图 identity support

大调研后的新叙事不再是“把 pose 注入网络的哪个位置”，也不是“用 learnable pose token 替代 CLIP query”。PAFormer、TSD 与 PGFL-KD 已经覆盖了 pose-token、privileged teacher 和 pose-free student 的常规故事。

仓库自己的反事实证据给出了更自然的起点：

- 局部结构描述子确实稳定涨点；
- 通用 canonical 人体布局已经解释了大部分收益；
- 正确逐图 pose 与解剖标签只提供较小增量；
- 把整个结构分支写回 global 没有收益。

所以新的问题是：

> 遮挡图只携带不完整的身份 support，但训练集中的同 ID 多图往往拥有互补可见部位。能否用训练期 pose 组织这些跨图证据，形成 leave-one-view-out 的完整 anatomical support，再让单图、无姿态 student 学会其身份关系？

exp371 CASD 保留 LGPA 作为 detached 训练期 extractor，但不保留 CLIP 语义 claim。它只用同 ID 其他视图的可见 part 构造 support，并严格排除 anchor 当前图；image-only student 只学习 support 相对 same-image teacher 真正新增的 identity relation。最终仍保留已经验证的 global+parts 标准 cosine 描述子；matching、GCN 与 CLIP 文本不进入贡献。

correct/uniform/shuffled/wrong-person pose 只作为 support-quality 因果控制。2022 年已有题名精确撞车的 Pose-guided Counterfactual Inference，因此 counterfactual routing 不作为 headline。

还必须承认 AAAI 2020 UMTS 已提出 multi-shot comprehensive teacher → single-shot student。CASD 的新颖性不能写成“多图补单图”，只能落在 pose-organized part support、hard leave-one-view-out 和 support-vs-self identity advantage 三者的联合机制上；若实验退化为完整 multi-shot feature KD，则论文 story 立即失效。

这条 story 目前只是有查新边界和证据动机的候选，尚无正结果。只有 leave-one-view-out CASD 明确优于 same-image KD 与伪 support controls、且无姿态 student 恢复至少 80% 的 LGPA 增益后，才允许写成论文主方法。

## 2026-07-14：CASD 门禁失败，LGPA 自有化故事正式停止

exp371 的唯一正式 frozen oracle 已完成，且不是边缘性失败：POSE-RESP 相对最强 PART-EQUAL 为 `-0.0766` mAP pp，五折方向全部为负；相对 POSE-SCALAR 与 RESP-PERM 也分别为 `-0.0162/-0.0372` pp。scene-merged 协议同样为负且五折一致。coverage、strict-three-donor、path/content disjoint、canonical matrix 与 wrong-ID fail-safe 均正常，因此不能把结果解释成实现或协议没有工作。

可写进论文素材的机制边界是：

- same-ID 多图 support 确实能形成强 identity geometry；
- 固定部位对应比 slot permutation 高 `+1.2347` pp；
- 但实例级 pose response 没有给 equal/scalar/permuted routing 增加独立价值。

这意味着 CASD 的 headline——“用训练期 pose 组织跨图互补 support，再蒸馏给 RGB-only student”——缺少最关键的 pose-specific 因果证据。由于普通 multi-shot support 又已有 UMTS/MVI²P 等直接邻居，继续训练 student 只可能验证一个非独占的 generic support 故事，不能解决 LGPA 的归属问题。

论文叙事必须据此止损：

1. 不写 CASD，不把未训练的 student 设计写成方法；
2. 不把 LGPA、matching、GCN 或 CLIP 文本换名后声称自有创新；
3. LGPA 可作为历史系统中的性能组件或对照，但其 `+0.82～0.85 mAP` 不能承担新论文主贡献；
4. 若继续写全新论文，应回到独立成立的 PSG/既有证据，或重新定义与 LGPA 无关的问题对象，而不是继续修 pose routing 小变体。

到此，IPER 的实例 pose effect、PBSR 的无姿态 read/write global、CASD 的跨实例 pose support 三条路线均被各自预注册门禁否定。LGPA 自有化目标在当前证据下未实现，诚实结论是停止，而不是放宽门槛。

## 2026-07-15：PCAR 不进入论文故事

最后一次正交尝试把作用位置从 LGPA head 移到 official CLIP-ReID 的 CLIP ViT 内部：用相对 canonical layout 的实例姿态 residual 修改少量 self-attention heads，同时保留标准 global descriptor。该候选在训练前查新阶段即被否定。

否定理由不是“完全相同的名字已出现”，而是机制可归约：`B(Pinstance)-B(Pcanonical)` 仍是普通 additive pose bias；PeVL/PAAB 已覆盖 pose-conditioned CLIP/ViT attention，MUVA 已在 ReID 中逐层修改 CLIP ViT attention。少量 heads、zero-init、untouched semantic heads和 global-only输出只能形成一种更克制的 adapter 配置，不能承担主贡献。

因此新论文不得写：

- “首次将姿态注入 CLIP attention”；
- “canonical-relative residual 本身是一种新 attention”；
- “六臂控制更严格，所以机制就是新的”；
- “把 LGPA 换到 official CLIP-ReID 后即可归属为我们的创新”。

当前最诚实的故事边界仍是：LGPA 是有稳定增益的结构化局部资产，但不是已经完成归属的新方法。PCAR 不进入方法、实验表或摘要；如果继续全新论文，应回到独立成立的 PSG/已有证据，或重新定义与 additive pose attention、part assembly 无关的问题对象。

## 2026-07-15：SA 正交 scale–shift 不进入论文故事

在 PCAR 之后又审计了 PSG/PAA 合并路线。首先，论文不能把 PAA 描述为“只在
最终层添加”：真实代码已经在每个启用 Swin block 后依次执行 PSG 与 PAA；
Stage2+3 同步注入也已经由 `exp073` 跑过且没有增益。其次，clean stage sweep
只支持“Stage3 最稳定、更多层边际递减”，不支持“调制层数越多越好”。

把两支直接统一为

\[
y=x\odot(1+g(H))+a(H)
\]

只是空间条件仿射，与 FiLM/SPADE 重合。进一步把 PAA residual 投影到 PSG
displacement 的正交补，虽然比普通 scale–shift 多了 content-conditioned hard
constraint，但 orthogonal residual update 已有直接 vision 先例，ReID 中也已有
人体 shape/pose 子空间与正交补 identity feature 的直接工作。因此该变化最多是
PSG+PAA 的辅助正则，不是新的主方法。

论文故事据此锁定：

1. PSG 仍可按既有跨数据集、跨 backbone 证据作为独立方法资产；
2. PAA 可作为历史辅助组件或消融，但其正信号主要来自含 GCN 的旧 scaffold，
   不能直接宣称改善纯 global descriptor；
3. 不写“首次统一 multiplicative/additive pose modulation”；
4. 不写“正交化使 PSG/PAA 可识别”或“正交补本身是新 ReID 机制”；
5. 若论文需要第二个真正贡献，必须另找独立问题对象，而不是继续修 scale/shift
   的层数、路由、阈值或投影形式。

## 2026-07-15：exp374 终止 PSG 自有化，下一故事转向姿态控制状态动力学

exp374 把 PSG 的两个命题彻底拆开。相对 true bypass，正确 PSG 平均提高
`+3.8577 mAP / +5.1433 R1`，所以它不是无效组件；但把当前实例姿态替换为严格 matched
的其他实例姿态后，平均只变化 `+0.0012 mAP / 0.0000 R1`，三 seed 中两 seed 的 mAP
方向还非正。因此论文不能再把 PSG 的涨点解释成网络利用了当前行人的精确姿态。

这条结果不是要删除 PSG：它仍可作为强基线、通用人体空间先验或系统性能组件。被终止的
是“继续修改 PSG gate 后把它写成我们的 instance-specific pose reasoning”这一故事。
后续也不再围绕 layer、temperature、canonical、centroid 或 anatomical group 做救场。

下一故事候选独立转向 pose-controlled state-space dynamics：姿态不再逐位置缩放特征，
而是控制视觉证据如何被状态保留、遗忘、更新和跨身体区域传递。它必须同时满足：机制上
超出普通 pose token/input fusion；实验上正确 pose 明确优于 shuffle；并相对参数匹配的
image-only Mamba 与 PSG/SFT 对照产生稳定增益。该方向目前是待查新、待实现的候选，尚未
形成论文正结论。

## 2026-07-15：exp375 终止 PRSM 论文故事

pose-controlled state-space 候选已经从“待实现”推进到完整 Gate A，而不是停在查新或
早期快照。PRSM 用实例姿态控制 6 个身体状态槽的 recurrent write/retain，RGB 独立读取
carried state；B0、canonical M0 与 instance-pose P0 在同机同运行时跑满 120 epoch。

最终 P0=`57.1 mAP / 66.3 R1`，低于 B0=`58.4/67.1` 和 M0=`58.8/67.5`。更关键的是，
同 checkpoint correct 与 matched-shuffle、foreground-uniform、zero-bypass 的 mAP 差都在
`0.002` 百分点以内，R1/R5/R10 完全相同。反事实切换、matched gate、zero identity、
correct start/end 与参数学习审计全部正常，所以这不是执行失败，而是实例姿态没有成为
状态动力学的有效身份控制信号。

因此论文叙事必须明确收口：

1. PRSM 不进入方法、贡献、摘要、主表或消融，不写成“pose-controlled Mamba”成功结果；
2. 不用 graph、scan order、更多状态槽或额外 loss 延长这条线；
3. 本结果不删除 PSG/LGPA 历史性能资产，但也没有解决它们的创新归属；
4. 当前新论文仍缺少经强控制成立的第二主创新，不能用 PRSM、matching、GCN 或 generic
   memory 补位；
5. 若未来再研究 pose×state-space，必须更换问题对象或可观测信号，并先过 correct-vs-matched
   的最小因果门禁，而不是继续围绕当前 heatmap routing 微调。

## 2026-07-16：Pose Hyper-LoRA 与 Selective SSM 均不进入论文故事

后续两次实验已经覆盖“也许只是旧门控太简单”的主要反驳。exp376 在 8 个 Swin block 后
加入由姿态生成的逐层低秩动态算子，e60 比 clean B0 低 `1.0 mAP`；exp377 使用真实稳定
状态矩阵与 `exp(ΔA)` recurrence，让实例姿态联合修正 selective `Δ/B/C`，e60 仍低
`0.7 mAP`，e120 也只有 `+0.2 mAP`。两者的动态参数都实际学习，不能解释为模块没工作。

因此新论文不能写：

1. “更强的姿态动态参数生成解决了 PSG 的对应性问题”；
2. “姿态控制 Mamba selective update 带来稳定增益”；
3. “RGB-only SSM 的普通容量收益证明姿态有效”；
4. “再拆 `Δ/B/C`、改 scan 或增加 state 就足以形成主创新”。

至此，逐位置仿射、结构槽 recurrent routing、逐层低秩动态变换和真实 selective SSM 四类
机制都没有让正确实例 heatmap 产生可报告的额外身份价值。PSG/LGPA 的历史性能资产仍可保留，
但 pose-state 线不能承担新论文的第二贡献。若继续新故事，应更换可观测信号或问题对象，而非
继续围绕同一单图 2D heatmap 增加函数复杂度。

## 2026-07-16：TAPF 提供新的正向故事候选，但主张仍处于归因阶段

exp378不再把外部ViTPose heatmap直接当作永久控制输入，而是在Swin Stage-2特征上bootstrap
一个轻量内生姿态场，推理期完全不读取外部pose，再用该场控制后续Stage-3 PSG。seed 1234
同机final中，residual-OFF的hard F0/MR-F0分别为`55.9/56.0 mAP`，相对B0=
`+0.8/+0.9`；当前geometry residual在hard/relax下均为`-0.3 mAP`，relaxation本身只有
描述性`+0.1 mAP`。R5/R10相对B0略降，因此当前信号集中在mAP，不能写成全面提升。

这使论文故事从“PSG是一个`x*(1+heatmap)`门控”向一个更完整、可部署的对象移动：训练期用
外部姿态建立解剖坐标，后期/推理由视觉特征自行产生姿态场，姿态场作为受限中间状态控制后续
视觉更新。与exp374–377失败的外部实例heatmap动态调制不同，这里的正信号来自
residual-OFF内生姿态场配置；当前没有证据支持把geometry residual或SGD relaxation写成贡献。

此时只能写成**候选叙事**，不能提前写摘要结论。F0仍同时包含bootstrap、anchor容量、Gaussian
renderer与PSG；单seed也不能证明稳定性或正确关节语义。必须先补同机D0/J0/R0与最小Gate B：
RG0隔离raw teacher和Gaussian renderer，N0/置换bootstrap隔离正确pose监督与普通17通道容量，
并对最佳residual-OFF checkpoint做joint/confidence permutation、错误姿态退化、teacher
agreement、flip equivariance与geometry-only ID probe。

若这些对照闭合后正信号仍成立，主贡献候选升级为Progressive Anatomical Field：一套17关节
姿态状态跨视觉层级传递，每层只作可靠性有界的坐标/尺度修正，并重渲染后控制下一层视觉更新；
共享decoder加stage-specific投影，避免把多个独立pose head或多层PSG堆叠包装成新意。随后用
ResNet-50验证backbone-agnostic，并把同一状态对象迁移到Video ReID：跨帧汇聚关节可靠性、
运动连续性和遮挡恢复，和普通时序pooling、逐帧TAPF、外部pose smoothing做强对照。

潜在论文headline应围绕“跨层/跨帧递进修正的可靠性有界内生姿态状态”，而不是“首次在backbone
中做pose gating”或“多层加入PSG”。在hierarchical pose estimation、multi-stage
pose-guided ReID、recurrent pose refinement、feature modulation与video pose-ReID专项查新
完成前，只把它列为B类潜力路线，不宣称新颖性已经成立。

## 2026-07-17：TAPF候选叙事经语义审计收紧，当前版本不能作为姿态方法正结论

D0 e90冻结审计严格复现`56.2984/67.6471/79.8190/83.5294`，external pose的correct、shuffle、
None和不可索引sentinel与correct descriptor逐位相同，确认部署态确实不依赖外部ViTPose。这一
部署优点成立，但姿态因果部分没有成立：matched wrong field、joint/confidence permutation、
spatial constant和zero field相对correct的mAP绝对差都小于`0.1`个百分点；真正旁路PSG则降到
`53.6154/63.9367/76.2896/80.4525`，四项下降`2.6829/3.7104/3.5294/3.0769`个百分点。

anchor同时保留不错的teacher agreement、flip equivariance与17通道占用，说明失败不是“完全没
学出pose-like field”，而是下游PSG没有因果使用这些语义。当前`+0.8/+0.9 mAP`只能归到训练后的
PSG模块/容量性重标定，不能写成姿态场贡献、解剖推理或Progressive Anatomical Field的先导成功。
这也解释了为什么N0正确通道置换、R0/RG0 renderer差异和D0/J0 geometry residual都没有形成独立
mAP贡献。

论文材料据此执行以下边界：

1. 当前单锚点TAPF不进入摘要主贡献，不补multi-seed、ResNet或Video迁移；部署态external-free只作
   工程性质，不单独包装为创新；
2. PSG可继续作为性能组件或强基线，但必须称为模块收益，不能称为instance-specific pose reasoning；
3. Hierarchical/Video方向只有在新consumer满足null field严格identity、parameter-matched static/RGB
   control、以及逐层correct-vs-matched/constant因果门禁后才能重新进入论文主线；
4. B类故事若重建，headline应是“可空分离且逐层可干预的内生结构状态”，而不是“逐层加入PSG”或
   “中间热图看起来像姿态”。时序姿态可靠性仍可作为后续统一扩展，但不能先于单图因果成立。

所以exp378的价值转为一次关键归因：它证明了外部pose-free内生场可以稳定训练，也证明了视觉上
pose-like的中间变量并不自动构成检索因果机制。下一版本必须从consumer定义上消除容量混淆，而不是
继续给当前PSG增加层数、backbone或视频数据。

## 2026-07-17：论文主对象修正为完整模块，恢复逐层与跨backbone验证

用户进一步明确了论文对象：原始PSG在测试期需要额外姿态模型持续提供heatmap，而exp378 D0把
这项依赖移到了训练期，最终部署只输入RGB。因而anchor和PSG可以作为一个完整方法单元，不必把
`+1.1 mAP`强行拆成“anchor单独多少、PSG单独多少”。这不是忽略消融，而是把主问题从“某个
关节通道是否在冻结推理时独立因果”调整为“能否消除测试期外部姿态依赖并保留PSG性能”。

当前可写入故事的事实为：同机B0=`55.1/66.7/79.5/83.8`，D0=
`56.2/67.6/79.8/83.4`，external-pose R0=`56.1/67.4/79.5/83.7`。D0相对B0提升
`+1.1 mAP/+0.9 R1`，并基本匹配R0，却在推理期完全不读取pose。冻结语义审计仍要求删去“精确
关节语义被PSG因果使用”之类强表述，但不再把D0从方法候选中删除。

下一章候选是exp379 Progressive Hierarchical TAPF：

```text
Stage-1 RGB feature → internal field-1 → Stage-2 PSG
Stage-2 RGB feature + field-1 prior → refined field-2 → Stage-3 PSG
```

两个节点用stage-specific轻量投影和单个共享decoder，训练期持续pose supervision，测试期仍然
RGB-only。主比较是逐层HT0对单点D0，而不是只对B0报更大的总增益；旧external multi-stage PSG
失败作为重要反例，说明创新不能只是复制同一个heatmap。

Swin-T首轮跑满后，按同一接口迁移ResNet-50和合适ViT。若弱backbone上的增益更清楚，可以形成
“强人体预训练吸收部分结构先验，但内部层级pose supervision在不同归纳偏置下仍可迁移”的证据；
必须依靠每个backbone内部B0/D0/HT0三臂，不能拿跨backbone绝对数值支撑。之后再把层级状态扩展
到Video ReID，用跨帧可靠性、运动连续性和遮挡恢复形成时序版本。

当前最合适的headline候选是：**Progressive pose distillation for pose-free person ReID：把测试期
外部姿态模型压缩为跨视觉层递进的内部结构模块。** 是否足以投B类，取决于逐层增益、多seed、
ResNet/ViT迁移、计算开销和Video时序扩展，不能仅凭单seed `+1.1`提前下结论。

## 2026-07-17：exp379跑满后的故事校正——先保住完整方法，不虚写逐层增益

HT0已在Swin-T上跑满并通过完整审计，final=`56.1/67.6/79.9/83.4`；相对单点D0=
`-0.1/+0.0/+0.1/+0.0`，相对B0=`+1.0/+0.9/+0.4/-0.4`。因此目前不能在论文中写
“progressive hierarchical refinement显著优于single-stage”，也不能只展示对B0的正差来回避D0。

现阶段更诚实也更稳的主叙事是两层：

1. **已由当前Swin证据支撑的主对象**：训练期姿态监督学习内部anchor，并与PSG组成原子模块；
   推理期不运行姿态模型。单点D0相对B0为`+1.1 mAP/+0.9 R1`，且基本匹配测试期依赖ViTPose的
   R0。冻结语义审计只限制“精确关节通道被因果使用”的措辞，不否定模块级部署收益。
2. **已实现但尚未获得增益证据的扩展**：每个anchor对应一个PSG，浅层state调制中层、refined
   state再调制深层。HT0证明该链路可训练、RGB-only且不损伤D0，但Swin-T上目前只是中性扩展。

由此，暂定headline中的`Progressive`应理解为“把外部pose监督逐步蒸馏到内部视觉路径”，不能
提前专指“多stage一定更强”。ResNet-50与后续ViT必须分别在内部训练B0/D0/HT0三臂：若HT0跨
backbone稳定超过D0，再把hierarchical refinement升为核心贡献；若仍中性，则最终方法应收敛为
更简洁的单anchor+PSG，逐层版本只进消融或扩展讨论。

Video ReID与时序姿态仍是后续最有潜力的增量，因为它引入单图没有的真实信息——跨帧可见性、
运动连续性和遮挡恢复。它必须晚于backbone迁移，并与逐帧方法、普通temporal pooling、外部pose
smoothing及RGB-only video backbone强对照，才能把收益归给时序姿态而非一般视频容量。

## 2026-07-17：exp380补上ResNet正证据——完整方法跨骨干，逐层贡献仍需ViT定性

ResNet-50三臂均在同一ImageNet初始化、seed、batch与120-epoch配方下fresh串行完成：B0=
`35.0/45.3/61.3/68.2`，D0=`38.1/49.4/64.6/71.1`，HT0=
`38.9/50.5/65.9/72.0`。D0−B0=`+3.1/+4.1/+3.3/+2.9`，说明把训练期pose监督、内部
anchor和后继PSG作为原子方法后，收益并非SOLIDER Swin-T特例；推理仍不运行姿态模型。

逐层结果也首次出现明确正差：HT0−D0=`+0.8/+1.1/+1.3/+0.9`。它支持“每个anchor对应一个
后继PSG，浅层state进入下一视觉层并被更深语义refine”的设计在ResNet上有额外价值。但Swin-T
对应差值仍为`-0.1/+0.0/+0.1/+0.0`，所以论文不能写成“hierarchical在所有backbone稳定更强”。

当前故事应保持两级证据结构：

1. **主贡献候选已增强**：完整pose-supervised、pose-free `anchor+PSG`在Swin-T和ResNet-50
   内部都优于各自B0，核心部署叙事站住；冻结语义审计仍禁止夸大精确关节通道的独立因果作用。
2. **层级贡献候选由中性变为条件性正证据**：ResNet支持HT0>D0，Swin中性。它可以进入方法设计
   与主消融，但在ViT结果前仍不升格为跨架构普适headline。

因此下一实验是合适ViT的同骨干B0/D0/HT0判别，而不是继续优化ResNet。如果ViT再次支持逐层正差，
headline可明确指向progressive hierarchical pose distillation；如果ViT中性，则headline回到更稳的
“以训练期姿态监督消除测试期姿态模型”的完整模块，逐层版本作为架构条件性扩展。ViT闭合后再进入
Video ReID，把跨帧可靠性、运动连续性和遮挡恢复作为新的信息源，而非普通时序容量。

## 2026-07-17：exp381闭合三骨干证据——论文中心回到原子方法

ViT-B三臂final为B0=`52.9/59.5/77.1/82.0`、D0=`54.9/61.4/78.9/84.0`、HT0=
`54.6/60.6/78.4/84.1`。因此D0−B0=`+2.0/+1.9/+1.8/+2.0`，而HT0−D0=
`-0.3/-0.8/-0.5/+0.1`。结合Swin和ResNet，完整单anchor+PSG相对各自B0的mAP差为
`+1.1/+3.1/+2.0`；逐层HT0相对D0则为`-0.1/+0.8/-0.3`。

这组证据给论文故事一个明确收口：

1. **中心方法**：把外部pose作为训练期privileged supervision，学习内部anchor，并与后继PSG
   作为不可强拆的原子模块；推理期只输入RGB。三个不同骨干内部均获得描述性正差。
2. **不再作为核心的扩展**：多层anchor/refinement在ResNet有效，但在Swin和ViT不优于单层，
   只能进入架构条件性消融，不能写成普适progressive hierarchy贡献。
3. **机理边界**：冻结语义审计仍禁止声称精确关节名称或field空间结构具有独立检索因果贡献；
   ViT post-block11又证明consumer必须拥有通向最终descriptor的真实下游路径。
4. **证据边界**：当前跨骨干结果均为单seed探索证据，不等于统计显著；正文必须报告同骨干差值，
   不能比较Swin/ResNet/ViT绝对指标。

因此更准确的headline候选是：**Pose-privileged training for pose-free person ReID：在训练期使用
姿态监督学习内部结构调制，部署时移除外部姿态模型。** `Hierarchical`不再放进headline。

下一章若进入Video ReID，必须真正利用单图没有的信息：跨帧可见性、运动连续性和遮挡恢复。主表
至少包含同video backbone的RGB temporal B0、逐帧原子D0和时序pose T0；论文只有在T0直接超过D0
时才讨论时序姿态贡献。普通temporal pooling、更多帧或参数量带来的提升属于视频基线能力，不能
并入方法增益。测试期仍以RGB-only为目标，外部pose只在训练期作为teacher。

## 2026-07-17：exp382关闭Video分支，正文转向原子机制与跨域证据

Video专项查新给出了必须吸收的负结论：GAE-Net已覆盖训练期人体时序结构特权教师向RGB-only
视频学生的蒸馏，PAFormer覆盖pose-supervised、pose-free inference，KPRTrack覆盖tracklet同部位
聚合；成熟video ReID又已有遮挡/干扰memory与多粒度temporal modeling。当前没有足够差分把
temporal pose state写成新的独立贡献，且远端没有可用视频数据。因此Video TAPF不进入方法、主表
或摘要，只保留为未来应用扩展。

这也进一步收紧正文headline。不能只写“首次pose-privileged training for pose-free ReID”，因为
PGFL-KD、TSD、PAFormer等已占这个大类。更准确的机制表达应是：

**在训练期用姿态target学习backbone内部anchor/state，并让该状态通过PSG调制仍通向最终descriptor
的后继视觉特征；推理时整条路径由RGB内生，不调用外部姿态模型。**

现有证据可以支撑“跨三骨干描述性正mAP差”和“严格RGB-only parity”，但尚不能支撑跨数据集、
统计稳定或低开销主张。论文证据缺口已单列在
`paper_materials/tables/tapf_claim_evidence_gap.md`。下一必要实验是exp383：Market上fresh matched
B0/D0，同时报告Market域内与Occluded-ReID跨域final，并补参数/FLOPs/训练与推理成本。
Occluded-ReID没有训练split，因此只能写独立遮挡target，不能写第二训练集。

如果exp383两域方向支持，再对最终论文主骨干补必要seed；如果跨域非正，则不能用继续重复
Occluded-Duke seed掩盖数据集泛化缺口。Hierarchical和Video都不再消耗主实验预算。

## 2026-07-18：官方干净重启后的论文故事——原子方法小幅成立，层级版删除

用户要求从 SOLIDER 官方最后提交重新开始，旧 runtime 与旧 pose cache 不再作为论文主实现。
这轮 clean execution 先在 Market 复现 official B0=`91.6/96.3/98.7/99.2`，再建立
Occluded-Duke official B0=`57.4/67.4/80.6/85.2`。两个数据集的 pose 都只从 train RGB 用
ViTPose-H fresh 提取；query/gallery 从未建立 pose target。

当前可进入 clean 主表的结果是：

| 数据集 | B0 | D0 | D0−B0（mAP/R1/R5/R10） |
|---|---|---|---|
| Market-1501 | 91.6/96.3/98.7/99.2 | 92.0/96.5/98.8/99.3 | `+0.4/+0.2/+0.1/+0.1` |
| Occluded-Duke | 57.4/67.4/80.6/85.2 | 57.6/67.7/80.8/84.6 | `+0.2/+0.3/+0.2/−0.6` |

D0 是不可强拆的完整 `anchor+PSG`：训练期 anchor 接受 pose target，student field 在 handoff 后由
RGB feature 内生，两个后继 PSG 改变仍有最终 descriptor 下游路径的视觉特征；测试期不读取
external pose。correct/shuffle/None/exploding 输入的 descriptor/field/gate 逐元素 exact，说明
“pose-free inference”是执行事实。参数约 `+0.375%`，supported-op FLOPs 约 `+0.242%`。

这组证据只能支撑克制表述：**同一轻量原子模块在两个训练域得到小幅 mAP 正差，并在部署时完全
删除姿态模型。** 它不能支撑“稳定四项提升”“统计显著”“精确关节语义具有独立检索因果作用”，
也不能继续把旧 runtime 的三骨干 `+1.1/+3.1/+2.0 mAP`直接并入 clean 主表。旧结果可在研究历程
或补充材料中说明，但当前论文数字应以官方干净实现为主。

clean hierarchical HT0=`56.9/65.9/80.0/84.1`，相对 D0=
`−0.7/−1.8/−0.8/−0.5`。六个 early 与两个 late PSG 全部学习且都能独立改变 final descriptor，
所以层级失败不是 dead consumer。`Hierarchical/Progressive`必须从标题、贡献和主方法中删除，只能
作为负消融说明更多层不自动受益。

MMAsia 当前最诚实的候选标题方向是：**Training-time pose targets for lightweight internal
modulation in pose-free person ReID**。但在效应只有 `+0.2/+0.4 mAP`且单 seed 的情况下，标题仍是
候选而非定稿。下一项必须是 clean Occ-Duke matched B0/D0 多 seed，报告逐 seed差值和 mean/std。
只有均值方向与方差支持，才继续补 clean 跨骨干和强先例差分；若不支持，就应降级当前 headline，
而不是复活 Video、hierarchical 或挑选中途 best。

## 2026-07-18：exp390三seed闭合——原子TAPF保留为克制的mAP-only证据

official clean Occluded-Duke三seed paired结果如下：

| seed | B0 | D0 | D0−B0（mAP/R1/R5/R10） |
|---:|---|---|---|
| 1234 | `57.4/67.4/80.6/85.2` | `57.6/67.7/80.8/84.6` | `+0.2/+0.3/+0.2/−0.6` |
| 4321 | `56.0/66.2/79.4/83.8` | `56.8/66.5/79.9/84.3` | `+0.8/+0.3/+0.5/+0.5` |
| 2025 | `57.5/67.9/81.1/85.7` | `57.9/67.0/80.4/85.2` | `+0.4/−0.9/−0.7/−0.5` |

paired mean±sample std为`+0.47±0.31/−0.10±0.69/+0.00±0.62/−0.20±0.61`。这允许正文写：
**TAPF在三个matched seed上都提高mAP，平均约+0.47；推理期仍为严格RGB-only。** 但正文必须紧接
着说明：rank指标没有稳定提升，效应幅度小，当前证据不等于统计显著、跨架构普适或精确关节语义
因果。最诚实的定位是“training-time pose targets带来的轻量内部调制，在mAP上小幅可重复”。

exp389层级版继续是负消融，而不是被新seed结果推翻。后续全stage探索只有在严格分解loss budget、
consumer balance和Stage-0 route后才有解释价值；因此exp391按H2-M→H2-B→H3-OFF/ON推进。
如果前两阶段失败，论文保留单层D0并把“更多stage不自动更好”写入消融；如果matched H3-ON真正
超过H3-OFF且至少两个stage有独立下游贡献，才重新讨论多阶段主方法。CLIP语义校准不与这轮结构
验证混跑，避免把语义teacher收益误归到stage topology。

## 2026-07-18：exp391封板——loss归一恢复层级路线，但不足以改写主方法

exp391 Phase A只把exp389的两层pose objective从sum改为mean，H2-M final=
`57.2/67.3/80.2/84.5`。它相对HT0为`+0.3/+1.4/+0.2/+0.4`，说明原sum确有额外优化惩罚；但
相对单层D0仍为`−0.4/−0.4/−0.6/−0.1`，触发预注册NO-GO。因此Phase B/C不进入实现或训练，
正文也不展示中途best来替代唯一final。

冻结层级消融补上了必要的机制边界：full−early-bypass的mAP为`+0.141`，early route不是dead；
full−late-bypass为`+1.546 mAP`，late仍是主要贡献。八个consumer逐一旁路都能改变descriptor，
全部参数轨迹与pose-free终审通过。所以最准确的论文表述是：**多阶段内部状态可以被真实执行，
loss预算归一可恢复性能，但当前6/2 topology仍未超过更简洁的单层D0。** 这是一项负消融，不是
progressive headline。

论文主证据仍是exp390三seed的原子TAPF mAP-only弱GO：`+0.47±0.31 mAP`、三个seed同向，rank
均值不正。下一创新候选转向joint-channel语义不可辨识，而不是继续增加stage。CLIP语义校准若要
进入新设计，必须由冻结image+text双编码器构造实例级局部语义teacher，只监督各stage内部anchor，
不直接蒸馏final descriptor，并在推理时完全移除CLIP、文本和external pose。它首先是只读查新与
机制审查对象；在能证明correct、channel-shuffle和wrong-field真正可分之前，不写入论文贡献或
启动正式实验。这不永久否定多阶段：语义门禁若成立，应以semantic single-stage为新基线，再独立
验证semantic multi-stage是否带来额外收益；不得把该新机制误记为exp391 Phase B/C续跑。

## 2026-07-18：Phase 0A给出新的论文问题证据——有效调制不等于解剖语义

exp392对clean D0 final做了不训练参数的内部field反事实。all-PSG bypass使mAP从`57.559`降到
`56.200`，两个PSG分别旁路也各降约`0.68/0.71 mAP`，所以内部调制路径是真实且有条件性价值的。
但是把17个joint channels循环置换后mAP为`57.583`，换成同camera、不同PID且field统计匹配的
wrong field后仍为`57.554`；两者都与correct近乎相同。更进一步，保留每通道均值、删除所有空间
geometry的constant field达到`57.905`，比correct高`0.346 mAP`。

因此clean论文不能再暗示“17个internal channels已经对应17个具体人体关节并据此改善检索”。现有
证据支持的是更弱但准确的说法：训练期pose target学习出一个有效的RGB内生条件调制器；它的mAP在
三seed上小幅可重复，但内部anatomical semantics尚不可辨识。冻结bypass的`−1.359`也不能替代
D0−B0的matched训练差，只解释已训练模型内部依赖。

这使下一方法贡献有了明确问题对象：**counterfactually identifiable executable anatomical
mediator**。候选方法必须让正确geometry-semantic binding优于channel shuffle、matched wrong、
spatial constant与generic adapter；CLIP只作为训练期双编码teacher校准内部state，不直接蒸馏global
descriptor。Phase 0B先判断该teacher是否真依赖当前RGB、pose mask和text，而不是固定part label。
若teacher失败，保留原子TAPF的克制mAP-only故事；若teacher与后续router都通过，才把论文中心升级
为“从generic pose-conditioned modulation到可辨识anatomical mediation”，并重新评估balanced
semantic multi-stage。

## 2026-07-18：Phase 0B封板——问题不在CLIP是否“强”，而在局部读出是否真的text-aligned

预注册的naive teacher把pose mask池化的ViT-L/14末层patch token直接与五类body-part text prototype
比较。全15,618图correct top-1只有square=`2.692%`、letterbox=`4.637%`，显著低于20% chance；
shuffle与wrong-text反而达到`15–34%`。因此该teacher不能进入任何训练，更不能用CLIP名称为当前
TAPF语义故事背书。

终审没有停在“CLIP不适合”。hook与OpenCLIP官方token输出逐元素exact，pose mask上下顺序正确，
bicubic也不能修复。关键对照是：同一pose region若改成tight crop、重新走CLIP真正受全局对比目标
监督的CLS，macro top-1升到`44.688%`；而原patch feature做image-only cluster也有`52.8–60.0%`
region结构。这说明patch含人体局部信息，却没有被命名到正确text轴。

论文叙事因此暂时停在诊断而非方法结果：**有效的局部视觉结构不等于text-aligned局部语义，正如
有效的TAPF调制不等于可辨识joint semantics。** 两个失败具有同一逻辑：不能根据模块名称解释内部
变量，必须用反事实证明绑定。

下一候选不再直接pool raw patch token，而是让pose region通过CLIP受监督CLS readout生成teacher：
共享早期trunk，在后段若干block限制CLS读取对应region，或研究可缓存的region-crop global CLS。
同时把固定anatomical slot identity（pose给出）与slot内appearance/support distribution（CLIP给出）
拆开。只有新的teacher-only门禁先证明correct优于shuffle/wrong RGB/wrong mask/wrong text，才恢复
Phase 0C和semantic single-stage；多阶段仍排在单阶段因果成立之后。

## 2026-07-19：Semantic C0不进入正面方法结果，但保留为CLIP耦合的关键负归因

Phase 0B2修复了naive raw-patch teacher最明显的接口错误：hard-owner固定解剖slot identity，
PC-MBCLS沿CLIP受监督CLS路径读取每slot局部support，并在128图遮挡反事实中表现出稳定的
sample-specific单调响应。随后首次single-stage Semantic TAPF按完整e120运行，final=
`56.9/67.1/80.6/85.0`，相对clean D0=`−0.7/−0.6/−0.2/+0.4`。因此它不能进入主表作为正增益，
也不能把相对HT0的rank优势包装成CLIP有效。

这次失败的机制证据比单个分数更重要。checkpoint严格有限且完全不含teacher；两个semantic router
都能改变final descriptor，NULL identity与RGB-only exact成立，所以不是实现没接上。真正的弱点是
CLIP support经过标量q readout后只剩mean约`0.512`、混合五slot pooled std约`0.0169`，q loss维持
`0.692`，router更新也只有`10^-6–10^-5`量级。当前模型实际主要学到了coarse mask/presence，而不是足以区别
static prior的强sample-specific语义证据。

因此论文当前仍以exp390原子TAPF的三seed mAP-only弱GO为正面边界，不新增“CLIP语义提升”贡献。
exp392可作为方法动机和负归因链：

1. 原D0 consumer有效，但joint identity/geometry不可辨识；
2. raw CLIP patch含局部结构，却未对齐text轴；
3. CLS路径能恢复局部响应，但绝对support标量动态过窄，尚未改善ReID final。

若后续继续CLIP深耦合，正文可争的升级点不是普通KD，而是把局部视觉—语言证据改造成**相对化、
非退化、可执行且可反事实验证的anatomical mediator**。下一步先用static-q、pose-only、generic-router
和wrong binding拆清当前checkpoint；只有新的single-stage correct arm同时超过这些强对照与clean D0，
才重写主故事并重新授权semantic multi-stage。当前Semantic C0的NO-GO只关闭本组合，不是对
CLIP–TAPF的永久否定。

## 2026-07-19：Phase 0D证明当前semantic route是“数值可达、检索失活”

冻结全验证集反事实给出了比final负差更精确的解释。五slot同slot跨图q std实际只有
`0.00009–0.00029`；把q换成slot均值或全1、把mask变成空间常量、循环错配slot、把五expert取均值，
乃至旁路全部router，mAP变化绝对值都小于`0.0007`，R1/R5/R10完全不变。correct start/end descriptor
与state SHA exact，排除了评测漂移。

这要求论文区分两种“有效”：final checkpoint里两个consumer置零会产生非零descriptor L2，说明代码
路径和参数不是dead；但这种差异不足以改变检索排序，因此不能称为有检索贡献。当前Semantic C0的
route对final几乎是identity，D0差值也不能归因于CLIP语义执行。

实现层面的关键断点是：CLIP loss监督anchor state，而state在router前detach；router只从global ReID
loss学习，并以zero expert启动。结果是CLIP语义在输入端存在，却没有直接拥有执行残差的梯度，形成
“拓扑深耦合、优化浅耦合”。后续故事若要升级，必须让centered CLIP局部视觉residual监督内部
router latent/delta，同时保持不直接蒸馏final descriptor和推理RGB-only。新的正面证据至少要同时
包括correct-vs-wrong evidence差、all-router-bypass final贡献和clean D0提升；否则exp392只保留为
诚实的失败诊断，不进入方法贡献。

## 2026-07-19：exp393是候选修复，不提前写成正面贡献

下一候选COER把证据链拆为两个独立问题。RZ-C0先不改变现有teacher、mask或q，只用nonzero branch与
zero ReZero scalar验证identity-safe route能否在final检索中留下`all-bypass`可见贡献。rich evidence
再把PC-MBCLS region CLS相对同图global与slot prior中心化为16维code，以RZ-C0为直接对照进入同一
router。teacher code失败只关闭该code，RZ route失败只关闭该route接口；不再用单一门禁否定整条
CLIP–TAPF方向。

“深耦合”的论文定义也被收紧：CLIP evidence不能只停在anchor head或pre-expert latent，它必须控制
推理保留的生产branch，并让internal relation loss的梯度实际到达token/context/evidence projection与
expert；另一方面ReZero alpha只由ReID loss打开，防止辅助loss制造大残差却不改善检索。只有final
同时满足clean D0提升、correct-vs-wrong/static差和all-router-bypass贡献，COER才有资格进入正面方法
故事。在此之前，论文主证据仍是exp390原子TAPF的三seed mAP-only弱GO，exp393只记录为预注册候选。

0E-S/C8/128已经提供了teacher侧的首段正证据：centered rich local residual在64个held-out PID上五slot
macro effective rank=`11.050/16`，且correct↔flip相对wrong RGB和same-RGB wrong mask的逐slot
PID-cluster CI均严格为正。slot-mean/global-only exact zero、raw uncentered较弱，说明中心化后的局部
残差不是exp392那种几乎固定的slot prior。

这仍不能进入正面方法表。fixed random orthogonal也保留强信号，故PCA只是压缩器；更关键的是teacher
richness没有证明生产route会改变检索。只有0E-FULL复核、独立RZ-C0 route activation与最终COER三段
证据连续成立，才能把故事从“失败归因与候选修复”升级为“CLIP-owned executable residual”。

0E-FULL已完成teacher侧全量封板：official 15,618图按PID严格fit/audit，341个held-out PID上的五slot
macro effective rank=`12.335/16`，wrong RGB与wrong mask的逐slot置信区间均严格为正。由此，论文
可以把“rich centered CLIP local residual真实存在”写成机制开发证据，而不再只依赖128图smoke。

但它仍不进入方法主结果，因为没有任何ReID训练或检索提升。故事下一段必须由Phase A证明
identity-safe route可被ReID loss打开，再由Phase B证明correct evidence相对wrong/static控制和
all-bypass final贡献同时成立；少任一段，COER仍只是被充分审计的候选机制。

## 2026-07-19：Phase A负结果把COER故事从ReZero修复推进到执行所有权

RZ-C0自然e120 final=`56.8/66.8/79.6/83.9`，并通过strict finite、teacher-free、RGB-only、NULL
identity及全部router参数轨迹审计；但all-router-bypass得到完全相同的一位小数四项，raw
full−bypass只有`-0.000249709 mAP point`。因此不能把“alpha和expert更新过”写成route对检索有贡献。

这项负结果与Phase 0E全量正证据并不矛盾：CLIP rich local residual真实存在，但当前执行幅度仍由只受
ReID loss的自由ReZero scalar控制，最终被压到`1e-4`，使production residual对排序近似identity。
论文主结果仍停留在exp390原子TAPF的mAP-only弱GO；exp393只作为机制诊断链，不进入正面方法表。

若继续CLIP深耦合，下一故事必须证明三件事同时成立：rich evidence控制推理保留branch的方向；执行预算
不会静默塌零；correct evidence相对wrong/static/generic与all-bypass产生可辨识检索差并不低于clean
D0。任何“固定非零scale”只能是接口条件，不能自身充当创新或成功证据。

## 2026-07-19：exp394目前只闭合“可执行实现”，尚未产生方法结果

证据预算化rich residual已通过独立CPU exact contract：旧四类TAPF默认路径不变，新接口在rho=0时
exact identity，NULL slot exact zero，evidence/mask/`L_exec`/ReID的梯度所有权互不越界，且teacher、
CLIP、codebook不进入checkpoint state。这说明“rich evidence拥有生产expert方向、固定预算只负责防
静默塌零”的实现可以被严格落地，而不是概念草图。

论文叙事仍不能把它写成正面贡献，因为目前没有真实AMP训练、final retrieval或wrong/static/generic
反事实结果。固定rho、RMS normalization、PCA-16和local CLIP distillation都只是接口部件。下一证据
必须先通过真实batch64 CUDA preflight，再由唯一fresh e120同时满足clean D0、all-bypass与semantic
counterfactual门槛；在此之前，主结果仍是exp390的mAP-only弱GO，exp394只属于候选机制开发链。

## 2026-07-19：exp394止于真实AMP首步，不进入训练或方法结果

production CPU contract曾证明rho schedule、NULL identity和四类梯度所有权都能按设计成立；但唯一
actual batch64 CUDA/AMP门在step 1 unscale后检出non-finite model gradient，并在optimizer update前
退出。成功更新为`0/24`，没有checkpoint、handoff、retrieval或counterfactual结果。因此论文不能把
exp394写成“已训练但效果未知”，更不能用CPU PASS暗示它具备真实可训练性。

这条负证据把故事边界进一步收紧：rich CLIP evidence全量稳定存在，固定预算代数也可执行，但当前
production computation graph尚未通过AMP finite门。由于失败资产没有定位具体parameter组，正文只应
写成当前接口的数值执行失败，不应猜测某个loss是唯一原因。论文正面边界仍是exp390 mAP-only弱GO；
exp394作为机制开发负证据，说明“梯度可达、预算非零、CPU exact”三者仍不足以构成可训练的
CLIP-owned mediator。

## 2026-07-19：exp395目前只建立AMP归因方法，不改变论文结果边界

为避免把exp394首步FAIL事后归到一个方便的loss，exp395预先冻结D0 baseline与rich图的逐loss、逐参数组
scaled/unscaled归因矩阵，并禁止任何optimizer update。CPU contract已证明reporter对11个loss、15组、
NaN/±Inf和动态范围的统计正确，但没有读取official batch或运行CUDA。

因此论文故事不新增“已找到AMP根因”或“已修复rich route”的内容。正面结果仍只有exp390的mAP-only
弱GO；Phase0E rich evidence、exp393 route FAIL、exp394 AMP FAIL组成机制开发证据链。只有未来独立
actual归因门先给出可复核支持子图，再由新的AMP-stable机制通过训练与检索反事实，才可能把这条链升级
为方法贡献。

归因脚本的静态门现已通过，但它仍是“测量工具已就绪”，不是“AMP根因已找到”。论文正文不报告
static gate为方法结果，也不据此恢复exp394训练。只有独立actual matrix在零更新条件下完成并通过
state/RNG/asset终审，才允许把数值失败从宽泛production图进一步收紧；在那之前正面story完全不变。

## 2026-07-19：exp395 actual没有给出AMP根因，只封板了reporter规模失败

唯一actual越过了fresh source、canonical runtime、official batch64与teacher target前置阶段，但在第一
行D0 `reid`的scaled梯度统计中，backbone组触发canonical `torch.quantile`大输入限制。异常发生在
unscale之前，D0与rich矩阵均未完成；因此不能把这次失败写成“shared D0不稳定”、也不能写成“rich
auxiliary已被定位”。

论文结果边界完全不变：exp394仍只证明当前production AMP接口首步失败，exp395只证明第一版测量器没有
覆盖真实参数规模。optimizer update与checkpoint均为0，不存在训练或检索结果。若继续，只能由独立
exp396先证明大张量reporter可执行，再在zero-update门内获得实际支持矩阵；任何后续机制结论仍需另立
实验，不能把诊断器工程修正当成方法贡献。

## 2026-07-19：exp396排除了“rich auxiliary导致首步AMP失败”的叙事

chunk-safe reporter完成了matched D0与rich的全矩阵。两者ReID loss相同，且`reid/total`只在backbone
出现完全相同的368 NaN、3,753 `+Inf`和4,183 `-Inf`；D0 pose与rich的mask、presence、evidence、
两个exec consumer及pose全部finite。这把exp394的宽泛失败收紧为shared ReID backbone现象，而不是
CLIP teacher或rich route新增梯度的特有失败。

论文仍不能写“production已经可训练”：exp394按原门确实FAIL，exp396也没有optimizer update或检索
结果。但故事中的负结论需要修正为“绝对首步finite门未经D0校准”。下一证据必须让D0与rich都使用
canonical GradScaler默认初值和自然skip/update，比较动态scale轨迹；不能手工降scale，也不能把D0同样
发生的overflow包装成rich方法缺陷。正面方法边界仍停留在exp390 mAP-only弱GO。
