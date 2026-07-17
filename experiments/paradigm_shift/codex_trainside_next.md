Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019f0b5f-e4f0-7333-9284-53f453d2e9f9
--------
user
调研: 钉死训练侧+多方向证伪后, ReID 训练侧还有哪些未探的范式级创新方向? 已死区(必避): occluded 内部/AG 视频/DG foundation-preserving/frozen-prior 弱(fine-tune 王道)/gallery(B/C/OSAC/Hubness/Rank-Regret)/open-set/SMPL 几何/FM-import(MLLM-DINO-SD)/test-time marginalization(LM-ReID)/single-support CVaR(worst 跨 camera/pose gap 不可训练改善, oracle-headroom 墙)。**六点定律(本 team 真 measure)**: 训练端塑形/对齐/分离 K 变体破坏 test-time marginalization 多样性(consistency/LCRS/LATS/LSRC/LRFD 全负, K-cos 升=坍缩)。问: 这些约束外, 还有哪些训练侧机制有真 novelty(过去没写出)+可消融+CCF-B 价值? 每方向: 机制一句/为何不撞死区/cheap kill-switch(per-query 必控 #false-in-topk)/CCF-B 分(1-10)/先例风险(web search)。排序 Top5。务实中文, 别套话。
codex
我会先读本仓库最新实验文档，确认你们已经证伪的边界和当前主线；然后用论文/代码源做快速先例排查，最后只给训练侧、可消融、能被 cheap kill-switch 快速打掉的 Top5。
exec
/bin/zsh -lc "sed -n '1,220p' experiments/decisions.md" in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc "sed -n '1,220p' experiments/paper_materials/story.md" in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc "sed -n '1,260p' experiments/innovation_brainstorm.md" in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc "sed -n '1,220p' experiments/results.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# 论文故事线（持续更新）

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

 succeeded in 0ms:
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


 succeeded in 0ms:
# 决策记录 — Phase 2: Pure Pose Heatmap

### [2026-03-07 19:30] 决策 #1

**上下文**: Phase 1 (33 个实验) 已充分探索了基于 ViTPose visibility 向量的所有方向。最佳训练端改进仅 +1.4%（GiLt+PCFC）。用户指示放弃 visibility 方向，转向纯 pose heatmap + mmpose 更鲁棒模型。

**选择**: 从 SOLIDER 作者原始代码重新开始，纯 pose heatmap 方向
**理由**:
1. Visibility 向量不够可靠（与 AP 相关性仅 0.237）
2. PCFC alpha suppression 限制了所有后续改进
3. 纯 pose 热图更原始、更可靠、更多论文验证过有效性
4. 从干净代码开始避免 33 个实验的代码污染

**执行计划**:
1. 先跑 baseline 确认可复现（预期 mAP ~56.6%）
2. 选择并部署 mmpose 模型进行 pose heatmap 提取
3. 设计新的 heatmap 利用方式

### [2026-03-07 19:22] 决策 #2

**上下文**: 用户补充了多条重要指导：
1. 姿态模型可以参与训练（不限于离线热图）
2. 尽可能做训练侧创新（NFC/RR 等 test-time 方法不够公平）
3. 可以大胆修改 backbone 中间层
4. 数据必须从 log 文件读取，不能凭记忆

**选择**: 以训练侧创新为核心方向
**理由**:
1. NFC/Re-ranking 等 test-time 方法所有 SOTA 都可以用，不算公平的对比
2. 训练侧创新才是论文的核心贡献
3. 用户允许 pose 模型参与训练 + 修改中间层 → 创新空间大大增加
4. 可以考虑：在线 pose 特征注入、pose-guided attention、pose 结构约束等

**潜在技术方向**:
- 冻结的 pose 模型提取特征 → 通过 cross-attention 注入 Swin 中间层
- Pose heatmap 作为 spatial attention bias 直接修改 window attention
- Pose 骨骼结构约束 part features 之间的关系

### [2026-03-09 11:55] 决策 #3

**上下文**: exp001 (Pose Part Pooling with sigmoid) 完成。结果：mAP 57.1% (+0.5%), R1 66.7% (+0.2%)。有效但提升有限。关键发现：id_part 收敛极慢（最终仍在 2.0 vs id_global 0.2），说明 sigmoid 热图在 12×4 分辨率的 soft attention pooling 不够 discriminative。

**选项**:
  A. 改进 part pooling：使用 spatial softmax 代替 sigmoid，增强热图峰值对比度
  B. 放弃 part pooling，转向 pose heatmap 作为 attention bias 注入 Swin backbone
  C. 使用更高分辨率的中间层特征（stage2: 24×8 而非 stage3: 12×4）

**选择**: 先试 A（spatial softmax 改进），如果 id_part 收敛改善但最终结果仍有限，再转 B
**理由**:
1. exp001 证明 part pooling 方向有效（+0.5% mAP），但 id_part 是瓶颈
2. Spatial softmax 是最小改动，只改一行代码就能验证 "热图对比度不够" 的假设
3. 如果 id_part 收敛问题解决，part pooling 可能有更大提升空间
4. 如果 A 验证后仍不够，则 B 是完全不同的方向，有更大的创新性

**执行结果**: exp002 结果 mAP 57.2% vs exp001 57.1%，几乎无差异。id_part 训练中期收敛更快但最终效果相同。**结论：归一化方式不是瓶颈，转向方案 B。**

### [2026-03-09 14:13] 决策 #4

**上下文**: exp001 和 exp002 结果对比完成。两种归一化方式（sigmoid vs spatial_softmax）效果几乎一致。特征模式消融发现 part-only > concat > global，说明 part 特征有效但融合方式有问题。

**关键发现**:
1. 两种 normalization 最终 part-only mAP 都是 57.5%（+0.9% vs baseline）
2. Concat 融合反而比 part-only 差（1/N scaling 稀释信号）
3. id_part 收敛慢不是 normalization 的问题，而是 12×4 分辨率下 part 区分度本身有限

**选项**:
  A. 改进 part pooling 的融合方式（如 learnable weights, attention-based fusion）
  B. 转向 pose heatmap attention bias 注入 Swin backbone 中间层
  C. 提高 part 特征图分辨率（使用 stage2 特征 24×8）

**选择**: A — 改进特征融合方式。Part 特征已被证明有效（+0.9% mAP），但融合方式拖累了整体效果。这是最直接的改进方向。

**理由**:
1. Part-only 已经超 baseline 0.9%，说明 part 学到了有用信息
2. 当前 concat 的 1/N scaling 太朴素，直接稀释了 part 信号
3. 改进融合方式是低风险高回报：不需要改 backbone，只需修改测试时的特征组合
4. 如果简单的融合改进有效，可以作为消融实验的重要证据
5. B 和 C 是更大的改动，作为备选

**具体方案**: exp003 — 移除 1/N scaling，等权拼接 global + parts；或测试只用 part-only 作为最终特征

**执行结果**: exp003 在 ep60 终止，mAP 50.2%（-6.4% vs baseline）。降低 global loss weight 严重伤害 backbone 特征质量。Part 分类器虽学得更快（id_part 2.08 vs exp001 ~3.3），但池化的 backbone 特征变差了。**结论：global 和 part 是共生关系，不能通过削弱 global 来强化 part。**

### [2026-03-09 15:32] 决策 #5

**上下文**: exp001-003 完成。核心发现：
1. Part pooling 有效（+0.9% mAP with part-only feature）
2. 归一化方式（sigmoid vs spatial_softmax）无差异
3. 融合方式：part-only > concat（1/N scaling 有害）
4. 降低 global weight 反而伤害 part（因为 backbone 质量下降）
5. id_part 收敛极慢是核心瓶颈（id_part≈2.0 vs id_global≈0.2）

**关键问题**: 如何在不削弱 backbone 的前提下增强 part 学习？

**选项**:
  A. 独立 Part BN+分类器 + 更高 LR（加速 part 收敛，不改 global loss weight）
  B. 转向 Direction B：Pose heatmap 作为 attention bias 注入 Swin backbone 中间层（全新方向）
  C. Part feature 使用更高分辨率特征图（stage2: 24×8 而非 stage3: 12×4）
  D. 在 Part head 加入额外的 self-attention 层增强 part 特征表达
  E. 改进 Part 学习信号：per-part triplet loss (GiLt) + part-specific augmentation

**选择**: E — Per-part triplet loss (GiLt)
**理由**:
1. Phase 1 中 GiLt 已证明有效（+0.5% on top of PCFC），这次在 pure heatmap 框架下重试
2. 当前 part triplet 是"所有 part 特征拼起来做一个 triplet"，每个 part 没有独立的 hard positive/negative mining
3. Per-part triplet 让每个 part 独立学习判别性特征，直接解决 id_part 收敛慢的问题
4. 最小改动：只改 loss 计算方式，不改 backbone 或 part pooling 模块
5. 如果 GiLt 有效，可以作为消融实验的重要证据（"per-part triplet vs global triplet"）

**方案 B 作为备选**：如果 GiLt + part pooling 组合无法超过 +1.5% mAP，则转向全新的 backbone attention 方向。

**执行结果**: exp004 PFM 是中性结果。mAP 与 exp001 part-only 相同（57.5%），R1 反而下降 0.8%。PFM 加速收敛但不改善最终表征。**结论：不要在同一处重复使用 pose 信息（PFM+part pooling 是冗余的）。**

### [2026-03-09 17:52] 决策 #6

**上下文**: exp001-004 已探索了当前 part pooling 架构的多个变体：
- exp001/002: 不同热图归一化（sigmoid vs spatial_softmax）→ 无差异
- exp003: 改变 loss 权重 → 负面
- exp004: 加 PFM feature modulation → 中性

当前最佳：mAP 57.5% (part-only), R1 67.1% (+0.9%/+0.6% vs baseline)

**核心瓶颈**: id_part ≈ 2.0 无法进一步降低，part 特征质量受限于 12×4 分辨率。

**选项**:
  A. 使用 stage 2 特征 (24×8, 384ch) 做 part pooling — 4× spatial resolution
  B. Part diversity loss — 惩罚 part 特征间的相似度
  C. 转向 backbone attention 注入 — 修改 Swin 中间层
  D. Part-specific data augmentation — 基于 pose 的部位级数据增强
  E. Adaptive global-part fusion — 学习动态融合权重

**选择**: A — 使用 stage 2 高分辨率特征做 part pooling

**理由**:
1. 当前 12×4 分辨率对 5 个 part 来说太粗（每个 part 只能覆盖 2-3 个 spatial position）
2. Stage 2 (24×8 = 192 positions) 提供 4× 空间分辨率，pose heatmap attention 可以更精确
3. 384 channels 虽然比 768 少，但仍然有丰富的语义信息
4. 实验简单：只需改一下 part pooling 使用的特征图来源
5. 如果分辨率是瓶颈，这个实验会看到 id_part 明显改善

**风险**: Stage 2 特征可能不够 semantic（还没经过 stage 3 的进一步抽象）

**执行结果**: exp005 明确负面。ep40 mAP 仅 37.0%（baseline 56.6%）。id_part 到 ep49 才降到 4.70（exp001 同期 ~2.0）。**确认：Stage 2 特征语义不足以支撑 part-level identity 分类。** 更高空间分辨率无法补偿语义信息的缺失。

### [2026-03-09 19:10] 决策 #7

**上下文**: exp005 证明浅层（stage 2）特征不够 semantic。exp001-005 总结：
- Part pooling 方向的上限在 +0.9% mAP（part-only mode）
- Fusion 方式（concat_scaled vs equal_concat）对最终结果影响约 0.4%
- PFM 是冗余的；stage 2 太浅
- id_part 收敛始终慢于 id_global

**核心问题**：如何突破 +0.9% 的瓶颈？

**选项**:
  A. 改进 test-time 融合（L2-norm concat）— 可能挤出 0.2-0.5%，但不需要训练
  B. 转向 backbone attention 注入 — pose 信息参与特征形成过程
  C. 多尺度 part pooling — stage 2 spatial + stage 3 semantic 的融合
  D. Part feature diversity loss — 惩罚 part 间的冗余
  E. Pose-guided token selection — 剔除无关 token 提高效率

**选择**: 先做 A（test-time L2-norm fusion，零成本验证），然后转 B（backbone attention）

**理由**:
1. A 不需要训练，5 分钟可验证，如果能把 +0.9% 提升到 +1.2%，对论文有价值
2. B 是全新方向，改变特征形成过程本身，可能突破当前 part pooling 的上限
3. C-E 仍在 part pooling 框架内优化，上限有限
4. B 的创新性更好（"pose-conditioned attention" vs "better pooling"），更适合论文

**执行结果**:
- exp006 (A): L2-norm concat 57.4% vs concat 57.2%，小改进但仍不如 part-only (57.5%)。融合方向上限已到。
- **exp007 (B): PSG backbone injection → mAP 58.3%, R1 67.9%。Phase 2 最佳结果！+1.7% mAP, +1.4% R1。超过 Phase 1 最佳 (58.0%/68.0%)。**

### [2026-03-09 21:25] 决策 #8

**上下文**: exp007 PSG 取得突破性结果 (58.3%/67.9%)。关键发现：
1. Backbone-level pose injection (+1.7%) 远优于 post-hoc part pooling (+0.9%)
2. 纯 global feature，无需 part branch，架构极简
3. 额外参数仅 102K（两个 PSG 模块），几乎不增加计算量
4. 已超过 Phase 1 最佳

**下一步方向**:
  A. PSG + Part Pooling 组合 — 让 backbone 和 part branch 同时利用 pose
  B. PSG 消融实验 — 证明 PSG 每个组件的必要性
  C. PSG 在不同 stage 注入 — Stage 2 vs Stage 3 vs 全部 stages
  D. PSG 超参数分析 — hidden_dim, 是否 sigmoid, etc.

**选择**: A — PSG + Part Pooling 组合

**理由**:
1. PSG global feat (58.3%) 和 part-only feat (57.5%) 都有各自的优势
2. PSG 改善了 backbone 特征质量 → part features 也应该受益
3. 组合后可能进一步提升（PSG backbone + enhanced part features）
4. 如果组合有效，这就是完整的方法（backbone injection + part pooling = 全方位 pose 利用）

**执行结果**: exp008 mAP 57.7%, R1 66.0%。**组合不叠加**，低于 PSG-only (58.3%/67.9%)。Part pooling 的 part_only 测试模式丢弃了 PSG 增强的 global feature，而 part features 本身无法匹配 PSG-global 的质量。**结论：backbone-level injection 是更有效的 pose 利用方式，post-hoc pooling 在 PSG 基础上没有增量价值。**

### [2026-03-09 23:35] 决策 #9

**上下文**: exp007 (PSG) 和 exp008 (PSG+Part) 的对比揭示了重要规律：
1. PSG backbone injection: mAP 58.3% (+1.7%) — 全局特征，无 part branch
2. PSG + Part Pooling: mAP 57.7% (+1.1%) — part_only 测试，丢弃 global
3. Part Pooling alone: mAP 57.5% (+0.9%) — exp001

**核心洞察**:
- PSG 的增益主要来自改善全局特征质量，而 part pooling 依赖的是局部特征
- 两种方法的增益来源有重叠：都利用 pose heatmap 做 spatial attention
- 在 part_only 测试模式下，PSG 增强的 global 特征被浪费了

**选项**:
  A. PSG + concat 融合 — 保留 PSG global + part features，不丢弃全局特征
  B. 多 stage PSG — 在 Stage 2 也注入 PSG，更早引入 pose 先验
  C. PSG 改进 — 更强的 gate 机制（如 channel attention, multi-head gate）
  D. Backbone freeze warmup — 冻结 backbone 前 5 epochs，防止随机初始化模块破坏预训练

**选择**: 先做 B（多 stage PSG），这是架构级改进，有更大创新潜力

**理由**:
1. 当前 PSG 只在 Stage 3（2 个 block）注入，信息利用有限
2. 多 stage 注入可以让 pose 信息更早参与特征形成（Stage 2 的 24×8 分辨率对 pose heatmap 更有利）

 succeeded in 0ms:
# 实验结果总表 — Phase 2: Pure Pose Heatmap

## 数据集: Occluded-Duke

### 无后处理（纯模型结果）

> 注：本节默认记录各实验的**首次单 seed / 单 checkpoint**结果，用于保留搜索轨迹；是否能作为论文主结论，以文末 **4090 多种子验证** 为准。

| ID | 方法 | mAP | R-1 | R-5 | R-10 | vs Baseline | 备注 |
|----|------|-----|-----|-----|------|-------------|------|
| 000 | Baseline (SOLIDER-Swin-Tiny, SW=0.2) | 56.6% | 66.5% | 79.4% | 83.4% | — | 120 epoch, 完美复现 |
| 001 | + Pose Part Pooling (sigmoid, 5 parts) | 57.1% | 66.7% | 78.4% | 83.0% | mAP+0.5%, R1+0.2% | Part 分类器收敛慢(id_part≈2.0 vs id_global≈0.2) |
| 001* | ↳ part-only 特征 | **57.5%** | **67.1%** | 79.1% | 83.5% | mAP+0.9%, R1+0.6% | Part 单独使用反而最好 |
| 002 | + Pose Part Pooling (spatial_softmax, T=1.0) | 57.2% | 66.4% | 79.2% | 83.5% | mAP+0.6%, R1-0.1% | 与 sigmoid 几乎相同 |
| 002* | ↳ part-only 特征 | **57.5%** | 66.8% | 79.6% | 84.0% | mAP+0.9%, R1+0.3% | 再次确认 part 单独使用更好 |
| 003 | Part-Dominant (67% part loss, part-only test) | 50.2% | 59.1% | 73.7% | 78.7% | mAP-6.4%, R1-7.4% | ❌ 降低 global weight 伤 backbone, ep60 终止 |
| 004 | + PFM (pose feature modulation, part-only test) | 57.5% | 66.3% | 79.6% | 84.1% | mAP+0.9%, R1-0.2% | 🟡 mAP 同 001*, R1 差 0.8%. PFM 加速收敛但不改善最终结果 |
| 004-g | ↳ global 特征 | 57.4% | 66.1% | 79.6% | 83.9% | mAP+0.8%, R1-0.4% | Global 略好于 001-global (57.1%), PFM 帮助 global |
| 005 | Stage 2 Part Pooling (24×8, 384ch, part-only) | 37.0%* | 44.8%* | 59.1%* | 65.2%* | mAP-19.6% | ❌ ep40 数据, ep49 OOM 终止. Stage 2 语义不足 |
| 006 | L2-norm concat (exp001 model, test-only) | 57.4% | 66.9% | 78.9% | 83.5% | mAP+0.8% | 🟡 比 concat(57.2%) 好, 但不如 part-only(57.5%) |
| **007** | **Pose Spatial Gate in Backbone (PSG)** | **58.3%** | **67.9%** | **80.8%** | **84.9%** | **mAP+1.7%, R1+1.4%** | **✅ 3-seed mean = 57.83% / 67.13%，所有 seed 均优于 baseline，PSG 有效** |
| 008 | PSG + Part Pooling (part_only test) | 57.7% | 66.0% | 78.3% | 82.8% | mAP+1.1%, R1-0.5% | 🟡 组合不叠加, 低于 PSG-only. Part pooling 拖累全局特征 |
| 009 | Multi-stage PSG (Stage 2+3) | 58.3% | 67.2% | 81.2% | 85.2% | mAP+1.7%, R1+0.7% | 🟡 mAP 匹配 exp007, R1 略低(-0.7%), R5/R10 略优. 多 156K params 无显著收益 |
| 010 | PSG + Backbone Freeze 5ep | 12.5%* | 17.5%* | 30.4%* | 36.7%* | — | ❌ ep30 终止. 冻结 backbone 导致灾难性特征损坏 |
| 011 | PSG Stage 3 (200 epochs) | 58.3% | 67.6% | 81.1% | 85.3% | mAP+1.7%, R1+1.1% | 🟡 与 exp007(120ep) mAP 相同, 75% 更多训练时间无收益 |
| 012 | Pose Attention Bias (PAB, Stage 3) | 57.4% | 67.3% | 81.4% | 86.2% | mAP+0.8%, R1+0.8% | 🟡 有效但弱于 PSG. 仅 5.4K params. 证明 feature gate > attn bias |
| 013 | PSG + PAB Combo (Stage 3) | 57.6% | 67.2% | 81.3% | 84.4% | mAP+1.0%, R1+0.7% | ❌ 双重注入互相干扰, 不如 PSG-only(-0.7% mAP). PAB 拖累 PSG 收敛 |
| 014 | PSG + Part Supervision (global test) | 57.6% | 65.8% | 77.9% | 82.6% | mAP+1.0%, R1-0.7% | ❌ 用 exp008 checkpoint 直接验证。Part supervision 梯度损害 PSG global feature |
| 015 | PSG Spatial (3×3 DWConv) | 58.3% | 67.1% | 81.4% | 85.8% | mAP+1.7%, R1+0.6% | 🟡 mAP 匹配 exp007, R1 低 0.8%. 3×3 conv 冗余，1×1 已足够 |
| 016 | PSG + Pose-Guided Erasing (PGE) | 54.8% | 65.0% | 77.7% | 82.2% | mAP-1.8%, R1-1.5% | ❌ PGE 替代 RE 严重有害 (-3.5% vs exp007). 身体部件级擦除过强 |
| 017 | PSG + Pose Channel Gate (PCG) | 58.0% | 67.3% | 80.9% | 85.3% | mAP+1.4%, R1+0.8% | 🟡 与 exp007 持平(-0.3% mAP). 通道级正交不干扰但无额外收益 |
| 018 | PCG-only (无 PSG) | 57.8% | 67.7% | 81.4% | 86.2% | mAP+1.2%, R1+1.2% | 🟡 PCG 有独立效果(+1.2%), 但低于 PSG(-0.5%). PSG+PCG 不叠加 |
| 019 | Pose Cross-Attention (PXA, 替代 PSG) | 57.3% | 66.9% | 80.4% | 85.3% | mAP+0.7%, R1+0.4% | 🟡 有效但弱于 PSG(-1.0% mAP). Cross-attn 过拟合严重, 简单门控更好 |
| 020 | PSG + Pose Reconstruction Aux (PRA) | 57.8% | 67.3% | 80.3% | 84.7% | mAP+1.2%, R1+0.8% | 🟡 中性. 辅助重建任务不改善 PSG(-0.5% mAP). 后期梯度干扰导致锯齿波动 |
| 021 | Content-Adaptive PSG (CAPSG) | 57.2% | 66.0% | 80.5% | 85.2% | mAP+0.6%, R1-0.5% | ❌ Content-dependent gate 弱于静态 PSG(-1.1% mAP). 过度参数化, PSG 简洁性即优势 |
| 022-g | PDS global-only (独立Stage3, PSG全局分支) | 57.9% | 67.1% | 80.0% | 84.2% | mAP+1.3%, R1+0.6% | 🟡 PSG 增益大部分保留(-0.4% vs exp007), Stage 3 解耦有效 |
| 022-cs | PDS concat_scaled | 57.5% | 66.5% | 79.4% | 83.8% | mAP+0.9%, R1±0% | 🟡 Part 缩放融合 = baseline 水平, Part 仍有噪声 |
| 022-eq | PDS equal_concat | 56.1% | 64.0% | 77.5% | 82.4% | mAP-0.5%, R1-2.5% | ❌ 维度比 5:1 过度稀释 Global 贡献 |
| 022-p | PDS part-only | 55.2% | 63.1% | 76.3% | 81.7% | mAP-1.4%, R1-3.4% | Part 分支独立效果不佳 (ID_part loss 2.02 未充分收敛) |
| **023-g** | **PDS+StopGrad global-only** | **59.5%** | **69.5%** | **81.8%** | **85.8%** | **mAP+2.9%, R1+3.0%** | **单 seed 最优 global；3-seed mean = 59.20% / 68.63%，后续被 exp007a 几乎等价复现** |
| 023-cs | PDS+StopGrad concat_scaled | 59.1% | 68.8% | 81.0% | 85.1% | mAP+2.5%, R1+2.3% | ✅ Part 特征提供补充信息 |
| 023-eq | PDS+StopGrad equal_concat | 57.5% | 66.2% | 79.1% | 83.6% | mAP+0.9%, R1-0.3% | 🟡 equal_concat 仍被维度比稀释 |
| 023-p | PDS+StopGrad part-only | 56.7% | 65.1% | 77.9% | 82.6% | mAP+0.1%, R1-1.4% | Part 在 frozen 共享特征上学好(+1.5% vs exp022-p) |
| 024-g | PDS+StopGrad noPSG global-only | 59.2% | 68.7% | 82.0% | 86.1% | mAP+2.6%, R1+2.2% | 单 seed 高点；后续 multi-seed 不支持“PSG 贡献很小”这一强结论 |
| 024-cs | PDS+StopGrad noPSG concat_scaled | 59.0% | 68.3% | 81.6% | 85.7% | mAP+2.4%, R1+1.8% | |
| 024-eq | PDS+StopGrad noPSG equal_concat | 57.1% | 65.4% | 78.8% | 83.1% | mAP+0.5%, R1-1.1% | |
| 024-p | PDS+StopGrad noPSG part-only | 56.4% | 64.9% | 77.9% | 82.4% | mAP-0.2%, R1-1.6% | |
| 025-g | PDS+DelayedStopGrad global-only | 58.9% | 68.4% | 80.6% | 84.8% | mAP+2.3%, R1+1.9% | 🟡 前30ep阻断+释放, -0.6% vs exp023-g, +1.0% vs exp022-g |
| 025-cs | PDS+DelayedStopGrad concat_scaled | 58.6% | 67.8% | 80.4% | 84.4% | mAP+2.0%, R1+1.3% | 🟡 -0.5% vs exp023-cs |
| 025-eq | PDS+DelayedStopGrad equal_concat | 57.3% | 65.8% | 78.9% | 82.8% | mAP+0.7%, R1-0.7% | 🟡 -0.2% vs exp023-eq |
| 025-p | PDS+DelayedStopGrad part-only | 56.4% | 64.9% | 78.0% | 81.9% | mAP-0.2%, R1-1.6% | 🟡 -0.3% vs exp023-p, +1.2% vs exp022-p |
| 026 | PSG + Stochastic Pose Dropout (p=0.3) | 57.9% | 66.2% | 80.5% | 85.2% | mAP+1.3%, R1-0.3% | 🟡 -0.4% vs exp007. SPD 正则化未超越 PSG, pose 信号一致有用 |
| 027 | PSG + PCRA (alpha=0.2, loss 距离调制) | 57.8% | 66.8% | 81.0% | 85.3% | mAP+1.2%, R1+0.3% | 🟡 -0.5% mAP vs exp007. Pose similarity 调制 triplet 距离中性偏负 |
| 028 | PDS+StopGrad + Part LR 3x (equal_concat) | 59.3% | 68.9% | 81.4% | 85.4% | mAP+2.7%, R1+2.4% | 🟡 vs exp023-eq(57.5%)+2.8%, 但 vs exp023-g(59.5%)-0.2%. Part 收敛改善(ID 0.4 vs 2.0)未转化为测试增益 |
| 029 | PSG + Pose-Weighted Pooling (PWP) | 57.9% | 67.5% | 81.1% | 85.3% | mAP+1.3%, R1+1.0% | 🟡 vs exp007(58.3%)-0.4%. PWP 替换 GAP 为 pose-weighted pooling, 效果中性. PSG 已做了空间选择, post-hoc weighting 冗余 |
| 030-g | PDS+StopGrad + Skeleton GCN (global-only) | 59.5% | 69.5% | 82.0% | 86.5% | mAP+2.9%, R1+3.0% | 与 exp023-g 持平，GCN 辅助训练未损害 Global |
| **030-cs** | **PDS+StopGrad + Skeleton GCN (concat_scaled)** | **60.5%** | **70.5%** | **83.4%** | **87.2%** | **mAP+3.9%, R1+4.0%** | **PDS 版单 seed 最佳 fusion；后续 030a multi-seed 已显示 equal_concat 更强** |
| 030-eq | PDS+StopGrad + Skeleton GCN (equal_concat) | 59.9% | 70.9% | 82.7% | 87.4% | mAP+3.3%, R1+4.4% | E120 训练时默认模式 (E110 peak: 60.0%/71.0%) |
| 030-p | PDS+StopGrad + Skeleton GCN (part-only) | 57.4% | 69.5% | 81.2% | 86.2% | mAP+0.8%, R1+3.0% | GCN part-only 效果好，R1 大幅超越 baseline |
| **007a** | **PSG + 0.5x Global Loss Scale** | **59.5%** | **69.8%** | **81.9%** | **86.0%** | **mAP+2.9%, R1+3.3%** | **✅ 3-seed mean = 59.37% / 69.43%；相对 PSG 稳定 +1.53%，且与 exp023-g 无显著差异** |
| 030a-g | PSG + Skeleton GCN (global-only, 无 PDS) | 59.8% | 69.5% | 81.9% | 86.1% | mAP+3.2%, R1+3.0% | 3-seed mean = 59.33% / 68.87%，≈ exp007a；说明 GCN 分支对 global 基本中性 |
| 030a-cs | PSG + Skeleton GCN (concat_scaled, 无 PDS) | 60.5% | 73.7% | 85.0% | 88.1% | mAP+3.9%, R1+7.2% | 3-seed mean = 60.20% / 73.13%，稳定优于 030a-global，但弱于 equal_concat |
| **030a-eq** | **PSG + Skeleton GCN (equal_concat, 无 PDS)** | **61.1%** | **73.7%** | **85.2%** | **87.8%** | **mAP+4.5%, R1+7.2%** | **✅ 3-seed mean = 60.73% / 72.57%；对 030a-global 稳定 +1.40 mAP，是当前最强且已确认的无后处理模式** |
| 030a-p | PSG + Skeleton GCN (gcn_only, 无 PDS) | 58.2% | 72.9% | 83.3% | 86.6% | mAP+1.6%, R1+6.4% | 3-seed mean = 57.97% / 71.77%；branch 本身强，但不如 fusion |
| 030b-g | PSG+GCN w_p=0.01 (global-only) | **60.6%** | 71.0% | 83.8% | 87.3% | mAP+4.0%, R1+4.5% | 单 seed 高点；现主要作为“低权重时 branch 几乎未学好”的反例，不宜再单独拿它否定 loss scaling |
| 030b-cs | PSG+GCN w_p=0.01 (concat_scaled) | 59.4% | 72.9% | 83.9% | 87.3% | mAP+2.8%, R1+6.4% | 单 seed；核心信息是低权重时 concat 无法稳定超越 global |
| 030b-eq | PSG+GCN w_p=0.01 (equal_concat) | 60.5% | 73.0% | 84.4% | 88.3% | mAP+3.9%, R1+6.5% | 单 seed；与 global 接近，说明未训练好的 branch 贡献有限 |
| 030b-p | PSG+GCN w_p=0.01 (gcn_only) | 56.9% | 70.9% | 82.4% | 86.2% | mAP+0.3%, R1+4.4% | 图传播几乎未训练，但 keypoint pooling 本身仍强 |
| 032-g | PSG + Keypoint Pooling Only (global-only) | 59.8% | 70.0% | 81.7% | 85.4% | mAP+3.2%, R1+3.5% | 单 seed；支持“branch 不解释 global 提升”，但精确结论应以 030a multi-seed 为准 |
| 032-cs | PSG + Keypoint Pooling Only (concat_scaled) | 59.3% | 72.4% | 85.1% | 88.4% | mAP+2.7%, R1+5.9% | 单 seed；说明 keypoint pooling 本身就有较强 fusion 价值 |
| 032-eq | PSG + Keypoint Pooling Only (equal_concat) | 60.2% | 72.5% | 85.1% | 88.3% | mAP+3.6%, R1+6.0% | 单 seed；现在更适合作为“keypoint pooling 强基线”的证据，而不是单独量化 GCN 增益 |
| 032-p | PSG + Keypoint Pooling Only (gcn_only 测试模式) | 54.7% | 69.9% | 82.4% | 86.0% | mAP-1.9%, R1+3.4% | 无图传播仍有高 R1，证明关键点采样+置信度池化本身就是强基线 |
| 035a | PSG+GCN score weight (bundled sanity check) | 61.1% | 73.8% | 85.1% | 87.9% | mAP+4.5%, R1+7.3% | = exp030a seed1234 结果（61.1/72.9），含 target-aware+vis aug fix, 无 regression |
| 035b | PSG+GCN score*visibility weight | 60.4% | 71.6% | 84.8% | 87.9% | mAP+3.8%, R1+5.1% | ❌ vs 035a: -0.7% mAP, -2.2% R1。当前只说明 `score*visibility` 未带来收益，不能上升为整条 visibility 路线结论 |
| 007b | PSG + 0.25x Global Loss Scale | 58.3% | 67.6% | 80.0% | 84.9% | mAP+1.7%, R1+1.1% | = exp007(1.0x)! 收敛慢但最终追平 |
| 007c | PSG + 0.75x Global Loss Scale | 58.6% | 67.6% | 81.6% | 85.6% | mAP+2.0%, R1+1.1% | 单 seed；现阶段不能再用 0.25x/0.75x 的单次结果否定 0.5x，多种子只确认了 0.5x vs 1.0x |
| **000b** | **Baseline (seed 42, 3090)** | **56.1%** | **65.8%** | **79.4%** | **83.8%** | — | 3090 vs 4090(55.9%) Δ=0.2%, 确认跨硬件一致 |
| 036 | PSG+GCN + Per-Keypoint Triplet Loss | 60.6% | 73.1% | 84.5% | 88.2% | mAP+4.0%, R1+6.6% | ❌ vs 035a: -0.5% mAP, -0.7% R1。该编号已偏离原 visibility 路线，实际属于 `exp035` 之后的 branch 内部探索 |
| 037 | PSG+GCN + Learnable Keypoint Attention | 60.7% | 71.7% | 83.8% | 87.1% | mAP+4.1%, R1+5.2% | ❌ vs 035a: -0.4% mAP, -2.1% R1。该编号已偏离原 visibility 路线，LKA 未显示稳定正增益 |
| 039a | PSG+GCN + CVK retrieval (`cvk_only`) | 59.3% | 72.9% | 84.1% | 87.1% | mAP+2.7%, R1+6.4% | 测试时诊断；vs 035a: -1.8% mAP, -0.9% R1。纯共同可见关键点距离不足以替代 `equal_concat` |
| 039b | PSG+GCN + CVK retrieval (`cvk_hybrid`) | 61.9% | 73.2% | 85.2% | 88.5% | mAP+5.3%, R1+6.7% | 测试时诊断；vs 035a: +0.8% mAP, -0.6% R1。共同可见关键点更适合作为 global 的 pair-specific 补充 |
| 040a | exp030a checkpoint recheck (`equal_concat`) | 61.1% | 73.7% | 85.2% | 88.0% | mAP+4.5%, R1+7.2% | 原始 `exp030a` checkpoint 的当前代码口径复核；为 `040b` 提供直接对照 |
| 040b | exp030a checkpoint + CVK retrieval (`cvk_hybrid`) | 61.9% | 73.2% | 85.2% | 88.6% | mAP+5.3%, R1+6.7% | ✅ vs 040a: +0.8% mAP, -0.5% R1。与 039b 高度一致，说明正信号可复核 |
| 041a | exp030a checkpoint + CVK retrieval (`2:1`) | 61.6% | 72.6% | 84.2% | 88.1% | mAP+5.0%, R1+6.1% | 权重敏感性；vs 040b(`1:1`): -0.3% mAP, -0.6% R1。偏向 global 会削弱收益 |
| 041b | exp030a checkpoint + CVK retrieval (`1:2`) | 61.6% | 73.6% | 85.1% | 88.6% | mAP+5.0%, R1+7.1% | 权重敏感性；vs 040b(`1:1`): -0.3% mAP, +0.4% R1。偏向 CVK 更像用 mAP 换 R1 |
| 045a | rebuilt seed42 checkpoint recheck (`equal_concat`) | 60.2% | 72.7% | 84.4% | 87.6% | mAP+3.6%, R1+6.2% | `exp044` 重建 checkpoint 的直接对照；mAP 与既有 seed42 记录一致 |
| 045b | rebuilt seed42 checkpoint + CVK retrieval (`cvk_hybrid`) | 61.1% | 73.2% | 84.2% | 88.1% | mAP+4.5%, R1+6.7% | ✅ vs 045a: +0.9% mAP, +0.5% R1。CVK 正 mAP 信号已在第二个 checkpoint 上复核 |
| 046 | rebuilt seed2024 checkpoint (`exp030a` recover) | 60.1% | 72.9% | 84.0% | 87.6% | mAP+3.5%, R1+6.4% | `exp030a seed2024` checkpoint 重建完成；第三个可复用 checkpoint 资产已补齐，可用于后续第三 checkpoint 复核 |
| 047 | PSG+GCN + CSGT (Common-Support-Guided Triplet) | — | — | — | — | ❌ 中止 | Epoch 60 中断无 checkpoint。根本问题：pos/neg overlap 几乎相同（≈0.65），机制无法区分正负 pair。pos_fallback≈0.7 说明大部分退化为标准 triplet |
| 048 | PSG+GCN + SGMKC (Skeleton-Guided Masked Keypoint Completion) | 58.9% | 72.1% | 84.2% | 87.5% | mAP+2.3%, R1+5.6% | ❌ 负面 (-1.6% vs exp030a)。SGMKC loss 与 ID 分类存在梯度冲突，GCN 容量不足以同时完成两个任务 |
| 050 | PSG+GCN + PAMC (Pose-Aware Masking Consistency) | 60.7% | 72.2% | 83.7% | 87.3% | mAP+4.1%, R1+5.7% | 🟡 中性 (vs exp030a-eq 3-seed: -0.03% mAP, -0.37% R1)。Consistency loss 未提供额外增益。连续第 3 个辅助 loss 方向失败 |
| 051-eq | PSG+GCN + PAML (Pose-Aware Metric Learning, equal_concat) | 60.7% | 72.7% | 84.6% | 88.2% | mAP+4.1%, R1+6.2% | 🟡 中性 (vs exp030a-eq 3-seed: -0.03% mAP, +0.13% R1)。逐关键点距离训练未带来增益。连续第 4 个辅助 loss 失败 |
| 051-cvk | PSG+GCN + PAML (cvk_hybrid) | 62.0% | 73.6% | 85.1% | 88.4% | — | 🟡 vs exp030a CVK (61.9%/73.2%): +0.1%/+0.4%。训练-测试 metric alignment 假设未得到验证 |
| 052-eq | PSG+GCN + KP-RPE (equal_concat) | 61.0% | 72.7% | 84.4% | 87.6% | mAP+4.4%, R1+6.2% | 🟡 中性 (vs exp030a-eq 3-seed: +0.27% mAP, +0.13% R1，在方差范围内)。mAP 训练全程 10/12 checkpoint 为正(均值+0.76%)，但最终结果在方差内 |
| 052-g | PSG+GCN + KP-RPE (global) | 59.5% | 68.4% | 81.6% | 85.7% | mAP+2.9%, R1+1.9% | 🟡 vs exp030a-g(59.8/69.5): -0.3%/-1.1%。KP-RPE 未改善 backbone 特征 |
| 052-cvk | PSG+GCN + KP-RPE (cvk_hybrid) | 61.7% | 72.6% | 84.3% | 88.2% | — | 🟡 vs exp030a CVK(61.9/73.2): -0.2%/-0.6%。KP-RPE + CVK 无正交增益 |
| 053-eq | PSG + XCAD (equal_concat) | 59.7% | 70.8% | 82.0% | 86.2% | mAP+3.1%, R1+4.3% | ❌ vs exp030a-eq 3-seed: -1.03% mAP, -1.77% R1。Cross-attention decoder 劣于 GCN |
| 053-g | PSG + XCAD (global) | 59.2% | 68.6% | 81.6% | 85.9% | mAP+2.6%, R1+2.1% | 🟡 vs exp030a-g 3-seed: -0.13%/-0.27%，几乎持平 |
| 053-cvk | PSG + XCAD (cvk_hybrid) | 60.7% | 71.8% | 82.9% | 86.9% | — | ❌ vs exp030a CVK(61.9/73.2): -1.2%/-1.4% |
| **054-eq** | **PSG+GCN + PGAM (equal_concat)** | **61.1%** | **73.8%** | **85.1%** | **87.9%** | **mAP+4.5%, R1+7.3%** | **🟢 vs exp030a-eq 3-seed: +0.37% mAP, +1.23% R1。首个 PSG+GCN 上正向叠加模块！** |
| 054-g | PSG+GCN + PGAM (global) | 59.8% | 69.5% | 81.9% | 86.1% | mAP+3.2%, R1+3.0% | 🟡 vs exp030a-g 3-seed: +0.47%/+0.63%，方差内 |
| 054-cvk | PSG+GCN + PGAM (cvk_hybrid) | 61.9% | 73.2% | 85.2% | 88.5% | — | 🟡 vs exp030a CVK: 0.0%/0.0%，完全持平 |
| 055-eq | PSG+GCN + PGAM t=0.5 (eq_concat) | 61.2% | 73.5% | 85.2% | 88.6% | mAP+4.6%, R1+7.0% | 🟢 vs exp054: ≈持平。阈值不敏感 |
| 055-g | PSG+GCN + PGAM t=0.5 (global) | 60.3% | 70.2% | 82.2% | 87.1% | mAP+3.7%, R1+3.7% | 🟢 vs exp054-g: +0.5%/+0.7%。t=0.5 global 更好 |
| 056-eq | PSG+GCN + PGAM S2+S3 (eq_concat) | 61.1% | 73.7% | 85.2% | 88.6% | mAP+4.5%, R1+7.2% | 🟡 vs exp054: ≈持平。多 Stage 无额外增益 |
| 057-eq | PSG+GCN + KDL w=0.1 (eq_concat) | 61.0% | 73.3% | 84.6% | 87.9% | mAP+4.4%, R1+6.8% | 🟡 中性。vs exp030a 3-seed: +0.27%/+0.73%。Dissimilar loss 无效 |
| **058-eq** | **PSG+GCN + ROA (equal_concat)** | **61.8%** | **72.8%** | **85.2%** | **88.3%** | **mAP+5.2%, R1+6.3%** | **🟢🟢 历史最高 mAP！vs 3-seed: +1.07%/+0.23%。超出方差！** |
| **058-g** | **PSG+GCN + ROA (global)** | **60.8%** | **70.0%** | **83.0%** | **87.0%** | **mAP+4.2%, R1+3.5%** | **🟢🟢 vs 3-seed: +1.47%/+1.13%。全局特征也显著提升！** |
| 059-eq | PSG+GCN + ROA + PGAM (eq_concat) | 61.8% | 72.8% | 85.2% | 88.3% | mAP+5.2%, R1+6.3% | 🟡 与 exp058 精确相同。PGAM 与 ROA 完全冗余 |
| 060-eq | PSG+GCN + PA-ROA (eq_concat) | 61.6% | 72.5% | 84.5% | 87.9% | mAP+5.0%, R1+6.0% | 🟡 vs random ROA: -0.2%/-0.3%。Pose-guided 放置不优于随机 |
| 061-eq | PSG+GCN + GKD 30% (eq_concat) | 60.8% | 73.0% | 84.3% | 87.8% | mAP+4.2%, R1+6.5% | 🟡 中性。vs 3-seed: +0.07%/+0.43%。GCN dropout 无效 |
| 062-eq | PSG+GCN + LKU (eq_concat) | 60.7% | 71.2% | 84.1% | 87.4% | mAP+4.1%, R1+4.7% | ❌ 负面。vs 3-seed: -0.03%/-1.37%。Learned uncertainty 损害 R1 |
| 063-eq | PSG + PTD (eq_concat) | 56.7% | 65.3% | 78.3% | 82.4% | mAP+0.1%, R1-1.2% | ❌❌ 严重负面。vs 3-seed: -4.03%/-7.27%。Pose-Token 无法替代 GCN |
| 058+nfc | PSG+GCN+ROA + NFC (eq_concat) | **64.0%** | **74.3%** | 84.3% | 87.2% | — | 🟢 NFC test-time boost on ROA。最强结果（含 NFC）|
| 058+cvk | PSG+GCN+ROA + CVK (cvk_hybrid) | 62.7% | 73.5% | 85.4% | 88.7% | — | 🟢 CVK 在 ROA 上也有效 |
| 064-eq | PSG+GCN + PKE (eq_concat) | 61.0% | 73.1% | 84.5% | 87.7% | mAP+4.4%, R1+6.6% | 🟡 微弱正向。vs 3-seed: +0.27%/+0.53%。Precision weighting 安全但不显著 |
| 065-eq | PSG+GCN + PKE+ROA (eq_concat) | 61.9% | 73.2% | 84.5% | 88.2% | mAP+5.3%, R1+6.7% | 🟡 ≈ROA alone。PKE+ROA 不正交 |
| **066-eq** | **PSG+GCN + PAA (eq_concat)** | **61.6%** | **74.2%** | **85.4%** | **88.4%** | **mAP+5.0%, R1+7.7%** | **🟢🟢🟢 历史最高 R1！vs 3-seed: +0.87%/+1.63%。训练端创新！** |
| **067-eq** | **PSG+GCN + PAA+ROA (eq_concat)** | **62.0%** | **73.7%** | **85.2%** | **88.6%** | **mAP+5.4%, R1+7.2%** | **🟢🟢🟢 历史最高 mAP！PAA+ROA 部分正交叠加。vs 3-seed: +1.27%/+1.13%** |
| 068-eq | PSG+GCN + RR-PAA (eq_concat) | 61.2% | 72.9% | 85.4% | 88.3% | mAP+4.6%, R1+6.4% | 🟡 vs PAA uniform: -0.4%/-1.3%。路由不优于 uniform |
| 069-eq | PSG+GCN + PAA b128 (eq_concat) | 61.3% | 74.6% | 85.2% | 88.3% | mAP+4.7%, R1+8.1% | 🟡 vs PAA b32: -0.3% mAP, +0.4% R1。R5/R10 改善但 mAP 未超。b32 仍是最优配置 |
| 070-eq | PSG+GCN + PAA S&C (eq_concat) | 61.4% | 73.4% | 85.4% | 88.5% | mAP+4.8%, R1+6.9% | 🟡 vs PAA scene: -0.2% mAP, -0.8% R1。target-only 热图不优于 scene 热图。消融价值 |
| 071-eq | PSG+GCN + PCL r=16 (eq_concat) | 60.7% | 72.0% | 84.6% | 88.1% | mAP+4.1%, R1+5.5% | ❌ vs PAA: -0.9% mAP, -2.2% R1。Feature-dependent LoRA 劣于 feature-independent PAA |
| 072-eq | PSG+GCN + PS-PAA (eq_concat) | 61.1% | 73.8% | 84.8% | 88.4% | mAP+4.5%, R1+7.3% | 🟡 vs PAA: -0.5% mAP, -0.4% R1。Body-part 分组不优于 generic 混合 |
| 073-eq | PSG+GCN + PAA Stage2+3 (eq) | 61.1% | 74.2% | 85.7% | 88.4% | mAP+4.5%, R1+7.7% | 🟡 vs PAA Stage3: -0.5% mAP, 0.0% R1。多 stage 不如单 stage |
| 074-eq | PSG+GCN + PAA+PGAM (eq) | — | — | — | — | — | ❌ 中止。PGAM 完全无效——结果与 exp066 精确相同。PGAM 为 no-op |
| 066-5060 | PAA 跨硬件验证 (5060 Ti) | 61.2% | 74.3% | 85.4% | 88.3% | — | ✅ 与本地 3090 结果一致 (Δ<0.4%)。远程可靠 |
| **066-s42** | **PAA seed42 (5060 Ti)** | **61.1%** | **74.4%** | **85.0%** | **87.6%** | — | **✅ vs seed1234(61.6%/74.2%): Δ-0.5%/+0.2%。PAA 跨 seed 确认** |
| **067-s42** | **PAA+ROA seed42 (3090)** | **62.1%** | **73.6%** | **85.2%** | **88.6%** | — | **✅ vs seed1234(62.0%/73.7%): Δ+0.1%/-0.1%。完美复现** |
| 076-eq | PSG+GCN+PAA+TDPC (eq) | 61.3% | 72.7% | 84.9% | 87.8% | mAP+4.7%, R1+6.2% | ❌ vs PAA(61.6/74.2): -0.3%/-1.5%。differential adapter 无收益 |
| 077-eq | PSG+GCN+ST-PAA 34ch (eq, 5060) | 61.0% | 73.6% | 84.4% | 88.6% | mAP+4.4%, R1+7.1% | ❌ vs PAA: -0.6%/-0.6%。scene+target concat 不优于 scene-only |
| 078-eq | PSG+GCN+PAA+APG (eq) | 60.5% | 72.5% | 84.3% | 87.9% | mAP+3.9%, R1+6.0% | ❌ vs PAA: -1.1%/-1.7%。adaptive gate 负面 |
| **079-eq** | **PSG+GCN+ROA 无PAA (eq, 5060)** | **62.0%** | **73.6%** | **85.0%** | **88.1%** | **mAP+5.4%, R1+7.1%** | **🟢🟢 ROA 独立有效！vs 3-seed: +1.27%/+1.03%。≈ exp067 PAA+ROA** |
| 081-eq | PSG+PAA+PQTD (eq) | 56.9% | 67.2% | 79.1% | 84.1% | mAP+0.3%, R1+0.7% | ❌❌ Decoder 严重不够收敛。GCN(400K) >> Decoder(2.5M) 在 120ep |
| 083-eq | PSG+GCN+PAA+PGFI (eq) | 61.1% | 73.4% | 84.7% | 88.1% | mAP+4.5%, R1+6.9% | 🟡 中性偏负 vs PAA(-0.5%/-0.8%)。Inpainter 未带来额外收益 |
| 084-eq | PSG+GCN+PAA+CIPGFR (eq) | 61.4% | 73.6% | 85.5% | 88.6% | mAP+4.8%, R1+7.1% | 🟡 中性 vs PAA(-0.2%/-0.6%)。Cross-instance recovery 未改善 |
| **085-eq** | **PSG+GCN+PAA+ROA p=0.7 (5060)** | **62.6%** | **75.3%** | **85.2%** | **88.4%** | **mAP+6.0%, R1+8.8%** | **🟢🟢🟢 历史最高！vs ROA p=0.5: +0.6%/+1.7%** |
| 085b-eq | PSG+GCN+ROA p=0.7 无PAA (5060) | 62.2% | 73.4% | 84.5% | 88.0% | mAP+5.6%, R1+6.9% | 🟡 vs p=0.5 无PAA: +0.2%. p=0.7 增益主要来自与 PAA 协同 |
| **086-eq** | **PSG+GCN+PAA+ROA+PA-PAT (3路)** | **62.7%** | **74.6%** | **85.3%** | **88.7%** | **mAP+6.1%, R1+8.1%** | **🟢🟢🟢 Peak 62.8%@Ep100。留作拼 SOTA recipe** |
| 087-eq | PSG+GCN+PAA+MM (momentum) | 61.5% | 73.0% | 84.5% | 88.2% | mAP+4.9%, R1+6.5% | 🟡 中性 vs PAA(-0.1%/-1.2%)。Memory contrastive 无额外收益 |
| **090-sgcfr** | **SGCFR on PAA (top_k=5, α=0.7)** | **64.2%** | **75.7%** | — | — | **mAP+7.6%, R1+9.2%** | **🟢🟢🟢🟢 +2.6% vs PAA baseline** |
| **090b-sgcfr** | **SGCFR on PAA+ROA (top_k=5, α=0.7)** | **64.9%** | **75.7%** | — | — | **mAP+8.3%, R1+9.2%** | **🟢🟢🟢🟢🟢 最强结果! +2.9% vs PAA+ROA** |
| 091-eq | PSG+GCN+PAA+TTSFR (eq) | 61.4% | 73.2% | 85.1% | 88.5% | mAP+4.8%, R1+6.7% | 🟡 中性 vs PAA(-0.2%/-1.0%)。Batch 内 recovery 信号不够（仅4张/ID） |
| 092-eq | PSG+GCN+PAA+LSRM w=0.5 (eq) | 60.9% | 73.3% | 85.0% | 88.1% | mAP+4.3%, R1+6.8% | 🟡 中性偏负 vs PAA(-0.7%/-0.9%)。Learned recovery 在 batch 内仍不够 |
| 092d-eq | PSG+GCN+PAA+LSRM BS128 (eq) | 61.3% | 73.5% | 84.8% | 88.4% | mAP+4.7%, R1+7.0% | 🟡 大batch帮助 (+0.4% vs BS64)，但仍 -0.3% vs PAA |
| 091b-eq | PSG+GCN+PAA+TTSFR BS128 (5060) | 60.8% | 73.0% | — | 88.6% | mAP+4.2%, R1+6.5% | 🟡 中性偏负。大 batch 对 simple recovery 无效 |
| 093-eq | PSG+GCN+PAA+PGTM (eq) | 56.7% | 68.0% | 80.9% | 85.2% | mAP+0.1%, R1+1.5% | ❌❌ Token merging 9.4M params 120ep 严重不够收敛 |
| 094 | PSG+GCN+PAA+PCQA (PTM) | — | — | — | — | 中性 (Ep74终止) | 🟡 PTM loss 不收敛(0.28→0.40)，Ep70: 59.2% vs 基线58.1%(+1.1%)，但 PTM 对照 exp030a 而非 exp066 |
| 094b | PSG+GCN+PAA+PCQA 归一化 (远程) | 61.2% | 74.0% | 84.8% | 88.2% | vs PAA: -0.4%/-0.2% | 🟡 PCQA 中性。PTM loss 0.41 不收敛 |
| 095-eq | PSG+GCN+PAA+DPF (热图池化) Ep100 | 60.0% | 71.8% | 83.5% | 87.1% | vs PAA: **-1.6%/-2.4%** | ❌ 12×4 分辨率太低，热图空间池化不如点采样 |
| 096-eq | PSG+GCN+PAA+MRKF (多尺度) Ep100 | 60.3% | 72.0% | 84.3% | 87.2% | vs PAA: -1.3%/-2.2% | ❌ Stage2(384d)+Stage3 融合不稳定，高方差震荡 |
| 098-eq | PSG+GCN+PAA+PKP (KPR式prompting) | 60.9% | 72.8% | 84.5% | 88.5% | vs PAA: -0.7%/-1.4% | 🟡 Swin window attention 限制早期 pose 传播 |
| 099 | OT Matching (测试时 Sinkhorn) | 59.0% | 71.0% | — | — | vs PAA: **-2.6%/-3.2%** | ❌ per-keypoint OT 不如 global cosine |
| 100-eq | PSG+GCN+PAA+FiLM (全阶段) | 61.0% | 73.3% | 84.6% | 88.3% | vs PAA: -0.6%/-0.9% | 🟡 PSG+PAA 已足够，更多 conditioning 不帮助 |
| 101-eq | PSG+GCN+PAA+SGMT (masking) | 61.0% | 73.8% | 85.0% | 88.5% | vs PAA: -0.6%/-0.4% | 🟡 中性，SGCFR 增益与基线相同 (+2.7% vs +2.6%) |
| 102-eq | PSG+GCN+PAA+SGMT-50% (masking) Ep110 | 60.6% | 73.1% | 84.7% | 87.9% | vs PAA: -1.0%/-1.1% | 🟡 50% masking 更激进，效果略差于 30%(exp101)。训练仅到 Ep110 |
| 104c-eq | PSG+GCN+PAA+PACD v3 (3×3 fm mask) | 61.3% | 74.5% | 85.4% | 88.6% | vs PAA: -0.3%/+0.3% | 🟡 中性。Feature map masking (8%) 太弱，GAP 鲁棒 |
| 104d-eq | PSG+GCN+PAA+PACD v4 (row fm mask) Ep100 | 60.4% | 73.3% | 84.5% | — | vs PAA: -1.2%/-0.9% | 🟡 中性偏负。33% 行级 mask 仍不够 |
| 105b-eq | PSG+GCN+PAA+SGRE (cross-attn) Ep90 | 60.7% | 73.3% | 85.1% | — | vs PAA: -0.3%/-0.2% | 🟡 中性。SGRE loss 收敛(3.28→0.30)但 detached kp 不影响 backbone |
| 106-eq | PSG+GCN+PAA+PISD (image mask) Ep28 | — | — | — | — | 提前终止 | 🟡 pisd loss 0.02-0.04 极小。GAP 全局特征天然遮挡不变 |
| 142-eq | PSG+GCN+SKC (Support-Supervised Keypoint Completion, eq) | 60.3% | 71.8% | 84.4% | 87.7% | vs exp030a-eq: -0.8%/-1.9% | ❌ 中性偏负。completion module 虽然活跃（gate=0.26, delta_norm=1.5），但 skc_pre≈skc_post 说明修改方向不是向 prototype 靠近。gate 无限制增长导致后期过度修改特征。feature-level completion 方向已被多轮验证为无效 |
| 143-eq | PSG+GCN+SASA (Skeleton-Aware Self-Attention, eq) | 61.1% | 73.7% | 85.1% | 88.5% | vs exp030a-eq: **0.0%/0.0%** | 🟡 完美中性。零参数骨架测地注意力偏置对最终结果无任何影响。与 KP-RPE(exp052) 结论一致：Swin window attention 的 RPE 已足够编码空间结构 |
| 141-cvk | PSG+GCN+LPCS comp_ctx (cvk_residual) | 55.8% | 68.1% | 78.3% | 82.4% | — | ❌ LPCS comp_ctx 失败。competition-context 未改善排序。LPCS 训练 loss 严重干扰主学习，最终远低于 exp030a (-5.3% mAP) |
| 144-eq | PSG+GCN+SASA α=1.0 (equal_concat) | 61.0% | 73.5% | 84.6% | 87.9% | vs exp030a-eq: **-0.1%/-0.2%** | 🟡 中性。10x更强的SASA偏置与α=0.1结果相同。确认skeleton attention信息对Swin完全冗余 |
| 145-eq | PSG+GCN+PAA+SASA (equal_concat) | 61.4% | 73.8% | — | 88.4% | vs PAA(exp066): **-0.2%/-0.4%** | 🟡 中性。SASA 与 PAA 组合无正交增益，确认 SASA 在任何配置下均无效 |
| 148-eq | PSG+GCN+PCVT (Pose-Complementary View Training, eq) | ~59.3%* | ~71.3%* | — | — | ❌ 负面。*ep100 数据，训练中。早期加速（ep30: +2.4 mAP）但后期被基线追平并反超。3-view 训练的 1/3 主损失稀释导致后期收敛不足。训练集 95.8% 全可见使 complementary masking 缺乏信号 |
| 149 | PSG+GCN+SCFA (Symmetry-Conditioned Feature Aggregation) | — | — | — | — | ❌ ep30 止损。ep30: 50.7/61.3 vs exp030a 52.2/66.0 (-1.5/-4.7)。bilateral gap case 太少(scfa_pg=0.09)，hand-crafted pooling trick 不够强 |
| 151-eq | PSG+GCN+PVAT (Pose-Visibility Adversarial Training, eq) | 进行中 | — | — | — | 🟡 中性趋势。ep70: 59.0/72.0 vs exp030a 58.1/70.9 (+0.9/+1.1)。但 pvat_acc=0.83 不降——训练集 95.8% 可见，adversarial 无信号。预计最终中性 |
| **maxsim** | **exp030a + MaxSim (ColBERT-style late interaction)** | **60.1%** | **74.4%** | **84.3%** | **87.5%** | **🟢 Test-time method。R1 74.4% 最高！但 mAP 低于 equal_concat (-1.0%)** |
| **maxsim_hybrid 1:1** | **exp030a + MaxSim Hybrid (global+maxsim)** | **62.2%** | **73.8%** | **84.9%** | **88.2%** | **🟢🟢 超越 CVK hybrid (61.9/73.2)！mAP+1.1% vs eq_concat** |
| **maxsim_hybrid 1:2** | **exp030a + MaxSim Hybrid (偏向 MaxSim)** | **62.2%** | **74.5%** | — | **88.6%** | **🟢🟢🟢 mAP+1.1, R1+0.8 vs eq_concat。ColBERT-style late interaction** |
| **maxsim_paa 1:2** | **PAA (exp066) + MaxSim Hybrid** | **62.6%** | **75.2%** | **85.6%** | **89.0%** | **🟢🟢 vs PAA eq_concat(61.6/74.2): +1.0/+1.0** |
| **maxsim_paa_roa 1:2** | **PAA+ROA (exp067) + MaxSim Hybrid** | **63.5%** | **75.4%** | **86.2%** | **88.9%** | **🟢🟢🟢🟢 vs PAA+ROA eq_concat(62.0/73.7): +1.5/+1.7。跨 checkpoint 稳定正向** |
| 152b-eq | MaxSim Hard Triplet Training (tau=0.005, eq_concat) | 57.8% | 69.7% | — | 86.8% | ❌ vs exp030a-eq: **-3.3/-4.0**。MaxSim training 严重损害特征 |
| 152b-ms | MaxSim Hard Triplet Training (maxsim_hybrid 1:2) | 59.0% | 71.0% | 83.8% | 87.2% | ❌ vs exp030a maxsim: **-3.2/-3.5**。即使 MaxSim test 也无法回补 |
| 152-eq | MaxSim Soft Triplet Training (tau=0.05, eq_concat) | 57.8% | 70.3% | — | 87.4% | ❌ vs exp030a-eq: **-3.3/-3.4**。与 hard 版结果一致 |
| 153-eq | MaxSim Additive w=0.25 (eq_concat) | 60.6% | 72.3% | — | 88.0% | 🟡 中性 vs exp030a-eq: **-0.5/-1.4**。不有害但无增益 |
| 153-ms | MaxSim Additive w=0.25 (maxsim_hybrid 1:2) | 61.8% | 74.3% | 85.1% | 88.4% | 🟡 中性 vs exp030a maxsim: **-0.4/-0.2** |
| 153b-eq | MaxSim Additive w=1.0 (eq_concat) | 57.6% | 70.0% | — | 87.1% | ❌ vs exp030a: **-3.5/-3.7**。w=1.0 崩了，与 replace 模式一致 |
| 155-eq | Evidential DL (GCN branch, eq_concat) | 60.7% | 72.9% | 84.4% | 88.4% | 🟡 中性 vs exp030a: **-0.4/-0.8**。Bayes Risk 梯度太弱(id_part=11 vs CE ~0.5) |
| 155-ms | Evidential DL (maxsim_hybrid 1:2) | 62.1% | 74.3% | 85.7% | 88.7% | 🟡 中性 vs exp030a maxsim: **-0.1/-0.2** |
| 155b-eq | Evidential DL kl=0.01 (eq_concat) | 61.0% | 73.0% | 84.9% | — | 🟡 中性 vs exp030a: **-0.1/-0.7**。中期 +1.4 peak(ep50)但最终追平 |
| 155b-ms | Evidential DL kl=0.01 (maxsim_hybrid) | 62.1% | 74.1% | 84.9% | 88.4% | 🟡 中性 vs maxsim: **-0.1/-0.4** |
| 156-eq | SPLADE sparse repr (eq_concat) | 60.5% | 72.3% | — | 87.5% | 🟡 中性 vs exp030a: **-0.6/-1.4** |
| **157-eq** | **PLBOA lower-body (VOC, p=0.7, eq_concat)** | **62.7%** | **74.0%** | **85.4%** | **89.0%** | **🟢🟢🟢 vs exp030a: +1.6/+0.3。vs ROA: +0.9。最强训练改进！** |
| **157-ms** | **PLBOA + MaxSim hybrid 1:2** | **64.1%** | **75.0%** | **86.4%** | **89.8%** | **🟢🟢🟢🟢 项目最高！vs baseline maxsim: +1.9/+0.5** |
| 157c-eq | PLBOA gradient bottom-heavy (eq_concat) | 60.8% | 73.5% | 85.2% | — | 🟡 中性 vs baseline: -0.3/-0.2。太激进 |
| **158-eq** | **PAA+PLBOA (eq_concat)** | **62.2%** | **74.7%** | **85.8%** | **89.0%** | **🟢🟢 vs baseline: +1.1/+1.0。R1 最高！** |
| **158-ms** | **PAA+PLBOA (maxsim_hybrid)** | **63.6%** | **75.8%** | **86.0%** | **89.2%** | **🟢🟢🟢 R1 最高！** |
| 157d-eq | Body-random occlusion (eq_concat) | 61.0% | 71.5% | 84.4% | 88.4% | 🟡 中性偏负 vs baseline: -0.1/-2.2。人体 bbox 随机遮挡不优于 ROA |
| 159-eq | PLBOA+ROA (eq_concat) | 62.4% | 73.7% | 85.4% | 88.7% | 🟢 vs baseline: +1.3/+0.0。但弱于 PLBOA-only (-0.3 mAP)。ROA+PLBOA 不正交 |
| 157-s42 | PLBOA seed42 (eq_concat) | 61.9% | 73.8% | 85.7% | 89.3% | ✅ 2-seed mean: **62.3%/73.9%** (+1.57/+1.33 vs baseline 3-seed) |
| 161-eq | **STD-PR (structural tokens, eq_concat)** | 58.7% | 67.4% | 81.1% | 85.0% | ❌ vs baseline: -2.4/-6.3。structural tokens 不如 GCN keypoint features |
| **161b-eq** | **STD-PR+PLBOA (eq_concat)** | **63.4%** | **73.4%** | **85.4%** | **88.5%** | **🟢🟢🟢 超 PLBOA+GCN mAP +0.7！vs baseline: +2.3/-0.3。STD-PR 替代 GCN 有效** |
| 161c-eq | STD-PR 17 parts (eq_concat) | 58.2% | 67.3% | 79.8% | 84.1% | 🟡 ≈6 parts (58.7)。token 数不是瓶颈 |
| 164-eq | STD-PR V2+PLBOA (anchor queries, eq) | 62.1% | 72.6% | 85.7% | 88.8% | ❌ vs V1: -1.3/-0.8。anchor 在遮挡位采噪声 |
| 164r-eq | STD-PR V2 alone (anchor, eq) | 57.9% | 68.0% | 81.5% | 85.0% | 🟡 vs V1: -0.8/**+0.6** R1。无 PLBOA 时 R1 改善 |
| 165-eq | STD-PR conf-pool+PLBOA (eq) | 61.8% | 71.9% | 84.5% | 88.5% | ❌ vs V1 mean: -1.6/-1.5。conf-pool 不帮 STD-PR+PLBOA |
| 165r-eq | STD-PR conf-pool alone (eq) | 58.2% | 68.9% | 81.5% | 85.7% | 🟡 vs V1 mean: -0.5/**+1.5** R1。无 PLBOA 时 R1 改善 |
| **157-3seed** | **PLBOA+GCN 3-seed mean** | **62.1±0.49%** | **73.9±0.12%** | — | — | **✅ +1.37/+1.33 vs baseline 3-seed** |
| **161b-3seed** | **STD-PR+PLBOA 3-seed mean** | **62.6±0.87%** | **72.7±0.67%** | — | — | **✅ +1.87/+0.13 vs baseline 3-seed** |
| **157+sgcfr** | **PLBOA+SGCFR (α=0.7)** | **65.2%** | **75.3%** | — | — | **🟢🟢🟢🟢 Test-time best! +4.5/+1.6 vs baseline** |
| 157+nfc | PLBOA+NFC (k=5) | 65.0% | 74.8% | 85.0% | 88.5% | 🟢🟢 +3.9/+1.1 vs baseline |
| 157+rr | PLBOA+Re-ranking | 78.8% | 79.7% | 87.8% | 90.0% | 🟢🟢🟢🟢🟢 含 re-ranking |
| 161d-eq | STD-PR+PLBOA+PAA (eq_concat) | 62.6% | 72.3% | 84.9% | 88.5% | 🟡 PAA 不帮 STD-PR (-0.8 vs 161b) |
| 161e-eq | STD-PR+PLBOA+ROA (eq_concat) | 63.2% | 72.9% | 85.5% | 88.8% | 🟡 ROA 不帮 STD-PR (-0.2 vs 161b) |

### Phase 4: SupCon + OA-SD + Parallel Aug (exp166-193)

| ID | 方法 | mAP | R-1 | R-5 | R-10 | 备注 |
|----|------|-----|-----|-----|------|------|
| 166 | STD-PR+PLBOA+PAPE+MS-PSG+CE (full arch) | 63.1% | 73.9% | 86.1% | 89.2% | CE baseline with full architecture |
| 166r | ↳ base arch (no PAPE/MS-PSG) | 60.3% | 72.8% | — | — | CE base for OA-SD comparison |
| 176 | +SupCon T=0.05 (1-view) | 64.1% | 75.5% | 85.9% | 89.2% | ✅ SupCon +1.0/+1.6 vs CE |
| **187** | **+SupCon T=0.05 + 3-view Parallel Aug** | **64.9%** | **76.6%** | **87.2%** | **90.0%** | **🟢🟢 Overall best! +1.8/+2.7 vs exp166** |
| **190** | **3-view Parallel Aug + CE (no SupCon)** | **64.2%** | **75.6%** | **86.2%** | **89.1%** | **✅ 3-view+CE ≈ 1-view+SupCon! +1.1/+1.7 vs exp166** |
| **191** | **OA-SD + CE (1-view, decay=0.999)** | **63.2%** | **75.4%** | **86.3%** | **89.1%** | **✅ OA-SD 独立有效! +2.9/+2.6 vs CE base** |
| 192 | OA-SD + CE (1-view, decay=0.99) | 62.6% | 74.9% | 86.0% | 89.1% | 🟡 vs exp191: -0.6/-0.5。decay 不敏感 |
| **193** | **OA-SD + 3-view + CE** | **64.4%** | **76.5%** | **86.3%** | **89.4%** | **✅ OA-SD+3-view additive! R1 ≈ exp187 SupCon, mAP +0.2/R1+0.9 vs exp190** |
| 194 | OA-SD + CE (weight=2.0) | 63.4% | 74.8% | 86.1% | 89.1% | 🟡 vs exp191 (w=1.0): +0.2/-0.6。weight 不敏感 |

codex
本地记录里已经能看到两个关键约束：后期大量“对齐/一致性/补全/结构偏置”都被打成中性或负面，真正还站得住的是 `PSG/GCN/PLBOA/OA-SD/多视图/SupCon` 这一类训练端资产；我再定位最新实验目录和监控，避免给出和现有证据冲突的方向。
exec
/bin/zsh -lc "find experiments -maxdepth 1 -type d -name 'exp*' | sort -V | tail -30" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/exp341base_noprompt
experiments/exp341_clip_id_prompt
experiments/exp342_clip_id_pose
experiments/exp343_pose_guided_clip
experiments/exp344_pose_cond_prompt
experiments/exp345_pose_part_clip
experiments/exp347_noparam_deocc
experiments/exp348_occ_repel
experiments/exp349b_small_undetach_clip
experiments/exp349_small_full_clip
experiments/exp350_undetach_clean
experiments/exp351_undetach_deocc
experiments/exp352_undetach_morepartw
experiments/exp353_undetach_noclip
experiments/exp354_pc_sor
experiments/exp355r_pgpd_random
experiments/exp355_pgpd
experiments/exp356r_pcmsc_random
experiments/exp356_pcmsc
experiments/exp357_pose_shuffle_ks
experiments/exp358_pose_channel_shuffle
experiments/exp359_lm_reid
experiments/exp360_intruder
experiments/exp361_psc_jepa
experiments/exp362_genoccl
experiments/exp363_ag_foundation
experiments/exp364_dg_foundation
experiments/exp365_wildlife
experiments/exp366_active_evidence
experiments/exp367_single_support

exec
/bin/zsh -lc 'rg -n "exp19|exp2|当前|进行中|中止|证伪|六点|marginal|CVaR|single|support|dead|LM-ReID|DINO|MLLM|foundation|gallery|open" experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md experiments/results.md' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/innovation_brainstorm.md:7:1. `PSG` 仍然是当前最稳的主创新点  
experiments/innovation_brainstorm.md:13:   - `exp009 / exp251 / exp253` 说明 multi-stage **不是普遍自动更优**
experiments/innovation_brainstorm.md:14:   - `exp255 vs exp255b` 又强烈说明：在 `GCN512` 结构分支下，`2-stage PSG` 是关键条件
experiments/innovation_brainstorm.md:16:3. `LGPA-D + GCN + OA-SD + MaxSim + flip` 更适合写成 supporting system，而不是主创新本体  
experiments/innovation_brainstorm.md:19:   - `exp257-259` 已说明 recipe 空间基本耗尽
experiments/innovation_brainstorm.md:21:### 当前真正该补的不是新故事，而是干净消融
experiments/innovation_brainstorm.md:29:### 当前推荐验证顺序
experiments/innovation_brainstorm.md:47:### 当前主线口径
experiments/innovation_brainstorm.md:52:- `2-stage PSG` = scalable extension / 当前最终版本
experiments/innovation_brainstorm.md:53:- `LGPA-D / GCN / OA-SD / MaxSim` = supporting assets
experiments/innovation_brainstorm.md:247:**问题 2: 当前方法组合天花板**
experiments/innovation_brainstorm.md:325:旧: image → single vector → pairwise cosine → triplet loss
experiments/innovation_brainstorm.md:343:### 当前进行中
experiments/innovation_brainstorm.md:398:2. **真正的 occlusion gap 在 test-time**: gallery/query 有严重遮挡，但训练集没有
experiments/innovation_brainstorm.md:406:### 当前新的判断
experiments/innovation_brainstorm.md:408:经过 `exp110-126`，当前最重要的收束不是“support-complete 有没有价值”，而是：
experiments/innovation_brainstorm.md:417:**让 support-complete prototype 以“可学习残差 prior”的形式进入 keypoint branch。**
experiments/innovation_brainstorm.md:433:  1. 问题层面仍锚定 `single-image support incomplete`
experiments/innovation_brainstorm.md:434:  2. 机制层面从 “memory bank / routing trick” 升级成了真正的 **support-conditioned completion**
experiments/innovation_brainstorm.md:444:- 这说明 common-support 不是噪声，而是真实的 pairwise 证据
experiments/innovation_brainstorm.md:445:- 但当前主收益仍停留在 test-time，论文主线不够完整
experiments/innovation_brainstorm.md:448:1. `exp041` 已说明 `1:1` 基本是当前 mAP sweet spot
experiments/innovation_brainstorm.md:454:### 当前最有价值的训练端候选
experiments/innovation_brainstorm.md:462:1. 用 `kp_weights` 构造 batch 内 pairwise common-support overlap
experiments/innovation_brainstorm.md:463:2. 在 global branch 上增加一条 support-aware triplet
experiments/innovation_brainstorm.md:481:**核心教训**: 把 retrieval-time 的 common-support 信号迁到训练端，不能简单用 overlap 做 mining filter。retrieval-time CVK 有效是因为它改变了距离计算方式（只在共同可见关键点上计算距离），而不是因为它筛选了更好的 pair。
experiments/innovation_brainstorm.md:485:2. 如果要做训练端 common-support，必须改变 loss 本身的距离计算（如只在共同可见区域上计算 triplet 距离）
experiments/innovation_brainstorm.md:488:### 当前结论（更新）
experiments/innovation_brainstorm.md:714:- GCN: **+1.3%** (single-seed, 待确认)
experiments/innovation_brainstorm.md:771:   - BPBreID / KPR 都把 visibility 用在 query-gallery 距离计算
experiments/innovation_brainstorm.md:774:3. **当前代码线的真实 gap 已经更清楚了**
experiments/innovation_brainstorm.md:776:   - 但当前测试仍用 `equal_concat`
experiments/innovation_brainstorm.md:791:- 只在 query-gallery 共同可靠的关键点上计算局部距离
experiments/innovation_brainstorm.md:803:#### 当前优先级更新
experiments/innovation_brainstorm.md:848:1. **`1:1` 是当前小范围内的 mAP 最优点**
experiments/innovation_brainstorm.md:851:2. **这条线更像 balanced correction，而不是 single-side domination**
experiments/innovation_brainstorm.md:858:   **global identity space + balanced common-support correction**
experiments/innovation_brainstorm.md:882:   当前最准确的表达不是：
experiments/innovation_brainstorm.md:885:   - **CVK 主要做 deeper-rank common-support correction**
experiments/innovation_brainstorm.md:909:### 三篇论文对当前研究的关键观察
experiments/innovation_brainstorm.md:917:#### 2. PartFeatureDecoder（ProFD）与当前方向的关系
experiments/innovation_brainstorm.md:925:**结论**：这个方向与 exp047 失败路线在本质上不同（exp047 是 training-end CSGT，而这是 feature extraction decoder），但当前 exp030a 的 GCN 方案已足够，不优先推进 PartFeatureDecoder 路线。
experiments/innovation_brainstorm.md:995:**共性教训**: 在 exp030a 的 GCN branch 上做额外训练端改进，都遇到了"小型 GCN 容量不足以承载第二个任务"的瓶颈。GCN 的最优角色就是其当前角色——纯 ID 分类的关键点特征传播器。
experiments/innovation_brainstorm.md:1007:### 当前方向决策
experiments/innovation_brainstorm.md:1016:**PAMC 仍是当前最有希望的候选**，因为：
experiments/innovation_brainstorm.md:1017:- 它不依赖 GCN branch 改进（避开已证伪的方向）
experiments/innovation_brainstorm.md:1144:| exp047 | CSGT (Common-Support-Guided Triplet) | 新 loss | ❌ 中止，pos/neg 无法区分 |
experiments/innovation_brainstorm.md:1159:1. 当前 GCN branch 的 ID loss + Triplet loss 已经足够——增量训练信号无法带来额外增益
experiments/innovation_brainstorm.md:1175:| 测试模式 | exp052 | exp030a single-seed | Δ | exp030a 3-seed mean | Δ vs 3-seed |
experiments/innovation_brainstorm.md:1198:### 当前有效模块清单（更新）
experiments/innovation_brainstorm.md:1222:**能继续的方向必须是架构级别的改变**，而不是在当前框架上"加一点东西"。
experiments/innovation_brainstorm.md:1259:### 当前确认有效的唯一模块组合
experiments/innovation_brainstorm.md:1283:### 当前格局
experiments/innovation_brainstorm.md:1336:**3. 下一步创新必须跳出当前框架**
experiments/innovation_brainstorm.md:1339:### 当前可行的下一步
experiments/innovation_brainstorm.md:1360:### exp065 PKE+ROA: 进行中，预计 ≈ ROA alone（不正交）
experiments/innovation_brainstorm.md:1362:### 当前最佳方法栈
experiments/innovation_brainstorm.md:1424:3. **Feature-independent** — 不需要依赖当前特征
experiments/innovation_brainstorm.md:1435:**当前需要的不是更多 PAA 变体，而是**：
experiments/innovation_brainstorm.md:1440:**当前最强方法栈**:
experiments/innovation_brainstorm.md:1450:### 当前最重要的判断
experiments/innovation_brainstorm.md:1493:1. 先补完 `exp075` 的 PAA 多 seed，确认当前 strongest baseline
experiments/innovation_brainstorm.md:1498:4. 若首轮无正信号，立即止损，回退到 retrieval-time `common-support recovery`
experiments/innovation_brainstorm.md:1524:### 已证伪的方向（exp076-078）
experiments/innovation_brainstorm.md:1533:### 已证伪的方向（exp081 PQTD）
experiments/innovation_brainstorm.md:1537:**结论**: Transformer decoder 在 15K 训练图 + 120ep 下严重不够收敛。GCN (400K params) 在当前数据规模上远优于 Decoder (2.5M params)。
experiments/innovation_brainstorm.md:1539:### 当前进行中：exp083 PGFI
experiments/innovation_brainstorm.md:1569:   **在 per-keypoint / common-visible support 层面做 duplicate-aware confuser reasoning**。
experiments/innovation_brainstorm.md:1571:### 当前更可信的新主线约束
experiments/innovation_brainstorm.md:1577:  3. per-keypoint / common-support 粒度
experiments/innovation_brainstorm.md:1589:  - 就连 `per-keypoint / common-support` 层面的 test-time confuser penalty 也不稳定
experiments/innovation_brainstorm.md:1611:  - “support-complete latent representation” 的 headroom 非常大
experiments/innovation_brainstorm.md:1616:   **模型没有学会从单图中逼近完整 identity support。**
experiments/innovation_brainstorm.md:1618:   - support 来源太弱
experiments/innovation_brainstorm.md:1622:   **same-ID support bank → single-image support-complete distillation**
experiments/innovation_brainstorm.md:1624:### 当前最值得赌的具体机制
experiments/innovation_brainstorm.md:1640:  - `support-complete` 不是只存在于上界分析里的幻觉
experiments/innovation_brainstorm.md:1644:1. 当前最值得继续赌的，不再是“有没有必要做 support-complete”，而是：
experiments/innovation_brainstorm.md:1645:   **怎样让 prototype teacher 更可靠、更接近真正的 multi-view support。**
experiments/innovation_brainstorm.md:1651:   **reliable-support bank / teacher reliability gating**
experiments/innovation_brainstorm.md:1655:  1. 问题不是简单 occlusion comparison，而是 single-image support incomplete
experiments/innovation_brainstorm.md:1656:  2. 方法不是通用补全 decoder，而是 identity-level support-complete distillation
experiments/innovation_brainstorm.md:1666:- 结果几乎等价，说明“要求多个 support 样本共同支撑 teacher”这件事本身，并没有把当前增益显著放大。
experiments/innovation_brainstorm.md:1669:1. 当前 `support-complete` 主线并没有被否定，因为结果仍保持正向区间。
experiments/innovation_brainstorm.md:1672:   **teacher purity / write quality / support cleanliness**
experiments/innovation_brainstorm.md:1676:- 基于 support 置信度的 soft reliability weighting
experiments/innovation_brainstorm.md:1683:- `exp112` 说明更干净的 support 写入有用，但当前只形成弱正向：
experiments/innovation_brainstorm.md:1691:1. 当前最值得讲的主创新，已经不只是 “support-complete distillation”。
experiments/innovation_brainstorm.md:1693:   **如何在 pose-aligned support-complete learning 中控制 teacher hardening / non-stationary target。**
experiments/innovation_brainstorm.md:1701:- Lagged / stale support bank
experiments/innovation_brainstorm.md:1742:SCKD 系列的结案意味着必须转向全新方向。当前最值得探索的不再是"如何让 prototype bank 更好"，而是：
experiments/innovation_brainstorm.md:1748:2. **接受当前配置**：PSG+GCN+PAA+ROA 作为训练端最强配置（~62.7% 单 seed），SGCFR 作为测试端独特创新（+2.6%）
experiments/innovation_brainstorm.md:1760:- 核心: 每个 body part 产生一个独立 embedding，匹配时用 MaxSim（每个 query part 找 gallery 中最佳匹配的 part，取 max cosine similarity 后求和）
experiments/innovation_brainstorm.md:1762:- 与我们框架的关系: 我们的 GCN branch 已经产出 17 个 per-keypoint embedding。当前用 weighted pooling 聚合为一个向量。改为 MaxSim 匹配可能释放 per-keypoint 信息的全部价值
experiments/innovation_brainstorm.md:1787:### 当前最推荐的下一步
experiments/innovation_brainstorm.md:1801:- 重新看当前最扎实的证据链：
experiments/innovation_brainstorm.md:1802:  1. `cvk_hybrid` 说明 common-support 的 pairwise 几何是真实的
experiments/innovation_brainstorm.md:1805:  4. `exp109-116` 说明 `support-complete` 若被压成 `per-ID prototype`，会损失 pair-specific 细节
experiments/innovation_brainstorm.md:1807:### 当前更值得赌的新机制
experiments/innovation_brainstorm.md:1812:- 不再把 support 压成 prototype
experiments/innovation_brainstorm.md:1826:2. 机制层面：pose/keypoint branch 作为 **common-support relational teacher**
experiments/innovation_brainstorm.md:1827:3. 训练目标：把 global embedding 蒸馏成更符合 common-support 几何的空间
experiments/innovation_brainstorm.md:1840:   **teacher 自身还是单图 `kp_feats`，并不 support-complete**
experiments/innovation_brainstorm.md:1842:### 当前最值得赌的下一跳
experiments/innovation_brainstorm.md:1849:3. 而是先用 `exp109` 方向的 support bank 补全 low-vis keypoint teacher，再用补全后的 teacher 去做 `CSRD`
experiments/innovation_brainstorm.md:1852:1. `exp109` 已证明 support-complete teacher 有巨大 headroom
experiments/innovation_brainstorm.md:1856:   **support-complete teacher + relational distillation**
experiments/innovation_brainstorm.md:1861:- 但这次不能简单说 `support-complete teacher` 失败，因为机制统计很清楚：
experiments/innovation_brainstorm.md:1869:3. 因而当前更合理的解释不是“teacher 还不够完整”，而是：
experiments/innovation_brainstorm.md:1870:   **support-complete 监督的收益主要属于 support-incomplete 样本，被 clean 样本等权平均后稀释掉了**
experiments/innovation_brainstorm.md:1872:### 当前最值得赌的下一跳
experiments/innovation_brainstorm.md:1877:1. 保持 `exp120` 的 support-complete relational teacher 完全不变
experiments/innovation_brainstorm.md:1880:   - 它有多少 keypoint 真正被 support-complete teacher 补全
experiments/innovation_brainstorm.md:1903:   **support-complete teacher 实际只改变了一部分 pairwise 关系，distillation 应聚焦这些 pair-change relations**
experiments/innovation_brainstorm.md:1905:### 当前最值得赌的下一跳
experiments/innovation_brainstorm.md:1910:1. 保持 `exp120` 的 support-complete teacher 完全不变
experiments/innovation_brainstorm.md:1914:   - support-complete teacher 几何
experiments/innovation_brainstorm.md:1915:4. 对那些 **被 support completion 真正改变过的 pair** 赋予更高 distillation focus
experiments/innovation_brainstorm.md:1927:  **teacher stability = supporting mechanism**
experiments/innovation_brainstorm.md:1931:  2. 当前更像是“兑现偏慢”，而不是“机制不对”
experiments/innovation_brainstorm.md:1936:  - 这意味着当前第一版 pair focus **放大力度过弱**
experiments/innovation_brainstorm.md:1938:### 当前最值得赌的下一跳
experiments/innovation_brainstorm.md:1946:4. 验证当前 delayed weak-positive 是否只是因为放大不够
experiments/innovation_brainstorm.md:1949:1. `exp121` 已说明 freeze 只是 supporting，不值得再扩成一条线
experiments/innovation_brainstorm.md:1952:4. 因而现在最有信息量的，不是换问题，而是测试 **pair focus strength 是否就是当前瓶颈**
experiments/innovation_brainstorm.md:1967:2. 训练监控的 delayed gain 没有清晰地转成最终 eval gain，说明当前第一版 focus 仍偏弱、偏散
experiments/innovation_brainstorm.md:1971:4. 因而当前最值得赌的新下一跳，不是继续平滑放大，而是：
experiments/innovation_brainstorm.md:1974:### 当前最值得赌的下一跳
experiments/innovation_brainstorm.md:1979:1. 保持 `exp123/124` 的 support-complete relational teacher 完全不变
experiments/innovation_brainstorm.md:1987:3. 当前更像是 teacher-change pairs 本来就稀疏，连续加权仍然被大量近零变化 pair 稀释
experiments/innovation_brainstorm.md:1995:  **stable teacher 只是 supporting mechanism，不再值得单独扩线**
experiments/innovation_brainstorm.md:1997:这会把当前主问题再收紧一步：
experiments/innovation_brainstorm.md:1998:1. `support-complete teacher` 的新增信息是真实存在的
experiments/innovation_brainstorm.md:2001:4. 于是 support-complete 带来的那部分新增 correction，极可能被 base teacher 的主体结构稀释掉
experiments/innovation_brainstorm.md:2003:### 当前最值得赌的下一跳
experiments/innovation_brainstorm.md:2008:1. 保留 `exp125` 当前最强的在线 relational 主线
experiments/innovation_brainstorm.md:2012:4. 让 global embedding 学习的不是“再复刻一遍 skeleton teacher”，而是只学 **support completion 真正带来的关系修正**
experiments/innovation_brainstorm.md:2018:   **当前瓶颈究竟是 routing 不够稀疏，还是 target 没把新增 correction 单独抽出来**
experiments/innovation_brainstorm.md:2032:3. 所以当前可以把假设进一步收紧为：
experiments/innovation_brainstorm.md:2036:### 当前最值得赌的下一跳
experiments/innovation_brainstorm.md:2041:1. 保留 `exp125` 当前最强的 online support teacher 与 `delta_top` routing
experiments/innovation_brainstorm.md:2053:   **让 student 在更大的 relation support 上学习 support-complete comparability correction**
experiments/innovation_brainstorm.md:2070:3. 所以当前可以把假设进一步收紧为：
experiments/innovation_brainstorm.md:2077:3. 因而“learned pair module”这条线 **还没有被真正做过，更没有被证伪**
experiments/innovation_brainstorm.md:2079:### 当前最值得赌的下一跳
experiments/innovation_brainstorm.md:2084:1. 不再强迫单个 global embedding 吃下 support-complete correction
experiments/innovation_brainstorm.md:2086:3. 让它根据当前 pair 的 `global / CVK / overlap / visibility` 描述，自适应预测：
experiments/innovation_brainstorm.md:2088:   - 该在多大程度上相信 common-support distance
experiments/innovation_brainstorm.md:2090:   - 用 `support-complete teacher` 提供更理想的 pairwise target
experiments/innovation_brainstorm.md:2099:   **当前 correction 不适合继续被压进单向量 embedding**
experiments/innovation_brainstorm.md:2115:3. 当前最合理的解释不是“learned pair module 没用”，而是：
experiments/innovation_brainstorm.md:2119:### 当前最值得赌的下一跳
experiments/innovation_brainstorm.md:2131:   - 必要时再加入更细的 keypoint-wise common-support statistics
experiments/innovation_brainstorm.md:2147:### 2026-03-21 上午补记：`exp133/134` 当前不能用于创新判断
experiments/innovation_brainstorm.md:2155:  1. 当前不能把 `exp133/134` 的数值拿来支持 “pair-specific correction scoring 成立”
experiments/innovation_brainstorm.md:2161:- 因而当前正确动作不是切题，而是：
experiments/innovation_brainstorm.md:2166:### 2026-03-21 晚间更新：`LPCS` 已经真正成立，但 sparse routing 最终只是 supporting 机制
experiments/innovation_brainstorm.md:2176:这批证据把当前创新判断收得更紧了：
experiments/innovation_brainstorm.md:2182:2. 但 sparse routing 不是当前最像论文主突破的部分
experiments/innovation_brainstorm.md:2188:   - 当前更像是 `LPCS` 的 **ranking objective / pair aggregation** 不够贴近最终检索目标
experiments/innovation_brainstorm.md:2190:因此当前最合理的主线升级不是：
experiments/innovation_brainstorm.md:2199:- `pose-defined common support`
experiments/innovation_brainstorm.md:2217:   - 而是当前 **hard-top 25%** 这种离散 hard selection 太激进
experiments/innovation_brainstorm.md:2230:基于 `exp136/137` 的双重负边界，当前最值得并行验证的不是同一机制的两个系数，而是两个不同的问题解释：
experiments/innovation_brainstorm.md:2238:   - 方案：给每个 pair descriptor 追加 query 的正负均值距离、margin、support 完整度与 teacher change 统计
experiments/innovation_brainstorm.md:2241:- `pose-defined common support`
experiments/innovation_brainstorm.md:2250:- `exp139 Query-Context LPCS` 当前版本被 Claude 明确驳回，原因不是“想法没新意”，而是：
experiments/innovation_brainstorm.md:2252:  2. 当前 query context 用到了 `row_pos_mean / row_neg_mean / row_margin` 等 label-dependent 统计，测试时天然不可得
experiments/innovation_brainstorm.md:2258:     - 当前 scorer 是否太短视，缺少 query 级语境
experiments/innovation_brainstorm.md:2261:3. 因而这条线下一步的正确形态，不是当前版 `exp139`，而是：
experiments/innovation_brainstorm.md:2266:### 2026-03-22 当前收敛：`rank-decay` 退为 supporting，`query-context correction` 升为主候选
experiments/innovation_brainstorm.md:2271:  - 它证明了“平滑 top-sensitive”比 `hard-rank` 合理，但最终只形成 supporting 级别的改进
experiments/innovation_brainstorm.md:2273:- `exp139 Query-Context LPCS` 则在当前阶段首次同时给出：
experiments/innovation_brainstorm.md:2283:3. 从而让同样的 common support 在不同 query 上被不同地解释
experiments/innovation_brainstorm.md:2286:- **共同可见的身体证据并不是孤立解释的，它依赖 query 当前的整体难度与上下文。**
experiments/innovation_brainstorm.md:2292:1. 当前 `LPCS` 也许不是不会修正
experiments/innovation_brainstorm.md:2307:1. pose 定义 common support
experiments/innovation_brainstorm.md:2308:2. support-complete teacher 提供 correction 方向
experiments/innovation_brainstorm.md:2311:### 2026-03-22 当前最强创新候选已收紧到两条互补线
experiments/innovation_brainstorm.md:2313:到 `exp139 ep50` 这一步，当前创新探索已经开始明显分层：
experiments/innovation_brainstorm.md:2317:   - 当前信号：
experiments/innovation_brainstorm.md:2322:     - 同一份 common support，是否需要放在 query-level 语境里解释
experiments/innovation_brainstorm.md:2326:   - 当前状态：
experiments/innovation_brainstorm.md:2328:     - 但当前版本的 gate 很快塌成接近常数 1，已判当前实现形态不成立
experiments/innovation_brainstorm.md:2334:在 `query-context` 与 `confidence-calibration` 之后，当前最新的本地候选不再问：
experiments/innovation_brainstorm.md:2341:**当前这个 candidate，在本 query 的所有候选竞争里到底处于什么位置？**
experiments/innovation_brainstorm.md:2351:3. `support_rank`
experiments/innovation_brainstorm.md:2357:- pose 定义 common support
experiments/innovation_brainstorm.md:2359:- 还要看当前 pair 在整个候选竞争中的相对位置
experiments/innovation_brainstorm.md:2362:- 都还紧扣 `exp109` 的核心发现：单图 support 不完整
experiments/innovation_brainstorm.md:2366:- `exp139` 强调 **如何解释 common support**
experiments/innovation_brainstorm.md:2369:### 2026-03-21 本地大转向：从 pair correction 切回 feature-space support completion
experiments/innovation_brainstorm.md:2380:当前新的大方向是：
experiments/innovation_brainstorm.md:2394:1. pose 不再只是用来构造 `common support distance`
experiments/innovation_brainstorm.md:2398:   - 哪些 support prototype 可作为跨图补全证据
experiments/innovation_brainstorm.md:2403:- `exp109` 暴露出的单图 support incomplete，能否在编码阶段被修复
experiments/innovation_brainstorm.md:2409:## 2026-03-22: feature-level completion 方向彻底证伪，转入注意力 inductive bias
experiments/innovation_brainstorm.md:2424:2. support bank / EMA prototype 的质量本身受限于数据量
experiments/innovation_brainstorm.md:2470:- 单图 support incomplete 能否通过 **pose-defined complementary pseudo-views** 改写成“伪多 support 学习”？
experiments/innovation_brainstorm.md:2481:- PCVT 直接改写训练对象，把单图变成“互补 support 组合体”
experiments/innovation_brainstorm.md:2486:- 当前 keypoint branch 是否浪费了 **单图内部的双侧同源冗余**？
experiments/innovation_brainstorm.md:2498:### 当前判断
experiments/innovation_brainstorm.md:2509:### `PCVT` 的当前价值
experiments/innovation_brainstorm.md:2522:2. `single-image support incomplete` 可能确实更适合被改写成“伪多 support 学习对象”，而不是继续做 scorer / completion 小修补
experiments/innovation_brainstorm.md:2523:3. 当前真正要验证的核心问题已收紧为：
experiments/innovation_brainstorm.md:2527:### `SCFA` 的当前结论
experiments/innovation_brainstorm.md:2532:3. 这意味着“单图内部双侧冗余”在当前 benchmark 上不是足够强的主问题
experiments/innovation_brainstorm.md:2660:### OA-SD 核心特性（exp191-194 消融）
experiments/innovation_brainstorm.md:2662:1. **OA-SD + CE = 强正向**: +2.9/+2.6 vs CE base (exp191)
experiments/innovation_brainstorm.md:2664:3. **EMA decay 不敏感**: 0.99 vs 0.999 最终差异 <1% (exp192)
experiments/innovation_brainstorm.md:2665:4. **Loss weight 不敏感**: 1.0 vs 2.0 最终差异 <1% (exp194)
experiments/innovation_brainstorm.md:2666:5. **OA-SD + 3-view 是 additive**: exp193 = 64.4/76.5 vs exp190 = 64.2/75.6 (+0.2/+0.9)
experiments/innovation_brainstorm.md:2681:**验证** (exp195): SupCon + OA-SD global-only ep70=60.2/73.4
experiments/innovation_brainstorm.md:2688:- 消融链: exp188 (冲突) → exp195 (分离) 是清晰的证据
experiments/innovation_brainstorm.md:2690:### 当前最强结果总表
experiments/innovation_brainstorm.md:2695:| 2 | exp193 | 3-view + OA-SD + CE | 64.4% | 76.5% |
experiments/innovation_brainstorm.md:2696:| 3 | exp190 | 3-view + CE | 64.2% | 75.6% |
experiments/innovation_brainstorm.md:2698:| 5 | exp194 | OA-SD + CE (w=2.0) | 63.4% | 74.8% |
experiments/innovation_brainstorm.md:2699:| 6 | exp191 | OA-SD + CE | 63.2% | 75.4% |
experiments/innovation_brainstorm.md:2702:### 待验证: exp196 终极配置
experiments/innovation_brainstorm.md:2735:**4. 已证伪的方向更新**
experiments/innovation_brainstorm.md:2736:- STM (Token Mixup): 只加速不改善天花板 (exp197/198)
experiments/innovation_brainstorm.md:2737:- OA-SD + SupCon: 互斥，即使 global-only 也无法叠加 (exp195/196)
experiments/innovation_brainstorm.md:2739:### 选定方向: OA-RD (exp199)
experiments/innovation_brainstorm.md:2755:| exp195 | OA-SD global-only + SupCon | SupCon | ~-2.8 mAP | ❌ 信号太弱 |
experiments/innovation_brainstorm.md:2756:| exp196 | OA-SD global-only + SupCon + 3v | SupCon | -2.5/-1.4 | ❌ 同上 |
experiments/innovation_brainstorm.md:2757:| exp199 | OA-RD relational + SupCon + 3v | SupCon | ~-1.5/-3.4 | ❌ 关系级也冲突 |
experiments/innovation_brainstorm.md:2758:| exp191 | OA-SD all-token + CE | CE | +2.9/+2.6 | ✅ CE 兼容 |
experiments/innovation_brainstorm.md:2759:| exp193 | OA-SD all-token + CE + 3v | CE | +0.2/+0.9 | ✅ CE 兼容 |
experiments/innovation_brainstorm.md:2760:| exp200 | OA-RD relational + CE | CE | ~-1.0/-3.4 | ❌ OA-RD 不如 OA-SD |
experiments/innovation_brainstorm.md:2768:   - OA-SD 路线: exp193 = 64.4/76.5 (不加 SupCon)
experiments/innovation_brainstorm.md:2788:**MaxSim Hybrid 在 exp206 checkpoint 上无需重训即可获得 +1.8% mAP！**
experiments/innovation_brainstorm.md:2799:- 对 query 的每个 keypoint feature，找 gallery 中最相似的 keypoint
experiments/innovation_brainstorm.md:2816:| Swin-Base (ep40) | 66.6% (3-view, 进行中) | ~68%+ |
experiments/innovation_brainstorm.md:2822:1. **Swin-Small + GCN+PAA+OA-SD + maxsim_hybrid = 72.4%** (exp210b with PKC=0.05)
experiments/innovation_brainstorm.md:2836:| exp210 | PKC w=0.5 | detached | 灾难 3.6% |
experiments/innovation_brainstorm.md:2837:| exp210b | PKC w=0.05 | detached | 无效 (=baseline) |
experiments/innovation_brainstorm.md:2838:| exp211 | MST w=0.5 | detached | 完全无效 (所有 loss 一致) |
experiments/innovation_brainstorm.md:2839:| exp213 | PKC+MST 组合 | detached | 灾难 40.6% |
experiments/innovation_brainstorm.md:2840:| exp215 | BA-PKC non-detach | non-det | 灾难 0.5% |
experiments/innovation_brainstorm.md:2841:| exp217 | OERL non-detach cosine | non-det | `62.2/75.2`，相对 `exp191 63.2/75.4` 为 `-1.0/-0.2` |
experiments/innovation_brainstorm.md:2842:| exp218 | PACI prototype bank | detached | `61.9/74.2`，相对 `exp191 63.2/75.4` 为 `-1.3/-1.2` |
experiments/innovation_brainstorm.md:2843:| exp219 | PACI without OA-SD | detached | 远程日志当前只确认到 `ep30=51.9/64.9`，早期即落后 baseline `52.2/65.2` |
experiments/innovation_brainstorm.md:2844:| exp220 | GSPB gradient scale 5% | 5% scale | `62.9/74.3`，相对 `exp191 63.2/75.4` 为 `-0.3/-1.1` |
experiments/innovation_brainstorm.md:2871:exp220 (scale=0.05) 完整对照:
experiments/innovation_brainstorm.md:2892:## 2026-04-03: BT-PKD 系列 — Non-Detached Gradient 全面证伪
experiments/innovation_brainstorm.md:2895:- exp229: BT-PKD constant (Tiny) → -1.0/-0.4
experiments/innovation_brainstorm.md:2896:- exp230: BT-PKD constant (Small, no PAUG) → ~0/-0.7 (ep110)
experiments/innovation_brainstorm.md:2897:- exp231: BT-PKD cosine decay (Tiny) → -1.5/-1.1
experiments/innovation_brainstorm.md:2898:- exp232: BT-PKD cosine decay (Small) → terminated ep37
experiments/innovation_brainstorm.md:2926:1. **不再试图让 Part 梯度到达 backbone** — 这条路已彻底证伪
experiments/innovation_brainstorm.md:3021:| Pose-Conditioned Masking | 6/10 | 类似 PLBOA 但在 feature 层 | 低风险, 可能有 marginal 增益 |
experiments/innovation_brainstorm.md:3027:当前系统 (73.0% mAP Small MaxSim) 距离 SOTA (KPR 75.1% ViT) 仅差 2.1%。
experiments/innovation_brainstorm.md:3030:### 当前最佳策略
experiments/innovation_brainstorm.md:3032:1. **短期**: 完成 exp249 (Small LGPA-D+GCN) → 可能 73-74% mAP
experiments/innovation_brainstorm.md:3035:4. **如需更强创新**: 需要跳出当前 Swin-Tiny/Small + detach 框架
experiments/results.md:63:| **030a-eq** | **PSG + Skeleton GCN (equal_concat, 无 PDS)** | **61.1%** | **73.7%** | **85.2%** | **87.8%** | **mAP+4.5%, R1+7.2%** | **✅ 3-seed mean = 60.73% / 72.57%；对 030a-global 稳定 +1.40 mAP，是当前最强且已确认的无后处理模式** |
experiments/results.md:74:| 035b | PSG+GCN score*visibility weight | 60.4% | 71.6% | 84.8% | 87.9% | mAP+3.8%, R1+5.1% | ❌ vs 035a: -0.7% mAP, -2.2% R1。当前只说明 `score*visibility` 未带来收益，不能上升为整条 visibility 路线结论 |
experiments/results.md:82:| 040a | exp030a checkpoint recheck (`equal_concat`) | 61.1% | 73.7% | 85.2% | 88.0% | mAP+4.5%, R1+7.2% | 原始 `exp030a` checkpoint 的当前代码口径复核；为 `040b` 提供直接对照 |
experiments/results.md:89:| 047 | PSG+GCN + CSGT (Common-Support-Guided Triplet) | — | — | — | — | ❌ 中止 | Epoch 60 中断无 checkpoint。根本问题：pos/neg overlap 几乎相同（≈0.65），机制无法区分正负 pair。pos_fallback≈0.7 说明大部分退化为标准 triplet |
experiments/results.md:126:| 074-eq | PSG+GCN + PAA+PGAM (eq) | — | — | — | — | — | ❌ 中止。PGAM 完全无效——结果与 exp066 精确相同。PGAM 为 no-op |
experiments/results.md:168:| 151-eq | PSG+GCN+PVAT (Pose-Visibility Adversarial Training, eq) | 进行中 | — | — | — | 🟡 中性趋势。ep70: 59.0/72.0 vs exp030a 58.1/70.9 (+0.9/+1.1)。但 pvat_acc=0.83 不降——训练集 95.8% 可见，adversarial 无信号。预计最终中性 |
experiments/results.md:218:| 192 | OA-SD + CE (1-view, decay=0.99) | 62.6% | 74.9% | 86.0% | 89.1% | 🟡 vs exp191: -0.6/-0.5。decay 不敏感 |
experiments/results.md:219:| **193** | **OA-SD + 3-view + CE** | **64.4%** | **76.5%** | **86.3%** | **89.4%** | **✅ OA-SD+3-view additive! R1 ≈ exp187 SupCon, mAP +0.2/R1+0.9 vs exp190** |
experiments/results.md:220:| 194 | OA-SD + CE (weight=2.0) | 63.4% | 74.8% | 86.1% | 89.1% | 🟡 vs exp191 (w=1.0): +0.2/-0.6。weight 不敏感 |
experiments/results.md:224:| 198 | OA-SD + CE + STM (base, remote) | 63.2% | 75.2% | — | — | 🟡 = exp191 (无 STM)。STM 只加速不改善天花板 |
experiments/results.md:226:| 200 | CE + OA-RD (base, remote) | 62.9% | 73.9% | 85.2% | 88.5% | ❌ vs exp191 OA-SD: -0.3/-1.5。OA-RD 不如 OA-SD |
experiments/results.md:369:| exp030a-eq vs exp023-g | **+1.53%** | (1.4, 1.0, 2.2) | 4.35 | 0.0491 | ✅ 当前最强模式已稳定超过 PDS global |
experiments/results.md:371:### 当前应采用的结论
experiments/results.md:377:5. **equal_concat 才是当前主模式**：3 个 seed 中都优于 `concat_scaled`，因此 `030a-eq` 应替代 `030-cs` 成为主结果。
experiments/results.md:411:  3. “target/distractor ambiguity”如果继续做，推理粒度必须回到 `per-keypoint / common-support`，不能停留在 pooled person embedding。
experiments/results.md:418:> 基于 `exp030a cvk_hybrid` 的 test-time 原型诊断，不计入训练端创新，只用于验证“per-keypoint / common-support 粒度的 duplicate-aware confuser penalty”是否存在独立 headroom。
experiments/results.md:432:  1. 即使把 confuser reasoning 下沉到 `per-keypoint / common-support` 粒度，当前 retrieval-time penalty 仍然整体负面。
experiments/results.md:439:### exp109: GT same-ID per-keypoint support bank
experiments/results.md:441:> 基于 `exp030a cvk_hybrid` 的 oracle 诊断，不计入正式结果；仅用于判断“training-time support-complete distillation”是否存在足够大的理论 headroom。
experiments/results.md:459:  1. `support-complete` 方向存在非常明显的 headroom，尤其集中在低可见 query。
experiments/results.md:461:  3. 因此下一步最值得推进的，不是新的 penalty/rerank，而是训练版的 **support-complete prototype distillation**。
experiments/results.md:468:> 基于 `exp030a equal_concat` 的训练端最小原型。该实验只新增 `per-identity / per-keypoint prototype bank`，并对低可见 keypoint 施加蒸馏；当前结果为单 seed 探索证据，不视为最终结论。
experiments/results.md:485:- 当前结论：
experiments/results.md:486:  1. `support-complete distillation` 训练版最小原型已经**成功转正**，但当前仍只是单 seed、弱增益证据。
experiments/results.md:487:  2. 当前瓶颈更像是 prototype teacher 的可靠性不够，而不是这条主线本身错误。
experiments/results.md:488:  3. 因此下一步应优先做 **reliable-support bank** 一类的单变量改进，而不是直接堆更重模块。
experiments/results.md:507:- 当前结论：
experiments/results.md:509:  2. `support-complete` 主线仍然成立，因为相对 `exp030a-eq seed1234` 依旧保持 `R1` 正增益。
experiments/results.md:510:  3. 但当前更关键的 teacher reliability 维度，大概率不是 support 数量门槛，而是 support 纯度 / 写入质量。
experiments/results.md:517:> 基于 `exp110_sckd` 的 teacher purity 单变量验证。该实验在 `ep84` 提前停表，原因是需要把资源切到更直接的 non-stationary teacher 验证；下面记录其当前最有信息量的观测点，不视为最终收敛结果。
experiments/results.md:525:- 当前结论：
experiments/results.md:526:  1. `UPDATE_THR=0.7` 说明 teacher purity 是有价值的方向，但当前只形成 **弱正向 / 近乎等价** 的证据。
experiments/results.md:534:> 该实验不是为了刷最终指标，而是为了回答：当前 `raw sckd` 上升究竟来自 coverage 变化，还是来自 teacher hardening。实验在 `ep44` 提前停表，因为机制信号已足够清楚。
experiments/results.md:545:- 当前结论：
experiments/results.md:547:  2. 更像是 bank 随训练持续积累 support，teacher 逐步变硬，导致 student 的平均对齐余弦下降。
experiments/results.md:574:- 当前结论：
experiments/results.md:592:- 当前结论：
experiments/results.md:610:- 当前结论：
experiments/results.md:646:### exp119: 把 common-support pairwise 几何蒸馏进 global embedding
experiments/results.md:648:> 基于 `exp030a` 的单变量训练端实验。`CSRD` 不再把 support 压成 `per-ID prototype`，而是直接用 skeleton/keypoint branch 计算出的 `CVK-style` pairwise 几何作为 detached relational teacher，蒸馏 global embedding 的 batch-wise 距离结构。
experiments/results.md:661:  2. 这说明当前更接近真实问题的，不是 overlap mining，也不是 prototype pointwise 蒸馏，而是 **common-support relational teacher**
experiments/results.md:662:  3. 但 `equal_concat` 仍只是近乎持平，表明当前 teacher 还不够强，单图 `kp_feats` 本身依旧受 `support incomplete` 限制
experiments/results.md:663:  4. 因此下一步不应回到 generic 模块叠加，而应把 `exp109` 的 support-complete headroom 引回 `CSRD`，验证 **support-complete relational teacher**
experiments/results.md:668:### exp120: support-complete teacher enhancement 已生效，但没有自动转成更好指标
experiments/results.md:670:> 基于 `exp119` 的单变量训练端实验。`exp120` 保留 `CSRD` 的 relational distillation 形式，只把 support-complete bank 用来补全 low-vis teacher keypoint，再用补全后的 teacher 去蒸馏 global 几何。该实验在 `ep90` 人工停表，结论来自训练监控口径，不是正式 eval 口径。
experiments/results.md:678:  1. `support-complete teacher` 的增强并没有失败，日志中 `csrd_sr≈0.145`、`csrd_sn≈157~159` 持续稳定，说明 low-vis keypoint 基本都拿到了补全 teacher
experiments/results.md:681:  4. 当前更合理的解释是：support-complete supervision 被大量本来就不缺 support 的 clean 样本稀释了
experiments/results.md:682:  5. 因而下一步不该继续盲目增强 teacher，而应测试 **只对 support-incomplete anchor 强化 relational distillation**
experiments/results.md:688:> 基于 `exp120` 的单变量训练端实验。`exp121` 仅把 support-complete teacher bank 的更新停止在 `epoch 30`，用于验证“teacher 稳定化”是否能改善 `SCRD`。该实验跑满 `ep120`，结论来自训练监控口径。
experiments/results.md:698:  2. 它也明显强于提前停表的 `exp120 online teacher`，说明 `support-complete relational teacher` 确实受 teacher 稳定性影响
experiments/results.md:699:  3. 但这个量级仍不足以单独支撑论文主创新，更合理的定位是 **supporting mechanism**
experiments/results.md:704:### exp122: sample-level selective weighting 没有把 support-complete teacher 的收益兑现出来
experiments/results.md:706:> 基于 `exp120` 的单变量训练端实验。`exp122` 保持 support-complete teacher 完全不变，只把 `CSRD` 的 anchor 权重改为 sample-level `replace_ratio`。该实验在 `ep43` 提前停表，结论以 `ep40` 首个关键验证点为准。
experiments/results.md:718:  4. 因而当前应放弃 sample-level weighting，转向更结构化的 **pair-level teacher-change focusing**
experiments/results.md:724:> 基于 `exp120` 的单变量训练端实验。`exp123` 保持 support-complete relational teacher 完全不变，只新增 pair-level `delta` focusing，让 `CSRD` 更聚焦于那些被 support-complete teacher 实际改变过的 pair。该实验已完成正式 eval。
experiments/results.md:741:  3. 当前更合理的解释不是“pair focus 不成立”，而是：
experiments/results.md:748:### exp130: `residual_kl` 没有把 `exp125` 推得更强，`target dilution` 不是当前主瓶颈
experiments/results.md:750:> 基于 `exp125` 的单变量训练端实验。`exp130` 保持 online support teacher、`delta_top` pair routing、KL distillation 与 `tau=0.10` 全部不变，只把 `CSRD` target 从完整 teacher distribution 改为 `residual_kl`。该实验已跑满 `ep120`，结论来自训练监控口径。
experiments/results.md:762:  3. 因而当前可以把结论收紧为：
experiments/results.md:763:     - `target dilution` 不是当前主瓶颈
experiments/results.md:771:> 基于 `exp125` 的单变量训练端实验。`exp131` 保持 online support teacher、`delta_top` pair routing 与 full teacher target 全部不变，只新增 `cross-batch relation queue`，把每个 anchor 可见的 candidate relations 从 batch 内扩展到 `batch + queue`。该实验已跑满 `ep120`，结论来自训练监控口径。
experiments/results.md:783:  3. 当前更合理的解释是：
experiments/results.md:785:     - 问题在于 **当前学生如何消费这些 pair-specific support-complete corrections**
experiments/results.md:793:> 基于 `exp131` 之后的新方向实验。`exp132` 不再把 support-complete correction 蒸进 embedding，而是在检索期引入真正挂进 checkpoint 与 evaluator 的 `pair-adaptive fusion head`，学习每个 pair 该在多大程度上相信 `global distance` 与 `CVK distance`。该实验已跑满 `ep120`，并完成同一 checkpoint 下的正式对照评估。
experiments/results.md:807:  3. 因而当前不能声称 “learned pair-adaptive fusion rule” 已经成立；第一版 `LTCS` 作为方法机制判负
experiments/results.md:816:### exp133 / exp134: 由于共享接线 bug，当前结果全部作废，不能用于支持或反驳 LPCS
experiments/results.md:820:| 方法 | 已观测数值 | 当前解释 |
experiments/results.md:826:  1. `exp133/134` 当前所有数值都 **不能** 进入 LPCS 的方法判断
experiments/results.md:836:### exp135: corrected full-pair `LPCS` 已真实成立，但更像 mAP-strong 的 supporting 线
experiments/results.md:847:     - 相对 `exp125 ep120 = 60.5 / 73.5`，当前是 `mAP +0.6 / R1 -1.2`
experiments/results.md:848:  3. 因而 full-pair `LPCS` 更适合作为 supporting 证据：
experiments/results.md:850:     - 但当前 loss 聚合方式还没有把收益转成更强的 `R1`
experiments/results.md:865:     - 相对 `exp135 ep120 = 61.1 / 72.3`，当前是 `-0.2 / -0.2`
experiments/results.md:866:  3. 这意味着 supervision dilution 也许存在，但当前还不能把它当作 `LPCS` 的主瓶颈
experiments/results.md:883:  3. 这说明当前问题不是“只要更关注 hardest ranked pairs 就会更强”，相反：
experiments/results.md:885:  4. 因而当前更合理的下一步不是继续加大 hard-top，而是：
experiments/results.md:890:### exp148: `PCVT` 早中期已形成稳定 `mAP` 正向，成为当前最值得继续追的训练端新方向
experiments/results.md:892:> `exp148` 把单图改写成 `full / complementary-view-a / complementary-view-b` 三视图训练对象，用 pose-defined complementary pseudo-views 验证“单图能否被改写成伪多 support 学习对象”。该实验当前仍在运行，以下结论来自 `ep10/20/30` 训练监控。
experiments/results.md:903:- 当前结论：
experiments/results.md:905:  2. 当前差值为：
experiments/results.md:915:  5. 当前唯一保留风险是：
experiments/results.md:918:### exp149: `SCFA` 快速诊断判负，双侧冗余前提在当前 benchmark 上不够强
experiments/results.md:931:- 当前结论：
experiments/results.md:940:     - 即当前 benchmark 上真正“一侧低一侧高”的 bilateral gap case 太少
experiments/results.md:943:## 2026-04-01/02: exp206r, exp207, exp209, exp210, exp210b, exp212, exp213
experiments/results.md:945:### exp206r: Small GCN+PAA+CE+OA-SD (Fixed OA-SD teacher)
experiments/results.md:946:> Repeat of exp206 with fixed OA-SD teacher (BN/Dropout/DropPath eval mode, clean teacher pose)
experiments/results.md:950:| exp206r equal_concat | 70.6% | 82.6% | 89.5% | 91.4% | ep120 final |
experiments/results.md:951:| **exp206r maxsim_hybrid** | **72.3%** | **82.9%** | **90.5%** | **92.2%** | ep120 + maxsim test |
experiments/results.md:953:- OA-SD fix: +0.1/+0.3 vs buggy exp206 (70.5/82.3). Fix 加速了早期收敛但不改变 final。
experiments/results.md:955:### exp207: Base GCN+PAA+CE+OA-SD 3-view (Fixed OA-SD)
experiments/results.md:959:| exp207 equal_concat | 70.7% | 80.7% | 89.5% | 91.7% | ep120 final |
experiments/results.md:960:| exp207 maxsim_hybrid | 72.2% | 82.0% | 90.4% | 92.3% | ep120 + maxsim test |
experiments/results.md:962:- Base (88M) 仅比 Small (50M) 高 +0.1% mAP。Base scaling 在当前配置下无效。
experiments/results.md:965:### exp209: Small STD-PR+CE+OA-SD — 终止 (ep30)
experiments/results.md:969:| exp209 | 56.0% | 69.3% | ep30 终止 |
experiments/results.md:973:### exp210: Small GCN+PAA+CE+OA-SD + PKC weight=0.5 — 灾难
experiments/results.md:977:| exp210 | 3.6% | 5.3% | ep10 终止 |
experiments/results.md:981:### exp210b: Small GCN+PAA+CE+OA-SD + PKC weight=0.05
experiments/results.md:985:| exp210b equal_concat | 70.6% | 81.8% | 89.9% | 92.4% | ep120 final |
experiments/results.md:986:| **exp210b maxsim_hybrid** | **72.4%** | **83.1%** | **90.8%** | **92.7%** | ep120 + maxsim test |
experiments/results.md:988:- PKC=0.05 不改变 equal_concat (= exp206r)，但 MaxSim 提升 +0.1/+0.2。
experiments/results.md:989:- **72.4/83.1 = 当前最佳 (无 NFC/reranking)！**
experiments/results.md:991:### exp212: Small GCN+PAA+CE+OA-SD LR=0.0008 — 灾难
experiments/results.md:995:| exp212 | 0.8% | 1.3% | ep10 终止 |
experiments/results.md:999:### exp213: Small + PKC(0.05) + MST(0.1) — 终止
experiments/results.md:1003:| exp213 | 40.6% | 54.8% | ep10 终止 |
experiments/results.md:1007:### MaxSim Hybrid 跨 checkpoint 分析 (exp206 local)
experiments/results.md:1019:## 2026-04-02: exp215, exp217, exp218, exp220, exp222, exp223
experiments/results.md:1023:| exp215 BA-PKC w=0.1 | 0.5% | 0.8% | 3.1% | 4.5% | ep10 终止 |
experiments/results.md:1024:| exp217 OERL + OA-SD | 62.2% | 75.2% | 86.0% | 89.0% | ep120 final |
experiments/results.md:1025:| exp218 PACI + OA-SD | 61.9% | 74.2% | 85.6% | 88.9% | ep120 final |
experiments/results.md:1026:| exp220 GSPB + OA-SD | 62.9% | 74.3% | 86.2% | 89.5% | ep120 final |
experiments/results.md:1027:| exp222 GSPB on Small (scale=0.05) | 2.3% | 3.9% | 9.9% | 14.3% | ep10 终止 |
experiments/results.md:1028:| exp223 PADPQ K=4 + OA-SD | 63.7% | 74.5% | 86.2% | 89.5% | ep120 final |
experiments/results.md:1030:- exp215 证实了 non-detached BA-PKC 会直接破坏 backbone 收敛。
experiments/results.md:1031:- exp217 / exp218 / exp220 都低于 `exp191 = 63.2 / 75.4`，因此不能写成训练端正向超越。
experiments/results.md:1032:- exp223 在 `equal_concat` 上给出 `mAP +0.5`，但 `R1 -0.9`；当前更适合作为 trade-off 证据，而不是“全面超越”。
experiments/results.md:1033:- exp219 的远程 `train_log` 已补回，但目前只确认到 `ep30 = 51.9 / 64.9`，尚无 final，因此暂不纳入正式结果表。
experiments/results.md:1034:- 注：`exp220/223` 的 `maxsim_hybrid` 数字目前只在各自 `monitor.md` 中留有测试记录，本地未发现独立 `test_log`，因此本总表仅登记训练日志可直接复核的 `equal_concat` 结果。
experiments/results.md:1036:## 2026-04-02/03: exp222c, exp224, exp225, exp226, exp227, exp228
experiments/results.md:1042:| exp222c GSPB Small scale=0.01 | 15.1% | 23.8% | 38.4% | 45.4% | ep10 终止 |
experiments/results.md:1043:| exp224 KAMP (random-init proj) + OA-SD | 60.7% | 73.0% | 85.1% | 88.3% | ep120 final |
experiments/results.md:1044:| exp225 GSPB(0.05) + PADPQ K=4 + OA-SD | 64.2% | 74.9% | 86.8% | 89.6% | ep120 final |
experiments/results.md:1045:| exp226 KAMP (zero-init proj) + OA-SD | 61.6% | 74.3% | 85.1% | 88.0% | ep120 final |
experiments/results.md:1047:- exp222c: GSPB scale=0.01 在 Small 上仍然灾难 (scale=0.05 → 2.3%, scale=0.01 → 15.1%)
experiments/results.md:1048:- exp224: KAMP (多尺度 keypoint 融合) random-init projection 造成 -2.5% mAP 噪声
experiments/results.md:1049:- exp225: **GSPB+PADPQ K=4 = 64.2/74.9 — Tiny 最佳 equal_concat！** (+1.0/-0.5 vs OA-SD)
experiments/results.md:1050:- exp226: KAMP zero-init projection 减少噪声但仍 -1.6% mAP。KAMP 方向失败。
experiments/results.md:1052:### exp227: Small GSPB(0.005) + PADPQ K=4 + OA-SD
experiments/results.md:1056:| exp227 equal_concat | 71.6% | 80.8% | 89.8% | 91.8% | ep120 final |
experiments/results.md:1057:| exp227 maxsim_hybrid | 71.8% | 80.6% | 89.9% | 91.9% | ep120 + maxsim test |
experiments/results.md:1059:- 对照 exp206r: **mAP +1.0, R1 -1.8** (equal_concat)
experiments/results.md:1061:- maxsim 71.8 < 当前最佳 72.4 (exp210b)。**GSPB+PADPQ 在 Small maxsim 上无优势。**
experiments/results.md:1064:### exp228: Tiny GSPB(0.05) + PADPQ K=8 + OA-SD
experiments/results.md:1068:| exp228 equal_concat | 64.1% | 74.3% | 86.4% | 89.5% | ep120 final |
experiments/results.md:1070:- 对照 exp225 K=4: **-0.1/-0.6**。K=8 ≈ K=4，无额外收益。
experiments/results.md:1073:### exp229: Tiny BT-PKD (w=0.01, constant) + OA-SD
experiments/results.md:1077:| exp229 equal_concat | 62.2% | 75.0% | 86.1% | 89.0% | ep120 final |
experiments/results.md:1079:- 对照 exp191 (OA-SD): **-1.0/-0.4**
experiments/results.md:1084:### exp230: Small BT-PKD (w=0.01, constant, no PARALLEL_AUG)
experiments/results.md:1088:| exp230 equal_concat | 70.8% | 81.9% | 89.7% | 91.9% | ep110 (OOM at ep120) |
experiments/results.md:1091:- 对照 exp206r (有 PAUG): 70.6/82.6 → **+0.2/-0.7** (mAP 持平, R1 差因缺 PAUG)
experiments/results.md:1093:### exp231: Tiny BT-PKD cosine decay (w→0 by ep60)
experiments/results.md:1097:| exp231 equal_concat | 61.7% | 74.3% | 85.5% | 88.6% | ep120 final |
experiments/results.md:1099:- 对照 exp191: **-1.5/-1.1**。Cosine decay 没有解决后期干扰。
experiments/results.md:1100:- 对照 exp229 constant: **-0.5/-0.7**。Decay 甚至略差。
experiments/results.md:1104:## 2026-04-04: exp235, exp236, exp237
experiments/results.md:1106:### exp235: FSDC (wrong ROA+PLBOA config)
experiments/results.md:1110:| exp235 | 61.7% | 74.5% | ep120 final |
experiments/results.md:1112:- 对照 exp191: **-1.5/-0.9**
experiments/results.md:1115:### exp236: FSDC (正确 ROA=False, PLBOA=0.7)
experiments/results.md:1119:| exp236 | 61.7% | 73.2% | ep120 final |
experiments/results.md:1121:- 对照 exp191: **-1.5/-2.2**
experiments/results.md:1122:- FSDC 正确配置仍然负面。**Feature completion 方向证伪。**
experiments/results.md:1124:### exp237: PPA (Pose-Prompted Part-Assignment Head) ⭐
experiments/results.md:1128:| exp237 equal_concat | **63.7%** | **75.0%** | ep120 final |
experiments/results.md:1129:| exp237 maxsim_hybrid | 64.1% | 75.1% | ep120 + maxsim |
experiments/results.md:1131:- 对照 exp191: equal_concat **+0.5/-0.4**, maxsim -0.1/-2.0
experiments/results.md:1136:### exp238: PPA assign_weight=0.1
experiments/results.md:1140:| exp238 | 62.1% | 74.0% | ep120 final |
experiments/results.md:1142:- 对照 exp191: **-1.1/-1.4**
experiments/results.md:1145:### exp239: PPA + GiLt (Part triplet only)
experiments/results.md:1149:| exp239 | 63.8% | 73.6% | ep120 final |
experiments/results.md:1151:- 对照 exp191: **+0.6/-1.8**
experiments/results.md:1154:### exp240: PPA on Small (w=0.5, no PARALLEL_AUG)
experiments/results.md:1158:| exp240 | 70.7% | 81.1% | ep120 final |
experiments/results.md:1160:- 对照 exp230 (no PAUG, ep110): -0.1/-0.8
experiments/results.md:1163:### exp241: PPA + GCN 双分支 on Tiny ⭐
experiments/results.md:1167:| exp241 equal_concat | **63.7%** | **75.3%** | **86.2%** | **88.9%** | ep120 final |
experiments/results.md:1169:- 对照 exp191: **+0.5/-0.1** — 最佳综合结果!
experiments/results.md:1170:- 对照 exp237 PPA-only: +0.0/+0.3 — GCN 改善 R1
experiments/results.md:1174:**exp241 MaxSim test**: 64.1/74.8 (MaxSim gain +0.4/-0.5 vs equal_concat)
experiments/results.md:1176:### exp242: PPA + GCN on Small ❌❌
experiments/results.md:1180:| exp242 | 60.9% | 73.4% | 88.9% | ep120 final |
experiments/results.md:1182:- 对照 exp206r (Small GCN): **-9.7/-9.2** — 灾难性失败!
experiments/results.md:1186:### exp243: LGPA (CLIP + Cross-Attention + Pose) on Tiny 🟡
experiments/results.md:1190:| exp243 ep80 | 60.9% | 72.5% | ep80 (GPU crash at ep88, 训练未完成) |
experiments/results.md:1192:- 对照 exp191 (GCN ep80): 62.0/74.4 = **-1.1/-1.9**
experiments/results.md:1198:### exp244: LGPA-Detach (CLIP + Detached Features) ⭐⭐⭐
experiments/results.md:1202:| exp244-R (detach, 无OASD) | 63.6% | 74.7% | 85.3% | 88.6% | ep120 remote final |
experiments/results.md:1203:| **exp244-L (detach+OASD)** | **65.3%** | **75.7%** | **86.8%** | **89.7%** | **ep120 local final** |
experiments/results.md:1205:- 对照 exp191 (GCN+OASD): **+2.1/+0.3** — **首个在 final 仍正向的 Part branch!**
experiments/results.md:1206:- 对照 exp243 (LGPA non-detach, ep80): +4.4/+3.2 — detach 完全解决后期干扰
experiments/results.md:1207:- 对照 exp244-R (无OASD): +1.7/+1.0 — OA-SD 与 LGPA-D 正交叠加
experiments/results.md:1212:**exp244 MaxSim test**: 66.0/76.4/87.2/90.5 (MaxSim hybrid on LGPA-D+OA-SD ep120)
experiments/results.md:1214:### exp245g: LGPA-Detach on Swin-Small ⭐⭐
experiments/results.md:1218:| **exp245g (Small LGPA-D+OA-SD)** | **70.2%** | **80.1%** | **89.8%** | **91.2%** | **ep120 local PT2+mmcv-full** |
experiments/results.md:1220:- 对照 exp206r (Small GCN+PAA+OA-SD): -0.4/-2.5 — mAP 接近, R1 差距
experiments/results.md:1221:- 对照 exp244 (Tiny LGPA-D+OA-SD): **+4.9/+4.4** — Small backbone 有效
experiments/results.md:1222:- LGPA-D 用更简单架构 (无 GCN, 无 PAA) 达到接近 exp206r 的 mAP
experiments/results.md:1225:**exp245g MaxSim test**: 71.9/82.2/91.0/92.8 (MaxSim hybrid on Small LGPA-D+OA-SD ep120)
experiments/results.md:1227:- vs exp206r (70.6/82.6): **mAP +1.3, R1 -0.4** — mAP 超越 Small baseline!
experiments/results.md:1229:### exp245h_v2: Small LGPA-D + OA-SD 远程复现 ⭐⭐⭐
experiments/results.md:1233:| **exp245h_v2 equal_concat** | **71.6%** | **81.6%** | **89.2%** | **91.2%** | **ep120 远程 5060Ti final** |
experiments/results.md:1235:- 对照 exp245g (本地 3090): **+1.4/+1.5** — 远程环境收敛更好
experiments/results.md:1236:- 对照 exp206r (Small baseline): **mAP +1.0, R1 -1.0**
experiments/results.md:1239:**exp245h_v2 MaxSim test**: 73.0/82.7/90.5/92.7 (MaxSim hybrid on ep120)
experiments/results.md:1241:- vs exp206r (70.6/82.6): **mAP +2.4, R1 +0.1** — **Small 全面超越!**
experiments/results.md:1242:- vs exp245g MaxSim (71.9/82.2): **+1.1/+0.5** — **Small 新最强!**
experiments/results.md:1244:### exp246: LGPA-D + GCN 双分支 (Tiny) 🟡
experiments/results.md:1248:| exp246 (ep83 crash) | 64.1% | 75.2% | — | — | ep80 (GPU 竞争 crash) |
experiments/results.md:1249:| **exp246b equal_concat** | **65.5%** | **77.2%** | **86.9%** | **90.1%** | **ep120 final** |
experiments/results.md:1251:- 对照 exp244 (LGPA-D only): **+0.2/+1.5** — GCN 主要贡献在 R1
experiments/results.md:1252:- 对照 exp191 (GCN only): **+2.3/+1.8** — LGPA-D 贡献巨大
experiments/results.md:1254:- ep10~ep70 全部与 exp246 精确匹配 (复现验证通过)
experiments/results.md:1256:**exp246b MaxSim test**: 66.3/77.7/87.6/90.6 (MaxSim hybrid on LGPA-D+GCN ep120)
experiments/results.md:1258:- vs exp244 MaxSim (66.0/76.4): **+0.3/+1.3** — **Tiny 新最强!**
experiments/results.md:1260:### exp247: VCSR — Visibility-Conditional Semantic Routing (Tiny, 无OA-SD)
experiments/results.md:1264:| **exp247 VCSR** | **63.6%** | **73.5%** | **84.2%** | **88.3%** | **ep120 远程 final** |
experiments/results.md:1266:- 对照 exp244-R (LGPA-D 无OA-SD): **0.0/-1.2** — VCSR ≈ LGPA-D, visibility gating 无效
experiments/results.md:1271:### exp248: PCFD — Pose-Conditioned Feature Differencing (Test-time) ❌
experiments/results.md:1275:| exp244 cosine baseline | 65.3% | 75.7% | — |
experiments/results.md:1281:- Learned pair-level matching 证伪 (训练端 exp152/153 + test-time PCFD 均失败)
experiments/results.md:1284:### exp249: Small LGPA-D + GCN 双分支 + OA-SD (进行中)
experiments/results.md:1288:| exp249 ep10 | 51.1% | 61.7% | 77.9% | 83.8% | 远程 5060Ti, ep10 |
experiments/results.md:1289:| exp249 ep20 | 60.9% | 73.2% | 85.5% | 88.6% | 远程 5060Ti, ep20 |
experiments/results.md:1290:| exp249 ep30 | 63.6% | 74.2% | 86.0% | 89.2% | 远程 5060Ti, ep30 |
experiments/results.md:1291:| exp249 ep40 | **68.0%** | **78.7%** | 88.8% | 90.7% | 远程 5060Ti, ep40 |
experiments/results.md:1292:| exp249 ep50 | 69.4% | 79.4% | — | 90.9% | 远程 5060Ti, ep50 |
experiments/results.md:1293:| exp249 ep60 | 70.2% | 80.7% | — | 91.1% | 远程 5060Ti, ep60 |
experiments/results.md:1294:| exp249 ep70 | 70.9% | 81.6% | — | 91.4% | 远程 5060Ti, ep70 |
experiments/results.md:1295:| exp249 ep80 | 71.5% | 81.4% | 89.4% | 91.5% | 远程 5060Ti, ep80 |
experiments/results.md:1296:| exp249 ep90 | 71.4% | 81.4% | 89.4% | 91.5% | 远程 5060Ti, ep90 |
experiments/results.md:1297:| exp249 ep100 | 71.7% | 82.3% | 89.6% | 91.8% | 远程 5060Ti, ep100 |
experiments/results.md:1298:| exp249 ep110 | 71.9% | 81.7% | 89.7% | 91.7% | 远程 5060Ti, ep110 |
experiments/results.md:1299:| **exp249 FINAL** | **71.9%** | **81.8%** | **89.5%** | **91.6%** | **远程 5060Ti, ep120 FINAL** ⭐⭐ |
experiments/results.md:1301:- **FINAL: mAP 71.9 (+0.3 vs exp245h_v2), R1 81.8 (+0.2 vs exp245h_v2)**
experiments/results.md:1303:- 对照 exp206r (Small GCN+PAA+OA-SD): 70.6/82.6 → **mAP +1.3, R1 -0.8**
experiments/results.md:1306:**exp249 MaxSim test (ep120 final)**:
experiments/results.md:1310:| exp249 equal_concat | 71.9% | 81.8% | 89.5% | 91.6% |
experiments/results.md:1311:| **exp249 MaxSim** | **73.3%** | **83.2%** | **90.9%** | **93.0%** |
experiments/results.md:1314:- **vs exp245h_v2 MaxSim (73.0/82.7): +0.3/+0.5 — 全面超越!**
experiments/results.md:1315:- **exp249 是项目新最佳: 73.3/83.2 (Small LGPA-D+GCN+OA-SD MaxSim)**
experiments/results.md:1317:### exp250: POT (Partial Optimal Transport) Test-time 评估 🟡
experiments/results.md:1319:在 exp246b (Tiny LGPA-D+GCN ep120) checkpoint 上测试:
experiments/results.md:1334:**exp245h_v2 (Small LGPA-D, best checkpoint) POT 结果:**
experiments/results.md:1349:### exp251: Tiny Multi-Stage PSG (Stage2+3) + PAA + LGPA-D+GCN
experiments/results.md:1353:| **exp251 FINAL** | **65.2%** | **76.2%** | 86.6% | 89.6% |
experiments/results.md:1354:| exp246b (Stage3 PSG+GCN) | 65.5% | 77.2% | — | — |
experiments/results.md:1358:- MSPSG+PAA vs single-stage: -0.3/-1.0 (seed variance 内)
experiments/results.md:1359:- 结论: multi-stage PSG 作为 novel design 有效，但不额外超越 single-stage
experiments/results.md:1361:### exp253: Tiny 3-Stage PSG (Stage1+2+3, 无 PAA) + LGPA-D+GCN
experiments/results.md:1365:| **exp253 FINAL** | **65.1%** | **76.2%** | 87.0% | 89.5% |
experiments/results.md:1366:| exp251 (2-stage+PAA) | 65.2% | 76.2% | 86.6% | 89.6% |
experiments/results.md:1367:| exp246b (1-stage) | 65.5% | 77.2% | — | — |
experiments/results.md:1371:- PAA 无贡献 (exp253 vs exp251 = -0.1/0.0)
experiments/results.md:1397:### exp255: Small GCN512 + 2-stage PSG + LGPA-D + OA-SD — NEW SMALL BEST
experiments/results.md:1399:| 方法 | mAP | R1 | R5 | R10 | vs exp249 |
experiments/results.md:1401:| **exp255 FINAL** | **73.2%** | **83.3%** | 90.4% | 92.3% | **+1.3/+1.5** |
experiments/results.md:1402:| exp255 MaxSim ep100 | 73.3% | 83.4% | — | — | +0.2/+0.3 (同epoch) |
experiments/results.md:1405:- exp255b (GCN512 + 1-stage): ≈ baseline — 2-stage PSG 是 GCN512 发挥的关键!
experiments/results.md:1406:- **exp255 seed42 FINAL: 73.1/83.1** (vs seed1234 73.2/83.3 = -0.1/-0.2)
experiments/results.md:1407:- **exp255 seed2024 FINAL: 72.6/82.0** (vs seed1234 73.2/83.3 = -0.6/-1.3)
experiments/results.md:1410:### exp256: Pose Prompt (KPR-style) — 负面/中性
experiments/results.md:1414:| exp256 (GCN512+2stage+Prompt, 进行中) | ep90: 72.4 | 82.3 | — | -0.5 vs exp255 |
experiments/results.md:1415:| **exp256b (GCN256+1stage+Prompt) FINAL** | **68.8** | **79.3** | **70.3/81.0** | **-3.1 vs exp249** |
experiments/results.md:1419:- **exp256 FINAL: 72.7/82.4** (vs exp255 73.2/83.3 = -0.5/-0.9)
experiments/results.md:1422:### exp257: ArcFace + Label Smoothing — 负面
experiments/results.md:1424:| 变体 | mAP | R1 | vs exp255 | 备注 |
experiments/results.md:1426:| exp257 (ArcFace m=0.35+LS, 远程) | 59.1% | 76.5% | -14.1/-6.8 | ep55 终止, ArcFace 严重不收敛 |
experiments/results.md:1427:| exp257b (Label Smooth only, 本地) | 71.5% | 81.7% | -1.7/-1.6 | ep86 终止, LS 稳定负面 |
experiments/results.md:1432:### exp258: ArcFace m=0.2 / GCN 3-layer — 负面/中性
experiments/results.md:1434:| 变体 | mAP | R1 | vs exp255 | 备注 |
experiments/results.md:1436:| exp258 (ArcFace m=0.2, 本地) | 67.7% | 81.2% | **-5.5/-2.1** | ArcFace 证伪 |
experiments/results.md:1437:| exp258b (GCN 3-layer, 远程) | 73.1% | 82.7% | -0.1/-0.6 | GCN 3-layer ≈ 2-layer |
experiments/results.md:1439:- ArcFace m=0.2: 比 m=0.35 好但仍 -5.5 mAP。ArcFace 在 Swin+SOLIDER pretrained 上完全证伪。
experiments/results.md:1442:### exp259: WD / OA-SD / DropPath 调参 — 全中性/负面
experiments/results.md:1444:| 变体 | mAP | R1 | vs exp255 | 备注 |
experiments/results.md:1446:| exp259 (WD=2e-4, 本地) | 72.2% | 82.1% | **-1.0/-1.2** | WD 过强负面 |
experiments/results.md:1447:| exp259b (OA-SD w=2.0, 远程) | 73.2% | 83.4% | 0.0/+0.1 | OA-SD=2.0 ≈ baseline |
experiments/results.md:1448:| exp259b MaxSim | 73.6% | 83.7% | +0.1/-0.1 | MaxSim 也持平 |
experiments/results.md:1449:| exp259c (dp=0.2, 本地, 进行中) | ep90: 72.6% | 82.7% | -0.6/-0.6 | dp=0.2 ≈ baseline |
experiments/results.md:1451:- **exp255 的 recipe (softmax CE, WD=1e-4, OA-SD=1.0, dp=0.1) 已是 SOLIDER Swin 上的最优 recipe。**
experiments/results.md:1452:- 所有 recipe 调参 (exp257-259) 均无法超越 baseline，recipe 空间已耗尽。
experiments/results.md:1454:### exp255 Test-Time Evaluations
experiments/results.md:1458:| exp255 equal_concat (baseline) | 73.2% | 83.3% | — | ep120 final |
experiments/results.md:1459:| exp255 global cosine | 72.7% | 82.3% | -0.5/-1.0 | global-only 模式 |
experiments/results.md:1460:| exp255 VisWeighted Part | 73.5% | 83.6% | +0.3/+0.3 | 可见部位加权 |
experiments/results.md:1461:| **exp255 MaxSim Hybrid** | **74.1%** | **84.6%** | **+0.9/+1.3** | **ep120 final, gw=1.0** |
experiments/results.md:1462:| **exp255 SGCFR α=0.5** | **74.0%** | **84.3%** | **+0.8/+1.0** | **top_k=5, vis_thr=0.3** |
experiments/results.md:1463:| exp255 SGCFR α=0.4 | 73.9% | 83.8% | +0.7/+0.5 | |
experiments/results.md:1464:| exp255 CVK hybrid α=0.7 | 72.2% | 82.6% | -1.0/-0.7 | CVK 无 recovery 反而负面 |
experiments/results.md:1465:| exp259b equal_concat | 73.2% | 83.4% | 0.0/+0.1 | OA-SD=2.0, ≈ exp255 |
experiments/results.md:1466:| exp259b MaxSim+flip | 75.1% | 85.4% | — | OA-SD=2.0, 略低于 exp255 (-0.1/-0.2) |
experiments/results.md:1468:| exp255 Global cosine+flip | 73.6% | 83.4% | +0.4/+0.1 | flip-test TTA |
experiments/results.md:1469:| **exp255 MaxSim+flip** | **75.2%** | **85.6%** | **+2.0/+2.3** | **⭐⭐⭐ 目标达成! flip-test+MaxSim** |
experiments/results.md:1474:### exp260: Base GCN512 + 2-stage PSG (LR=4e-4) — 未超 Small
experiments/results.md:1476:| 方法 | mAP | R1 | R5 | R10 | vs exp255 (Small) |
experiments/results.md:1478:| **exp260 FINAL** | **72.6%** | **81.6%** | — | 92.5% | **-0.6/-1.7** |
experiments/results.md:1481:- exp260 MaxSim+flip: 74.7/84.6 (仍低于 Small 75.2/85.6)
experiments/results.md:1483:### exp260b: Base GCN512 + 2-stage PSG (LR=8e-4) — 超越 Small!
experiments/results.md:1485:| 方法 | mAP | R1 | R5 | R10 | vs exp255 (Small) |
experiments/results.md:1487:| **exp260b FINAL** | **73.9%** | **83.2%** | — | — | **+0.7/-0.1** |
experiments/results.md:1488:| exp260b MaxSim+flip ep100 | 75.4% | 84.9% | — | — | +0.2/-0.7 (非final) |
experiments/results.md:1492:- **exp260b MaxSim+flip FINAL: 75.4/84.8** (vs Small 75.2/85.6 = +0.2/-0.8)
experiments/results.md:1495:### exp260b Market: Base GCN512 + 2-stage PSG (LR=8e-4, 无PLBOA)
experiments/results.md:1499:| **exp260b Market FINAL** | **94.4%** | **97.1%** | — | 99.4% | Base backbone |
experiments/results.md:1503:| exp260b Market MaxSim+flip | 94.7% | 97.2% | — | — | |
experiments/results.md:1504:| exp260b Market→Occluded-ReID (eq) | 86.0% | 88.5% | 95.3% | 97.9% | 跨数据集 equal_concat |
experiments/results.md:1505:| **exp260b Market→Occluded-ReID (MaxSim+flip)** | **88.0%** | **90.6%** | — | — | **跨数据集 MaxSim+flip** |
experiments/results.md:1511:> 机器: srvA/B/C = 5060 Ti 16G。本地 3090 挂，Base 3 run (exp263/266/269) DEFERRED。
experiments/results.md:1515:| exp261 | Swin-Tiny | Occ-Duke | **65.9% / 77.4%** | TBD | ✓ e120 FINAL @ 2026-04-19 04:16 srvB |
experiments/results.md:1516:| exp262 | Swin-Small | Occ-Duke | **73.8% / 83.1%** | TBD | ✓ e120 FINAL @ 2026-04-19 09:59 srvA (R5=90.2 R10=92.2). **略优 KPR w/o prompt 73.3/82.5** (+0.5/+0.6) |
experiments/results.md:1517:| exp263 | Swin-Base | Occ-Duke | **e100 eff FINAL: 72.5 / 81.8 (Global+flip), 74.5 / 84.0 (MaxSim+flip)** | ✓ @ 2026-04-20 09:01 srvB | ⚠️ e100 eval OOM-killed (内存 13.2G 触 16G),ckpt 100 完整,不重训。MaxSim hybrid+flip **74.5/84.0** 超 KPR w/o prompt +1.2/+1.5 |
experiments/results.md:1518:| exp263c | Swin-Base | Occ-Duke | ~~abandoned @ e31~~ | — lab3090 pwrlim280 seed 42 | seed 42 轨迹异常 (e10 2.7 / e20 17.0),用户指示换 seed 41 → 切 exp263d |
experiments/results.md:1519:| exp263d | Swin-Base | Occ-Duke | **74.1 / 83.3** | ✓ e120 FINAL @ 2026-04-21 14:27 lab3090 pwrlim 280W (R5=90.8 R10=93.0). **vs exp263 old e100 eff 72.5/81.8 Δ=+1.6/+1.5**. exp263 系列 PRCV 主表用此数字 (seed 41 替代 seed 42) |
experiments/results.md:1520:| exp263b | Swin-Base | Occ-Duke (seed 42 restart full 120) | **73.5 / 81.5 (train eq_concat), 74.8 / 84.0 (MaxSim+flip)** | ✓ e120 FINAL @ 2026-04-23 16:47:17 lab4090 4090 TEST.IMS_PER_BATCH 64 (R5=90.2 R10=92.3). MaxSim Global 72.4/81.4, hybrid 74.8/84.0 (+1.3/+2.5 vs eq_concat)。**vs exp263 old e100 eff 72.5/81.8 (eq) / 74.5/84.0 (MaxSim) Δ=+0.3/0** (MaxSim 侧 full 120 微优)。vs exp263d s41 MaxSim 75.2/84.8 Δ=-0.4/-0.8 (**seed 41 > seed 42 再次 confirmed**)。论文 Base OD 主表仍用 exp263d (seed 41 最强), exp263b 作 seed 42 full 120 复现点 |
experiments/results.md:1521:| exp294 | Swin-Base | Occ-Duke (LGPA-only / Full-GCN s41) | **74.0 / 82.6 (eq+flip), 75.0 / 84.4 (MaxSim+flip)** | ✓ e120 FINAL @ 2026-04-24 02:18:48 lab4090 TEST.IMS_PER_BATCH 64 (R5=90.5 R10=92.4). Global 73.5/83.3, **MaxSim 75.0/84.4** (+1.0/+1.8 vs eq_concat)。**vs exp263d Full+GCN s41**: eq 74.1/83.3 → -0.1/-0.7, **MaxSim 75.2/84.8 → -0.2/-0.4** (GCN 冗余双评测模式都验证)。vs exp263b Full+GCN s42 MaxSim 74.8/84.0: **+0.2/+0.4** (Full-GCN s41 > Full+GCN s42)。补 Phase 3-C Base 行, **3-backbone 统一结论 GCN 可移除** |
experiments/results.md:1522:| exp264 | Swin-Tiny | Occ-PTrack | **76.7% / 85.1%** | TBD | ✓ e120 FINAL @ 2026-04-19 07:15 srvC (R5=94.1 R10=97.0) |
experiments/results.md:1523:| exp265 | Swin-Small | Occ-PTrack | **78.4% / 86.2%** | TBD | ✓ e120 FINAL @ 2026-04-20 04:45 srvC (R5=94.8 R10=97.3, Small >> Tiny 76.7/85.1) |
experiments/results.md:1524:| exp266 | Swin-Base | Occ-PTrack | **e60 eff FINAL: 78.4 / 86.2 (peak e50: 78.5/86.3)** | ✓ @ 2026-04-20 21:27 srvC | ⚠️ e70 后 silent exit (非 OOM 非 CUDA, 推测 hy-tmp 平台 kill)。**Base 对 Small (exp265 78.4/86.2) 0 增益**, 不重训 |
experiments/results.md:1525:| exp265b | Swin-Small | Occ-PTrack (seed 41) | **78.5% / 85.9%** | ✓ e120 FINAL @ 2026-04-22 09:03 srvA 5060Ti (R5=94.7 R10=97.1) | **vs exp265 s42 78.4/86.2 Δ=+0.1/-0.3**。seed 41 微优 mAP 略弱 R1, 论文主表仍用 exp265 s42 (更高 R1), exp265b 作跨 seed 鲁棒性 supplementary |
experiments/results.md:1526:| exp266b (srvA) | Swin-Base | Occ-PTrack (seed 41) | **78.7% / 86.3%** | ✓ e120 FINAL @ 2026-04-23 13:18:50 srvA 5060Ti TEST.IMS_PER_BATCH 128 (R5=94.5 R10=97.1). **vs exp266b_3090 s41 78.5/86.2 Δ=+0.2/+0.1** (srvA 5060Ti 微优, 跨设备方差 0.2)。vs exp266 s42 e60 eff 78.4/86.2 Δ=+0.3/+0.1。vs exp265b Small s41 78.5/85.9 Δ=+0.2/+0.4 (**Base vs Small 同 s41 首次 R1 显著领先**)。**论文 Base OP 主表更新用此数字 78.7/86.3** (替代原 78.5) |
experiments/results.md:1527:| exp266b_3090 | Swin-Base | Occ-PTrack (seed 41) | **78.5% / 86.2%** | ✓ e120 FINAL @ 2026-04-22 09:29 lab3090 pwrlim 280W (R5=94.4 R10=96.9). **vs exp266 s42 e60 eff 78.4/86.2 Δ=+0.1/0** (持平)。vs exp265 Small 78.4/86.2 Δ=+0.1/0。vs exp265b Small s41 78.5/85.9 Δ=0/+0.3 |
experiments/results.md:1528:| exp267 | Swin-Tiny | Market | **92.5% / 96.4%** | TBD | ✓ e120 FINAL @ 2026-04-19 13:45 srvB (R5=98.9 R10=99.3) |
experiments/results.md:1529:| exp268 | Swin-Small | Market | **94.3% / 97.3%** | TBD | ✓ e120 FINAL @ 2026-04-20 00:39 srvA (R5=99.1 R10=99.5) |
experiments/results.md:1530:| exp269 | Swin-Base | Market | **e80 eff FINAL: 94.4 / 97.0 (Global+flip), 94.5 / 97.1 (MaxSim+flip)** | ✓ @ 2026-04-20 13:xx srvA | ⚠️ e80 eval OOM-killed 同 exp263 模式,ckpt80 完整,不重训。Base 对 Small 优势小(Market 已饱和) |
experiments/results.md:1531:| exp269b | Swin-Base | Market (seed 42 restart full 120, PLBOA OFF) | **94.5 / 97.2 (eq+flip), 94.6 / 97.2 (MaxSim+flip)** | ✓ e120 FINAL @ 2026-04-24 01:17:24 srvC 5060Ti TEST.IMS_PER_BATCH 64 (R5=99.1 R10=99.5). Global+flip 94.4/97.1, **MaxSim 94.6/97.2** (+0.1 mAP vs eq_concat)。**vs exp269 orig e80 eff**: eq 94.4/97.0 → +0.1/+0.2; MaxSim 94.5/97.1 → +0.1/+0.1。vs exp268 Small 94.3/97.3 Δ=+0.2/-0.1。vs exp293b Base PLBOA ON 93.8/97.2 Δ=+0.7/0 (**PLBOA 净 -0.7 mAP**)。**论文 Market Base 主数字升级 eq 94.5/97.2 / MaxSim 94.6/97.2** |
experiments/results.md:1539:| exp270 | Swin-Tiny | Occ-Duke | 无 (baseline) | **59.2 / 68.4** | ✓ e120 FINAL @ 2026-04-20 12:29 srvB (R5=82.2 R10=85.8). vs exp000 旧协议 56.6/66.5 → +default flip 贡献 +2.6/+1.9 |
experiments/results.md:1540:| exp271 | Swin-Tiny | Occ-Duke | `[-1]` (1-stage) | **60.2 / 69.5** | ✓ e120 FINAL @ 2026-04-20 16:36 srvB (R5=81.8 R10=85.9). vs exp270 Δ=+1.0/+1.1 = stage 3 PSG 独立贡献 |
experiments/results.md:1541:| exp272 | Swin-Tiny | Occ-Duke | `[-2,-1]` (2-stage) | **60.5 / 69.7** | ✓ e120 FINAL @ 2026-04-20 20:19 srvB (R5=82.6 R10=86.2). vs exp271 Δ=+0.3/+0.2 = stage 2 边际贡献微弱;vs exp270 Δ=+1.3/+1.3 = 2-stage 累计 |
experiments/results.md:1542:| exp273 | Swin-Tiny | Occ-Duke | `[-3,-2,-1]` (3-stage) | **60.5 / 69.9** | ✓ e120 FINAL @ 2026-04-21 00:05 srvB (R5=82.8 R10=87.0). vs exp272 2-stage Δ=0/+0.2 (stage 1 边际贡献 ~0 mAP). **Phase 3-A Tiny 矩阵完整**: 边际收益递减 +1.0 → +0.3 → 0 |
experiments/results.md:1543:| exp274 | Swin-Small | Occ-Duke | 无 (baseline) | **68.1 / 76.8** | ✓ e120 FINAL @ 2026-04-20 21:34 lab4090 (R5=87.8 R10=90.9). vs Tiny exp270 Δ=+8.9/+8.4 = Small vs Tiny backbone 容量差 |
experiments/results.md:1544:| exp275 | Swin-Small | Occ-Duke | `[-1]` (1-stage) | **68.8 / 76.8** | ✓ e120 FINAL @ 2026-04-20 23:37 lab4090 (R5=87.2 R10=90.4). vs exp274 no-PSG Δ=**+0.7/0** (mAP 涨 R1 持平). vs Tiny 1-stage 增益 (+1.0/+1.1),Small 上 +0.7/0 缩水 |
experiments/results.md:1545:| exp276 | Swin-Small | Occ-Duke | `[-2,-1]` (2-stage) | **68.3 / 77.2** | ✓ e120 FINAL @ 2026-04-21 01:41 lab4090 (R5=87.2 R10=90.1). vs exp275 1-stage Δ=-0.5 mAP/+0.4 R1 (**Small 上 2-stage 不同 Tiny,mAP 不涨但 R1 涨**) |
experiments/results.md:1546:| exp277 | Swin-Small | Occ-Duke | `[-3,-2,-1]` (3-stage) | ~~49.0 / 57.7 (seed 42 偶发塌缩)~~ | abandoned @ 2026-04-21 03:47 (e2 id_global 卡 3.277 classifier uniform). **改 exp277b seed 41 重跑** (用户判断偶发) |
experiments/results.md:1547:| exp277b | Swin-Small | Occ-Duke | `[-3,-2,-1]` (3-stage) | **68.3 / 77.6** | ✓ e120 FINAL @ 2026-04-21 23:34 lab4090 (R5=87.4 R10=89.8). **R1 最强 Phase 3-A Small!** vs exp277 s42 塌缩 49.0/57.7 Δ=+19.3/+19.9. vs exp276 2-stg 68.3/77.2 Δ=0/+0.4. **seed 41 完全验证 exp277 塌缩是偶发** |
experiments/results.md:1555:| exp281 (= exp261) | Swin-Tiny | 512 | `[-2,-1]` | **65.9 / 77.4** | Phase 1 共享,不重跑 |
experiments/results.md:1556:| exp278 | Swin-Tiny | 256 | `[-1]` | **65.7 / 76.7** | ✓ e120 FINAL @ 2026-04-21 10:42 srvB (R5=86.7 R10=89.6). vs exp261 GCN512+2stg 65.9/77.4 Δ=-0.2/-0.7. vs exp286 LGPA-only 66.0/76.6 Δ=-0.3/+0.1 (GCN256 略弱于 no GCN) |
experiments/results.md:1557:| exp279 | Swin-Tiny | 256 | `[-2,-1]` | **65.7 / 76.9** | ✓ e120 FINAL @ 2026-04-21 21:32 srvB (R5=86.6 R10=90.1). vs exp278 GCN256+1stg 65.7/76.7 Δ=0/+0.2 (mAP 持平 R1 +0.2). vs exp261 65.9/77.4 Δ=-0.2/-0.5 |
experiments/results.md:1558:| exp280 | Swin-Tiny | 512 | `[-1]` | **65.7 / 76.2** | ✓ e120 FINAL @ 2026-04-22 08:07 srvB (R5=86.7 R10=89.7). **vs exp261 GCN512+2stg 65.9/77.4 Δ=-0.2/-1.2** (最弱 R1 格), vs exp278 GCN256+1stg 65.7/76.7 Δ=0/-0.5. **Phase 3-B Tiny 2×2 闭合: GCN256+1stg=GCN256+2stg=GCN512+1stg mAP 全 65.7, GCN512+2stg 唯一 65.9**。和 Small 2×2 GCN512+1stg 最弱同模式 |
experiments/results.md:1559:| exp285 (= exp262) | Swin-Small | 512 | `[-2,-1]` | **73.8 / 83.1** | Phase 1 共享, srvA 5060Ti (原始), 已 re-eval flip fix 后 73.8/83.1 no-op |
experiments/results.md:1560:| exp285b | Swin-Small | 512 | `[-2,-1]` | **73.8 / 83.8** | ✓ e120 FINAL @ 2026-04-22 06:04 lab4090 (R5=90.7 R10=92.7). **vs exp262 (srvA old) 73.8/83.1 Δ=0/+0.7** (mAP 持平, R1 +0.7 lab4090 > srvA). **Phase 3-B Small 矩阵 gold-standard**, 论文主表用此数字 |
experiments/results.md:1561:| exp282 | Swin-Small | 256 | `[-1]` | **73.7 / 83.9** | ✓ e120 FINAL @ 2026-04-21 09:33 lab4090 (R5=90.5 R10=92.5). **vs exp262 73.8/83.1: mAP -0.1 R1 +0.8** → low-cap ≥ high-cap, Small Full Scaffold 容量饱和 |
experiments/results.md:1562:| exp283 | Swin-Small | 256 | `[-2,-1]` | **73.5 / 83.2** | ✓ e120 FINAL @ 2026-04-21 15:38 lab4090 (R5=90.7 R10=92.5). vs exp262 73.8/83.1 Δ=-0.3/+0.1. vs exp282 73.7/83.9 Δ=-0.2/-0.7 |
experiments/results.md:1563:| exp284 | Swin-Small | 512 | `[-1]` | **73.4 / 82.9** | ✓ e120 FINAL @ 2026-04-21 21:23 lab4090 (R5=89.9 R10=92.2). vs exp262 73.8/83.1 Δ=-0.4/-0.2. **Phase 3-B Small 2x2 完整: GCN256+1stg (83.9) 最 R1, GCN512+2stg (73.8 mAP) 最 mAP; GCN512+1stg 反而最弱** |
experiments/results.md:1567:> Phase 3-C: **LGPA-only + 变量 PSG_STAGES** (关 GCN, 保留 LGPA/OA-SD/ParAug/LOWER_BODY_OCC)。回答"2-stage PSG 的收益是偏 structural 还是 semantic branch 也吃"。srvC exp266 silent exit 后空闲,利用上。
experiments/results.md:1571:| exp286 | Swin-Tiny | `[-1]` | **66.0 / 76.6** | ✓ e120 FINAL @ 2026-04-21 10:03 srvC (R5=86.4 R10=89.7). **vs exp261 Full Scaffold 65.9/77.4 Δ=+0.1/-0.8** → GCN 对 Tiny 几乎无贡献, LGPA-only 等价 Full |
experiments/results.md:1572:| exp287 | Swin-Tiny | `[-2,-1]` | **65.9 / 77.0** | ✓ e120 FINAL @ 2026-04-21 20:48 srvC (R5=87.0 R10=89.7). vs exp286 LGPA-only 1stg 66.0/76.6 Δ=-0.1/+0.4 (2-stg R1 微优). vs exp261 Full 65.9/77.4 Δ=0/-0.4 (GCN 主要给 R1) |
experiments/results.md:1573:| exp288 | Swin-Small | `[-1]` | **73.8 / 83.8** | ✓ e120 FINAL @ 2026-04-22 12:51 srvC (R5=90.5 R10=92.0). 🔥 **vs exp285b Full Scaffold 73.8/83.8 完全持平** (mAP/R1 identical, R5/R10 微差 0.2/0.7)。vs exp282 Full GCN256+1stg 73.7/83.9 Δ=+0.1/-0.1。**证实 GCN 对 Small OD 零贡献**, LGPA 单独达 Full Scaffold 性能 |
experiments/results.md:1574:| exp289 | Swin-Small | `[-2,-1]` | **73.8 / 83.3** | ✓ e120 FINAL @ 2026-04-23 05:39 srvC (R5=90.5 R10=92.4). **vs exp288 1-stg 73.8/83.8 Δ=0/-0.5**, vs exp285b Full Scaffold 73.8/83.8 Δ=0/-0.5 — **mAP 完全持平 Full Scaffold, GCN 零贡献 reconfirmed**. 和 Tiny Phase 3-C (exp287 2-stg 65.9/77.0 vs exp286 1-stg 66.0/76.6) 方向相反 (Small 1-stg R1 微优, Tiny 2-stg R1 微优), 但 mAP 均持平 |
experiments/results.md:1580:| exp290 | Swin-Small | Occ-PTrack | 42 | **78.4 / 86.2** | ✓ e120 FINAL @ 2026-04-23 09:22 srvB (R5=94.8 R10=97.4). 🔥 **严格持平 exp265 scene baseline 78.4/86.2/94.8/97.3** (Δ 0/0/0/+0.1). target-heatmap 3 数据集全 near-no-op, OP 多人场景预期增益未实现 |
experiments/results.md:1581:| exp291 | Swin-Small | Occ-Duke | 42 | **73.5 / 82.9** | exp285b 73.8/83.8 (Δ -0.3/-0.9) | ✓ e120 FINAL @ 2026-04-22 18:13 lab4090 (R5=90.7 R10=92.5). OD 多单人场景 near no-op, 机制无显著回归 |
experiments/results.md:1582:| exp292 | Swin-Small | Market | 42 | **e90 eff FINAL: 94.2 / 97.1** | exp268 FINAL 94.3/97.3 (Δ -0.1/-0.2 持平) | ✓ 停于 e93 @ 2026-04-22 23:25 用户让出 lab3090。R5 99.2 R10 99.5 = exp268 FINAL R5 99.1 R10 99.5 |
experiments/results.md:1583:| exp293 | Swin-Base | Market + **PLBOA** | 42 | **e120 FINAL (restart): 93.8 / 97.2** (完整 120ep) | exp269 e80 eff 94.4/97.0 (Δ -0.6/+0.2); first run e80 eff 94.1/96.9 (Δ -0.3/+0.3 跨 restart 方差) | ✓ restart full 120 @ 2026-04-23 08:24 lab4090 (R5=98.9 R10=99.5). First run e80 eval OOM, 重启 w/ TEST.IMS_PER_BATCH 64. **PLBOA 在 Market full 120 net -0.6 mAP / +0.2 R1** (vs exp269 PLBOA OFF e80) — 主表待 exp269b FINAL 公平对比 |
experiments/results.md:1585:## Post-PRCV 消融/复现/扫参 runs (exp295–321b, 2026-04-25~28)
experiments/results.md:1590:### exp295–304: 复现 / multi-seed / LR sweep / Phase 3-D LGPA 消融
experiments/results.md:1596:| exp295 | Swin-Small | Full Scaffold 复现 exp255 | lab4090 / 1234 | **74.2 / 84.0** | **75.2 / 85.4** | vs exp255 hist 75.2/85.6: **0 / -0.2** | ✅ 完全重现 exp255 75.2 mAP，证历史数字真实可复现（非 eval bug）。**Small OD 主表新 reference** |
experiments/results.md:1597:| exp296 | Swin-Base | LR 8e-4 复现 exp263d | lab4090 / 41 | 73.7 / 81.7 | 74.9 / 83.8 | vs exp263d 75.2/84.8: **-0.3 / -1.0** | reproducibility 接近但 R1 系统性偏低（lab4090 vs lab3090 硬件差）；主表仍用 exp263d |
experiments/results.md:1598:| exp297 | Swin-Base | **LR 4e-4** | srvA(5060Ti) / 41 | 73.2 / 82.4 | 74.6 / 84.1 | vs exp296 LR8: **-0.3 / +0.3**（近 tie） | LR4 vs LR8 接近持平，**非显著 underfit**；比 hist exp260 LR4(72.6) 高 0.6 mAP |
experiments/results.md:1599:| exp298 | Swin-Base | **LR 2e-4**（下界） | srvB(5060Ti) / 41 | 68.6 / 78.6 | 69.6 / 79.1 | vs exp296 LR8: **-5.3 / -4.7** | LR2 严重 underfit（e10 mAP 1.3 near-random），LR ablation 下界，证 LR8 不能再降 |
experiments/results.md:1600:| exp299 | Swin-Base | **PLBOA OFF** | srvC(5060Ti) / 41 | 70.9 / 78.0 | 72.7 / 80.5 | vs exp296 PLBOA ON: **-2.2 / -3.3** | OD 上 PLBOA net positive **+2.2 mAP MaxSim**；与 Market 上 PLBOA 有害形成 dataset-specific claim |
experiments/results.md:1601:| exp300 | Swin-Base | Full Scaffold seed 1234 | lab4090 / 1234 | 74.0 / 83.8 | 75.0 / 85.0（e100 ckpt 75.0/85.2） | vs exp263d 75.2/84.8: **-0.2 / +0.2**（e120） | 未破 exp263d SOTA mAP，但 R1 +0.2~0.4 微超；e100 ckpt R1 peak 85.2 |
experiments/results.md:1602:| exp301 | Swin-Small | **LGPA OFF**（Phase 3-D） | lab4090 / 42 | 71.9 / 83.0 | 71.9 / 83.0（MaxSim **0 boost**） | vs exp285b Full 74.7/84.8: **-2.8 / -1.8** | LGPA 贡献 +2.8 mAP MaxSim；移除 LGPA → MaxSim 失去 boost（LGPA 是 MaxSim 主驱动） |
experiments/results.md:1603:| exp302 | Swin-Base | Full Scaffold seed 42（multi-seed 第3） | srvA(5060Ti) / 42 | 73.3 / 81.4 | 74.4 / 83.6 | vs exp263d 75.2/84.8: **-0.8 / -1.2** | Base 3-seed(41/1234/42) MaxSim mAP mean **74.87 std 0.42**；主行仍用 exp263d |
experiments/results.md:1604:| exp303 | Swin-Tiny | **LR 4e-4** | srvB(5060Ti) / 41 | 64.4 / 74.8 | 65.7 / 76.1 | vs exp261 LR8 67.2/78.6: **-1.5 / -2.5** | Tiny LR4 underfit -1.5 mAP；LR8 仍 sweet spot（Tiny 比 Base 更 LR 敏感） |
experiments/results.md:1605:| exp304 | Swin-Small | Full Scaffold seed 2024（multi-seed 第3） | srvC(5060Ti) / 2024 | 73.3 / 82.7 | 74.3 / 84.0 | vs exp295 75.2/85.4: **-0.9 / -1.4** | Small 3-seed(42/1234/2024) MaxSim mAP mean **74.7 std 0.45**；主行仍用 exp295 |
experiments/results.md:1611:| exp305 | Swin-Tiny | **LGPA OFF**（mirror exp301） | lab4090 / 42 | 64.5 / 76.0 | 64.5 / 76.0（**0 boost**） | vs exp261 67.2/78.6: **-2.7 / -2.6** | LGPA 贡献 +2.7 mAP MaxSim（+1.4 eq）；Phase 3-D Tiny+Small 双 backbone 完整 |
experiments/results.md:1612:| exp307 | Swin-Tiny | **PLBOA OFF**（mirror exp299） | srvB(5060Ti) / 42 | 62.8 / 71.8 | 64.5 / 73.5 | vs exp261 67.2/78.6: **-2.7 / -5.1** | Tiny PLBOA net positive **+2.7 mAP**；与 Base(+2.2) 一致。PLBOA dataset-specific 2-backbone evidence |
experiments/results.md:1616:> commit `c059dca` 修复 GLOBAL_LOSS_SCALE 只在 no-part 路径生效的 bug（Full Scaffold 此前完全忽略，effective=1.0）。exp311+ 后 scale 才真在 part-path 生效。Tiny sweep seed 42 / baseline exp261(67.2/78.6)；Small 验证 seed 1234 / baseline exp295(75.2/85.4)。
experiments/results.md:1620:| exp311b | Swin-Small | **GLOBAL_LOSS_SCALE 0.5**（bugfix 后真生效） | lab4090 / 1234 | 73.5 / 83.2（e100 eff，e101 OOM） | 74.5 / 84.8 | vs exp295: **-0.7 / -0.6** | 0.5× global 真生效后 net **-0.7 mAP**，非有效改进；effective 1.0 更好 |
experiments/results.md:1621:| exp312 | Swin-Tiny | **GLOBAL_LOSS_SCALE 2.0** | lab4090 / 42 | 65.7 / 76.6 | 66.8 / 77.2 | vs exp261: **-0.4 / -1.4** | 2.0× 也 net negative。结合 exp311b(0.5×负)，**双向都负 → 1.0 sweet spot**（推翻早期 0.5） |
experiments/results.md:1622:| exp313 | Swin-Tiny | **POSE_PART_WEIGHT 2.0**（ID favor part） | srvA(5060Ti) / 42 | 65.8 / 77.0 | 66.9 / 77.9 | vs exp261: **-0.3 / -0.7** | favor part 微 negative |
experiments/results.md:1623:| exp314 | Swin-Tiny | **POSE_PART_WEIGHT 0.5**（ID favor global） | srvB(5060Ti) / 42 | 65.8 / 77.5 | 67.2 / 78.6 | vs exp261: **0 / 0**（完全相等） | favor global net neutral；default 1.0 双 sweet spot |
experiments/results.md:1624:| exp315 | Swin-Tiny | **POSE_LGPA_ASSIGN_WEIGHT 1.0**（LGPA aux ×2） | srvC(5060Ti) / 42 | 65.8 / 76.9 | 67.0 / 77.4 | vs exp261: **-0.2 / -1.2** | LGPA aux 加倍 net negative；default 0.5 sweet spot |
experiments/results.md:1625:| exp316 | Swin-Tiny | **POSE_OA_SD_WEIGHT 2.0** | lab4090 / 42 | 66.0 / 77.6 | 67.2 / 78.0 | vs exp261: **0 / -0.6** | OA-SD ×2 net neutral；default 1.0 sweet spot |
experiments/results.md:1626:| exp317 | Swin-Tiny | **POSE_LGPA_ASSIGN_WEIGHT 0.25**（LGPA aux ÷2） | lab3090 / 42 | 66.2 / 77.4 | 67.4 / 78.6 | vs exp261: **+0.2 / 0** ⭐ | sweep 中**唯一 MaxSim 超 baseline**(+0.2)，但在 multi-seed std 内，需 Small 验证 |
experiments/results.md:1627:| exp318 | Swin-Tiny | **POSE_PART_TRI_WEIGHT 0.5**（Tri favor global） | srvB(5060Ti) / 42 | 65.9 / 77.7 | 67.1 / 78.3 | vs exp261: **-0.1 / -0.3** | Tri-side favor global slight neg；与 exp314 合证 default 双 sweet spot |
experiments/results.md:1628:| exp319 | Swin-Tiny | **POSE_OA_SD_WEIGHT 0.5** | srvC(5060Ti) / 42 | 65.8 / 76.8 | 67.1 / 78.1 | vs exp261: **-0.1 / -0.5** | OA-SD ÷2 slight neg；与 exp316(×2) 合证 default 1.0 sweet spot |
experiments/results.md:1629:| exp320 | Swin-Small | **POSE_LGPA_DETACH=False**（LGPA aux 反传 backbone） | lab4090 / 1234 | 68.1 / 79.3 | 68.8 / 79.6 | vs exp295: **-6.4 / -5.8** | **catastrophic -6.4 mAP**（e10 46% underfit）；证 LGPA detach 必要。强 negative 消融素材 |
experiments/results.md:1630:| exp321b | Swin-Small | **POSE_LGPA_ASSIGN_WEIGHT 0.25**（验证 exp317） | lab4090 / 1234 | 73.9 / 83.7 | 74.9 / 85.4 | vs exp295: **-0.3 / 0** | Tiny exp317 的 +0.2 **未迁移到 Small**（slight -0.3）→ 判 seed noise，保持 default 0.5 |
experiments/results.md:1632:> 跳号说明：exp306/308/309/310/321a/321c 无目录（实验号跳过/未跑，非数据丢失）。exp311(s42) e10 即被 kill，以 exp311b(s1234) 计入。exp296/exp302 R1 跨设备系统性偏低 1-1.6（5060Ti/lab4090 vs lab3090），主表用 lab3090 exp263d 不受影响。
experiments/results.md:1634:### exp323: MLLM 视觉裁剪 A/B 廉价首验（inference-only，非训练）
experiments/results.md:1650:- **结论**：frozen 小 MLLM + pose 视觉裁剪/文字提示这条廉价首验**不正向，建议砍**（kill-switch 信号明确）。
experiments/results.md:1652:### exp324: frozen DINOv2 emergent correspondence + pose-anchored part-MaxSim（inference-only，非训练）
experiments/results.md:1654:> post-PRCV「搬范式」#2 路线。frozen DINOv2-base（lab-3090-d, RTX 3090），全量 Occluded-Duke（2210 query × 17661 gallery，无后处理、无训练）。脚本 `scripts/exp324_dino.py`。输入 224W×448H → patch grid 32×16。keypoints 缩放到 grid → 每部位 3×3 窗均值池化成 5 个 part 向量 + per-part visibility，跨图只比 mutually-visible part 的 per-part cosine（part-MaxSim）。重遮挡子集 = query visibility_binary.sum()≤8（989/2210）。**training-free，不计入主表增益。**
experiments/results.md:1665:- 绝对分低（pose-part heavy 1.86 mAP）符合 DINO 零样本 ReID 文献区间（0.3-4.7 mAP）。
experiments/results.md:1667:- **结论**：机制**有明确相对信号**，pose-anchored DINO correspondence 在重遮挡上 3-4 倍超整图基准且 pose 锚定占绝对主导 → kill-switch 命中正向条件，**值得 exp324b 上轻量 part-projection 头 / LoRA**。
experiments/results.md:1669:### exp327: 更强/更新冻结对应特征源（DINOv2-with-registers）— pose-part-MaxSim training-free 天花板 check（inference-only）
experiments/results.md:1671:> 同 exp324 pipeline（pose 锚定 5-part + mutually-visible part-MaxSim + 重遮挡 vis.sum()≤8），**唯一变量=特征源**。hyy GPU1（5060 Ti），slim pose data（剥 heatmap，数值与 exp324 一致）。脚本 `scripts/exp327_dinov3.py`。**training-free，不计入主表增益。** DINOv3-vitb16 gated（hf-mirror 需 token）下不了，改用 ungated 的 `facebook/dinov2-with-registers-base`（registers 去 high-norm artifact token，更干净 dense 特征，patch14 grid 32×16，nreg=4）。
experiments/results.md:1680:- **vs exp324 DINOv2-base（heavy pose-part 1.86/3.54）**：dinov2reg-b heavy **2.15/3.84（+0.29 mAP / +0.30 R1）**；ALL 3.85/8.60 vs 3.21/7.87（+0.64/+0.73）。
experiments/results.md:1694:- **vs exp324 DINOv2-base（heavy pose-part 1.86）**：DIFT heavy **0.73（−1.13 mAP）**——**DIFT 全量明显劣于 DINOv2-base**，更不及 dinov2-registers 的 2.15。
experiments/results.md:1695:- 机制方向仍在（pose 0.73 > grid 0.35 > holistic 0.22，pose vs grid +0.38），但绝对判别性远低于 DINO。
experiments/results.md:1696:- **smoke 误导**：smoke（500 gallery）DIFT heavy 9.92，full（17661 gallery）塌到 0.73。DINO 从 smoke 2.55→full 1.86 仅小降，DIFT 从 9.92→0.73 灾难性塌 → **SD 特征 category-level 语义对应强（PCK 高）但 instance-level 身份判别弱**（SD-DINO/Tale-of-Two-Features 文献一致）。
experiments/results.md:1697:- **结论**：**SD/DIFT 特征不值得上头**（决定性问题答案=否），SD 线止损。教训：训练-free probe 必须用全量 gallery 判定，小 gallery smoke 只看流程不看绝对值。耗时 2065s（feature 1650s ensemble4 慢 + rep 405s）。
experiments/results.md:1699:### exp324d / exp324i: LoRA-unfreeze DINOv2 + 解相关对照（破冻结天花板 + 张力鲁棒性）
experiments/results.md:1701:> exp324d = LoRA 解冻 DINOv2-base/large + 可微 pose-part-MaxSim（破 exp324b 冻结天花板）。exp324i = 在其上加跨网络跨协方差解相关损失（逼 DINO-global 与 frozen-Swin-global 线性无关）。Occluded-Duke，BS=64，rank16 除非标注。**单分支 part-MaxSim = 纯模型；fusion(⊕Swin) = test-time 后处理(NFC 级)，不计训练端增益。**
experiments/results.md:1714:- **LoRA 解冻决定性破冻结天花板**（8.65→40+ heavy，~4.7×）→ 瓶颈是 "frozen" 不是 DINO 表征。
experiments/results.md:1732:| single-branch heavy | 39.05 | 38.69 | 38.18 |
experiments/results.md:1754:| **exp349** | 强系统 exp255(73.2) + CLIP | **71.4/71.3**(eq/global) | **CLIP有害 -1.8** |
experiments/paper_materials/story.md:7:### 当前一句话故事
experiments/paper_materials/story.md:11:### 当前重审结论
experiments/paper_materials/story.md:15:当前更稳的写法是：
experiments/paper_materials/story.md:20:5. `MaxSim / POT / flip` 只作为 supporting evaluation，不当训练端主贡献
experiments/paper_materials/story.md:22:### 当前主判断
experiments/paper_materials/story.md:32:   - `exp249` 与 `exp246` 已经说明 `LGPA-D + GCN` 双分支具备稳定互补性
experiments/paper_materials/story.md:35:   - `exp009`、`exp251`、`exp253` 都说明：multi-stage 不会在所有 scaffold 上自动更强
experiments/paper_materials/story.md:36:   - 但 `exp255 vs exp255b` 明确说明：在 `GCN512` 这类高容量结构分支上，`2-stage PSG` 是关键条件
experiments/paper_materials/story.md:40:### 当前最强系统 scaffold
experiments/paper_materials/story.md:42:当前训练端最强实验是 `exp255`：
experiments/paper_materials/story.md:46:当前最关键的结构证据是：
experiments/paper_materials/story.md:47:- `exp255`: `GCN512 + 2-stage PSG = 73.2 / 83.3`
experiments/paper_materials/story.md:48:- `exp255b`: `GCN512 + 1-stage PSG = 71.5 / 81.9`
experiments/paper_materials/story.md:87:   - 写法：test-time supporting evaluations
experiments/paper_materials/story.md:93:3. 在 Occluded-Duke 上系统验证该框架，并采用 `2-stage PSG` 作为最终实现；实验表明该设计能够更稳定地支撑高容量结构分支，最终在 Swin-Small 上得到当前最佳训练端结果之一
experiments/paper_materials/story.md:109:   - 在 Occluded-Duke 等基准上，该框架取得了当前项目最优结果之一，其中 `Swin-Small` 配置达到 `73.2 / 83.3`；消融进一步表明，最终采用的 `2-stage PSG` 更适合支撑高容量结构分支。
experiments/paper_materials/story.md:131:### 当前最佳结果
experiments/paper_materials/story.md:140:| *Small* | *LGPA-D+GCN+OA-SD (exp249, 进行中)* | *TBD* | *TBD* | *TBD* | *TBD* |
experiments/paper_materials/story.md:170:   - Non-detached (exp243): ep80 -1.1 mAP → 后期干扰
experiments/paper_materials/story.md:171:   - Detached (exp244): ep120 +2.1 mAP → 全程正向
experiments/paper_materials/story.md:194:### 当前最佳结果 (Phase 4 时期)
experiments/paper_materials/story.md:211:4. **per-keypoint training loss 全面证伪**: PKC, MST, PACI, OERL, BA-PKC — 10 个实验全部失败。根本原因: detached GCN 阻断梯度到 backbone，non-detached 与 CE 冲突。
experiments/paper_materials/story.md:254:- MaxSim training: exp152 进行中
experiments/paper_materials/story.md:255:- Ablation: soft vs hard MaxSim (exp152 vs exp152b) 进行中
experiments/paper_materials/story.md:264:### 当前最可靠的核心发现
experiments/paper_materials/story.md:287:### 2026-03-16 周度评估：当前 story 仍不够支撑 B 类主线
experiments/paper_materials/story.md:310:- ROA 是当前最有效的单一改进（数据增强级）
experiments/paper_materials/story.md:314:#### 发现 3: TDPC 方向全面证伪
experiments/paper_materials/story.md:318:#### 发现 4: Transformer Decoder 在当前数据量不可行
experiments/paper_materials/story.md:321:#### 当前进行中
experiments/paper_materials/story.md:332:### 当前主结果表（Occluded-Duke, Swin-Tiny, 4090）
experiments/paper_materials/story.md:342:| **PSG + GCN** | **equal_concat** | **60.73±0.47%** | **72.57±0.58%** | **当前最强且已确认的无后处理模式** |
experiments/paper_materials/story.md:370:### 当前可 claim 的贡献
experiments/paper_materials/story.md:375:### 当前不应再主张的结论
experiments/paper_materials/story.md:395:3. **因此下一阶段的候选主线应转向 retrieval-time common-support reasoning**
experiments/paper_materials/story.md:399:   - 检索阶段基于 query-gallery 共同可见关键点进行距离推理
experiments/paper_materials/story.md:411:当前可得出的更细判断是：
experiments/paper_materials/story.md:413:2. 但它 **不适合单独替代** 当前主距离，因为 `cvk_only` mAP 明显下降。
experiments/paper_materials/story.md:434:3. 因而当前更有把握的表述是：
experiments/paper_materials/story.md:435:   **Skeleton branch 的价值不只是在 embedding-level concat，更可能在 retrieval-time 提供 pair-specific common-support correction。**
experiments/paper_materials/story.md:451:因此当前更稳的叙事可以写成：
experiments/paper_materials/story.md:453:- CVK reasoning 提供 pair-specific common-support correction
experiments/paper_materials/story.md:470:   **CVK 不是 top-1 booster，而是 deeper-rank common-support correction。**
experiments/paper_materials/story.md:474:- 它是当前最完整的机制证据，但不应被写成所有 checkpoint 都必须出现的固定代价
experiments/paper_materials/story.md:492:3. 这样 qualitative 部分就能和当前 story 保持一致：
experiments/paper_materials/story.md:512:因此当前更稳的论文叙事应写成：
experiments/paper_materials/story.md:523:2. 它说明当前最该推进的已经不是继续做资产恢复，而是把 common-support 机制真正推进到训练端验证。
experiments/paper_materials/story.md:530:### 2026-03-13 下一跳候选：CSGT（训练端化 common-support）
experiments/paper_materials/story.md:531:基于当前两类事实：
experiments/paper_materials/story.md:535:当前最合理的训练端候选，不是再做一个融合模块，而是：
experiments/paper_materials/story.md:536:- 用 skeleton branch 的 `kp_weights` 构造 batch 内 common-support overlap
experiments/paper_materials/story.md:537:- 在 global triplet 上加一条 support-aware hard mining 约束
experiments/paper_materials/story.md:698:  **如果我们要讲 ambiguity / confuser 这条主线，机制必须发生在 per-keypoint / common-support 粒度，而不是 person-level pooled feature 粒度。**
experiments/paper_materials/story.md:704:  “只要把 confuser reasoning 下沉到 per-keypoint / common-support，再做 duplicate-aware penalty，就能在 retrieval-time 把 ambiguity 解开。”
experiments/paper_materials/story.md:710:  2. 有效信息更像是 target-target common-support 的正向匹配，而不是额外的反事实 penalty
experiments/paper_materials/story.md:719:  如果我们给每张图一个 GT same-ID 的 per-keypoint support bank，`cvk_hybrid` 会从 `61.88 / 73.26` 直接跃升到 `66.15 / 77.87`；
experiments/paper_materials/story.md:722:  **主要缺口不是缺一个更聪明的比较公式，而是缺一个更完整的 latent support representation。**
experiments/paper_materials/story.md:728:  2. GCN/CVK 暴露出 common-support reasoning 的真实性
experiments/paper_materials/story.md:729:  3. 但真正的缺口在于：单图只包含 partial support
experiments/paper_materials/story.md:731:     **用 same-ID multi-view support 作为 teacher，把 single-image representation 蒸馏成更接近 support-complete 的关键点表征**
experiments/paper_materials/story.md:740:  **support-complete 不再只是 oracle 上界，而是已经能作为真实训练信号转正。**
experiments/paper_materials/story.md:743:  2. oracle support bank 证明缺口来自 incomplete support
experiments/paper_materials/story.md:745:     训练端确实可以把 same-ID support 转成正向监督
experiments/paper_materials/story.md:747:     **如何让 support teacher 更可靠、更接近真正的 multi-view support-complete representation**
experiments/paper_materials/story.md:761:  **当前最值得讲的，不只是 reliable support，更是 stable / non-hardening support teacher。**
experiments/paper_materials/story.md:764:  1. pose 定义 keypoint-level support units
experiments/paper_materials/story.md:765:  2. same-ID bank 提供 support-complete teacher
experiments/paper_materials/story.md:767:  4. 因而需要一个更可靠、更稳定的 support-complete learning 机制
experiments/paper_materials/story.md:774:  **如何把 `cvk_hybrid` 已验证过的 pairwise common-support 几何，迁到训练端并写进 global embedding。**
experiments/paper_materials/story.md:776:- 这使得当前的候选主叙事进一步变成：
experiments/paper_materials/story.md:779:  3. prototype bank 失败，说明问题不该被压成 `per-ID average support`
experiments/paper_materials/story.md:781:     **把 common-support pairwise 几何作为 privileged relational teacher，蒸馏到 global embedding**
experiments/paper_materials/story.md:802:  2. 所以它虽然能传递 `pair comparability`，但 teacher 自身还受 `support incomplete` 限制
experiments/paper_materials/story.md:804:     **需要一个 support-complete relational teacher，而不是 prototype pointwise teacher**
experiments/paper_materials/story.md:806:- 因而当前最合理的主叙事升级为：
experiments/paper_materials/story.md:807:  1. 单图遮挡导致 `support incomplete`
experiments/paper_materials/story.md:810:  4. 但 teacher 还必须被 support-complete 化，才能把 `exp109` 的 headroom 真正转成训练端收益
experiments/paper_materials/story.md:815:  它没有把指标推高，却把当前瓶颈说得更清楚了。
experiments/paper_materials/story.md:822:  1. support-complete teacher 真实在工作
experiments/paper_materials/story.md:827:  “support-complete relational teacher 不对”
experiments/paper_materials/story.md:830:  1. `support-complete teacher` 是必要的，但**不是充分的**
experiments/paper_materials/story.md:831:  2. oracle headroom 主要属于低可见 / support-incomplete 样本
experiments/paper_materials/story.md:835:  **pose-guided support-complete relational distillation 必须是 selective 的。**
experiments/paper_materials/story.md:837:- 这让当前最合理的下一跳不再是“更强 teacher”，而是：
experiments/paper_materials/story.md:838:  **按 sample-level support gap 分配 distillation 强度。**
experiments/paper_materials/story.md:843:  不是 `support-complete teacher` 不对，
experiments/paper_materials/story.md:857:  1. `support incomplete` 的影响不是均匀落在整张图/整个人样本上
experiments/paper_materials/story.md:859:  3. 所以 `support-complete relational distillation` 不能只做 sample-level selective
experiments/paper_materials/story.md:864:  收紧成了“只蒸馏那些被 support completion 真正改变过的关系”。
experiments/paper_materials/story.md:866:## 2026-03-20: exp121/123 把 story 进一步分成“supporting mechanism”与“主突破口”
experiments/paper_materials/story.md:874:  2. support-complete relational teacher 的确受 teacher stability 影响
experiments/paper_materials/story.md:875:  3. 但这个量级更像 supporting mechanism，而不是足以单独支撑整篇论文的方法核心
experiments/paper_materials/story.md:891:- 所以当前 story 最合理的下一步不是再讲：
experiments/paper_materials/story.md:895:  **pair-level teacher-change focusing 是对的，但当前第一版 focus 强度太弱。**
experiments/paper_materials/story.md:898:  1. pose/keypoint branch 定义 common-support relations
experiments/paper_materials/story.md:899:  2. support-complete bank 只负责增强 relational teacher
experiments/paper_materials/story.md:900:  3. stable teacher 是 supporting mechanism
experiments/paper_materials/story.md:904:### 当前最自然的下一跳
experiments/paper_materials/story.md:930:- 所以当前 story 最合理的新收束不再是：
experiments/paper_materials/story.md:937:     **sparse pair routing for support-complete relational distillation**
experiments/paper_materials/story.md:939:- 换句话说，当前主创新点已经越来越不像“再做一个 loss 权重”，而更像：
experiments/paper_materials/story.md:940:  **只把被 support completion 真正改变过的 comparability relations 蒸进 global embedding。**
experiments/paper_materials/story.md:945:  1. `support-complete relational teacher` 有价值
experiments/paper_materials/story.md:953:- 所以当前本地主线补上的不是“再一个 routing 变体”，而是：
experiments/paper_materials/story.md:957:  1. 继续坚持 `single-image support incomplete` 这个问题定义
experiments/paper_materials/story.md:959:     **support prior 参与 feature formation 的可学习残差补全**
experiments/paper_materials/story.md:964:  “如何让 support-complete prior 真正进入遮挡部位的表征形成”
experiments/paper_materials/story.md:969:- 更重要的是，它的 `gate` 几乎塌到 `1.0`，说明当前 learned residual completion 在 late-stage 实际退化成了近似 hard replace
experiments/paper_materials/story.md:971:  1. `single-image support incomplete` 的问题定义仍然成立
experiments/paper_materials/story.md:972:  2. 但 per-ID prototype 的 direct feature completion 兑现方式当前不成立
experiments/paper_materials/story.md:975:  - `stable teacher` 只是 supporting mechanism
experiments/paper_materials/story.md:978:## 2026-03-20: 当前 story 进一步收紧到 “新增 correction 如何被学到”
experiments/paper_materials/story.md:982:  2. `exp120/121` 证明 support-complete teacher 会改变 teacher 几何，但仅靠 teacher 变强不够
experiments/paper_materials/story.md:985:- 这三件事拼起来，当前更像是在说：
experiments/paper_materials/story.md:986:  **真正难学的不是完整 teacher 几何，而是 support completion 相对 base teacher 带来的那部分新增 relation correction。**
experiments/paper_materials/story.md:997:  1. 不再让 student 去复刻整份 support-complete teacher
experiments/paper_materials/story.md:998:  2. 而是只学习 `support completion` 真正引入的那部分 **pairwise correction**
experiments/paper_materials/story.md:1000:     **support-complete relation correction learning**
experiments/paper_materials/story.md:1015:  3. 所以当前不能再把 “teacher target 太完整、稀释了新增 correction” 当作主矛盾
experiments/paper_materials/story.md:1018:  1. `support-complete relational teacher` 有价值
experiments/paper_materials/story.md:1028:  **在更大的 relation support 上学习 support-complete comparability correction**
experiments/paper_materials/story.md:1047:  3. 也就是说，当前不是“没看见足够多的 changed pairs”
experiments/paper_materials/story.md:1049:     **现有 student 形式不适合承载 pair-specific support-complete correction**
experiments/paper_materials/story.md:1057:  2. pose/keypoint branch 定义 common-support evidence
experiments/paper_materials/story.md:1058:  3. support-complete teacher 负责告诉模型“哪些 pair 真的需要 correction”
experiments/paper_materials/story.md:1066:  - 所以“learned pair module”并没有被做过，更谈不上被证伪
experiments/paper_materials/story.md:1068:- 这让当前最自然的方法升级变成：
experiments/paper_materials/story.md:1076:     - 什么时候更该相信 common-support distance
experiments/paper_materials/story.md:1077:  4. 这个 decision rule 由 `support-complete teacher` 监督，而不是人工固定成 `1:1`
experiments/paper_materials/story.md:1080:  “support-complete relational distillation”
experiments/paper_materials/story.md:1082:  **support-complete guided pair-adaptive correction**
experiments/paper_materials/story.md:1093:  3. 也就是说，当前不是“检索期 head 没必要”
experiments/paper_materials/story.md:1095:     **当前 head 太弱，只能学到接近固定 `1:1` 融合的行为**
experiments/paper_materials/story.md:1105:  2. keypoint/common-support 分支提供 pair-specific 可比较证据
experiments/paper_materials/story.md:1106:  3. support-complete prior 告诉模型“哪些 pair 的比较关系应被修正”
experiments/paper_materials/story.md:1111:  **support-complete guided pair-adaptive correction**
experiments/paper_materials/story.md:1113:  **support-complete guided pair-specific correction scoring**
experiments/paper_materials/story.md:1117:- `exp133 LPCS` 与 `exp134 Sparse LPCS` 当前都被判定为失效 run。
experiments/paper_materials/story.md:1126:  1. 当前我们还**没有真正测到** `LPCS`
experiments/paper_materials/story.md:1133:## 2026-03-21 晚间更新：`LPCS` 终于被真正测到，但当前瓶颈更像 ranking 而不是 routing
experiments/paper_materials/story.md:1151:   这说明当前 head 更像在做 deeper-rank correction，而不是更直接地修 top-1 错误。
experiments/paper_materials/story.md:1159:因此，当前 story 最合理的收束不再是：
experiments/paper_materials/story.md:1207:     - 让每个 pair correction 同时感知 query 的整体难度、margin 与 support 完整度
experiments/paper_materials/story.md:1210:- pose 负责定义 common support
experiments/paper_materials/story.md:1221:但 `exp139` 的审查结论很重要：当前版 query-context 不能直接进入 story，因为它的 context 依赖 label，测试阶段天然不可得，而且 evaluator 仍在构造 6 维 descriptor。也就是说，**它不是结果不好，而是实验定义本身还没闭环。**
experiments/paper_materials/story.md:1233:这条修正后的上下文线仍然和 pose 主线一致，因为它不是在替换 common support，而是在问：
experiments/paper_materials/story.md:1234:**给定 pose 定义出的 common support 之后，pair correction 是否还需要 query 级语境。**
experiments/paper_materials/story.md:1236:## 2026-03-22 当前 story 收紧：主候选从“平滑 rank 强调”转向“query-context pair correction”
experiments/paper_materials/story.md:1248:     - supporting evidence
experiments/paper_materials/story.md:1263:**pose 定义哪些身体证据是共同可比的；query context 决定这些共同证据在当前检索 pair 中应该被如何解释。**
experiments/paper_materials/story.md:1269:- 如何理解同一个 common-support signal 在不同 query 语境下的意义
experiments/paper_materials/story.md:1289:- pose 定义 common support
experiments/paper_materials/story.md:1290:- support-complete teacher 提供 correction 信号
experiments/paper_materials/story.md:1300:当前 story 已经不再平均分散在多条线之间，而是出现了比较清楚的主次关系：
experiments/paper_materials/story.md:1308:     - **同一份 common-support signal，必须放在 query 的整体语境中解释**
experiments/paper_materials/story.md:1312:   - 当前版本的问题不是没接上，而是 gate 很快塌成接近常数 1
experiments/paper_materials/story.md:1321:还可能缺“当前这个候选相对其他候选到底排在哪”的 competition context。**
experiments/paper_materials/story.md:1345:   - pose-defined common support + query-context pair correction
experiments/paper_materials/story.md:1347:   - pose-defined common support + confidence-calibrated correction
experiments/paper_materials/story.md:1350:而 `exp140` 更像是在验证：当前剩余的 `R1` 缺口，究竟是解释问题，还是应用问题。
experiments/paper_materials/story.md:1364:而是单张图里 pose-aligned support 本身不完整。**
experiments/paper_materials/story.md:1368:- pose-defined common support
experiments/paper_materials/story.md:1373:- **pose-defined support incompleteness**
experiments/paper_materials/story.md:1374:- **support-conditioned keypoint completion**
experiments/paper_materials/story.md:1384:   - 上界来自 support completion
experiments/paper_materials/story.md:1387:所以当前论文主叙事开始出现新的分工：
experiments/paper_materials/story.md:1394:   - 直接回答：能否把 `support incomplete` 在特征层修掉
experiments/paper_materials/story.md:1413:   - 去学习“互补 support 的联合表示如何逼近完整表示”
experiments/paper_materials/story.md:1417:   - 但当前 benchmark 上真正有用的 bilateral gap case 太少
experiments/paper_materials/story.md:1428:- support incomplete
experiments/paper_materials/story.md:1429:- support completion / pair correction
experiments/paper_materials/story.md:1433:- **把单图改写成伪多 support 训练对象**
experiments/decisions.md:75:2. 当前 concat 的 1/N scaling 太朴素，直接稀释了 part 信号
experiments/decisions.md:105:2. 当前 part triplet 是"所有 part 特征拼起来做一个 triplet"，每个 part 没有独立的 hard positive/negative mining
experiments/decisions.md:116:**上下文**: exp001-004 已探索了当前 part pooling 架构的多个变体：
experiments/decisions.md:121:当前最佳：mAP 57.5% (part-only), R1 67.1% (+0.9%/+0.6% vs baseline)
experiments/decisions.md:135:1. 当前 12×4 分辨率对 5 个 part 来说太粗（每个 part 只能覆盖 2-3 个 spatial position）
experiments/decisions.md:166:2. B 是全新方向，改变特征形成过程本身，可能突破当前 part pooling 的上限
experiments/decisions.md:219:1. 当前 PSG 只在 Stage 3（2 个 block）注入，信息利用有限
experiments/decisions.md:225:**执行结果**: exp009 mAP 58.3%, R1 67.2%, R5 81.2%, R10 85.2%。Multi-stage PSG (Stage 2+3) 与 single-stage (Stage 3 only) mAP 持平，R1 略低（-0.7%），R5/R10 略优（+0.4%/+0.3%），但增加了 156K 额外参数。**结论：Stage 2 PSG 无显著收益，pose spatial gating 在 Stage 3 已足够。后续聚焦于改进 PSG 机制本身，而非扩展注入范围。**
experiments/decisions.md:235:**核心发现**: PSG Stage 3 (2 blocks, 102K params) 是当前最优配置。进一步改进需要改变 PSG 的内部机制或训练策略。
experiments/decisions.md:248:2. 当前 PSG 零初始化，但训练初期 backbone 的梯度（来自 ID loss 和 triplet loss）会同时更新 backbone 和 PSG，可能让 PSG 来不及学到好的 gate pattern 就被 backbone 适应掉了
experiments/decisions.md:258:- exp007 PSG Stage 3: mAP 58.3% (+1.7%) — 当前最佳
experiments/decisions.md:295:**核心反思**: 当前 PSG 是一个简单的 spatial gate (17→64→768 的 1×1 conv)。它做的是"根据 pose heatmap 在空间维度上调制特征"。这个方法的局限性：
experiments/decisions.md:406:2. 当前 1×1 conv 每个位置独立计算 gate，没有空间连贯性
experiments/decisions.md:506:**新思路**: 不添加模块到 PSG，而是**改进 PSG 本身**。核心问题：PSG 的门控是**静态的**——给定相同的 heatmap，不同图像得到相同的 gate。如果让 gate 同时依赖 pose 和当前特征内容（Content-Adaptive PSG / CAPSG），可能打破这个限制。
experiments/decisions.md:767:- 🔵 蓝队（方案 B: 论文整合）核心论点: 29 个实验的统计信号已足够清晰——模块级改进的边际收益趋近于零。exp028 证明 Part 瓶颈在信息内容而非提取方法（收敛 5x 改善但测试无增益）。PXA(exp019) 已证明 cross-attention 在本框架中效果差。当前 +2.9% 已有竞争力，论文需要完整的实验体系（多 seed、可视化、跨数据集）而非更多模块。4M 参数是 Swin-Tiny 的 14%，过重。方案 B 成功概率 ~100% vs 方案 A <15%。信心: 8/10
experiments/decisions.md:974:**上下文**: 下一阶段准备把 `visibility` 引入当前的 skeleton branch / keypoint pooling 路线。但当前多人图中仍存在一个更基础的问题：`pose_data` 和 branch 侧并没有稳定地区分“crop 中的目标人”和“旁边靠得很近的其他人”。如果先做 visibility，再去解决 target assignment，visibility 语义会被污染，后续所有结论都不可靠。
experiments/decisions.md:976:**当前问题定义**:
experiments/decisions.md:1062:**当前不做的事情**:
experiments/decisions.md:1101:- 但当前证据只覆盖 `score*visibility` 这一条实现路径，不能把单个负结果直接上升成整条 visibility 路线的最终结论
experiments/decisions.md:1103:**关键教训（收紧版）**: 目前只能说 `score*visibility` 在当前 keypoint-level 加权池化中未带来正向证据。Visibility 是否能在其它位置（如 retrieval-time reasoning、pairwise masking、target-aware setting）发挥作用，仍需后续更精确的问题定义来判断。
experiments/decisions.md:1138:**上下文**: exp148 PCVT、exp149 SCFA、exp151 PVAT 三条线同时或先后推进，试图从不同角度解决 "single-image support incomplete"。
experiments/decisions.md:1151:2. 真正的 occlusion gap 在 test-time（gallery/query 有严重遮挡）
experiments/decisions.md:1198:需要选择下一个实验方向。当前 GCN 分支的核心瓶颈不在"特征质量"，而在"融合方式"——equal_concat 是一个固定的、非自适应的融合策略。
experiments/decisions.md:1224:1. 近年的强路线把问题定义在 **target ambiguity / common visible support / retrieval-time reasoning**，而不是“再学一个融合权重”。
experiments/decisions.md:1225:2. BPBreID / KPR 的 visibility 主要落在 **query-gallery pairwise distance**，不是只改 pooling。
experiments/decisions.md:1226:3. 我们当前代码线的真实缺口是：
experiments/decisions.md:1245:1. 由于用户新增规则“不要主动停下来”，`exp037` 继续自然跑完，不主动中止。
experiments/decisions.md:1260:2. 这与 `exp035b`、`exp036` 形成一致信号：**继续在 branch 内部调关键点权重/损失，不是当前最值得推进的主线。**
experiments/decisions.md:1282:1. 纯共同可见关键点距离不能替代当前主距离：
experiments/decisions.md:1289:   这符合“keypoint common-support 更适合作为补充项”的判断。
experiments/decisions.md:1317:2. 当前收益模式也更稳定了：两次都表现为 **mAP 提升、R1 小幅回落**。
experiments/decisions.md:1337:1. `1:1` 是当前测试点中的 **mAP 最优点**。
experiments/decisions.md:1344:1. 当前已经得到足够清晰的机制信号：`1:1` 不是偶然点，方向解释也比较稳定。
experiments/decisions.md:1345:2. 再做更细权重搜索会逐渐滑向 test-time 参数调优，不符合当前主线要求。
experiments/decisions.md:1366:3. 这意味着当前最准确的机制描述应改成：
experiments/decisions.md:1404:2. `045b` 的 mAP 增幅与 `exp040` 的 `+0.8%` 非常接近，因此当前最稳定的信号已经从“单 checkpoint 正例”推进到“至少两个 checkpoint 上都能转正的 mAP 信号”。
experiments/decisions.md:1408:1. 当前最稳妥、且已经被多 checkpoint 支撑的结论是：
experiments/decisions.md:1413:   - common-support reasoning 对整体排序的修正作用
experiments/decisions.md:1431:2. 当前 test-time 权重敏感性已经基本收敛，继续细调性价比很低。
experiments/decisions.md:1438:2. 但这不等于可以把当前 `cvk_hybrid` 直接包装成训练端创新；中间还缺一个清晰机制。
experiments/decisions.md:1439:3. 当前最值得优先尝试的训练端候选不是 AFF 或新的局部权重模块，而是：
experiments/decisions.md:1441:   - 用 `kp_weights` 构造 batch 内 pairwise common-support overlap
experiments/decisions.md:1442:   - 在 global triplet 上增加 support-aware hard mining 约束
experiments/decisions.md:1471:3. 当前最高优先级应从资产恢复切换到 `exp047 CSGT` 的实际训练验证。
experiments/decisions.md:1479:2. 后续如果 `exp047` 或 `cvk_hybrid` 需要第三 checkpoint 复核，当前资产已经足够支撑。
experiments/decisions.md:1480:3. 继续停留在 checkpoint 恢复不会新增论文机制证据，而 `CSGT` 才是当前真正待验证的训练端创新候选。
experiments/decisions.md:1488:  B. **放弃训练端创新，转入 1-2 天文献精读和新问题定义**: 47 个实验已充分说明当前框架训练端改进空间极小，应寻找全新方向。
experiments/decisions.md:1495:1. GPU 当前空闲，不用是浪费
experiments/decisions.md:1499:5. 但我认同蓝队的核心判断：SGMKC 更可能是 supporting experiment 而非 main contribution
experiments/decisions.md:1603:- 🔵 蓝队（方案 B）核心论点: 7 次增量修改全部中性/失败（5 loss + 2 attention bias），信号极其清楚——当前框架已饱和。继续做变体是沉没成本谬误。新方向有更高的创新潜力和论文价值。可能的新方向：pose-guided contrastive learning、cross-attention decoder、MoE routing。信心: 8/10
experiments/decisions.md:1610:3. 需要全新的问题定义或机制类别才能突破当前上限
experiments/decisions.md:1629:  A. **Pose-Guided Token Selection + Cross-Attention (PGTCA)**: 用 PSG 热图做 token 重要性评分，选出可靠 token，再用 keypoint-guided cross-attention 提取 part 特征。本质上替换当前 GCN branch 为更强大的 cross-attention 解码器。
experiments/decisions.md:1644:**上下文**: `exp066-074` 已完成 `PAA` 系列探索，`exp075` 正在补多 seed。联网复盘了 KPR、ProFD、Pose2ID、DPEFormer、SSSC、FCFormer、TTPM 等 2024-2025 工作，并重新评估当前成果是否足以支撑 B 类会议/期刊主线。
experiments/decisions.md:1647:1. 当前最强训练端结果是：
experiments/decisions.md:1655:   而我们当前主线还没有显式处理 target 与 distractor 的冲突。
experiments/decisions.md:1659:1. **当前成果还不够支撑 B 类主线**。
experiments/decisions.md:1662:3. 当前最合理的新方向应切到：
experiments/decisions.md:1684:3. 它没有被 `exp070` 直接证伪，因为 `exp070` 试的是 hard switch，不是 `scene + target-distractor` 的联合机制。
experiments/decisions.md:1690:3. 若 `TDPC` 在 2-3 天内无明显正信号，则 fallback 到 retrieval-time `common-support recovery`，不继续做 `TDPC` 小修小补。
experiments/decisions.md:1697:1. PAA 是 multi-person specialist (+1.69% multi / -1.61% R1 single)
experiments/decisions.md:1733:1. `target/distractor ambiguity` 这个问题定义本身还没有被证伪。
experiments/decisions.md:1734:2. 但当前这条具体机制：
experiments/decisions.md:1738:   真正有效的 pair-specific reasoning 很可能必须发生在 `per-keypoint / common-support` 粒度，而不是 pooled person feature 粒度。
experiments/decisions.md:1740:**选择**: 停止 `exp107` 的当前实现，不继续在该公式上做小调参。
experiments/decisions.md:1749:  **duplicate-aware / confuser-aware 的 per-keypoint common-support reasoning**
experiments/decisions.md:1750:- 不再继续 `exp107` 当前公式的参数扫点。
experiments/decisions.md:1755:**上下文**: `exp108 DACCM` 完成了第二轮 retrieval-time 原型验证。该实验把 `exp107` 的思路从 pooled person embedding 下沉到 `per-keypoint / common-support` 粒度，并以 `exp030a cvk_hybrid` 为主基线，比较：
experiments/decisions.md:1771:   - per-keypoint common-support penalty 仍负面
experiments/decisions.md:1773:   **当前 retrieval-time 反事实 penalty 机制本身不构成稳定可用的排名信号。**
experiments/decisions.md:1790:**上下文**: `exp109` 完成了 `Oracle Support Bank` 上界诊断。该实验使用 `exp030a cvk_hybrid` 的 target keypoint features，在 query+gallery 上用 GT same-ID 样本构造 leave-one-out 的 per-keypoint prototype。
experiments/decisions.md:1803:   **当前性能缺口里有一大块确实来自“support 不完整”，而不是 confuser suppression 失败。**
experiments/decisions.md:1804:3. 因而 `support-complete distillation` 已从“想法”升级为“有强 headroom 支撑的训练主线候选”。
experiments/decisions.md:1829:3. 因而当前单 seed 相对提升为 `+0.1 mAP / +0.8 R1`
experiments/decisions.md:1835:2. 但当前增益不大，且仍是单 seed；还不能把它写成“已确认主方法”。
experiments/decisions.md:1836:3. 从实现细节看，当前更可能的瓶颈是 teacher 可靠性，而不是蒸馏是否应该存在：
experiments/decisions.md:1838:   - 这和“support-complete”要表达的 multi-view support 概念并不完全一致
experiments/decisions.md:1840:**选择**: 继续 `support-complete` 主线，但下一步只做 teacher reliability 的单变量改动。
experiments/decisions.md:1844:2. 用户特别提醒过 `0.5x global loss` 很关键；当前实验已经保留了这一点，因此不应随意动 global/part 主损失平衡。
experiments/decisions.md:1849:2. 若 `exp111` 转强，说明当前 gap 主要来自 noisy teacher。
experiments/decisions.md:1864:2. 这说明当前 teacher reliability 的主要瓶颈，不像是“prototype 需要更多支持样本数”。
experiments/decisions.md:1872:2. 当前并行的远程 `exp112`（`UPDATE_THR=0.7`）在 `ep50` 已给出更强的正信号，因此更值得优先跟进。
experiments/decisions.md:1873:3. 这与当前论文主线也更一致：关键不只是“有多少 support”，而是“teacher support 是否足够干净可信”。
experiments/decisions.md:1878:**上下文**: `exp112` 与 `exp113` 已完成当前阶段使命并被提前停表。
experiments/decisions.md:1892:1. 当前 `support-complete` 主线没有被否定；相反，它的瓶颈已比之前更清楚。
experiments/decisions.md:1894:3. 当前更核心的问题是：
experiments/decisions.md:1902:2. 若冻结 teacher 后表现变好，就能把当前主创新从“prototype distillation”进一步推进到：
experiments/decisions.md:1903:   **reliable support-complete learning**
experiments/decisions.md:1936:   - 围绕已确认的 `support incomplete` 问题重新设计新机制
experiments/decisions.md:1942:**上下文**: 接手复核后发现，`exp117` 与 `exp118` 已经偏离当前主线。
experiments/decisions.md:1948:2. `exp118` 改用了 `exp085` 作为对照，不再锚定当前主基线 `exp030a`。
experiments/decisions.md:1949:3. `exp118` 还是一个组合实验，不符合当前阶段“围绕论文核心机制做单变量推进”的要求。
experiments/decisions.md:1950:4. 这条线会把 story 从 `support incomplete / support-complete learning` 拉回到“GCN 小模块 + 组合扫点”。
experiments/decisions.md:1954:2. `exp118` 明显偏离当前方向，继续跑完只会增加噪声，不会提升主线清晰度。
experiments/decisions.md:1955:3. 当前更需要的是回到最近一轮有效问题定义之上，再决定下一个真正值得做的新机制。
experiments/decisions.md:1957:**选择**: 停止 `exp118`，并把 `exp117/118` 明确标记为旁路线探索，不作为当前主线继续推进。
experiments/decisions.md:1962:3. 当前最宝贵的是论文叙事的一致性，而不是继续积累一个组合结果。
experiments/decisions.md:1966:2. 若要切到新方向，必须先说明它相对 `support incomplete` 主线的关系，而不是直接跳到模块叠加。
experiments/decisions.md:1974:- `exp109-116` 则说明 `support-complete` 若被压成 `per-ID prototype`，会丢失太多 pair-specific 细节
experiments/decisions.md:1978:2. 当前最值得继续赌的，不是 feature prototype，也不是 generic local matching，而是：
experiments/decisions.md:1979:   **用已经被 `cvk_hybrid` 验证过的 common-support pairwise 几何，直接蒸馏 global embedding 的关系结构。**
experiments/decisions.md:1992:   - global embedding 需要被蒸馏成更符合 common-support geometry 的空间
experiments/decisions.md:2014:2. 当前最清楚的增益落在 `global`（`+0.6 / +0.4`），说明它确实把 common-support 几何迁进了 backbone/global 空间。
experiments/decisions.md:2015:3. `equal_concat` 仍接近持平，说明第一版 teacher 还不够强；瓶颈更像 teacher 的 `support incompleteness`，而不是 relational distillation 这件事本身无效。
experiments/decisions.md:2016:4. 因而 `exp109` 的高价值结论仍应保留：真正缺的不是再换一个 loss 形式，而是 **更 support-complete 的 teacher**。
experiments/decisions.md:2019:**把 `exp109` 的 support-complete bank 降级为 teacher enhancer，而不是 pointwise distillation target，构造 support-complete relational teacher。**
experiments/decisions.md:2052:1. `support-complete teacher` 并没有“没生效”，相反，它已经稳定地增强了 teacher 几何。
experiments/decisions.md:2053:2. 但这种增强没有自动转成更好的检索指标，说明当前瓶颈已经不再是 “teacher 够不够完整” 本身。
experiments/decisions.md:2055:   **support-complete 监督的价值集中在 support-incomplete 样本；如果对所有 anchor 等权蒸馏，clean 样本会稀释掉这份增益。**
experiments/decisions.md:2065:   - 单图遮挡带来 support incomplete
experiments/decisions.md:2066:   - pose branch 提供 support-complete relational teacher
experiments/decisions.md:2067:   - 但 distillation 必须 **selective**，聚焦真正存在 support gap 的 anchor
experiments/decisions.md:2090:2. 但它没有把 `support-complete teacher` 的增强转成更好的指标，反而更像削弱了有效监督总量。
experiments/decisions.md:2091:3. 这说明当前问题不该再写成“监督该打给哪些样本”，而应进一步收紧成：
experiments/decisions.md:2093:4. `support-complete` 主线本身仍然成立；被否定的只是 sample-level `replace_ratio` 作为路由信号太粗。
experiments/decisions.md:2099:2. 它直接回应 `exp122` 的失败：真正该被强调的不是“这个样本补了多少 keypoint”，而是 **support-complete teacher 实际改变了哪些 pair 几何**。
experiments/decisions.md:2101:   - 单图遮挡带来 support incomplete
experiments/decisions.md:2102:   - support-complete teacher 改变一部分 pairwise comparability
experiments/decisions.md:2128:1. `stable teacher` 已经被 `exp121` 明确坐实为有效 supporting mechanism，但它不是当前主突破口。
experiments/decisions.md:2130:3. 当前最明显的瓶颈不是“pair focus 没用”，而是 **focus 放大力度偏弱**：
experiments/decisions.md:2139:2. `exp123` 已给出“方向对”的证据，因此下一跳最有信息量的不是换题，而是放大当前有效信号
experiments/decisions.md:2140:3. 远程 `exp121` 已收尾，当前最合理的资源利用方式是并行验证 `alpha` 是否就是当前瓶颈
experiments/decisions.md:2168:3. 因而当前更合理的瓶颈不再是“有没有 pair focus”或“alpha 够不够大”，而是：
experiments/decisions.md:2176:3. 它直接回应当前最具体的证据：
experiments/decisions.md:2180:   **只把被 support completion 真正改变过的 comparability relations 蒸进 global embedding。**
experiments/decisions.md:2185:3. 不同时改 `alpha`、不改 teacher bank、不断开 `support-complete` teacher，避免再次混入多个变量。
experiments/decisions.md:2203:1. `exp125` 的 late-stage 表现已经把 `pair routing` 从“弱正向候选”提升成了当前最强的训练主线候选之一。
experiments/decisions.md:2204:2. 相对 `exp124`，更结构化的 `delta_top` 当前已经证明比“仅增大 alpha”更强，至少在 late-stage 的 `R1` 上明显占优。
experiments/decisions.md:2205:3. 但 `exp125` 也同时明确暴露出：当前实现依然没有形成真正的 sparse routing，`top-25%` 在阈值式实现下被严重 tie 扩散。
experiments/decisions.md:2214:2. `exp125` 说明当前主线是有效的，因此不能中途停掉。
experiments/decisions.md:2215:3. `exp126` 是相对 `exp125` 的最小单变量下一跳，直接检验当前最关键的机制缺口。
experiments/decisions.md:2220:3. 后续文档与 story 必须把“当前收益来自结构化 pair focus，但不等于已证明真稀疏 routing”写清楚。
experiments/decisions.md:2236:1. `exp125` 已把 “结构化 pair routing 有效” 这件事坐实，当前它是已完成训练中最强的 pair-routing 版本。
experiments/decisions.md:2237:2. `exp124` 证明了单纯增大 focus 强度也有效，但最终不如 `exp125`，因此它应退居 supporting branch。
experiments/decisions.md:2244:2. 远程继续监控 `exp126` 到 `ep30/40`，这是当前最有信息量的主线实验。
experiments/decisions.md:2255:2. `exp125` 评估必须沿当前正式配置完成，不额外加 test-time trick。
experiments/decisions.md:2267:- 同时当前 `exp109` 主线已经暴露出另一个未被打透的缺口：
experiments/decisions.md:2270:  - 二者都没有把 oracle support-complete 上界真正兑现出来
experiments/decisions.md:2273:1. 现在继续在本地扫 `alpha / top_ratio / freeze epoch` 属于低价值调参，不符合当前阶段目标。
experiments/decisions.md:2274:2. 但这不意味着要离开 `exp109`；相反，最合理的下一步仍然是沿 `support incomplete -> support-complete learning` 这条主线，直接测试更强的 feature-level 兑现机制。
experiments/decisions.md:2275:3. `SCFR≈SCKD` 只能说明 “hard replace 不优于 loss-only”，不能说明 “feature-level support completion 整体无效”。
experiments/decisions.md:2279:2. 该实验保持 `bank`、`warmup`、`threshold` 与 `exp116` 同量级，只改 low-vis keypoint 如何利用 support-complete prototype：
experiments/decisions.md:2310:1. `SCRC` 没有把 feature-level support completion 推成更强结果，反而 late-stage 基本塌成了“近似 hard replace”。
experiments/decisions.md:2311:2. 因而 `exp109` 被否定的不是 `support incomplete` 问题定义，而是：
experiments/decisions.md:2313:3. `freeze20/30` 的既有证据已经足够说明它只是弱 supporting mechanism，不值得继续占用本地算力。
experiments/decisions.md:2314:4. 当前最有价值的缺口不再是 “teacher 还该不该更稳定”，而是：
experiments/decisions.md:2315:   **support-complete teacher 的新增 correction 仍被完整 teacher target 稀释。**
experiments/decisions.md:2324:   - support-complete teacher 的增量信息是真实存在的
experiments/decisions.md:2325:   - 但当前 full-teacher distillation 没把这部分新增修正单独抽出来学
experiments/decisions.md:2337:   - `exp129`: target dilution 是否是当前主瓶颈的因果验证
experiments/decisions.md:2356:2. 但它到收敛都没有压过 `exp125`，因此当前不能再把“target dilution”当作主瓶颈。
experiments/decisions.md:2357:3. 这意味着当前最值得继续推进的，不是 `target form`，而是：
experiments/decisions.md:2365:2. 保留 `exp125` 作为当前本地最强的在线 `SCRD` 版本。
experiments/decisions.md:2395:  - `exp089 PAMN` 只有 design/review 草案，从未真正接入 checkpoint 与测试检索流程，因此**不能**算作“learned pair module 已被证伪”
experiments/decisions.md:2398:1. `cross-batch changed-pair coverage` 不是当前主瓶颈；queue 明显参与了监督，但没有带来实质性的 mAP 提升。
experiments/decisions.md:2400:3. 当前更合理的主假设应收紧为：
experiments/decisions.md:2401:   **pair-specific support-complete correction 不能被当前单向量 student 充分吸收。**
experiments/decisions.md:2422:3. `exp040/045` 的固定 `cvk_hybrid` 已经证明 pair-specific common-support correction 在检索时能转成稳定正信号。
experiments/decisions.md:2423:4. 因而当前最值得赌的新机制不是“再蒸一次”，而是：
experiments/decisions.md:2432:   - `exp125`（作为“蒸进 embedding”的当前最强对照）
experiments/decisions.md:2443:- 当前日志里没有落出 `ltcs_*` 统计，因此这轮实验只能靠正式结果做主要判定
experiments/decisions.md:2447:2. 这不等于 learned pair module 大方向被证伪；真正被证伪的是更具体的实现：
experiments/decisions.md:2451:3. 当前最合理的解释是：
experiments/decisions.md:2453:   - 且当前监督不够 ranking-aligned
experiments/decisions.md:2504:1. `exp133/134` 的当前数值全部 **不能** 用于支持或反驳 `LPCS`
experiments/decisions.md:2506:3. 当前最重要的决策不是“继续观察曲线”，而是立即止损并重跑干净实验
experiments/decisions.md:2522:1. 继续跑当前进程只是在浪费 3090 和 5060 Ti 算力
experiments/decisions.md:2524:3. 既然方法还没真正被测试到，就不能把当前线判负，更不能切题
experiments/decisions.md:2559:4. 因而当前最值得收紧的判断不是“routing 是否有效”，而是：
experiments/decisions.md:2560:   **当前 `LPCS` 的主瓶颈更像 ranking objective 本身，而不是 pair coverage / sparse routing 语义**
experiments/decisions.md:2563:1. `exp136` 到此结案，保留为 supporting 证据
experiments/decisions.md:2574:   - 比监督该选哪些 pair 更像当前瓶颈
experiments/decisions.md:2592:3. 这说明当前负边界已经很清楚：
experiments/decisions.md:2594:   - 它不是当前 `LPCS` 的正确升级方向
experiments/decisions.md:2605:3. 因而当前最合理的升级方向是：
experiments/decisions.md:2620:1. 当前 `LPCS` 主线还值得继续，但不能再沿着“更稀疏”或“更硬选择”推进
experiments/decisions.md:2639:2. `exp138/139` 相对当前主线都满足单变量原则
experiments/decisions.md:2648:## [2026-03-21 14:24] 决策：放行 `exp138`，驳回当前版 `exp139` 并重构为无标签 context
experiments/decisions.md:2653:- `exp139` 当前暴露的不是小实现问题，而是两个 blocking:
experiments/decisions.md:2659:2. 当前版 `exp139` 不能进入远程训练，否则即使勉强补零也无法解释结果
experiments/decisions.md:2667:2. 远程暂不启动当前版 `exp139`
experiments/decisions.md:2671:1. `exp138` 仍在 `LPCS` 主线内，并且当前最接近“平滑 top-sensitive”这一合理升级
experiments/decisions.md:2672:2. `exp139` 的当前失败不是点子无效，而是设计没有闭环到测试路径
experiments/decisions.md:2675:## [2026-03-22 00:14] 决策：终止 `exp138`，将 `exp139` 升为当前唯一主候选
experiments/decisions.md:2702:3. 因而当前最值得押注的不是“更平滑的 rank 强调”，而是：
experiments/decisions.md:2713:1. `exp138` 已经提供了足够的负边界：平滑 top-sensitive 只能算 supporting 机制
experiments/decisions.md:2714:2. `exp139` 是当前唯一同时拥有机制证据与中期正信号的升级线
experiments/decisions.md:2716:   - pose 定义 common support
experiments/decisions.md:2717:   - query context 决定 pair correction 应如何解释该 support
experiments/decisions.md:2722:- `exp138` 已停表，结论为 supporting 线
experiments/decisions.md:2723:- `exp139` 正在远程继续跑，并已成为当前唯一主候选
experiments/decisions.md:2731:2. 当前更值得测试的不同创新点是：
experiments/decisions.md:2747:1. 这是当前最合理、且与 `exp139` 机制不同的并行探索点
experiments/decisions.md:2754:## [2026-03-22 00:39] 决策：`exp139` 在 `ep50` 前后已强化为当前唯一主候选，`exp140` 作为本地并行线正式接上
experiments/decisions.md:2771:   - 当前唯一真正接近论文主创新点的 active run
experiments/decisions.md:2782:3. 当前不再新开第三条线，先看这两条并行主候选的中期走势
experiments/decisions.md:2785:1. `exp139` 已经给出当前最硬的正信号
experiments/decisions.md:2798:1. `exp139` 到 `ep70` 为止仍是当前唯一主候选
experiments/decisions.md:2807:1. 继续保持 `exp139` 为当前主候选并盯后续 `ep80/90`
experiments/decisions.md:2808:2. 不把 `exp140` 当前 run 计入结论
experiments/decisions.md:2814:2. `exp140` 与 `exp139` 仍然是当前最合理的双线并行：
experiments/decisions.md:2819:## [2026-03-22 02:00] 决策：继续双线推进，但当前优先级仍是 `exp139`
experiments/decisions.md:2829:1. `exp139` 到 `ep80` 为止，已经基本追平当前最强 supporting 线 `exp135`
experiments/decisions.md:2831:3. 但当前 `exp140` 的 `conf_mean` 明显高于 `conf_target_mean`，说明这版 confidence gate 仍偏激进，暂时不能提前下正结论
experiments/decisions.md:2835:2. 本地继续盯 `exp140`，当前关键节点是 `ep30`
experiments/decisions.md:2839:1. `exp139` 仍是当前最有机会收敛成论文主机制的方向
experiments/decisions.md:2841:3. 当前最重要的是看：
experiments/decisions.md:2845:## [2026-03-22 02:38] 决策：`exp140` 当前版本止损，远程 `exp139` 继续跑完
experiments/decisions.md:2858:1. `exp139` 到 `ep100` 为止仍是当前唯一可持续推进的主候选
experiments/decisions.md:2859:2. `exp140` 当前版本虽然真实接上了，但 `confidence gate` 已明显退化：
experiments/decisions.md:2869:3. 本地下一步不再沿当前 `confidence target` 形式修小补小
experiments/decisions.md:2873:2. `exp139` 目前虽未显著超过 `exp135`，但它仍是当前最接近论文主机制的 active line
experiments/decisions.md:2880:- `exp140` 当前版本已止损，原因是 confidence gate 退化为接近常数 1
experiments/decisions.md:2888:2. 当前更合理的不同创新点是：
experiments/decisions.md:2892:   - `exp141` 问“这个 pair 在当前 query 的候选竞争里处于什么位置”
experiments/decisions.md:2906:1. 这是当前与 `query_ctx / confidence-gate` 都不同的真正新点
experiments/decisions.md:2920:  - 真正 headroom 来自 `single-image support incomplete`
experiments/decisions.md:2925:   - 但它当前更像 supporting 机制，而不是确定的论文主方法
experiments/decisions.md:2928:3. 当前更合理的本地大转向应回到 `exp109` 根问题本身：
experiments/decisions.md:2930:   - 而在特征层直接补全 keypoint-level support
experiments/decisions.md:2947:**上下文**: exp142 SKC 训练完成。最终结果 mAP 60.3% / R1 71.8%（equal_concat），相对 exp030a -0.8% mAP / -1.9% R1。feature-level support-supervised completion 方向确认失败。
experiments/decisions.md:2950:1. SKC completion 模块虽然活跃（gate≈0.26, delta_norm≈1.5），但 skc_pre≈skc_post 说明修改方向不是向 support prototype 靠近
experiments/decisions.md:2961:1. feature-level completion 方向已被彻底证伪（5+ 次尝试），不值得继续做 ablation
experiments/decisions.md:2980:1. `single-image support incomplete` 这个问题定义没有被推翻
experiments/decisions.md:2996:   - 单图能否被改写成“伪多 support 学习”对象
experiments/decisions.md:3005:- 当前进入“先设计、后实现、再广范围 Claude 审查”的新阶段
experiments/decisions.md:3018:   - `SCFA` 则在当前数据集上缺少足够强的 bilateral gap case
experiments/decisions.md:3022:3. `PCVT` 当前最主要风险不是方法失效，而是：
experiments/decisions.md:3033:2. 当前更重要的是：
experiments/decisions.md:3046:**上下文**: exp190-195 系列实验完成，揭示了 OA-SD 和 3-view parallel aug 的组合关系，以及 OA-SD global-only 解决 SupCon 梯度冲突的新机制。
experiments/decisions.md:3049:- exp190 (3-view+CE): 64.2/75.6 — 3-view 是最强单一技术
experiments/decisions.md:3050:- exp191 (OA-SD+CE): 63.2/75.4 — OA-SD 独立有效
experiments/decisions.md:3051:- exp192 (decay=0.99): 62.6/74.9 — decay 不敏感
experiments/decisions.md:3052:- exp193 (3-view+OA-SD+CE): 64.4/76.5 — additive! R1 追平 SupCon
experiments/decisions.md:3053:- exp194 (weight=2.0): 63.4/74.8 — weight 不敏感
experiments/decisions.md:3054:- exp195 (SupCon+OA-SD global-only): ep70=60.2/73.4 — 梯度冲突解决!
experiments/decisions.md:3057:  A. exp196: 3-view + SupCon + OA-SD global-only（终极组合，验证所有创新 additive）
experiments/decisions.md:3064:2. 如果 exp196 > exp187 (64.9/76.6)，则创论文主表新高
experiments/decisions.md:3065:3. exp195 已验证 SupCon+OA-SD global-only 兼容，exp193 已验证 3-view+OA-SD additive
experiments/decisions.md:3072:**上下文**: exp196 (3-view + SupCon + OA-SD global-only) 在 ep70 持续落后 exp187 (3-view + SupCon) -1.8/-0.9。OA-SD global-only 的 distillation 信号 (oa_sd=0.01) 过弱。
experiments/decisions.md:3077:- OA-SD + SupCon (global-only) 无梯度冲突但信号太弱 (exp195/196)
experiments/decisions.md:3082:  B. 最终配置用 OA-SD+CE (exp193: 64.4/76.5) — R1 几乎一样
experiments/decisions.md:3113:**上下文**: exp199 (OA-RD+SupCon) ep60=-1.5/-3.4 vs exp187，exp200 (OA-RD+CE) ep60=-1.1/-3.4 vs exp191。OA-RD (relational distillation) 也是负结果。
experiments/decisions.md:3117:- OA-RD (relation-level): exp199 失败
experiments/decisions.md:3137:**上下文**: exp197-201 连续 5 个负结果。所有在 exp187 (64.9/76.6) 基础上的改进尝试都失败。
experiments/decisions.md:3140:- exp197 (STM + SupCon): -0.8/-0.6 — token mixup 只加速不改善
experiments/decisions.md:3141:- exp198 (STM + OA-SD): ±0 — 同上
experiments/decisions.md:3142:- exp199 (OA-RD + SupCon): -1.5/-2.1 — relational distillation 也与 SupCon 冲突
experiments/decisions.md:3143:- exp200 (OA-RD + CE): -0.3/-1.5 — OA-RD 不如 OA-SD
experiments/decisions.md:3144:- exp201 (global SupCon): ~-1.5/-3.6 — global SupCon 压缩特征空间
experiments/decisions.md:3173:| 当前 | Small GCN+PAA+CE+OA-SD | 70.5% |
experiments/decisions.md:3175:| +2 | **Swin-Base** (exp207 进行中) | 74-75% |
experiments/decisions.md:3180:1. exp207 Base 跑完后确认 Base 增益
experiments/decisions.md:3185:- exp208 (0.5x global loss) = NO-OP（GCN list-loss 已隐含 0.5x），取消
experiments/decisions.md:3186:- exp209 (STD-PR+CE+OA-SD) ep30=56.0/69.3，落后 5%，终止
experiments/decisions.md:3192:**上下文**: MaxSim hybrid 在 exp206 checkpoint 上无需重训给 +1.8% mAP (70.3→72.1)。OA-SD teacher bug 已修复。PKC (Per-Keypoint Contrastive) 开始测试。
experiments/decisions.md:3199:| exp210 | + PKC (进行中) | 73-74% |
experiments/decisions.md:3200:| exp207 | Swin-Base 3-view (进行中) | 74-76% |
experiments/decisions.md:3211:| exp210 | PKC weight=0.5 (detached GCN) | 灾难 (3.6%) |
experiments/decisions.md:3212:| exp210b | PKC weight=0.05 (detached GCN) | 无效 (= baseline) |
experiments/decisions.md:3213:| exp211 | MST weight=0.5 (detached GCN) | 无效 (= baseline, 所有 loss 完全一致) |
experiments/decisions.md:3214:| exp213 | PKC+MST 组合 (detached) | 灾难 (40.6%) |
experiments/decisions.md:3215:| exp215 | BA-PKC weight=0.1 (NON-detached backbone) | 灾难 (0.5%) |
experiments/decisions.md:3222:1. **per-keypoint loss 路线已证伪** — 架构约束使其不可能有效
experiments/decisions.md:3224:3. **当前最佳: 72.4/83.1 (exp210b + maxsim)**
experiments/decisions.md:3240:| exp210 | PKC w=0.5 on detached GCN | Yes | 灾难 3.6% |
experiments/decisions.md:3241:| exp210b | PKC w=0.05 on detached GCN | Yes | 无效 (=baseline) |
experiments/decisions.md:3242:| exp211 | MST w=0.5 on detached GCN | Yes | 无效 (所有 loss 完全一致) |
experiments/decisions.md:3243:| exp213 | PKC+MST combo | Yes | 灾难 40.6% |
experiments/decisions.md:3244:| exp215 | BA-PKC w=0.1 non-detached | No | 灾难 0.5% |
experiments/decisions.md:3245:| exp212 | LR=0.0008 | — | 灾难 0.8% |
experiments/decisions.md:3246:| exp217 | OERL w=1.0 non-detached cosine | No | `62.2/75.2`，相对 `exp191 63.2/75.4` 为 `-1.0/-0.2` |
experiments/decisions.md:3252:4. **per-keypoint training loss 路线已全面证伪**
experiments/decisions.md:3259:### [2026-04-02 09:45] 决策 — PACI 证伪 + MaxSim Ceiling 发现
experiments/decisions.md:3261:**PACI (exp218/219) 结果:**
experiments/decisions.md:3262:- PACI + OA-SD (exp218): `61.9 / 74.2` (vs `exp191 63.2 / 75.4` = **-1.3 / -1.2**)
experiments/decisions.md:3263:- PACI-only (exp219): 已从远程补回 `train_log`，当前可直接复核到 `ep10=37.7/50.5`、`ep20=47.5/60.4`、`ep30=51.9/64.9`；但尚无 final，因此它仍只能作为 early stop-loss 证据，不能写成正式最终结果
experiments/decisions.md:3264:- **PACI 证伪。Consistency loss on detached GCN = 无效。**
experiments/decisions.md:3285:4. 后续 `exp220` 已把 Tiny `maxsim_hybrid` 推到 `64.6`，因此这里原先的 `~64.4` / `~64.2` ceiling 表述应视为阶段性误判
experiments/decisions.md:3301:| PADPQ K=8+OA-SD | 进行中 | 进行中 |
experiments/decisions.md:3304:1. GSPB: 早期加速 +5.8% at ep10，按当前测试记录 `maxsim_hybrid` 相对 OA-SD 为 `+0.4`，是目前 Tiny 线上最高的 `maxsim` mAP
experiments/decisions.md:3315:### [2026-04-03 20:40] 决策 — BT-PKD 系列证伪，Non-Detached Gradient 方向关闭
experiments/decisions.md:3317:**上下文**: exp229-232 全面测试了 BT-PKD (Backbone-Through Per-Keypoint Distillation):
experiments/decisions.md:3330:**已证伪的 non-detached 变体汇总**:
experiments/decisions.md:3340:### [2026-04-04 15:40] 决策: exp242 PPA+GCN Small 灾难性失败
experiments/decisions.md:3347:2. 对比: PPA on Small (exp240) 也是中性 (70.7/81.1 vs 70.6/82.6 = +0.1/-1.5)
experiments/decisions.md:3355:### [2026-04-04 15:40] 决策: 启动 exp243 LGPA
experiments/decisions.md:3365:### [2026-04-04 21:10] exp243 LGPA 结果分析 (GPU crash at ep88)
experiments/decisions.md:3379:### [2026-04-05 04:10] exp244 LGPA-Detach — 突破性结果! ⭐⭐⭐
experiments/decisions.md:3381:**结果**: 65.3/75.7 (+2.1/+0.3 vs exp191 GCN+OA-SD)
experiments/decisions.md:3387:2. detach 完全消除了 non-detach 的后期干扰 (exp243 -1.1 → exp244 +2.1)
experiments/decisions.md:3393:- 消融故事清晰: non-detach (exp243) vs detach (exp244) 证明 detach 必要性
experiments/decisions.md:3418:### [2026-04-08 16:45] 决策 — exp249 完成后下一步
experiments/decisions.md:3420:**上下文**: exp249 (Small LGPA-D+GCN) 完成: 71.9/81.8 equal_concat, 73.3/83.2 MaxSim。
experiments/decisions.md:3429:1. Tiny 消融数据 (exp244, exp246b) 已经足够完整
experiments/decisions.md:3435:1. 所有 "安全" 创新方向已试完或被证伪
experiments/decisions.md:3437:3. 当前结果 (73.3/83.2) 已可投稿
experiments/decisions.md:3439:**当前论文素材已具备**:
experiments/decisions.md:3449:已完成 VCSR (exp247, 失败) 和 PCFD (exp248, 失败) 两个创新尝试。
experiments/decisions.md:3463:3. 当前系统 (73.0% MaxSim) 距 SOTA (75.1%) 仅差 2.1% (Swin vs ViT 差异)
experiments/decisions.md:3466:A. 短期: 完成 exp249, 快速测试 POT (test-time, 无训练需求)
experiments/decisions.md:3472:2. LGPA-D 虽然 single novelty 4.5/10, 但与 PSG+OA-SD+MaxSim 组成完整 framework novelty 更高
experiments/decisions.md:3473:3. exp249 (LGPA-D+GCN on Small) 有潜力达到 73-74% → 与 SOTA 竞争力足够
experiments/decisions.md:3479:- `exp109` oracle support bank 仍是仓库内最强问题证据
experiments/decisions.md:3480:- `exp257-259` 已基本说明当前 `exp255` recipe 空间耗尽
experiments/decisions.md:3486:  B. 回到 `exp109`，把主线改成“single-image support incomplete”的训练对象重写
experiments/decisions.md:3493:3. `MaxSim / POT / flip` 主要仍是 test-time supporting evidence，不能作为训练端主贡献
experiments/decisions.md:3501:1. 用 pose 定义互补 support 伪视图，而不是随机多视图分类
experiments/decisions.md:3502:2. 用互补视图组装 support-complete teacher token set
experiments/decisions.md:3509:3. 若 Tiny 为正，再上 `exp255` Small scaffold
experiments/decisions.md:3517:- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
experiments/decisions.md:3522:**上下文**: 用户明确表示当前目标是“把现在的探索结果包装成一个故事和创新点，先发 C 类”；并进一步确认：  
experiments/decisions.md:3528:  A. 继续沿刚提出的 `PSCD/support-complete` 新路线展开
experiments/decisions.md:3535:2. 当前最强系统 `exp255` 使用的就是 `2-stage PSG`
experiments/decisions.md:3536:3. `exp255 vs exp255b` 给出最强信息：在 `GCN512` 高容量结构分支下，`2-stage PSG` 带来 `+1.7 / +1.4`
experiments/decisions.md:3537:4. 虽然 `exp009 / exp251 / exp253` 不支持“multi-stage 普遍更强”，但这恰好说明需要**重跑干净消融**，而不是放弃 PSG 主线
experiments/decisions.md:3542:2. `2-stage PSG` = 当前最终版本 / scalable extension
experiments/decisions.md:3543:3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets
experiments/decisions.md:3562:**上下文**: 本地 3090 已挂；`phase1_design.md` 原把 Base 3 个 run（exp263/266/269）全部排在 3090 上。剩余资源仅 srvA/B/C 三台 5060 Ti 16G，已在跑 Phase 1 前 3 个 Tiny/Small run（exp261/262/264）。
experiments/decisions.md:3574:1. deadline 2026-04-30，11 天。Tiny/Small 6 run 三机并行约 22–28h，Base 3 run @ with_cp 三机并行约 18h，Phase 3 消融约 30h。仍能在 deadline 前出全稿
experiments/decisions.md:3575:2. `exp260b Base = 73.9/83.2`（旧协议，本地 3090）可作 Base 行 reference
experiments/decisions.md:3582:- Phase 1 当前运行: srvA=exp262(Small OD) e70, srvB=exp261(Tiny OD) e106, srvC=exp264(Tiny OP) e83；接下来按 srvB→exp267, srvC→exp265, srvA→exp268 顺序排队；Tiny/Small 6 run 完成后立即评估是否把 Base 3 run 并入 Phase 1
experiments/decisions.md:3609:- **Phase 3-A exp271** 刚起 15min,kill + restart 用新代码(`POSE_TEST_FEAT='global'` 单块,实际上受 bug 影响极小,但 restart 是对的)
experiments/decisions.md:3610:- **exp269 / exp266** 还在训中,Python 进程里缓存的是旧 code,e120 eval 会走 broken path → 完成后 test.py 重测
experiments/decisions.md:3611:- **exp270** `POSE_ENABLED=False` 单块模式,bug 不生效,数字 59.2/68.4 仍有效
experiments/decisions.md:3612:- **Phase 1 其余已完成** (exp261/262/264/265/267/268 + exp263 e100): 全部 test.py + 新 code 重测,在机器空闲时批量跑
experiments/decisions.md:3615:- [x] exp262 Small OD transformer_120.pth re-eval → **73.8/83.1 (与原训练内部 eval 完全一致)**
experiments/decisions.md:3616:- [x] exp268 Small Market transformer_120.pth re-eval → **94.3/97.3 (与原训练内部 eval 完全一致)**
experiments/decisions.md:3624:- exp262 Small OD: 73.8/83.1 (fixed) vs 73.8/83.1 (broken) — 完全一致
experiments/decisions.md:3625:- exp268 Small Market: 94.3/97.3 (fixed) vs 94.3/97.3 (broken) — 完全一致
experiments/decisions.md:3635:- exp263 e100: Global+flip 72.5/81.8, MaxSim+flip 74.5/84.0
experiments/decisions.md:3636:- exp269 e80: Global+flip 94.4/97.0, MaxSim+flip 94.5/97.1
experiments/decisions.md:3641:- srvA (gpushare i-2:29162) 用户忘续费,SSH refused 持续 >1.5h。ckpt (exp262/268/269.pth) 和原始 train_log.txt 在 /hy-tmp/log/ 上,是否保留取决于 gpushare 平台对 expired 实例的处理策略(未确认)
experiments/decisions.md:3642:- exp274(Phase 3-A Small baseline)刚启 40min 丢失,无重要损失
experiments/decisions.md:3643:- 同时用户的实验室 3090 复活(tailscale 100.115.252.80:22,容器 `18fbbab202e1`),git pull 到 `f69b61c`(flip fix 版),正在跑 `exp263b_best_b_od_s42_3090`(Base OD 完整重跑)
experiments/decisions.md:3649:   - srvB: Phase 3-A Tiny 全 4 格(exp270 ✓ / exp271 → exp272 → exp273)
experiments/decisions.md:3650:   - srvC: exp266 Base OP(完成后 → Phase 3-B 6 格)
experiments/decisions.md:3651:   - lab3090: exp263b Base OD 完成后(~2026-04-21 02:30 CST)→ Phase 3-A Small 4 格(exp274 重启+275/276/277)
experiments/decisions.md:3652:4. Phase 1 数字:exp262/268/269 FINAL 已 committed,不受 ckpt 丢失影响;若 gpushare 宽限期内能救回 ckpt 再说
experiments/decisions.md:3653:5. lab3090 3090 24GB 显存足以容纳 Base + full scaffold + default flip eval 不 OOM,原来在 5060Ti 16GB 上 OOM 的问题在 3090 上不复现;exp263b 完成后将给出干净的 e120 FINAL,作为 exp263 e100 salvage 的升级替代
experiments/decisions.md:3660:**exp263b vs exp263 对照**:
experiments/decisions.md:3661:- exp263 新协议 e100 eff-FINAL(5060Ti srvB,OOM 后 salvage):Global+flip 72.5/81.8, MaxSim 74.5/84.0
experiments/decisions.md:3662:- exp263b 3090 完整 e120 FINAL(将来):预期 MaxSim 75+,超 KPR w/ prompt 75.1/84.3 可能性大
experiments/decisions.md:3693:| srvC | 5060Ti 16G | exp266 Base OP + Phase 3-B |
experiments/decisions.md:3694:| lab3090 | 3090 24G | exp263b Base OD + Phase 3-A Small 4 格(exp274-277 重启) |
experiments/decisions.md:3704:- lab3090 (tailscale 100.115.252.80 docker container `18fbbab202e1`) 跑 exp263b (Base OD, Full Scaffold) 从 2026-04-20 ~08:00 本地起,到 e42 Iter 100 (10:14:56 UTC = 18:14 local) 卡住
experiments/decisions.md:3711:1. kill -9 189605 + `pkill -9 -f 'exp263b_best_b_od_s42_3090'` ✅
experiments/decisions.md:3714:4. e40 eval 未产生(本 run 只到 e30=某 mAP),需 GPU 恢复后用 `test.py` 跑 ckpt40 得 interim FINAL 作为 exp263 e100 salvage 的升级替代
experiments/decisions.md:3718:- 如短期未恢复,exp263b e40 作为可接受 fallback(Base OD 中段,预期 ~70-72 mAP,低于完整 e120 但比 srvA exp263 e100 salvage 稍低)
experiments/decisions.md:3719:- 如长期挂机,lab4090(24G,pose_data 同步完成后)可接替 exp263b resume from ckpt40
experiments/decisions.md:3741:3. `scripts/extract_visibility.py` 在 lab4090 GPU: train(15618)+query(2210)+gallery(17661),4 分钟完成
experiments/decisions.md:3747:- _val_merged 也补了 target 字段(persons 指 absolute paths 到 gallery/query,compute_target 脚本支持)
experiments/decisions.md:3754:**结论**: lab4090 Occluded-Duke pose_data **production-ready**,可接 Phase 3-A Small baseline(exp274 重启)。
experiments/decisions.md:3756:### [2026-04-20 21:35] 事件 — lab4090 queue_on_ckpt daemon python3 bug,exp275 crash 重启
experiments/decisions.md:3759:- 21:34 exp274 FINAL (68.1/76.8/87.8/90.9) ckpt 生成
experiments/decisions.md:3760:- daemon 3580255 立即触发 exp275,但 1 分钟内 crash
experiments/decisions.md:3761:- `/tmp/exp275.log` 只有 `ModuleNotFoundError: No module named 'torch'`
experiments/decisions.md:3766:- exp274 当初**手动**启动用的是完整 conda path,没用 daemon,所以 OK
experiments/decisions.md:3773:5. 手动启动 exp275 用 mmpose-abu python (PID 3653199)
experiments/decisions.md:3780:### [2026-04-20 22:41] 事件 — srvC exp266 silent exit @ e70 (非 OOM)
experiments/decisions.md:3783:- exp266 Base OP Full Scaffold 从 04:46 启动,稳定跑到 e70 (~21:27 CST)
experiments/decisions.md:3795:**决策**: **不重训 exp266**。
experiments/decisions.md:3797:- 与 exp265 Small FINAL 78.4 / 86.2 **完全持平** → Base 对 Small 在 Occ-PTrack 上 0 增益
experiments/decisions.md:3801:**同 exp263/exp269 OOM 处理模式**: effective FINAL 用最后一次 eval 数字,不重训。
experiments/decisions.md:3805:- rsync Occ-Duke + pose_data ~5.5GB from srvB 会影响 exp273 磁盘 I/O
experiments/decisions.md:3810:**发现**: 审查 srvC `/hy-tmp/data/occluded_duke/` 时发现 **数据已齐备** (4.9GB, train 22059 + query 4152 + gallery 24770, pose_data 四分区全),与我之前"需要 rsync"的假设相反。Pretrained swin_{tiny,small,base} + clip_part_text_features 也都在。
experiments/decisions.md:3812:**立即决策**: srvC 启动 **Phase 3-C exp286/287** (LGPA-only Tiny 2 runs,phase3_design.md L111-134 已规划),填补 srvC 空闲。
experiments/decisions.md:3815:- exp286 (LGPA-only + 1-stg PSG + Tiny, PID 59845) @ 23:32 CST,config load + dataset load OK
experiments/decisions.md:3816:- daemon 59846 挂 exp286 → exp287 (2-stg PSG) auto-chain
experiments/decisions.md:3818:- Small 2 runs (exp288/289) 等 lab4090 Phase 3-B 完成后接
experiments/decisions.md:3822:### [2026-04-20 23:30] 决策 — lab3090 exp263 系列 seed 切换
experiments/decisions.md:3825:- exp263c (lab3090 Base OD Full Scaffold pwrlim 280W seed 42) 跑到 e31,trajectory 异常:
experiments/decisions.md:3832:**决策**: 切换 **seed 42 → seed 41**,新命名 `exp263d_best_b_od_s41_3090_pwrlim`。
experiments/decisions.md:3833:- kill exp263c main PID 266
experiments/decisions.md:3834:- 启动 exp263d seed 41 at 23:34 CST (PID 8248)
experiments/decisions.md:3837:**用户指示**: "报告时就报告这个是 seed 41 就行" — PRCV 主表 exp263 行用 exp263d seed 41 的数字。
experiments/decisions.md:3844:**Monitor 更新**: stop b9h22bdiy (old exp263c tail) → bizb8v35k (new exp263d tail)
experiments/decisions.md:3851:- 启动 `exp265b_best_s_op_s41` (Small Full Scaffold OP seed 41) on srvA @ 12:00:30 CST, PID 633
experiments/decisions.md:3853:- 相对 exp265 (seed 42, srvC) 单变量 SEED 42→41
experiments/decisions.md:3855:- 用途: 和 exp265 组成 2-seed ensemble 或 max, 强化 OP SOTA 声明 (vs KPR w/o prompt 73.3/82.5)
experiments/decisions.md:3858:- 历史 exp263 Base OD 在 5060Ti e100 eval OOM (13.2G → 16G),  exp269 Base Market e80 eval OOM, exp266 Base OP silent exit (不确定 OOM)
experiments/decisions.md:3865:- 立即 apply: kill exp265b (12:00 版 TEST=256) + restart (12:08 版 TEST=128 PID 1151)
experiments/decisions.md:3868:**挂 daemon exp265b → exp266b (5060Ti Base OP seed 41, 带 TEST BATCH 降)**:
experiments/decisions.md:3870:queue_on_ckpt.sh /hy-tmp/log/occluded_posetrack/exp265b_.../transformer_120.pth \
experiments/decisions.md:3872:  /hy-tmp/log/occluded_posetrack/exp266b_best_b_op_s41 \
experiments/decisions.md:3873:  /tmp/exp266b.log exp265b_to_266b \
experiments/decisions.md:3876:exp266b FINAL 预计后天上午,覆盖 exp266 silent exit 留下的 OP 主表瑕疵。
experiments/decisions.md:3878:Monitor b8y4oohc4 arm for srvA exp265b。
experiments/decisions.md:3880:### [2026-04-21 03:47] 事件 — exp277 Small 3-stage PSG 训练塌缩 (negative result,不重训)
experiments/decisions.md:3883:- exp277 Small + PSG 3-stage `[-3,-2,-1]` 自 01:42 CST 启动
experiments/decisions.md:3884:- e10 eval **0.3 / 0.3** (接近 random), e120 FINAL **49.0 / 57.7** (远低 exp274 no-PSG 68.1/76.8, Δ=-19.1)
experiments/decisions.md:3893:- Tiny 3-stage (exp273) **60.5/69.9 正常** — Tiny backbone 容量小,features 较稀疏不易被 gate 归零
experiments/decisions.md:3894:- Small 3-stage (exp277) **49.0/57.7 塌缩** — Small backbone 容量大,features dense 更易被 multi-stage gate 压缩
experiments/decisions.md:3896:**决策**: **不重训**, 用 exp277 FINAL 作为 **negative result** 有效数据点。
experiments/decisions.md:3909:**Phase 3-A 科学结论** (初版, exp277 seed 42 塌缩):
experiments/decisions.md:3914:### [2026-04-21 04:30] 决策更新 — exp277 塌缩重审为偶发 seed 问题,exp277b seed 41 重跑
experiments/decisions.md:3917:- 3:47 CST exp277 FINAL 49.0/57.7 归因为 "Small 3-stage PSG 系统塌缩"
experiments/decisions.md:3922:- 新建 `exp277b_psg3_s_od_s41` 用 seed 41 重跑 (其他参数同 exp277)
experiments/decisions.md:3923:- daemon 3909905 挂 lab4090: exp284/transformer_120.pth → exp277b
experiments/decisions.md:3924:- 预计 tmr 11:50 CST FINAL (exp284 ~tmr 10:00 + 1h50min)
experiments/decisions.md:3925:- **exp277b 数字替代 exp277 作为 PRCV Table 2 Small 3-stage 行的数字**
experiments/decisions.md:3926:- exp277 (seed 42) 降级为 decisions.md 里 "偶发 seed 塌缩" 记录, results.md 标 strikethrough
experiments/decisions.md:3934:**不预判**, 等 exp277b 数据再下结论。当前 Phase 3-A 结论暂定 (基于 exp275/276 稳定的 1/2-stage 收益)。
experiments/decisions.md:3936:### [2026-04-22 08:08] 事件 — exp280 FINAL 65.7/76.2, Phase 3-B Tiny 2×2 闭合 + srvB idle
experiments/decisions.md:3939:- exp280 Swin-Tiny + GCN512 + PSG `[-1]` FINAL @ 08:07 CST srvB
experiments/decisions.md:3946:| PSG `[-1]` | 65.7/76.7 (exp278) | **65.7/76.2** (exp280, **weakest R1**) |
experiments/decisions.md:3947:| PSG `[-2,-1]` | 65.7/76.9 (exp279) | **65.9/77.4** (exp261) |
experiments/decisions.md:3950:- **GCN512+1stg 必弱**: Tiny 76.2 R1 最弱, Small exp284 82.9 R1 最弱
experiments/decisions.md:3957:**srvB GPU 状态**: exp280 是 Phase 3-B Tiny chain 最后一个, 无 daemon 继承 → **srvB idle**。
experiments/decisions.md:3960:- Task #12 (批量 MaxSim+flip) 用户指令"等当前队列跑完再起", srvC Phase 3-C exp288→exp289 ~12.5h 后才全 FINAL
experiments/decisions.md:3964:**监控链 idle 判定**: monitor `boairmoh9` (srvB 事件) 保持 armed, 将捕获任意意外事件。exp280 FINAL 处理完毕 (monitor.md + results.md + ablation.md + decisions.md + memory + git push `08de230`)。
experiments/decisions.md:3966:### [2026-04-22 09:29] 事件 — exp266b_3090 FINAL 78.5/86.2 (Base OP s41 完整 120ep) + lab3090 idle
experiments/decisions.md:3969:- exp266b_3090 Swin-Base + Full Scaffold + Occ-PTrack + seed 41 FINAL @ 09:29 CST lab3090 (docker, pwrlim 280W)
experiments/decisions.md:3973:- exp266 s42 srvC e60 eff: 78.4/86.2 → Δ +0.1/0 (持平, seed 41 微优 mAP)
experiments/decisions.md:3974:- exp265 Small OP s42: 78.4/86.2 → Δ +0.1/0 (**Base vs Small OP 0 mAP 增益**)
experiments/decisions.md:3975:- exp265b Small OP s41: 78.5/85.9 → Δ 0/+0.3 (Base 略优 R1 over Small 同 seed)
experiments/decisions.md:3982:**论文主数字**: Base OP 用 exp266b_3090 78.5/86.2 (完整 120ep), 替换 exp266 s42 e60 eff。
experiments/decisions.md:3985:- Phase 3-B Small GCN512+2stg rerun (exp285b) ✓
experiments/decisions.md:3986:- Base OD seed 41 (exp263d) ✓
experiments/decisions.md:3987:- Base OP seed 41 (exp266b_3090) ✓
experiments/decisions.md:3990:- Task #12 MaxSim eval (Base ckpts exp263d + exp266b_3090, 只需 test.py, ~5 min/ckpt)
experiments/decisions.md:3991:- 等 srvA exp266b + srvC Phase 3-C 都 FINAL 后统一批跑 (用户 wait 指令)
experiments/decisions.md:3994:**当前五机**: srvA exp266b (刚启动 e2), srvB idle, srvC exp288 (e95), lab3090 idle, lab4090 idle。3 idle。
experiments/decisions.md:4001:- srvB (5+4 batches): exp261 Tiny OD, exp267 Tiny Market (retry 后成功), exp278/279/280 Phase 3-B Tiny, exp271/272/273 Phase 3-A pure PSG Tiny
experiments/decisions.md:4002:- lab3090 (2 ckpts): exp263d Base OD, exp266b_3090 Base OP
experiments/decisions.md:4003:- lab4090 (4+5 batches): exp282/283/284/285b Phase 3-B Small, exp275/276/277/277b Phase 3-A pure PSG Small (exp274 POSE_ENABLED False crash)
experiments/decisions.md:4016:**跨 eval 验证**: Phase 3-A pure PSG 所有 Global+flip 数字和训练 FINAL eq+flip 精确对齐 (差 ≤ 0.1 R1), **exp277 seed 42 塌缩 49.0/57.6 跨 eval 复现确认偶发 seed 训练塌缩**。
experiments/decisions.md:4019:- Tiny 2×2: GCN512+2stg peak (exp261 66.4/77.7), GCN512+1stg 最弱 (exp280 66.1/76.7)
experiments/decisions.md:4020:- Small 2×2: GCN512+2stg peak mAP (exp285b 74.0/84.1), 四格方差 ≤ 0.3 mAP / 0.4 R1
experiments/decisions.md:4023:- srvC local: exp264 Tiny OP, exp265 Small OP, exp286/287 Phase 3-C Tiny LGPA-only, exp288/289 Phase 3-C Small LGPA-only
experiments/decisions.md:4024:- srvA local: exp262 (原始 srvA), exp265b Small OP s41, exp268 Small Market, exp269 Base Market
experiments/decisions.md:4032:- 等 srvA exp266b FINAL ~14:00 → srvA idle → 补 exp262/265b/268/269 eval
experiments/decisions.md:4033:- 等 srvC exp288/289 FINAL ~17:00 → srvC idle → 补 exp264/265/286/287/288/289 eval
experiments/decisions.md:4034:- 可选: lab3090 上跑 cross-domain Market→Occ-ReID (Occ-ReID 数据集已解压, 需 rsync exp267/268/269 Market ckpt)
experiments/decisions.md:4036:### [2026-04-22 12:51] 🔥 exp288 FINAL 73.8/83.8 — GCN 对 Small OD 零贡献确认
experiments/decisions.md:4039:- exp288 Swin-Small + LGPA + OA-SD + ParAug + LOWER_BODY_OCC + PSG `[-1]` (**无 GCN**) FINAL @ 12:51 CST srvC
experiments/decisions.md:4043:- exp285b Full Scaffold (GCN512 + LGPA + 2-stg PSG): **73.8 / 83.8 / 90.7 / 92.7** → Δ 0/0/-0.2/-0.7
experiments/decisions.md:4044:- exp282 Full GCN256+1stg: 73.7/83.9 → Δ +0.1/-0.1
experiments/decisions.md:4045:- exp284 Full GCN512+1stg: 73.4/82.9 → Δ +0.4/+0.9 (LGPA-only 反超!)
experiments/decisions.md:4049:2. 和 Tiny 结论 (exp286 LGPA-only 66.0 ≈ exp261 Full 65.9) **跨 backbone 一致**
experiments/decisions.md:4059:- exp289 LGPA-only 2-stg 自动启动 (srvC PID 86783), FINAL ~16:50 对照 PSG stage in LGPA-only 配置
experiments/decisions.md:4060:- 建议: 跑完 exp289 → 评估是否也加 GCN 做 Market/OP 对照 (exp267 + exp264 本就无 GCN 配置?)
experiments/decisions.md:4063:- results.md Phase 3-C section 已填 exp288 FINAL
experiments/decisions.md:4067:### [2026-04-22 14:31] ⚠️ exp292 CUDA OOM @ e20 eval, restart with TEST.IMS_PER_BATCH 64
experiments/decisions.md:4070:- exp292 Small Market target-heatmap 启动 12:52, 训练 e1-e20 顺利, Loss 14.77→4.08, Acc 0.001→0.607
experiments/decisions.md:4081:- 重启 exp292 with `TEST.IMS_PER_BATCH 64` (从 default 256 降 4x)
experiments/decisions.md:4087:- 新启动 PID 通过 /tmp/exp292.log 验证 e1-e20 都过关, 特别关注 e20 eval
experiments/decisions.md:4092:- lab4090 exp291 目前 TEST 默认 256, 如果 e20 eval 失败也同样降
experiments/decisions.md:4094:### [2026-04-22 18:13] exp291 FINAL 73.5/82.9 (target-heatmap OD) + exp293 auto-chain launched
experiments/decisions.md:4096:**exp291 FINAL** @ 18:13:30 CST lab4090:
experiments/decisions.md:4098:- vs exp285b Full Scaffold scene baseline 73.8/83.8 → Δ -0.3/-0.9/0/-0.2
experiments/decisions.md:4101:**三数据集 target-heatmap 横向对比 (partial, exp290/exp292 还在跑)**:
experiments/decisions.md:4104:| OP (多人, exp290 e30) | -0.1 / +0.1 | R1 持平/微优, 符合预期机制有效场景 |
experiments/decisions.md:4105:| OD (多单人, exp291 FINAL) | -0.3 / -0.9 | 接近 no-op, 微差 eval noise |
experiments/decisions.md:4106:| Market (全单人, exp292 e30) | 对照待 FINAL | 预期严格持平 (目前 e30 92.7 正常轨迹) |
experiments/decisions.md:4110:- 作为 supplementary 消融: 机制在 single-person 数据集无回归, 论文主表 Small OD 仍用 exp285b 73.8/83.8
experiments/decisions.md:4112:**auto-chain → exp293 触发成功**:
experiments/decisions.md:4113:- daemon 706372 detected ckpt @ 10:14:09 UTC (18:14 CST), 20s 安全 + no-crash 检查 → launch exp293 PID 724112 @ 10:14:29 UTC
experiments/decisions.md:4114:- exp293 config 确认 PLBOA=True 激活, OA-SD WARNING 消失 (teacher/student 现有差异)
experiments/decisions.md:4117:### [2026-04-22 23:25] exp292 e90 eff FINAL + exp293 e80 eff FINAL — target-heatmap Market + PLBOA Base 双消融收尾
experiments/decisions.md:4119:**exp292 Small Market target-heatmap** (lab3090 PLBOA OFF default):
experiments/decisions.md:4122:- vs exp268 FINAL 94.3/97.3: Δ **-0.1 / -0.2** (essentially 持平)
experiments/decisions.md:4123:- 结论: target-heatmap 在 Market 全 single-person 严格 no-op, 和 exp291 OD (-0.3/-0.9) / exp290 OP (-0.1/0) 结论一致 — **机制 3 数据集都 near-持平**
experiments/decisions.md:4125:**exp293 Base Market + PLBOA** (lab4090, OA-SD 激活):
experiments/decisions.md:4130:- vs exp269 e80 eff FINAL 94.4/97.0 (PLBOA OFF): Δ -0.3 / -0.1 (Global)
experiments/decisions.md:4133:  - 主表 Base Market 主数字 **仍用 exp269 94.4/97.0**
experiments/decisions.md:4134:  - exp293 作 supplementary "PLBOA on Market" 消融
experiments/decisions.md:4140:- **定位**: supplementary 消融, 证明机制 **backward-compat** (single-person 无回归) 和 **target disambiguation 语义正确**
experiments/decisions.md:4141:- 不作主创新, 主表 Small OD/OP/Market 仍用 exp285b/exp265/exp268 scene baseline
experiments/decisions.md:4144:- OD (exp285b etc): PLBOA True, OA-SD 蒸馏有效, +性能
experiments/decisions.md:4145:- OP (exp265 etc): PLBOA True, OA-SD 蒸馏有效, +性能
experiments/decisions.md:4146:- **Market: PLBOA False** (exp293 验证), 分布不匹配 → 保留关闭
experiments/decisions.md:4155:- exp289 完成后 Phase 3-C 2x2 闭合 (srvC ~05:30 tmr FINAL)
experiments/decisions.md:4156:- exp266b srvA FINAL (~13:00 tmr) 作 Base OP seed 41 srvA 对照 (cross-device with lab3090)
experiments/decisions.md:4157:- exp290 srvB FINAL (~09:00 tmr)
experiments/decisions.md:4160:### [2026-04-23 05:40] exp289 FINAL 73.8/83.3 — Phase 3-C Small 2×2 闭合 + exp269b auto-chain 启动
experiments/decisions.md:4162:**exp289 FINAL**:
experiments/decisions.md:4165:- vs exp288 LGPA-only 1-stg 73.8/83.8/90.5/92.0: Δ 0 / -0.5 / 0 / +0.4
experiments/decisions.md:4166:- vs exp285b Full Scaffold 73.8/83.8: Δ 0 / -0.5
experiments/decisions.md:4171:| LGPA-only | exp288 73.8/83.8 | exp289 73.8/**83.3** |
experiments/decisions.md:4177:**exp269b auto-chain 启动成功** @ 05:40 srvC via daemon 94420:
experiments/decisions.md:4178:- Base Market PLBOA OFF full 120 epoch (公平对比 exp293 restart PLBOA ON)
experiments/decisions.md:4186:- Phase 3-C (LGPA-only × PSG): **4/4 FINAL ✓** (刚刚 exp289 闭合!)
experiments/decisions.md:4188:- PLBOA Market (exp293 restart + exp269b) 进行中 (~06:00-11:40 FINAL)
experiments/decisions.md:4189:- exp263b (Base OD s42 restart) queued on lab4090 after exp293
experiments/decisions.md:4191:srvC 接下来: exp269b FINAL ~11:40 → 再 idle (无 chain). 或可 queue exp263b_s42 之类。
experiments/decisions.md:4193:### [2026-04-23 08:24] exp293 FINAL 93.8/97.2 (restart full 120) + exp263b auto-chain launched
experiments/decisions.md:4195:**exp293 restart FINAL**:
experiments/decisions.md:4200:**对比 original exp269 (PLBOA OFF, e80 eff 94.4/97.0)**:
experiments/decisions.md:4202:- 但 exp269 只有 e80, 对比不公平 — 等 exp269b FINAL (~11:40) 才有公平 120ep vs 120ep
experiments/decisions.md:4204:**cross-restart noise (exp293 first run e80 eff vs restart e80)**:
experiments/decisions.md:4209:**exp263b auto-chain 启动 @ 08:24 lab4090**:
experiments/decisions.md:4210:- Base OD seed 42 full 120 restart (原 exp263 e100 eff 72.5/81.8 OOM 截断)
experiments/decisions.md:4215:1. ✅ exp293 (PLBOA ON Base Market) FINAL 93.8/97.2
experiments/decisions.md:4216:2. 🔄 exp269b (PLBOA OFF Base Market) srvC e17, FINAL ~11:40
experiments/decisions.md:4217:3. 🔄 exp263b (Base OD s42) lab4090 e1 NEW, FINAL ~15:30
experiments/decisions.md:4218:4. ⏳ exp266c (Base OP s42) queued srvB after exp290 FINAL (~09:15)
experiments/decisions.md:4221:### [2026-04-23 09:22] exp290 FINAL 78.4/86.2 — target-heatmap OP 严格持平 scene + exp266c chain
experiments/decisions.md:4223:**exp290 FINAL**:
experiments/decisions.md:4226:- **严格持平 exp265 scene baseline 78.4/86.2/94.8/97.3** (Δ 0/0/0/+0.1)
experiments/decisions.md:4231:| OD (exp291) | 73.5/82.9 | 73.8/83.8 | -0.3/-0.9 |
experiments/decisions.md:4232:| **OP (exp290)** | **78.4/86.2** | **78.4/86.2** | **0/0 严格持平** |
experiments/decisions.md:4233:| Market (exp292 e90 eff) | 94.2/97.1 | 94.3/97.3 | -0.1/-0.2 |
experiments/decisions.md:4240:- 主表 Small OP 数字用 exp265 78.4/86.2 (= exp290, 等价)
experiments/decisions.md:4242:**exp266c chain**:
experiments/decisions.md:4243:- daemon 109773 detected exp290 ckpt @ 09:21
experiments/decisions.md:4248:- ✅ exp289 FINAL 73.8/83.3 (Phase 3-C Small 2-stg)
experiments/decisions.md:4249:- ✅ exp290 FINAL 78.4/86.2 (target-heatmap OP)
experiments/decisions.md:4250:- ✅ exp293 FINAL 93.8/97.2 (Base Market PLBOA ON full 120)
experiments/decisions.md:4251:- 🔄 exp269b e20 (Base Market PLBOA OFF full 120)
experiments/decisions.md:4252:- 🔄 exp263b e8 (Base OD s42 full 120)
experiments/decisions.md:4253:- ⏳ exp266c queued (Base OP s42 full 120) chain soon
experiments/decisions.md:4255:### [2026-04-23 13:20] 决策 #exp266b FINAL 78.7/86.3 — Base OP 新 SOTA
experiments/decisions.md:4257:**exp266b srvA s41 FINAL (2026-04-23 13:18:50 CST)**:
experiments/decisions.md:4263:| **srvA 5060Ti** | exp266b | **78.7/86.3** | baseline |
experiments/decisions.md:4264:| lab3090 | exp266b_3090 | 78.5/86.2 | -0.2/-0.1 |
experiments/decisions.md:4272:| Small (exp265/265b) | 78.4/86.2 | 78.5/85.9 |
experiments/decisions.md:4273:| Base (exp266/266b) | 78.4/86.2 e60 eff | **78.7/86.3** (srvA) |
experiments/decisions.md:4276:- 原方案: exp266b_3090 78.5/86.2 (lab3090 完整 120 epoch)
experiments/decisions.md:4277:- **更新方案**: **exp266b srvA 78.7/86.3** (srvA 完整 120 epoch, +0.2 mAP / +0.1 R1 更强)
experiments/decisions.md:4281:- exp266b 78.7/86.3 vs exp265b 78.5/85.9 → Δ **+0.2 mAP / +0.4 R1**
experiments/decisions.md:4288:### [2026-04-23 16:50] 决策 #exp263b FINAL 73.5/81.5 — seed 42 full 120 restart 有效但不如 seed 41
experiments/decisions.md:4290:**exp263b lab4090 s42 FINAL (2026-04-23 16:47:17 CST)**:
experiments/decisions.md:4292:- ckpt: `/home/afr/SOLIDER-REID/log/occluded_duke/exp263b_best_b_od_s42_full120/transformer_120.pth`
experiments/decisions.md:4297:| exp263 orig | 42 | e100 eff (OOM) | 72.5/81.8 | baseline |
experiments/decisions.md:4298:| **exp263b restart** | **42** | **e120 FINAL** | **73.5/81.5** | +1.0/-0.3 |
experiments/decisions.md:4299:| exp263d | 41 | e120 FINAL | 74.1/83.3 | +1.6/+1.5 |
experiments/decisions.md:4302:1. **full 120 epoch > e100 eff**: +1.0 mAP 提升, 说明原 exp263 因 OOM 中断的确损失了数字
experiments/decisions.md:4304:3. **R1 异常**: exp263b R1 (81.5) 略弱于 exp263 orig (81.8) 尽管 mAP 更高。可能 full 120 epoch 在末期轻微 overfit R1 top-1。
experiments/decisions.md:4308:- 主表仍用 **exp263d 74.1/83.3** (seed 41, 最强)
experiments/decisions.md:4309:- exp263b 作 **seed 42 full 120 复现数据点** (证明 restart 机制有效, 证明 seed 42 天然弱)
experiments/decisions.md:4314:- 对照 exp263 orig MaxSim 74.5/84.0, exp263d MaxSim 75.2/84.8
experiments/decisions.md:4317:- ✅ exp263b (Base OD s42 full 120) lab4090 FINAL 73.5/81.5
experiments/decisions.md:4318:- 🔄 exp266c (Base OP s42 full 120) srvB e30 eval 76.5/84.7
experiments/decisions.md:4319:- 🔄 exp269b (Base Market PLBOA OFF full 120) srvC e60 eval 94.1/97.0
experiments/decisions.md:4321:### [2026-04-24 01:20] 决策 #exp269b FINAL 94.5/97.2 — Market Base 新 SOTA, full 120 restart 策略验证
experiments/decisions.md:4323:**exp269b srvC s42 FINAL (2026-04-24 01:17:24 CST)**:
experiments/decisions.md:4326:**vs exp269 original (OOM 前 e80 eff)**:
experiments/decisions.md:4330:**vs exp268 Small**: Δ +0.2/-0.1 (Base vs Small Market 已饱和)
experiments/decisions.md:4331:**vs exp293b Base PLBOA ON**: Δ +0.7/0 → **PLBOA 净 -0.7 mAP 代价确认**
experiments/decisions.md:4334:- 原: exp269 orig 94.4/97.0 (Global+flip) 或 94.5/97.1 (MaxSim+flip)
experiments/decisions.md:4335:- **新: exp269b 94.5/97.2** (两者等价, 直接 eq_concat 就达 MaxSim 水平)
experiments/decisions.md:4338:- ✅ exp263b (Base OD s42 full 120) lab4090 FINAL 73.5/81.5, MaxSim 74.8/84.0
experiments/decisions.md:4339:- ✅ exp266c (Base OP s42 full 120) srvB **running** (e60 77.9/85.6)
experiments/decisions.md:4340:- ✅ exp269b (Base Market PLBOA OFF full 120) srvC FINAL 94.5/97.2
experiments/decisions.md:4342:**等等, 是 2/3 FINAL. srvB exp266c 仍在训练, FINAL ETA ~13:22 today。**
experiments/decisions.md:4346:### [2026-04-24 02:20] 决策 #exp294 FINAL 74.0/82.6 — GCN 冗余假设 3-backbone 统一验证
experiments/decisions.md:4348:**exp294 lab4090 s41 FINAL (2026-04-24 02:18:48 CST)**:
experiments/decisions.md:4352:| Exp | GCN | mAP | R1 | Δ vs exp263d |
experiments/decisions.md:4354:| exp263d | **ON** | **74.1** | **83.3** | baseline |
experiments/decisions.md:4355:| **exp294 (本)** | **OFF** | **74.0** | **82.6** | **-0.1 / -0.7** |
experiments/decisions.md:4360:| Tiny | exp287 65.9/77.0 | exp261 65.9/77.4 | **0/-0.4** |
experiments/decisions.md:4361:| Small | exp289 73.8/83.3 | exp285b 73.8/83.8 | **0/-0.5** |
experiments/decisions.md:4362:| Base | exp294 74.0/82.6 | exp263d 74.1/83.3 | **-0.1/-0.7** |
experiments/decisions.md:4374:- Base OD 主数字仍用 **exp263d 74.1/83.3** (最强)
experiments/decisions.md:4375:- exp294 作 **Phase 3-C Base 补齐行 + GCN 冗余 claim 证据**
experiments/decisions.md:4378:- lab4090 idle, 启动 exp294 MaxSim+flip eval (预期 ~74.8-75.2/83-84, 对标 exp263b 74.8 / exp263d 75.2)
experiments/decisions.md:4379:- 若 MaxSim < exp263d, 补 claim: "GCN 对 MaxSim 也冗余"
experiments/decisions.md:4382:- ✅ exp263b (Base OD s42) 73.5/81.5
experiments/decisions.md:4383:- ✅ exp266b (Base OP s41) 78.7/86.3 (SOTA)
experiments/decisions.md:4384:- ✅ exp269b (Market Base PLBOA OFF) 94.5/97.2
experiments/decisions.md:4385:- 🔄 exp266c (Base OP s42 full 120) srvB running, FINAL ~13:22 today
experiments/decisions.md:4386:- ⭐ **exp294 (Base Full-GCN s41 ablation)** FINAL 74.0/82.6 (用户新加 ablation)
experiments/decisions.md:4390:# Post-PRCV exp295–321b 决策回填（2026-06-15 补文档债）
experiments/decisions.md:4398:**双向 sweep 证据**：0.5 真生效(exp311b Small) **-0.7 mAP**；1.0 default(exp295/exp261) baseline ⭐；2.0(exp312 Tiny) **-0.4 mAP**
experiments/decisions.md:4406:**8 个 sweep 点（vs exp261 67.2/78.6 MaxSim）**：GLS2.0 -0.4；PartW2.0 -0.3；PartW0.5 0；lgpaW1.0 -0.2；oasdW2.0 0；lgpaW0.25 **+0.2**；partTriW0.5 -0.1；oasdW0.5 -0.1
experiments/decisions.md:4423:**结果（exp320 Small s1234 vs exp295）**：eq 68.1/79.3 vs 74.2/84.0（-6.1/-4.7）；MaxSim **68.8/79.6 vs 75.2/85.4（-6.4/-5.8）**
experiments/decisions.md:4430:**上下文**：overnight Base OD LR sweep（exp296/297/298）+ PLBOA 消融（exp299）。
experiments/decisions.md:4431:**LR sweep（Base s41 vs exp296 LR8 74.9/83.8 MaxSim）**：LR8 baseline；LR4(exp297) -0.3/+0.3（近 tie）；LR2(exp298) **-5.3/-4.7**（下界）
experiments/decisions.md:4432:**PLBOA 消融**：exp299(OFF) 72.7/80.5 vs exp296(ON) 74.9/83.8 → OD 上 **PLBOA net +2.2 mAP**；配 Tiny exp307(+2.7) 2-backbone 一致
experiments/decisions.md:4434:**理由**：LR4≈LR8（非显著 underfit），LR2 严重 underfit -5.3。PLBOA 在 Occ-Duke +2.2-2.7 mAP，但 Market→Occ-ReID 跨域 -25.4 mAP（exp293 vs exp269）。
experiments/decisions.md:4435:**执行结果**：Base OD 主表保持 exp263d 75.2/84.8；exp296-298 作 LR ablation，exp299/exp307 作 PLBOA dataset-specific evidence。
experiments/decisions.md:4440:**multi-seed 统计（MaxSim+flip）**：Small(42/1234/2024) mean **74.7 std 0.45** 主行 exp295；Base(41/1234/42) mean **74.87 std 0.42** 主行 exp263d
experiments/decisions.md:4445:### [2026-06-16] 决策 #exp323 — MLLM 视觉裁剪 A/B 廉价首验（3B 退化，不可判）
experiments/decisions.md:4454:**选择**：(a) always-NO 使一个词格式下 A/B/C **不可判**，非方法被证伪；
experiments/decisions.md:4467:**选择**：判 "frozen 小 MLLM + pose 视觉裁剪/文字提示" 首验**不正向**。
experiments/decisions.md:4470:**执行结果**：建议砍 frozen-MLLM-reasoner 廉价首验，转 exp324（DINO-correspondence，更 frontier-independent）或换机制。
experiments/decisions.md:4471:保留 escape hatch：若坚持 MLLM 线需 LoRA 让模型学会用裁剪/grounding，但 frozen 证据偏负+沉没成本警告。
experiments/decisions.md:4473:### [2026-06-16] 决策 #exp324 — DINO emergent correspondence + pose-anchored part-MaxSim 首验偏正
experiments/decisions.md:4475:**上下文**：exp323 frozen-MLLM 线偏负后，按搬范式 #2 路线做 frozen DINOv2-base 廉价首验（training-free）：
experiments/decisions.md:4482:ALL 子集同向更明显（pose-part 3.21/7.87 vs holistic 0.64/0.90）。绝对分低（heavy 1.86 mAP）但落在 DINO 零样本 ReID 文献区间（0.3-4.7）。
experiments/decisions.md:4484:**理由**：(1) 三种表征**单变量隔离干净**——(b)/(c) 都是 5 同序 part 向量在 common-visible part 求均值，唯一差别是锚定方式（pose vs 固定带），grid 几乎不涨而 pose 大涨，直接证明"姿态把 DINO token 约束到身体部位语义"是涨点来源，不是部位分解 trivial 效果；
experiments/decisions.md:4486:(3) 与 exp323（frozen 干预无效）形成对照——同样 frozen + 同样 pose，但 DINO dense correspondence 这条**有信号**，差别在表征端而非 LLM-reasoning 端。
experiments/decisions.md:4487:**执行结果**：exp324b 候选——冻结 DINO，仅训一个轻量 part-projection 头（或 LoRA）把 token 投到 ReID-judiciable 空间，
experiments/decisions.md:4491:### [2026-06-16] 决策 #exp327 — 更强冻结对应源（DINOv2-with-registers）止损
experiments/decisions.md:4493:**上下文**：exp324 frozen DINOv2-base pose-part 重遮挡 1.86，天花板低。问"换更新/更干净的冻结 SSL 源能否抬过 1.86"。hyy GPU1，唯一变量=特征源。DINOv3-vitb16 gated（hf-mirror 需 token）下不了，改用 ungated 的 dinov2-with-registers-base（registers 去 high-norm artifact token，更干净 dense 特征）。
experiments/decisions.md:4499:**理由**：registers 更干净特征只蹭出 +0.29 mAP（heavy），远不足以独立可用（exp324b 头已到 14）；印证 exp324 假说**训练-free 天花板瓶颈在 "frozen" 本身，不在 SSL 模型新旧/registers**。换更强冻结 DINO 源不是天花板解。
experiments/decisions.md:4504:**上下文**：exp324 frozen DINOv2-base pose-part 重遮挡 1.86。对应特征综述称 SD UNet 中间特征（DIFT）在遮挡/姿态对应基准上比 DINO 高 14-19 PCK。问"换 SD-DIFT 特征源能否超 1.86"。hyy GPU0，唯一变量=特征源（DINOv2→SD-v1.5 UNet up_blocks[1] DIFT，t=100 ensemble=4）。
experiments/decisions.md:4507:  B. 不超 → SD 训练-free 不优于 DINO，止损。
experiments/decisions.md:4508:**结果**：DIFT smoke（500 gallery）pose-part heavy **9.92**（趋势第一，误导），但 **FULL（17661 gallery）塌到 0.73（−1.13 vs 1.86）**，更不及 dinov2-registers 2.15。机制方向仍在（pose 0.73 > grid 0.35 > holistic 0.22）但绝对判别性远低于 DINO。
experiments/decisions.md:4510:**理由**：(1) DINO 从 smoke 2.55→full 1.86 仅小降，DIFT 从 9.92→0.73 **灾难性塌**——证明 **SD/DIFT 特征 category-level 语义对应强（PCK 高）但 instance-level 身份判别弱**（与 SD-DINO / Tale-of-Two-Features 文献一致：SD 与 DINO 互补、SD 不主导 instance retrieval）；(2) instance-discrimination 是 SD 特征的**结构性短板**（非超参问题），扫 t/up_block/ensemble 不会救；(3) 训头起点 0.73 远低于 DINO（1.86→14），不值得上 exp326b。
experiments/decisions.md:4511:**执行结果**：SD/DIFT 线止损，不上头。**重要方法论教训写入铁律：训练-free probe 必须用全量 gallery 判定绝对值，小 gallery smoke 只验流程不验数值**——DIFT 是活教材（smoke 排第一、full 垫底）。结合 exp327（registers +0.29 小幅、不破天花板）：**换特征源（更新 DINO / 换 SD 范式）都不是 frozen 天花板的解**，瓶颈在 frozen 本身（需 LoRA/解冻，即 exp324d 线）或换"DINO 补 Swin"重量级角度（planner #1 oracle）。
experiments/decisions.md:4513:### [2026-06-16] 决策 #exp324i — 做"解相关感知 DINO-LoRA"作 FM-import 方向最后一个真 method shot
experiments/decisions.md:4515:**上下文**：夜间 FM-import 全线证负，headline = 判别性-互补性张力（adaptation 让 DINO 判别化但趋同 Swin，融合只 +0.37）。lab-3090-d 空闲。用户睡前铁令"整夜不停务必找一个有用创新点"。问：直接用解相关损失攻击该张力，能否换来真互补、融合超 SOTA？
experiments/decisions.md:4517:  A. 跑 exp324i（跨协方差解相关 DINO-LoRA，λ=0 vs λ=1 单变量）——真机制介入，成则 method、败则把张力升级为强结论。
experiments/decisions.md:4526:**上下文**：exp324i（解相关感知 DINO-LoRA）e10 matched oracle 出（λ=0 vs λ=1）。
experiments/decisions.md:4540:**资产**: 现成 Market-trained ckpt `log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth`(Swin-Base+PSG+LGPA+GCN512), 现成跨域 eval `test_on_occluded_reid.py`(Market→Occluded-ReID 86.0/88.5 已存), 两域数据+pose 齐。脚本 `scripts/uce_calib_probe.py`, 结果 `log/uce_calib_probe.json`。
experiments/decisions.md:4551:**资产**: Market-trained ckpt `log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth`(Swin-Base+PSG+LGPA+GCN512, Occ-ReID baseline 88.0 mAP MaxSim+flip), 数据 Occluded-ReID(1000q/1000g+pose), env solider-reid(torch1.13+mmcv)。脚本 `scripts/vcnorm_probe.py`(主)+`scripts/vcnorm_probe_control.py`(对照), 结果 `experiments/vcnorm_probe/*.json` + README。
experiments/decisions.md:4563:**上下文**: 夜间范式调研唯一过审强 bet=burstiness(VLAD-BuFF/face-set import)。0-GPU 前提在 frozen DINO 成立(occluded +0.0206 更 bursty)。e120 弱 baseline(TransReID 53.5)训练模型真实判据。
experiments/decisions.md:4599:**调查**: 派 10 个 Codex 并行深挖(用户 rate-limit 不让开 300 Claude 子agent,Codex token 无限)→ 挖出**热图 bug**:exp335 喂 target-only 热图(`heatmaps[:,0]`+POSE_USE_TARGET_HEATMAP=True)→ LGPA assign KL 坍缩=0 → 部位退化。修(scene-merged)→ assign 0→7.02≈原版。但 ViT 仍只 +0.5、不翻盘。深挖发现 **LGPA-D 从未单独跑过**(exp244/245g 全是 PSG+LGPA+OASD+aug+384+Swin 全系统)。
experiments/decisions.md:4640:- 姿态**外挂** (exp342 detached LGPA): 60.0 +0.2 marginal
experiments/decisions.md:4691:**上下文**: 三个独立 codex(终身 d3 / 开集 d9 / 长尾 d10)收敛到同一 re-framing: ReID 失败由 GALLERY 组成(规模/膨胀/分布)驱动, 非只看 query/模型。用户要求零训练验证, ★铁律=每个 per-query 相关都控 trivial 代理(吸取 HUBNESS §7.6 教训: 上个诊断被漏控 #false-in-topk 证伪)。脚本 `cvpb_gallery_killswitch.py`, 复用 hubness 缓存特征, Market exp260b + Occluded-Duke exp255。双审(Claude broad 5 blocking 全修 + Codex)。
experiments/decisions.md:4693:**结果**(frozen, numpy, log `/tmp/cvpb_gallery_{market,oduke}.log`):
experiments/decisions.md:4694:- **测试 A Gallery-Growth Tax = LIVE**: frozen 模型旧 query mAP 随同域 gallery 膨胀结构性下降(Market 1x→10x −4.4, **OD −12.9**, 量级≈LReID 报的 forgetting)。CONTROL1(#false-in-topk, 杀 Hubness 的代理): ρ(−dAP,d#false)+0.74 大部分是 trivial 计数, 但"#false 完全不变"子集仍 −1.2(Market)/−2.6(OD) mAP, partial(OD)+0.28——结构成分过了致命代理。CONTROL2 ★决定性: real distractor −4.45(Market)/−13.16(OD) vs 列洗牌毁方向同 count −0.00 → tax 是结构性(distractor 身份几何咬人), 非机械 count。
experiments/decisions.md:4696:- **测试 C Singleton Merge = DEAD**: NN-is-head 0.72 只反映 head 占 72% 图像质量。per-head-ID(n=450/311 真功效)Spearman(support, attraction-PER-IMAGE)+0.003/+0.005≈0, 分箱 per-image 甚至下降。support-calibrated 阈值几乎无增益(d≈−0.003)且 40-60% level 退回 global。被 "head 图多→NN 彩票多" trivial count 吃掉。
experiments/decisions.md:4699:**理由**: A 是唯一过了 #false-in-topk + 列洗牌双控的信号, headline 干净: "frozen 强 ReID 旧 query 随同域 gallery 膨胀结构性掉点, LReID 误记为 catastrophic forgetting"。这正是上个 Hubness 诊断没做到的(被 trivial 代理吃光)——本次 A 的两个对照专为此教训设计且活下来。
experiments/decisions.md:4700:**执行(待办)**: A 当前是诊断, remedy(distractor-aware continual training)未验证, 需独立实验且警惕撞 backward-compatible LReID(arxiv 2403.10022)。诚实写明 CONTROL2 是主证据(CONTROL1 的结构残差 Market partial 仅+0.05 偏弱)。跨 backbone 普适性 + 与 re-rank 互补性未测。交付=`cvpb_gallery_result.md` 原始数字。
experiments/decisions.md:4702:### [2026-06-26] 决策 #99: LM-ReID(低分辨率=采样格点) session — 6.5 成稿 + 训练端穷尽 + 冲 7.0 失败 + d17 KILL = 探索收敛
experiments/decisions.md:4704:**上下文**: 探索 LM-ReID(d8 演化): 低分辨率 ReID 重定义为采样格点 sampling-lattice 隐变量, test-time decision marginalization(K=9 phase/bbox/kernel 变体边缘化)。autonomous mandate 找 B 类方法稿, 全自主无休止。脚本 cvpb_lattice_killswitch.py(全参数)/cvpb_lm_reid_train.py/cvpb_d17_killswitch.py。
experiments/decisions.md:4707:- **LM-ReID test-time 成立(6.5/10)**: LM-S2 5 分辨率全 beat 普通 TTA / LM-S2-strong 全 beat 强 TTA(+0.76~7.28, severe LR 处强 TTA 反有害) / LM-S4 bbox 检测框不确定性主导 +2.84 / K-sweep K=5 达 87% / LM-S3 logsumexp(soft decision marginalization)severe LR 最优 / backbone 泛化 Swin +3。
experiments/decisions.md:4708:- **训练端穷尽(8 机制 + 4 codex 8.5/10 无空间)**: embedding-invariance(consistency −1.73)/frozen-adaptation(LS-MRT +0.028/LPA +0.075/LATS)/backbone-loss(LSRC −1.9 损判别)/robust-ERM(Hard-Lattice 76.9<77.44)/input-canonicalize(BLC 数据证伪)全负 → "Why Training-Time Invariance Fails" 强论点。审查纪律: LSRC full-finetune codex 审出 Critical(默认混旧 loss)+High(train/test 不对称)已修[[pre-experiment-review-discipline]]。
experiments/decisions.md:4709:- **冲 7.0 三条腿全失败(codex push7 6/10 路径)**: ①detector-jitter σ-sweep 单调衰减到负(h12 +5.49→+2.18→−5.85, marginalization 是 sub-pixel sampling-lattice 非 detector 鲁棒性=诚实机制范围界定) ②MSMT17 跨数据集 config 缺失止损(msmt17_split 数据读对但 swin_small_pose.yml 被删 SANITY 2.67) ③adaptive-K 中性(per-query≈fixed K=5)。
experiments/decisions.md:4711:- **codex meta-eval 确认探索充分收敛**: 全新范式(event-camera 5.5 需新数据/federated 4.5/text 3 撞 FM/3D-SMPL 2.5 撞 SMPL/group 2 撞#false)非当前代码线方法点。
experiments/decisions.md:4713:**决策**: **LM-ReID 6.5 收尾投 B 类(唯一存活候选; codex 三层 push7/d17-eval/meta 都判务实)**。
experiments/decisions.md:4715:**执行**: paper 素材完整(experiments/exp359_lm_reid/paper_skeleton+results_tables 7 表+monitor; memory [[post-pivot-20domain-gallery-bet]])。正式 multi-seed/train/MLR benchmark/真实 detector 数据集(CUHK-SYSU/PoseTrack)留用户。event-camera 唯一边际全新方向留用户全新 sprint。codex 报告 litreview2/(lmreid_push7/d17_eval/meta_converge/train{2,3,4})。
experiments/decisions.md:4717:### [2026-06-27] 决策 #100: 换量级跳出盒子探索诚实终点 — AG+DG+open-set 全证伪, 现有 ReID 训练端机制探透
experiments/decisions.md:4721:**探索链(一整天全证伪)**:
experiments/decisions.md:4722:- **AG(exp363) aerial-ground/RGB-IR video foundation adaptation**: 视频证据积累路死(frozen DINOv2-reg 全 8 protocol mean-single<+5, 最高 +1.82, exp5 还 −1.82); frame-quality selection 坑(oracle 是 retrieval-label upper bound 无监督学不到); 换 foundation 不救。frozen DINOv2 Market 2.71。
experiments/decisions.md:4724:- **open-set/gallery-growth(codex 核查)**: 3.5/10 主线(撞 gallery 死区 B/C/Hubness 都被 #false-in-topk 吃 + 先例卡紧 Open-set ReID 2014/Open-Set Biometrics 2024, 唯一窄缝是 LReID re-attribution 分析非新机制)。
experiments/decisions.md:4726:**共同墙**: frozen foundation(DINOv2 通用/SOLIDER 行人专门)对 ReID instance prior 都弱(frozen 2.71/15.56 << fine-tuned 88+), **fine-tune 是王道**→解释 ReID 都靠 fine-tune 非 frozen retrieval; frozen-preservation/foundation-adaptation 类在 ReID 没燃料。
experiments/decisions.md:4731:**出路(用户拍板)**: ①收 LM-ReID(#99, 6.5 现有最强 B 类, 文档已全)/PSG 已成型线(最务实) ②换真实新数据规模(下新大规模数据集) ③换问题领域。
experiments/decisions.md:4733:**关联**: memory [[ag-dg-paradigm-killed-frozen-prior-weak]] [[paradigm-shift-occluded-reid-wall]]; monitor exp363_ag_foundation/exp364_dg_foundation; codex 报告 paradigm_shift/(codex_full_reid_gap/ag_verify/dg_deepen/openset_check)。

exec
/bin/zsh -lc "rg --files experiments | rg 'experiments/exp[0-9]+/(design|monitor|claude_review)\\.md"'$'"' | sort -V | tail -40" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/exp249/monitor.md
experiments/exp250/design.md
experiments/exp251/claude_review.md
experiments/exp251/design.md
experiments/exp251/monitor.md
experiments/exp252/claude_review.md
experiments/exp252/design.md
experiments/exp252/monitor.md
experiments/exp253/claude_review.md
experiments/exp253/design.md
experiments/exp253/monitor.md
experiments/exp254/claude_review.md
experiments/exp254/design.md
experiments/exp254/monitor.md
experiments/exp255/claude_review.md
experiments/exp255/design.md
experiments/exp255/monitor.md
experiments/exp256/claude_review.md
experiments/exp256/design.md
experiments/exp256/monitor.md
experiments/exp257/claude_review.md
experiments/exp257/design.md
experiments/exp257/monitor.md
experiments/exp258/claude_review.md
experiments/exp258/design.md
experiments/exp258/monitor.md
experiments/exp259/claude_review.md
experiments/exp259/design.md
experiments/exp259/monitor.md
experiments/exp260/claude_review.md
experiments/exp260/design.md
experiments/exp260/monitor.md
experiments/exp322/design.md
experiments/exp323/design.md
experiments/exp323/monitor.md
experiments/exp324/design.md
experiments/exp324/monitor.md
experiments/exp325/claude_review.md
experiments/exp325/design.md
experiments/exp325/monitor.md

codex
最新决策记录已经把用户列的死区基本写实了：LM-ReID 训练端八类负，AG/DG/open-set/foundation 一整天负，single-support 也被判墙。接下来我会只保留“训练时改变可学习过程、但不再强迫 test-time 单向量坍缩”的候选。
exec
/bin/zsh -lc 'find experiments/exp359_lm_reid experiments/exp360_intruder experiments/exp361_psc_jepa experiments/exp362_genoccl experiments/exp363_ag_foundation experiments/exp364_dg_foundation experiments/exp365_wildlife experiments/exp366_active_evidence experiments/exp367_single_support -maxdepth 2 -type f | sort' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/exp359_lm_reid/claude_review.md
experiments/exp359_lm_reid/codex_lcrs_review.md
experiments/exp359_lm_reid/codex_lrfd_review.md
experiments/exp359_lm_reid/codex_review.md
experiments/exp359_lm_reid/codex_review_final.txt
experiments/exp359_lm_reid/codex_review_lsrc.md
experiments/exp359_lm_reid/codex_review_raw.md
experiments/exp359_lm_reid/codex_review_raw_v2.md
experiments/exp359_lm_reid/codex_review_v2_final.txt
experiments/exp359_lm_reid/codex_train_correctness.md
experiments/exp359_lm_reid/design.md
experiments/exp359_lm_reid/design_blc.md
experiments/exp359_lm_reid/design_lcrs.md
experiments/exp359_lm_reid/design_lrfd.md
experiments/exp359_lm_reid/design_lsrc.md
experiments/exp359_lm_reid/monitor.md
experiments/exp359_lm_reid/paper_skeleton.md
experiments/exp359_lm_reid/results_tables.md
experiments/exp359_lm_reid/train_time_options.md
experiments/exp359_lm_reid/train_time_pipeline.md
experiments/exp360_intruder/codex_h2fail_decision.md
experiments/exp360_intruder/design.md
experiments/exp360_intruder/monitor.md
experiments/exp361_psc_jepa/claude_review.md
experiments/exp361_psc_jepa/codex_pivot_decision.md
experiments/exp361_psc_jepa/codex_pscjepa_diagnosis.md
experiments/exp361_psc_jepa/codex_raw_r1.md
experiments/exp361_psc_jepa/codex_raw_r2.md
experiments/exp361_psc_jepa/codex_raw_r3.md
experiments/exp361_psc_jepa/codex_raw_stageb_r1.md
experiments/exp361_psc_jepa/codex_raw_stageb_r2.md
experiments/exp361_psc_jepa/codex_raw_stageb_r3.md
experiments/exp361_psc_jepa/codex_review.md
experiments/exp361_psc_jepa/design.md
experiments/exp361_psc_jepa/monitor.md
experiments/exp361_psc_jepa/psc_jepa_pretrain.py
experiments/exp362_genoccl/codex_gap_decision.md
experiments/exp362_genoccl/gap_audit.py
experiments/exp362_genoccl/monitor.md
experiments/exp363_ag_foundation/ag_frozen_baseline.py
experiments/exp363_ag_foundation/codex_download_cmds.md
experiments/exp363_ag_foundation/codex_frozen_verdict.md
experiments/exp363_ag_foundation/design.md
experiments/exp363_ag_foundation/monitor.md
experiments/exp364_dg_foundation/codex_dinov2_verdict.md
experiments/exp364_dg_foundation/codex_directft_review.md
experiments/exp364_dg_foundation/codex_directft_review2.md
experiments/exp364_dg_foundation/codex_review.md
experiments/exp364_dg_foundation/codex_solider_verdict.md
experiments/exp364_dg_foundation/design.md
experiments/exp364_dg_foundation/frozen_xdomain_probe.py
experiments/exp364_dg_foundation/monitor.md
experiments/exp364_dg_foundation/solider_frozen_probe.py
experiments/exp365_wildlife/.venv-wl/.gitignore
experiments/exp365_wildlife/.venv-wl/.lock
experiments/exp365_wildlife/.venv-wl/CACHEDIR.TAG
experiments/exp365_wildlife/.venv-wl/pyvenv.cfg
experiments/exp365_wildlife/__pycache__/cvpb_wildlife_localverify.cpython-311.pyc
experiments/exp365_wildlife/__pycache__/cvpb_wildlife_probe.cpython-311.pyc
experiments/exp365_wildlife/cvpb_wildlife_localverify.py
experiments/exp365_wildlife/cvpb_wildlife_probe.py
experiments/exp365_wildlife/design.md
experiments/exp365_wildlife/monitor.md
experiments/exp366_active_evidence/cvpb_active_evidence_probe.py
experiments/exp366_active_evidence/design.md
experiments/exp366_active_evidence/monitor.md
experiments/exp367_single_support/codex_laststage_design.md
experiments/exp367_single_support/codex_review.md
experiments/exp367_single_support/codex_review2.md
experiments/exp367_single_support/codex_review_laststage.md
experiments/exp367_single_support/codex_review_laststage2.md
experiments/exp367_single_support/codex_review_laststage3.md
experiments/exp367_single_support/codex_review_laststage4.md
experiments/exp367_single_support/codex_review_smoke.md
experiments/exp367_single_support/codex_train_design.md
experiments/exp367_single_support/cvpb_single_support_probe.py
experiments/exp367_single_support/cvpb_ss_cvar_smoke.py
experiments/exp367_single_support/design.md
experiments/exp367_single_support/monitor.md
experiments/exp367_single_support/ss_cvar_laststage.py

exec
/bin/zsh -lc "sed -n '4700,4745p' experiments/decisions.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
**执行(待办)**: A 当前是诊断, remedy(distractor-aware continual training)未验证, 需独立实验且警惕撞 backward-compatible LReID(arxiv 2403.10022)。诚实写明 CONTROL2 是主证据(CONTROL1 的结构残差 Market partial 仅+0.05 偏弱)。跨 backbone 普适性 + 与 re-rank 互补性未测。交付=`cvpb_gallery_result.md` 原始数字。

### [2026-06-26] 决策 #99: LM-ReID(低分辨率=采样格点) session — 6.5 成稿 + 训练端穷尽 + 冲 7.0 失败 + d17 KILL = 探索收敛

**上下文**: 探索 LM-ReID(d8 演化): 低分辨率 ReID 重定义为采样格点 sampling-lattice 隐变量, test-time decision marginalization(K=9 phase/bbox/kernel 变体边缘化)。autonomous mandate 找 B 类方法稿, 全自主无休止。脚本 cvpb_lattice_killswitch.py(全参数)/cvpb_lm_reid_train.py/cvpb_d17_killswitch.py。

**结果**:
- **LM-ReID test-time 成立(6.5/10)**: LM-S2 5 分辨率全 beat 普通 TTA / LM-S2-strong 全 beat 强 TTA(+0.76~7.28, severe LR 处强 TTA 反有害) / LM-S4 bbox 检测框不确定性主导 +2.84 / K-sweep K=5 达 87% / LM-S3 logsumexp(soft decision marginalization)severe LR 最优 / backbone 泛化 Swin +3。
- **训练端穷尽(8 机制 + 4 codex 8.5/10 无空间)**: embedding-invariance(consistency −1.73)/frozen-adaptation(LS-MRT +0.028/LPA +0.075/LATS)/backbone-loss(LSRC −1.9 损判别)/robust-ERM(Hard-Lattice 76.9<77.44)/input-canonicalize(BLC 数据证伪)全负 → "Why Training-Time Invariance Fails" 强论点。审查纪律: LSRC full-finetune codex 审出 Critical(默认混旧 loss)+High(train/test 不对称)已修[[pre-experiment-review-discipline]]。
- **冲 7.0 三条腿全失败(codex push7 6/10 路径)**: ①detector-jitter σ-sweep 单调衰减到负(h12 +5.49→+2.18→−5.85, marginalization 是 sub-pixel sampling-lattice 非 detector 鲁棒性=诚实机制范围界定) ②MSMT17 跨数据集 config 缺失止损(msmt17_split 数据读对但 swin_small_pose.yml 被删 SANITY 2.67) ③adaptive-K 中性(per-query≈fixed K=5)。
- **d17 rank-leverage KILL**: RLC 现象成立(mean 0.357>uniform 0.167)但 raw spearman 反向(−0.100)+partial|#false −0.049≈0 → 重演 Hubness/evidence #false-trivial-proxy 坑(本 session 第 N 次验证: 任何 per-query 解释变量必控 #false-in-topk)。
- **codex meta-eval 确认探索充分收敛**: 全新范式(event-camera 5.5 需新数据/federated 4.5/text 3 撞 FM/3D-SMPL 2.5 撞 SMPL/group 2 撞#false)非当前代码线方法点。

**决策**: **LM-ReID 6.5 收尾投 B 类(唯一存活候选; codex 三层 push7/d17-eval/meta 都判务实)**。
**理由**: 探索充分收敛——训练端无空间(8 机制+4 codex)/冲 7.0 机制范围被 detector 误差封顶/d17 撞 #false 坑/全新范式需新数据非方法点。6.5 是诚实天花板=中等偏强 B 类(重定义+干净 test-time 机制+训练端系统反例+多维证据+σ-sweep 诚实界定)。负结果大多成论文素材。
**执行**: paper 素材完整(experiments/exp359_lm_reid/paper_skeleton+results_tables 7 表+monitor; memory [[post-pivot-20domain-gallery-bet]])。正式 multi-seed/train/MLR benchmark/真实 detector 数据集(CUHK-SYSU/PoseTrack)留用户。event-camera 唯一边际全新方向留用户全新 sprint。codex 报告 litreview2/(lmreid_push7/d17_eval/meta_converge/train{2,3,4})。

### [2026-06-27] 决策 #100: 换量级跳出盒子探索诚实终点 — AG+DG+open-set 全证伪, 现有 ReID 训练端机制探透

**上下文**: 用户点醒"没限定 occreid+solider"→换量级跳出盒子(训新预训练范式/换 backbone/换问题, 不限现有 occluded+SOLIDER)。codex 全 ReID gap analysis 选 AG(8/10)/DG(#2 6.5)。

**探索链(一整天全证伪)**:
- **AG(exp363) aerial-ground/RGB-IR video foundation adaptation**: 视频证据积累路死(frozen DINOv2-reg 全 8 protocol mean-single<+5, 最高 +1.82, exp5 还 −1.82); frame-quality selection 坑(oracle 是 retrieval-label upper bound 无监督学不到); 换 foundation 不救。frozen DINOv2 Market 2.71。
- **DG(exp364) Camera-Pair Foundation-Preserved Residual**: direct-FT SOLIDER swin Market 30ep 让所有行人域涨(Market 15.56→88.70 / MSMT held-out 4.18→11.37 **+7.19** / Occ-Duke 3.27→14.47), fine-tune 不破坏 held-out 反提升→保护弱 frozen topology(15.56)无意义, 无 U-shape sweet spot(no-op, PSC-JEPA 同质死另一形式)。
- **open-set/gallery-growth(codex 核查)**: 3.5/10 主线(撞 gallery 死区 B/C/Hubness 都被 #false-in-topk 吃 + 先例卡紧 Open-set ReID 2014/Open-Set Biometrics 2024, 唯一窄缝是 LReID re-attribution 分析非新机制)。

**共同墙**: frozen foundation(DINOv2 通用/SOLIDER 行人专门)对 ReID instance prior 都弱(frozen 2.71/15.56 << fine-tuned 88+), **fine-tune 是王道**→解释 ReID 都靠 fine-tune 非 frozen retrieval; frozen-preservation/foundation-adaptation 类在 ReID 没燃料。

**codex 诚实判**: 现有 ReID 标准数据 + 4 单卡 + 强 backbone, 训练端新机制空间已经很窄。

**决策**: 换量级在 ReID 现有约束内基本探透。诚实停在用户拍板节点。
**出路(用户拍板)**: ①收 LM-ReID(#99, 6.5 现有最强 B 类, 文档已全)/PSG 已成型线(最务实) ②换真实新数据规模(下新大规模数据集) ③换问题领域。
**方法论价值**: cheap kill-switch(frozen probe 零训练几分钟 + 一次 30ep bounded)一天验死 AG+DG 两范式, 没浪费多日训练; codex 审 backbone fine-tune(抓预处理口径不一致+ckpt+checkpoint); 诚实定位每个工程 bug(patch14/.eval返回None/MSMT camid/0.5 norm/python env mmcv); 不绑架沉没成本。负向结论(钉死共同墙)避免后续重复盲试。
**关联**: memory [[ag-dg-paradigm-killed-frozen-prior-weak]] [[paradigm-shift-occluded-reid-wall]]; monitor exp363_ag_foundation/exp364_dg_foundation; codex 报告 paradigm_shift/(codex_full_reid_gap/ag_verify/dg_deepen/openset_check)。

exec
/bin/zsh -lc "sed -n '2300,2520p' experiments/innovation_brainstorm.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:

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

codex
我已经确认 `exp359/363/364/367` 是最近这批约束的落点。现在补读这几份设计/监控的结论细节，尤其是 single-support CVaR 和训练端坍缩规律。
exec
/bin/zsh -lc "sed -n '1,260p' experiments/exp367_single_support/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp367 Single-Support CVaR — monitor

## cheap kill-switch（frozen SOLIDER Market exp260b, 零训练, 2026-06-28）

### v1 → codex 审抓 3 High → v2 修

| | full | best-support | random-support | worst-support | best-worst gap |
|---|---|---|---|---|---|
| v1（污染）| 94.43 | 98.61（>full!）| 96.46 | 88.17 | 10.44 |
| **v2（干净）** | 94.43 | 76.08 | 73.36±0.22(20seed) | 63.82 | **12.27** |

★v1 污染（codex 审抓）：single-support 跳无 positive query（比不同子集）+ distractor 压 1 张（负样本池变）→ best/random>full 假象。
★v2 修（codex 3 High）：common-valid query 共用 + distractor 全量 + 20 seed + missing 记 0。**single-support 都 <full（合理少正样本），best-worst 12.27 + random-worst 9.54，false10 best0.923≈worst0.927 → gap 不被 #false-in-topk 解释**。

### codex 两轮审（用户要审查交 codex）
- v1：needs-attention，3 个 High（valid-query 污染 / 负样本池变 / kill-switch 不硬）。
- v2：needs-attention（轻微残留非致命）：false10 没给 random mean/std + go 没检查 false10 + missing 可能混 camera-coverage。best/worst oracle 用 query-label 可接受。

## ★VERDICT GO（基本可信）

support 选择有 oracle headroom（best-worst 12.27，不被 #false 解释），单图 support representation 是真训练瓶颈。**诚实标注**：best/worst 用 query-label oracle 上下界，证 headroom 存在；训练能否学到（不用 query）要 Single-Support CVaR train 验。

## 下一步

codex 调研 Single-Support CVaR 训练设计 + novelty 确认（63517）：novelty 真空白（episodic single-support+CVaR worst-support tail 标准 ReID 无直接先例），two-level CVaR 设计，cheap 验证路径，CCF-B 6.5/10。详见 design.md。

## frozen head smoke（codex cheap 路径 #1，2026-06-28）——失败

frozen backbone + projection head 训 episodic single-support CVaR 20ep（codex 审 loss 实现基本对，two-level 一致，不退化 hard-mining）：

| | frozen baseline(probe v2) | frozen head CVaR smoke | Δ |
|---|---|---|---|
| full-gallery | 94.43 | 93.89 | **−0.54** |
| random-support | 73.36 | 72.98±0.28 | **−0.38** |
| worst-support | 63.82 | 62.09 | **−1.73** |

**全部掉**（codex 成功线 worst/random +0.8~1.0 未达，反而掉）。

★诚实诊断：① **train loss 几乎 0（0.004）= episode 太易**（N=16 id 分类，support-query 同 id 分类到 16 id 太易）→ **CVaR worst tail≈0，CVaR term 没起作用**；② head 学 episode 分类过拟合 → eval 掉（frozen+projection 只能旋转特征，codex 预言）。

★codex 明确"frozen 失败不判死"（frozen 不够，可能要改 backbone）。但 loss 0 是 episode 设计问题，要修（增大 N / 用 gallery distractor 当负样本，让分类难、CVaR 起作用）才能真验机制。

## frozen head N=128（episode 修难，2026-06-28）——cvar≈random

N=16 loss 0（CVaR 空转）→ 增 N=128 让 episode 难、CVaR 起作用（loss 0.085→0.056）：

| mode | full | random-support | worst-support |
|---|---|---|---|
| frozen baseline(probe v2) | 94.43 | 73.36 | 63.82 |
| N=128 **cvar** | 94.25 | 73.28 | 63.36 |
| N=128 **random**(无 CVaR) | 94.24 | 73.26 | 63.29 |

★**cvar ≈ random**（三项几乎一样）→ CVaR term 在 frozen 特征上不带来差异。cvar/random 都 ≈ baseline（略掉 0.1-0.5）→ frozen head 训练没提升。

★诊断：① frozen head（projection）不够（≈baseline，codex 预言单线性头只能旋转改不了特征）② CVaR term 在 frozen 旋转空间没用（cvar≈random）。只有 last-stage（解冻 backbone 改特征）能区分"frozen 不够"vs"CVaR 机制本身弱"。

## last-stage backbone 训练（codex 四轮审 approve, 2026-06-28）

解冻 swin base.stages[-1]+norm3+bottleneck/classifier，episodic single-support CVaR loss，3 mode。codex 四轮审 approve（Critical make_optimizer → High eval/train 口径 → 修 → approve；ss_cvar_laststage.py + codex_review_laststage1-4.md）。

### cvar mode（epoch 20）—— DEAD

| | full | best | random | worst | best-worst gap |
|---|---|---|---|---|---|
| frozen baseline(probe v2) | 94.43 | 76.08 | 73.36 | 63.82 | 12.27 |
| cvar e20 | 94.41 | 76.05 | 73.40 | **63.62** | **12.42** |

★ss_cvar 不空转（~0.025-0.14，比 frozen smoke N=16 的 0 好），但 **worst 63.62<63.82（略掉）+ gap 12.42>12.27（略增）**——Single-Support CVaR 没改善 single-support 鲁棒性，反略负。codex 成功线 worst+2 完全未达（反向）。

★机理：worst-support 难来自 query-support 跨 camera/pose gap，训练优化 support 选择改不了本质难度（像 exp109 oracle headroom 墙：best-worst gap 是 identity-conditioned 不可训练实现）。lam=0.3 base 主导 + ss_cvar 信号太小。

### random/plain 对照（bbavvtwpy 确认中）

若 cvar≈random≈plain（都 worst 不改善）→ 坐实 Single-Support CVaR DEAD（worst 不可训练改善，非 episode/FT 本身）。

## 决定

cvar DEAD（worst 不可训练改善）。等 random/plain 对照坐实 → 记 memory（Single-Support CVaR：probe oracle headroom 12.27 存在但训练不可达，worst 跨 camera/pose gap 不可训练改善，又一个 exp109-style oracle-headroom 墙）→ 转 codex 训练侧 #2 Equivariant Routing。严谨 build + 四轮 codex 审跑出干净负结果，比脏 GO 强。

exec
/bin/zsh -lc "sed -n '1,260p' experiments/exp359_lm_reid/train_time_options.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# LM-ReID 训练端创新候选（2026-06-26，3-codex 头脑风暴综合）

用户问："是不是多想一个 train-time 方法，两个结合更有把握？" → 启 3 codex 各从一个 lens 深挖（litreview2/train_lens{1,2,3}_*.md），收敛到两个机制。

## 硬约束（已试死的）
训练端做 invariance / feature-collapse 压判别性 = 死路：consistency −1.73、lattice-aug 不特殊 +0.54、L_marg 训练有害大元凶。新机制必须绕开"逼 K 变体特征塌成一个点"。

## 机制 A：学怎么加权边缘化（LPA / LUD）— codex 最高 7.5/10
- **LPA（Lattice Posterior Amortization，lens3 #1）**：冻结 backbone，只训小头预测 posterior qφ(z|x)（z=bbox/phase/kernel 格点），测试时 `score = logsumexp_z[log qφ(z|q) + cos(fθ(Tz q), fθ(g))/τ]`，uniform → posterior 加权。
- **LUD（Lattice Uncertainty Distillation，lens2）= 同族**：训头预测 lattice 风险 risk_k（stopgrad teacher 的 spread + margin），测试 per-variant 加权。
- **关键洞察（lens2）**：query 级 scalar confidence **不能改 gallery 排序**，必须 per-variant 权重 pi_k 才涨 mAP。
- **不塌缩**：主 embedding 完全不动（stopgrad），只训加权头。
- **kill-switch**：冻 no-LM-loss ckpt + 训 qφ 1-3 epoch；weighted 比 uniform 多 ≥+0.4 mAP(h12/16) + 格点预测 acc ≥35% + LATTICE−TTA 不降 → 活，否则杀。
- **不撞 PFE**：PFE=generic data-quality uncertainty；LPA=supervised sampling-lattice posterior（隐变量来源明确 = LR 采样格点，监督来自 K-hypothesis empirical spread）。

## 机制 B：学怎么对齐检测框（LC-STN / BLC）— 7/10（lens1）/ 6.5/10（lens3）
- 训小模块**估计 LR crop 格点偏移 (dx,dy) 并 grid_sample 重采样到 canonical**（类 STN，但监督几何参数非身份特征），残差留 test-time marginalization。
- 第一版 translation-only sub-pixel re-centering（max_shift=1.25 LR px，tanh bound），只打 bbox 主因子。
- **更有分量**：改进模型本身、能提 K=1 单图（不只测试加权）。**但风险高**：可能替代而非叠加边缘化；可能学 dataset center bias 致 HR sanity/h32 掉点。
- **不塌缩**：压的是已知几何扰动参数、不是身份特征；3 保险（HR gallery bypass / 硬边界 / 冻 backbone 先 probe）。
- **kill-switch**：冻 backbone 只训 canonicalizer 预测注入 offset；injected shift MAE<0.35 LR px + K-spread 降≥20% + K=1 mAP +1.0 OR K=9 再+0.3/0.5 + HR sanity 掉<0.2 → 活。看 θ 别 saturate 到边界（伪信号）。
- **novelty**：STN/AlignedReID/PAN/STNReID/CDPM 先例多 → 卖点不是"用 STN"，是"LR detector crop lattice 当可边缘化隐变量 + 监督 canonicalization 非 invariance + 残差 decision marginalization"组合无先例。

## codex 明确不建议（现在）
- raw Hard-Lattice ERM/CVaR（太像更狠 lattice-aug，续压判别性）→ 只做 cheap ablation（= 正在跑的 LM-S5，验证后即停）
- TTT（无标签自我确认错误）/ SR-auxiliary（撞车 + 拉回"LR=缺细节"削弱 re-frame）/ DEQ（不贴）/ EM-backbone（复现 L_marg 有害）

## 决策
- 两个 kill-switch 都冻结 backbone、廉价 → **两个都验，先 A（置信高）后 B（分量大）**。过的那个 = 训练端第二 contribution，和 test-time marginalization 组成完整 train+test 方法（推 7-8/10）。
- A、B 甚至可叠加（B 对齐降 spread + A 加权残差）。
- 当前 GPU 被 Hard-Lattice(3090)/Hard-ordinary(4090) 占 → 先实现两个 kill-switch 代码，GPU 一空即 cheap probe（不直接堆训练）。

exec
/bin/zsh -lc "sed -n '1,260p' experiments/exp359_lm_reid/train_time_pipeline.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# LM-ReID 训练端创新 pipeline（2026-06-26，用户指令"训练端不许停"后）

## ★★★LSRC 死 → 训练端三大类全死（2026-06-26 早）

LSRC（backbone set-loss）full fine-tune eval（4090 lam0.5）vs no-LM-loss（single 77.44 / lattice 79.90 / HR-sanity 88.92）：

| 指标 | no-LM-loss | LSRC lam0.5 | Δ |
|---|---|---|---|
| HR sanity | 88.92 | **85.84** | **−3.08** |
| h16 single | 77.44 | 75.70 | −1.74 |
| h16 lattice(MaxSim) | 79.90 | **77.98** | **−1.92** |
| h24 single | — | 82.31 | |
| h24 lattice(MaxSim) | — | 83.27 | |

**LSRC 全面拉低**：训练 acc 1.000（过拟合训练集）但测试全掉 = backbone set-loss 损害判别力。marginalization 在受损 backbone 上仍 +2.288（机制工作但起点被拉低，证 lattice marg 本身没问题，是 backbone 被训坏）。3090 lam1.0 必死（更大权重）。

**★非对称 LSRC 已实测（2026-06-26 用户二次质疑"LM-ReID 没训练端创新吗"后, --lsrc_asym query-set × gallery-single, lam0.5, 40ep）= 死 −0.33**（h16 lattice MaxSim 79.57 vs no-LM 79.90, 没过 +0.3 线; HR sanity 87.63 −1.29; single 77.29 −0.15; h24 lattice 84.73 d+0.893）。**但比对称温和太多**（−0.33 vs 对称 −1.92）: 我之前"对称死→非对称必死"**结论对但严重低估温和度**——外推只给死/活, 实测才知非对称几乎不伤 single(−0.15)、lattice 仅 −0.33（差点到 0 但仍 net negative）。机理证实 backbone set-loss 损害判别力对称/非对称都有, 非对称(不给 gallery oracle)温和但回不到正。**→ 非对称 LSRC 从"推理判死"升级"实测死"，至此 LM-ReID 训练端 100% 实测穷尽零外推**（frozen LS-MRT/LATS + backbone-loss consistency/LSRC对称/LSRC非对称 + robust-ERM Hard-Lattice + input BLC证伪）。

**→ 训练端三大类全死**：
1. **frozen-feature 重投影/重加权**（LS-MRT +0.028 / LPA +0.075）— no-LM-loss 特征对边缘化已最优、无 headroom。
2. **backbone 改 loss**（LSRC −1.9 / consistency −1.73 / L_marg 有害）— 改 backbone 损害判别力。
3. **robust ERM**（Hard-Lattice 76.9<77.44）— worst-case 也救不了。

**强结论**：no-LM-loss backbone 已是 LR-ReID 好特征，**test-time decision marginalization 是唯一有效杠杆**；训练端干预要么无用（frozen）要么有害（改 backbone）。备选 BLC（input canonicalize bbox crop, design 已写）market 只裁好框受限 6.5、未验。启 2 codex（train3_{fourthclass 找第四类机制, paperstrategy 评估反例论文策略}）。

## ★★★★第四类也死/证伪（2026-06-26，codex round3）

**fourthclass codex**：第四类候选 LATS（frozen-token sidecar 6.5）/ BLC（input）/ LVAS（data sampler 5）/ residual-diversity（4.5）。**paperstrategy codex**：明确"别再碰 backbone training loss"，论文 test-time + 反例 **5.5→6.5（补跨数据集/强TTA/MLR）→7.5（仅当 BLC 过线）**；核心论点 *"Learning to be invariant is the wrong objective; marginalizing decisions over plausible observations is the right one"*。

- **BLC 被现有 LM-S4 数据证伪**（不用跑）：BLC = canonicalize bbox（固定 canonical）+ marginalize 残差。但 bbox 是主因子（marg bbox +2.84），canonicalize 它 = 放弃这大收益，只剩 phase/zoom marg ≈ **+1.7 < marginalize-all +2.557**；且 market 测试用标注框（已 canonical），BLC 让 K=9 marg 退化成 single 77.44 < 79.90。**canonicalize 主因子 < marginalize 它 → BLC 死。**
- **LATS**：frozen-adaptation 子类（LS-MRT final-feat 已证 +0.028 无 headroom），token map cache ~18GB **不 cheap**，codex 期望 +0.0~0.2。**★实测（2026-06-26 用户质疑"LM-ReID 没训练端创新吗"后, cvpb_lats_probe.py cheaper stripe-pooled 版, frozen backbone + 6-stripe token sidecar）= 死 −5.147**（uniform-global marg 80.23 → stripe-LATS 75.09；K-cos rise +0.023 变体趋同）。机理: stripe sidecar set-retrieval 训练把 K 变体拉相似破坏 marginalization 多样性, **和 LSRC 一个死法**（训练端塑造/对齐变体即损害 test-time marginalization 多样性）。**→ LATS 从"外推判死"升级"实测死"**：frozen-adaptation 类两个实测点（LS-MRT final-feat +0.028 / LATS stripe-token −5.147），连 pool-前 token 空间信息都试过。用户质疑价值=穷尽从外推→实测, "Why Training-Time Invariance Fails"论点更硬。
- **paperstrategy 论文可投性**：test-time + 反例上限 **6.5**（BLC 证伪 → 够不到 7.5）。论文转 "Why Training-Time Invariance Fails"（三类反例写成"自然但错误的解法"）+ 补实验。

**train4_final codex 判决（8.5/10）：当前设定下没有值得追的 cheap 训练端机制，别硬凑。** 4 类全封：① frozen/head/sidecar 无 headroom（含 LATS）；② backbone-loss 伤判别性（consistency/LSRC）；③ robust-ERM 没赢（Hard-Lattice）；④ BLC 逻辑封住（bbox 收益来自 marginalize 非 canonicalize，market 框已 canonical）。codex 强论点：*"sampling-lattice uncertainty 不适合训练端消除/内化；强 no-LM backbone 已近最佳判别表征；有效杠杆是 test-time decision marginalization，不是 invariance/frozen-adaptation/robust-ERM/canonicalization"*。

**→ 训练端定论穷尽**（8 机制实测/证伪 + 4 codex 收敛，8.5/10）。这不是放弃，是有证据的结论，**本身是论文强论点**。转向：test-time 论文（paperstrategy 6.5）+ 训练端反例补强（"Why Training-Time Invariance Fails" controlled-alternatives 节）。GPU 空跑 K-sweep（compute-accuracy 素材）。

---

## ★更新（2026-06-26 凌晨）：frozen-feature 类全死 → 转 LSRC（改 backbone）

- **LS-MRT 死**：冻 backbone probe，smoke 时 P(D×D linear ~1M params)过拟合暴跌 −8.694；修复（identity-reg + 降 lr + 全量 116k samples）后 **+0.028 clean FAIL**（K-cosine 不升，但 P 不帮忙）。
- **LPA 死**：query-side 加权 +0.075，预测最佳变体 acc 12.4%≈chance。
- **关键发现**：LS-MRT(+0.028)+LPA(+0.075) 两个冻结特征 probe 都 ~0 → **no-LM-loss 特征对 test-time 边缘化已近最优，重投影/重加权救不了**（oracle +4.338 是 gallery 真 ID 上界，frozen-feature 够不着）。**LCRS/LRFD/DeepSets 也是 frozen-feature 重投影，大概率同样死**。
- → 训练端价值**必须改特征本身（backbone）或改输入**。启 2 新 codex（train2_{backbone,input}）：

| 机制 | 信心 | 核心 | 依赖 | 状态 |
|---|---|---|---|---|
| **LSRC** | 7.5/10 | full fine-tune，`L_id + lam_lsrc*(bag-to-bag set-supcon logsumexp + **neg-tail 压负样本假高 lattice**)`。打 marginalization 数学瓶颈（更少假高负例+更多可复用正证据），**解释了 frozen 为何没空间**（lattice union 已固定） | 不依赖原图 ✓ | **进行中** 4090 lam0.5 + 3090 lam1.0，acc 0.98/0.97 健康，判据 lattice mAP > 79.90 +0.3 且 single 不掉 |
| BLC/LC-STN++ | 8/10→6.5 | bbox crop refiner 改输入（最贴 bbox 主导发现） | market 只裁好框→6.5 | 备选 |

---
## （旧）5 候选 frozen-feature pipeline（LS-MRT 已死，其余大概率同死）

LPA(A 加权头)定死（oracle headroom +4.338 不可达，最佳变体 query+gallery 共定单看 query 预测不出 acc≈chance）。Hard-Lattice ERM eval 中。按用户"这三个不行就找更多"启 4 新 codex（litreview2/train_more_*.md），收敛到 **5 个候选**，全避开已死的（invariance-collapse / query-side 预测最佳变体 / L_marg 分类头边缘化）。

| 机制 | 信心 | 核心 | kill-switch | 状态 |
|---|---|---|---|---|
| **LS-MRT**（set-wise 检索） | 7/10 | 把 test-time 边缘化写进**训练检索损失**：`S=logmeanexp_k sim(z_q,k, z_g)`，supervised contrastive over gallery。**在检索决策层边缘化非分类头**——直接修 L_marg 失败因（L_marg 在 train-ID classifier posterior 求均值→塌缩；LS-MRT 在 q-g 相似度证据边缘化，denominator 有真负 gallery） | 冻 backbone + cached K=9 features 训小 P(linear/BNNeck+τ)，**最廉价**；活=h12/16 ≥+0.3 且 K-cosine 不升 | **先跑** |
| LCRS（互补 residual） | 7/10 | `z_k = norm(P_shared(g_k) + α·P_k(g_k))`，shared identity core + lattice-specific residual subspace + decorr(只在分类正确后)。"准确后分工"非硬推开。test K=9 边缘化拿更富 union | 冻 backbone 训 P_shared/P_k/cls；活=K-error correlation 降+K=9 gain ≥+0.5，individual variant mAP 不掉>0.8 | 排队 |
| LRFD（disentangle） | 7/10 | `z_l=P_id, r_l=P_lat`(lattice nuisance sink, 必须能预测 lattice), per-variant CE+triplet, 推理丢 r_l 用 z_l 边缘化 | 冻 backbone 训 heads；活=with-lattice-code >+0.2 over without + r_l lattice-pred acc>60% | 排队 |
| LC-STN（对齐 canonicalize） | 7/10 | tiny localization net 预测 bbox offset(dx,dy)+grid_sample 重采样到 canonical, L_geo=SmoothL1 监督几何非身份, 残差留 marginalization | 冻 backbone 训 canonicalizer；活=offset MAE<0.35 LR px + K=1 +0.8 OR K=9 +0.3 | 排队 |
| DeepSets Marginalizer（LPA 修正） | 6.5/10 | 不预测 query 最佳变体(LPA 死因)，改 pairwise: 每个(q,g)的 K 个 cos 经 DeepSets φ/ρ 合成校准分, α 初 0 限幅, embedding 冻 | cached features 小 scorer；活=+0.4 over uniform, 且 ordinary-TTA scorer 追不平(否则只是 generic learned TTA) | 排队 |

**不做**（codex 共判）：TTT/TENT(无标签自我确认 3/10), DRO/Hard-Lattice(≈将死 4/10), PFE-Gaussian(query-side uncertainty 不够 4/10)。

**执行顺序**：LS-MRT 冻 probe 先（最廉价+最强故事）→ 活则全量；死则 LCRS→LRFD→LC-STN→DeepSets 逐个 cheap probe。**任何一个过线即训练端第二 contribution，凑成 train+test 完整方法（→7-8/10）。全死也不停——再启 codex 找更多。** novelty 共识：卖点不是各机制本身(decorr/disentangle/STN/DeepSets 都老)，是"**LR 采样格点当可枚举隐变量 + 训练端 X + 测试端 decision marginalization**"这个 problem reformulation + train/test 对齐。

exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp367_single_support/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp367 Single-Support CVaR Episodic Loss（训练侧创新，2026-06-28）

## 动机

用户 goal：找训练侧创新发 CCF-B，不收手，不轻易说穷尽，审查调研交 codex，严谨，文档记好。codex 训练侧深度调研 #1（最务实）：训练时每 ID 只用单图 support 定义身份，对 worst-support 选择做 CVaR 优化。回应 exp109 根问题（single-image support incomplete）。**纯训练侧**（episodic loss，输出常规 descriptor），严格非 test-time/检索侧/范式重定义。

codex 已避所有死区（visibility/masked/CLIP-align/synthetic/topology/DG-foundation/noisy-label/long-tail），2024-26 novelty 空白：few-shot/DG 有先例，但**标准 Market/MSMT/Occluded 监督训练里"单图 support 是否足够定义身份"做成主训练目标，2024-2026 没看到直接占位**。

## 核心假设

ReID 训练用 multi-shot gallery（每 ID 多图），但模型学到的身份边界可能依赖"见过该 ID 多个 view"。部署常 single-shot（单图 support 定义新身份）。训练时**强制单图 support + CVaR worst-support 优化**，逼模型学"从任意单图恢复完整身份边界"的鲁棒表征，而非依赖 multi-view 平均。

## cheap kill-switch（零训练，cvpb_single_support_probe.py）

复用 Market 特征 cache（frozen SOLIDER exp260b）。每 gallery ID 只留 1 图：
- full-gallery：上界
- best-support：每 ID 选最好单图（同 ID query 平均 sim 最高，oracle 上界）
- random-support：每 ID 随机 1 图
- worst-support：每 ID 选最差单图（CVaR worst-case 目标针对的）

**GO**（support 选择是真训练瓶颈）：worst 比 full 掉 > 3 mAP 且 **best−worst gap > 3 mAP**（哪张 support 图很重要 = support 选择 matters，值得 CVaR 优化）。
**DEAD**：best≈worst（哪张 support 都一样，没 support 选择价值）或 single≈full（单图够）。

★诚实设计要点：单图 vs 多图必掉 mAP（少正样本）是 trivial，所以**关键判据是 best−worst gap**（同样单图，选择重不重要），不是 single<full。codex 审 probe 验这个设计是否真有意义（用户要审查交 codex）。

## 审查（codex，用户要求）

codex 审 probe（codex_review.md）：kill-switch 设计是否有意义、best/worst per-ID 选择逻辑、#false-in-topk 控制。

## 预期

- GO → 设计 Single-Support CVaR episodic loss 训练（每 ID 单图 support + worst-case 风险优化），训练侧第一 contribution，full fine-tune 前 codex 三审 diff。
- DEAD → support 选择无训练价值，转 Equivariant Routing（codex 训练侧 #2，routing 等变非 embedding 一致）。

## 训练设计（codex 调研 63517，probe GO 后）

★**novelty 真空白（codex 确认）**：2024-26 标准监督 person ReID 没有"episodic single-support training + CVaR worst-support tail optimization"直接先例（检索 single-support/worst-support/CVaR-ReID/support-selection 都没命中）。邻近但不同：CFReID(continual few-shot)/DG-episodic(domain-invariant)/ProtoNet(novel-class 优化 prototype 平均非 tail)/batch-hard(hard pair mining 非 support tail)。**claim 写窄**：不发明 episodic/CVaR，是"标准 ReID 优化单图 support 定义身份的 tail risk"。

★**训练设计（two-level CVaR，加项不替换）**：
- episode N ID × K 图，每 ID 1 support + K-1 query。
- `risk(y,s)` = 该 support 对同 ID 多 query 的 CE 失败。
- `L_cvar_y = CVaR_α(support risks)`，`L_ss_cvar = mean_y`。α=0.7/0.8。
- `L = L_id_ce + L_triplet + λ·L_ss_cvar`（λ 0.1→0.3 warmup）。
- support/query 都梯度 + 保 CE+Triplet 防 collapse。两级聚合(support risk→CVaR)非 CVaR over all pairs(避 batch-hard)。

★**避坑（避六点定律）**：不写 support completion/feature alignment/prototype compression/query-dependent selector。训练用 label 算 worst support 可以，测试不选 best/不用 query label，输出常规 descriptor。

★**cheap 验证路径**：① frozen head smoke(10-20ep, worst/random +0.8~1.0, 失败不判死) ② last-stage(20-40ep, worst+2/random+1/gap 缩≥2/full 不降>0.5, 对照普通 CE+Triplet continued FT) ③ full FT。

★**风险+对照（防退化 hard-mining）**：报 batch-hard/pair-CVaR/random episodic CE 三对照，只 support-level CVaR 独立赢才站得住。报 missing-positive/cross-cam 覆盖/false10 random std。

★CCF-B 6.5/10（last-stage/full FT 抬 worst/random + 赢 hard-mining → 7.5；只改 diagnostic 但 full-gallery 不涨 → 4 附录）。

## 状态

probe v2 GO（best-worst 12.27 不被 #false 解释，codex 两轮审）。codex 训练设计 GO（novelty 空白 6.5/10）。下一步：写 frozen head smoke 训练（cheap 第一步，复用 Market cache + projection head + episodic CVaR loss）+ codex 三审 diff（full FT 铁律 + 审查交 codex）。

codex
我现在做先例排查，不会只看标题；重点查 2023-2026 的 ReID 训练目标、listwise/ranking、camera/causal、part-mix/compositional、equivariant/multi-hypothesis、自监督预训练这些邻近区，排除“其实早就有人写过”的方向。
web search: 
web search: arXiv 2024 2025 person re-identification training loss ranking listwise differentiable AP person ReID
web search: 
web search: site:arxiv.org person re-identification differentiable sorting AP loss listwise ranking training
web search: 
web search: "person re-identification" "Average Precision" loss differentiable
web search: 
web search: arXiv "Differentiable Average Precision" metric learning loss
web search: 
web search: site:arxiv.org "person re-identification" "uncertainty" "training" 2024
web search: 
web search: "person re-identification" "equivariant" representation
web search: 
web search: "person re-identification" "camera" "adversarial" "training" 2024 2025
web search: 
web search: "person re-identification" "masked image modeling" "self-supervised" 2024
web search: 
web search: "person re-identification" "CutMix" "training"
web search: 
web search: "CutMix" "person re-identification"
web search: 
web search: "person re-identification" "part" "swap" augmentation
web search: 
web search: "person re-identification" "negative samples" "augmentation"
web search: 
web search: 2025 survey person re-identification training paradigms foundation models supervised self supervised language aligned
codex
先例搜索显示几个坑很密：camera debias/meta-learning、part mix、MIM/PersonViT、uncertainty/noisy-label、ranking/listwise 早已有论文。能剩下的不是“完全没人碰过的关键词”，而是要把 claim 缩到一个未被直接写成 ReID 训练目标的机制缝隙。
exec
/bin/zsh -lc "sed -n '1,260p' experiments/exp361_psc_jepa/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp361 PSC-JEPA — monitor

## Stage-A continued-pretrain（2026-06-26，4090，50 ep）
- 健康收敛：L 3.0→0.13，**tok_std 0.49→0.998（C1 防坍缩成功，全程不坍缩）**，cos_drop 0.95，var hinge 满足。
- ckpt pscjepa_10/20/30/40/50.pth（205 keys，backbone. prefix，H1 修复生效）。
- smoke 抓到 SOLIDER swin `.train()/.eval()` 不返回 self 的运行时 bug（审查覆盖不到），已修。

## Stage-A fine-tune kill-switch（2026-06-26，4090 PSC-JEPA vs 3090 plain，Occ-Duke 120ep，同 config 只差 PRETRAIN_PATH）

**★负结果（同 epoch 对比）**：
| epoch | PSC-JEPA mAP | plain mAP | 差 |
|---|---|---|---|
| 10 | 15.9% | 33.1% | **−17.2** |
| 20 | 27.8% | 42.8% | **−15.0** |
| 70 | 39.9%（平台）| — | — |

PSC-JEPA epoch 70 才 39.9%（平台），plain epoch 20 就 42.8%（趋势更高）。**continued-pretrain 让 fine-tune 更差，不是更好。**

**诊断：Stage-A 裸 continued-pretrain 破坏 SOLIDER backbone 判别性（catastrophic forgetting）** —— Stage-A **无 L_solider_anchor 防遗忘**（design 里 Stage-B 才加）。partial-view JEPA 把 backbone 从 ReID 判别表征拉偏，fine-tune 50+ ep 拉不回。

**★final 确认（2026-06-26）**：PSC-JEPA **120ep final mAP = 41.0%**（已 ENDED，平台）；plain @ epoch 60 已 **52.9%**（还在涨，120ep ~55%）。差 **−12 且会更大**。

**结论**：kill-switch Stage-A **FAIL（严重）**（PSC-JEPA 41.0 << plain ~55，差 ~−14，远不是 ≥+0.7）。**catastrophic forgetting 坐实** = 裸 continued-pretrain（无防遗忘）严重破坏 SOLIDER 判别性，fine-tune 拉不回。但**诊断清楚 = forgetting**，design 预期内（Stage-A = 骨架/防坍缩验证，不主张 novelty；Stage-B 才防遗忘 + support bank）。不是死路，是诊断明确的迭代。

## Stage-B 修复方向（防遗忘 + 真 novelty）
1. **L_solider_anchor（防遗忘，关键）**：frozen SOLIDER backbone（swin_tiny.pth 不更新）= anchor teacher；student 可见区 part token 蒸 frozen SOLIDER token（cos，gvis 掩码 visible）。锚住可见区判别性不遗忘，JEPA 只在 dropped 区学 completion。
2. **pseudo same-ID support bank（B 类 novelty）**：T_bank 同 ID NN 的 body-part prototype，dropped 区预测 support。
3. 重训 continued-pretrain（3 backbone：student + EMA teacher + frozen SOLIDER anchor）→ fine-tune 再验 kill-switch（≥+0.7 vs plain）。

## Stage-B 重训 + fine-tune（防遗忘 v2，2026-06-27）

- **continued-pretrain 50ep 健康**：防遗忘 sol_p 0.6→0.11 / sol_g 0.05→0.03 活跃，tok_std 不坍缩，L 收敛。codex 三审（R1 抓"只锚 5 part token 覆盖窄"→补 global GAP distillation→R2/R3 approve）。
- **★fine-tune early signal（epoch 10）**：Stage-B **23.0%** vs Stage-A 15.9%（防遗忘 **+7.1**，机制部分生效）vs plain 33.1%（仍 **−10.1**，没完全修）。
- **诚实判读**：防遗忘 anchor（part + global GAP）**减轻 forgetting 但不充分**——continued-pretrain 仍损害判别性。完全符合 codex 守的诚实"修复尝试成立非 forgetting 已解决"。
- 可能原因：anchor 权重不够 / part+global GAP 还不够（codex 提 dense/stage-wise distill）/ partial-view JEPA 与 ReID 判别性本质张力。
- 待：epoch 50 趋势（追平 plain or 平台卡 < plain）→ final 判 kill-switch。趋势平行 plain（差固定 −10）=防遗忘不够；收敛追平=够。

## ★PSC-JEPA continued-pretrain 主范式判死（2026-06-27，codex 诊断 8/10）

**kill-switch FAIL**：Stage-B 防遗忘 fine-tune 趋势 23→36.7→44→46.2→46.7（epoch 10-50 平台）<< plain 58.5（−11.5 平台差）。防遗忘缩 early gap（−10→−2）但**没改平台**。

**codex 诊断（非代码问题，范式本质冲突）**：
1. partial-view JEPA 补"**不可观测身份细节**"（被遮 part 真实 identity 单图不存在）→ 学生学上下文均值/人体先验 → 特征推向"可预测平滑不变"，而 ReID 要"细粒度可区分"（纹理/颜色/局部差异）。**目标方向冲突**。
2. continued-pretrain 覆盖 SOLIDER 已调好的 appearance/semantic 平衡，anchor 拉不回全部 dense feature/attention/层间几何/margin。
3. 防遗忘越强 JEPA 越 no-op（终点≈原 SOLIDER 不超 plain）。
- 外部先例支持：TransReID-SSL(Catastrophic Forgetting Score) / PersonMAE/HAP(需完整预训练体系) / continued-pretrain 普遍报 forgetting。

**结论：不再 pretrain backbone**。尸检 4-run mini grid（救回 2/10，跳过）。
**Pivot（codex）**：① support bank 改 fine-tune 侧 **detached auxiliary**（5.5，plain SOLIDER fine-tune + support 只监督轻量 completion head/part branch/pair scorer，global backbone hard guard ≥plain−0.3，避 pretrain harm 保 exp109 信号）② **生成数据引擎**（6.5，扩训练分布让监督 fine-tune 自学遮挡鲁棒，不蒸不可观测 support，更长期范式空间）。

exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp361_psc_jepa/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp361 — PSC-JEPA：Pose-defined Support-Complete JEPA pretraining（范式转向核心 build）

## 选定经过（2026-06-26）

范式转向（用户授权换量级）→ 3 路 gap → Intruder(C-#2) 选中 → **Intruder DEAD**（exp360 阶段0.5 因果证伪：donor-ID 可读但压它不救排序，H2 #false 控制 + rank-r projection 双证）→ codex 裁决转 **B PSC-JEPA(6.5)**。这是真正的 continued-pretrain 换量级，不是 frozen probe。

## 核心 idea

给一张**不完整人体图**（pose-defined 部分遮挡），让 backbone 在 **latent body-part token 空间预测"完整身份 support"**——target 来自 EMA full-view teacher + 高置信 pseudo same-ID support bank。**不补像素、不补语义比例，而是补"身份证据"**。JEPA 路线：预测 embedding target 而非像素，避开 ReID 中重建背景/遮挡物的污染。

贴项目 exp109 oracle 证据：support-complete teacher 有真实 headroom（oracle 61.88/73.26 → 70.40/81.36），PSC-JEPA 把这个 headroom 尝试**蒸进预训练**（不像 exp109/FGEU 是测试端不可得的 oracle）。

## Novelty 边界（codex 标，诚实）

**已被占**：SOLIDER(human SSL+语义可控) / PersonMAE(occlusion masking + pixel+semantic 重建, 报 Occ-Duke) / HAP(part-guided MIM) / PersonViT 2024(MIM+contrastive) / SAIP 2025(cross-scale)。
**没吃掉的窄缝**：① 补身份 support 非像素 ② support bank/pseudo cross-view teacher 非 single-image MAE ③ pretraining 学"缺部位时如何形成可检索身份证据" ④ JEPA latent prediction。
→ **B 类空间只在"latent support completion 讲清 + 赢过 plain continued-pretrain/random-mask/PersonMAE-lite"才成立**。

## Pipeline（codex 设计）

1. **init**：SOLIDER/Swin-Tiny continued-pretrain（保持 fine-tune 兼容）；DINOv2 可选 frozen dense teacher（稳的 latent target，不全量 fine-tune）。
2. **data**：**train split only**（防 query/gallery 泄漏）Market+MSMT17+Occluded-Duke+Occluded-ReID+Occluded-PoseTrack；预缓存 pose/keypoint visibility/body-group mask + pose-defined complementary masks。
3. **pretext**：输入 partial view（保留一部分 body support）；teacher target = `T_full`(EMA full-view body-part latent) + `T_bank`(pseudo same-ID/NN support bank 的 body-part prototype)；student 输出 visible/missing/union tokens。
4. **loss**：`L_part_jepa`(missing token 预测 teacher/support, cos/L2) + `L_union`(union token ≈ full-view identity) + `L_gram`(part-token 关系矩阵对齐) + `L_visible_anchor`(可见 part 不被改坏) + `L_solider/dino_anchor`(可见区蒸原 backbone, 防遗忘) + 可选 `L_cluster_contrast`(高置信 pseudo cluster)。
5. **fine-tune**：continued-pretrain 完 → 标准 ReID fine-tune → 评估。

## 4-slot 排布（codex；对照是 novelty 生命线）

| slot | 任务 | 作用 |
|---|---|---|
| 4090 | **PSC-JEPA 主跑**（SOLIDER/Swin-Tiny multi-dataset 50-100ep）| 主结果 |
| 3090 | 去 support bank，只 same-image full teacher | 对照"是否只是 OA-SD/PCVT 换名" |
| 5060Ti-1 | random mask / PersonMAE-lite latent baseline | 对照"是否只是普通 MIM" |
| 5060Ti-2 | support bank 质量诊断 / DINOv2 frozen teacher variant / Occ-Duke smoke | 诊断 |

## Build 阶段（deep work，v0 5-8 天）

- **阶段 A 骨架**：continued-pretrain loop（SOLIDER init + EMA teacher + partial-view pretext + `L_part_jepa`+`L_union`+`L_visible_anchor`，先不加 support bank）→ smoke 跑通 + 不崩。
- **阶段 B support bank**：加 pseudo same-ID support bank（`T_bank`）+ 质量诊断（pseudo same-ID top-k 精度）+ `L_solider_anchor` 防遗忘。
- **阶段 C 对照 + 全量**：4-slot 并行主跑 + 3 对照 → fine-tune Occ-Duke/Market/MSMT → 判 novelty（主跑赢过对照 + Occ-Duke ≥+0.7）。

## Kill-switch（codex）

- continued-pretrain 后 Occluded-Duke fine-tune **≥+0.7 mAP** 或 heavy-occ 子集明显正向；
- **plain continued-pretrain / random mask / same-image teacher 不能同涨**（否则 novelty 塌，只是"更好的 MIM/continued-pretrain"）；
- pseudo support bank top-k 精度过线（否则噪声主导）。

## 风险

- pseudo support bank 噪声（第一风险）；novelty 拥挤（PersonMAE/HAP/PersonViT，靠对照证差异）；continued-pretrain 不涨（45-55% 成功率, codex）。

## 训练前纪律

continued-pretrain = backbone 训练 → **启动前 codex 审 diff**（[[pre-experiment-review-discipline]]，用户分工：审查交 codex）。

关联：`experiments/paradigm_shift/`、`experiments/exp360_intruder/`（DEAD 因果记录）、memory [[exp109-headroom-is-a-wall]]。

exec
/bin/zsh -lc "sed -n '1,240p' experiments/exp360_intruder/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp360 Intruder — monitor

## 阶段 0 地基机制验证（frozen probe，2026-06-26）

**脚本**: `experiments/cargo_cvpb/cvpb_intruder_probe.py`（market noLMloss frozen baseline + 合成 target+donor，occ_frac 0.45 下半身遮挡）
**log**: 4090 `/tmp/exp360_intruder_full.log`（50 donor×20 + 700 query vs full gallery）

| 判据 | 结果 | 通过 |
|---|---|---|
| H1 donor-ID 泄漏可测 | probe acc 73% vs chance 2% = **36.5x** | ✅ PASS（泄漏巨大确凿）|
| person >> rand control | leak person 0.15 >> rand −0.01 | ✅ PASS |
| **H2 leak ↔ AP drop（控 #false）** | raw spearman **+0.120** → **partial\|#false = −0.028（≈0）** | ❌ **FAIL** |

AP drop 大（clean 0.835 → mix 0.409，−0.43，遮挡确实大幅损害检索）。

### 诚实判读
- donor 泄漏**存在且巨大**（H1 36.5x，能从被遮挡 target 的 embedding 73% 认出遮挡者），但**泄漏量不独立于 #false-in-topk 预测检索损害**（H2 控 #false 后 ≈0）。
- 即：leak 和检索损害都是遮挡的"症状"，但 leak 不是损害的**独立原因**（损害主由 top-k 混入错 ID 驱动）。
- **memory 铁律再次发挥**（#false 控制把 +0.120 打回 −0.028），避免 over-claim "泄漏导致损害"。Hubness/evidence/d17 同款。
- 对 Intruder 的实质打击：核心假设"压 donor 泄漏 → 救检索"因果地基不稳（= codex 头号风险=退化 target ambiguity 墙）。

### 不收敛停（deep work 模式）
- H2 是 frozen per-query 相关（哪个 query 损害大），FAIL ≠ "训练压 donor-ID 无效"（相关 ≠ 干预效果，H3 训练才是终判）。
- codex 评估（`codex_h2fail_decision.md`）：H2 FAIL 不数学杀死(但杀强叙事)，建议 Stage0.5 frozen donor-null projection 因果测试（GRL 上界代理）再裁决。

## 阶段 0.5 donor-null projection 因果测试（2026-06-26，codex 7/10）

frozen rank-r 抠掉 donor 判别方向，看 mix AP 是否回升（= GRL 软压的上界代理）：

| r | donor acc | mixAP Δ | cleanAP Δ | #false |
|---|---|---|---|---|
| 5 | 0.72→0.575 | +0.018 | −0.002 | 5.89→5.71 |
| 10 | →0.500 | +0.025 | −0.005 | →5.63 |
| 20 | →0.480 | +0.028 | −0.008 | →5.57 |
| 40 | →0.480 | **+0.031** | −0.009 | →5.58 |

**裁决：Intruder DEAD（因果证伪）**：
- donor 信息**高度分布式**：r=40 抠 40 维，donor acc 只 0.72→0.48（r=20→40 饱和，抠不干净）。
- 即使抠 40 维，mixAP 只回 **+0.031**（mix 0.409→0.440 vs clean 0.835，gap 0.4 几乎没动），clean 被伤 −0.009，#false 几乎不降。
- **坐实 codex 头号风险 + memory target ambiguity 墙**：donor-ID 可读(H1 36x)但压它不救排序。
- **codex Stage0.5 完美奏效**：frozen 因果测试(零训练)裁决 Intruder，省多日 GRL 训练 + 审查。上界 projection 救 0.031，GRL 软压更不行。

**→ Intruder DEAD（有据：H2 #false 控制 + Stage0.5 r-sweep 双重因果证伪）。按 codex 裁决转 B PSC-JEPA（6.5，从 SOLIDER continued-pretrain + pose-defined pseudo-support-bank latent JEPA，真正换量级）。**

exec
/bin/zsh -lc "sed -n '1,240p' experiments/exp366_active_evidence/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp366 Active Evidence Acquisition — monitor

## cheap kill-switch（frozen SOLIDER, 零训练, 2026-06-28）

验 codex 范式 #1：query 主动获取第二证据（同 ID 不同 camera），policy（margin 选 hard query 花预算）能否接近 oracle 且 >> random。

| 数据集 | baseline | oracle-all (2nd) | policy (hard 20%) | random (20%) | frac | verdict |
|---|---|---|---|---|---|---|
| Market（exp260b 强）| 94.43 | +2.51 | **+0.31** | +0.48 | 0.12 | DEAD（policy<random）|
| Occluded-Duke（exp004_pfm⚠️）| 3.09⚠️ | +6.45 | **+1.29** | +1.31 | 0.20 | DEAD（policy≈random）|

⚠️ Occluded baseline 3.09 异常低（exp004_pfm ckpt 配 market config 的 FrozenExtractor 加载不匹配）；但 policy vs random 是相对比较、不依赖 baseline 绝对值，结论成立。

## ★VERDICT DEAD（两数据集坐实）

margin（top1-top2 检索不确定性）**不是好的"值得获取证据"预算信号**：Market policy+0.31<random+0.48；Occluded policy+1.29≈random+1.31。两数据集 policy 都 ≈/< random。

**★范式根本困难（诚实诊断）**：
1. 系统不知道哪个 query 的第二证据有用（要获取才知道=鸡生蛋）。
2. margin 小（难 query）给证据没用——第二证据也可能难（occluded query 另一张也 occluded）。
3. 任何"检索不确定性"policy（margin/entropy 同质）都救不了，因为不确定 ≠ 第二证据能救。

oracle headroom 真实（occluded +6.45 R1 大涨），但**没有 cheap policy 信号能逼近它**。主动获取证据要价值，需要预测"第二证据质量"，而那本身要先获取（鸡生蛋）。

## 决定

Active Evidence policy DEAD（cheap kill-switch 半小时验透，没浪费训练）。转 **Generative Index ReID（codex 范式 #2，6.5/10 真空白）**：gallery identity 离散 token，query 生成 code prefix，kill-switch=PQ code recall<95% 则杀。

codex
本地还补充了两个重要负例：`PSC-JEPA` 这种 continued-pretrain 会显著遗忘 ReID 判别性；`Intruder` 的 donor 泄漏虽然存在，但控制 `#false-in-topk` 后不是独立因果；`Active Evidence` 有 oracle headroom 但 policy 选不出该加证据的 query。这些会直接影响排序。
exec
/bin/zsh -lc "sed -n '1,240p' experiments/exp362_genoccl/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp362 gap-measured occlusion engine — gap 审计（negative control）

## 由来
PSC-JEPA continued-pretrain（exp361）判死后，codex pivot 决策选生成数据引擎（6/10，窄缝=gap-measured occlusion distribution engine，必须赢 PLBOA）。第一步 cheap kill-switch：先 gap 审计验证前提（train-test 遮挡分布 gap 是否存在）。

## gap 审计（2026-06-27，cheap，no training，no diffusion）

**pose visibility threshold sweep（occluded_duke train N=15618 vs query N=2210）**：

| VIS_THR | legs gap | arm gap | heavy-occ(可见组≤2) gap |
|---|---|---|---|
| 0.3 | +9.2% (q9.5/t0.4) | rarm +0.5 | +0.1% |
| 0.5 | +16.4% (q17.2/t0.8) | larm +2.0 / rarm +2.1 | +1.0% |
| 0.7 | +26.1% (q28.1/t2.0) | larm +8.9 / rarm +5.9 | +5.1% |

**gap 形态确认：各 threshold 下始终主导 lower-body（legs），arm 次之（高 thr 才显现），heavy-occ 始终少。**

## ★结论：生成引擎窄缝被 PLBOA 占 → 转 LM-ReID（codex 2.5-3/10）

- gap 几乎全在 legs（lower-body），**正是 PLBOA（Pose-guided Lower-Body Occlusion，3-seed +1.37 mAP）已经在补的**。
- 生成引擎要成立 = 在 PLBOA 已覆盖的 lower-body gap 上**净增益**（否则只是"更贵的 PLBOA"）。窄缝太窄。
- caveat：pose visibility ≠ 真遮挡（codex），但 occluded_duke **无官方 mask**（真 mask audit 要跑 human parsing，中等成本）；且最强相对信号始终是 legs（非全身 heavy occlusion），caveat 不足以救生成线。
- **codex 综合判：转 LM-ReID**（exp359，诚实 6.5 B 类候选，表链全：强 TTA 对照/聚合消融/因子消融/K-sweep/backbone 泛化/σ-sweep/训练端反例）。exp362 保留为 audit/negative control。

## ★★换量级在 occluded ReID 内部的诚实困境（重大节点）

范式转向（换量级）在 occluded ReID 内部探索了多个 build，**都接近墙**：
- Intruder（exp360）DEAD：donor 可读但压它不救排序。
- PSC-JEPA continued-pretrain（exp361）DEAD：partial-view JEPA 与 ReID 判别性本质冲突（41/50 << 58.5）。
- 生成引擎（exp362）2.5-3/10：gap 被 PLBOA 占。
- support-bank aux 4/10 死：撞 SCKD 穷尽（+0.1）/FGEU 16.3%。

cheap kill-switch（Stage0.5 frozen 因果 / gap 审计纯 numpy）省了多次大投入——这是分工（先验证再大 build）的价值。**用户拍板节点**：接受"occluded ReID 内部换量级接近墙"+ 回 LM-ReID 6.5 收尾投 B 类（最务实）/ 生成引擎 human parsing 最后一搏 / 转非遮挡 reframe。

exec
/bin/zsh -lc "sed -n '1,240p' experiments/exp365_wildlife/monitor.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp365 Wildlife species-conditioned — monitor

## GiraffeZebraID cheap kill-switch（frozen MegaDescriptor-L-384, 零训练, 2026-06-28）

数据：GiraffeZebraID（lila, 非 kaggle, 9.7G）6925 图 / 2056 个体 / 2 物种（zebra_plains 6286 + giraffe_masai 639）。split q=1142 g=4869（每 id 第1张 query 其余 gallery）。本地 MPS 抽 MegaDescriptor 特征(1536维, bbox crop)。

| 测量 | 值 | 判定 |
|---|---|---|
| baseline all-species mAP | 70.95 (R1 78.02) | |
| **wrong-species in top10** | **0.001** | MegaDescriptor 已把 species 分开 |
| per-species centering oracle | **+0.15** | <+3 → Kill |
| same-species vs all-species (species 干扰) | **+0.05** | <+1 → Kill |
| false in top10 | 0.712 | 错误多但都同物种内 |

**VERDICT DEAD**：codex Kill 线命中——MegaDescriptor（动物 ReID SOTA）frozen 已 species-agnostic 强（wrong-species 0.001），per-species centering 没空间（+0.15），species 干扰不存在（+0.05）。

## ★核心发现

Wildlife ReID 真难点 = **同物种内细粒度个体区分**（false_top10=0.712 都是 same-species hard negatives，wrong-species 仅 0.001），**不是 species 干扰**。codex 的 species-conditioning（SCREA: Species-Conditioned Residual Evidence Adapter）**解决错方向**——species 已被 MegaDescriptor 分开，需要解决的是同物种内细粒度区分，那是标准 ReID 问题，无 species-conditioning 新机制。

## 局限（诚实标注）

GiraffeZebraID 只 2 物种（zebra/giraffe 差异大，species 本来好分 → wrong-species 0.001 符合预期）。近缘多物种（WildlifeReID-10k 10k 个体含近缘 species，kaggle 需 key 没有）的 species 干扰没验。**但即使近缘物种 species 干扰更大，真难点（同物种内细粒度）species-conditioning 也解决不了**——core argument 不依赖物种数。

## 决定

species-conditioning 方向偏死（不补 SCREA adapter rank/head/loss 小变体，codex 明确）。换量级 codex 给的方向（第一名 lattice real-data 偏 test-time + 第二名 Wildlife species-conditioning cheap probe 偏死）都偏弱。LM-ReID 6.5（现有最强 B 类，exp359 文档全）是兜底。

下一步：① 可求用户 kaggle key 严谨验 WildlifeReID-10k 近缘多物种（确认 species-conditioning 死，但 core argument 已指向同物种内细粒度）；② 或转新方向（codex 再调研换量级）。cheap kill-switch 价值：零训练几分钟（GiraffeZebraID lila）证 species-conditioning 偏死，没下 kaggle 全量 24.7G 才发现。

## local-verifier cheap probe（codex 建议最后一搏, LoFTR rerank, 2026-06-28）

针对真难点（同物种内细粒度 hard neg），验 LoFTR local matching 能否纠正 MegaDescriptor 的 same-species false。180 query（top-k 内有正样本的）采样：

| | false_top10 | R1(in-topk) |
|---|---|---|
| baseline (MegaDescriptor) | 0.677 | 90.56 |
| LoFTR rerank | 0.677 | **97.22** |
| Δ | +0.000 | **+6.67** |

**★脚本 verdict bug 诚实纠正**：脚本 GO 条件用了 false_top10<0.60，但 **false_top10 对 rerank 无意义**（rerank 只重排 top-k 内顺序，集合不变 → false 比例必然 +0.000），导致误判 DEAD。**真信号 = ΔR1(in-topk)=+6.67**（LoFTR 把 top-k 内正样本提到 top1，远超 codex Top1 +1~2 标准）→ **local verifier 实际有效**：LoFTR local matching 在 same-species hard neg 上有 MegaDescriptor 没用好的判别信息。

**但诚实面对**（为何仍不主推）：
1. upper-bound（LoFTR expensive 双图 matching），训练端要蒸馏成 pattern-token 单次前向。
2. **撞 WildFusion**（arxiv 2408.12934：MegaDescriptor + LoFTR/LightGlue 融合，17 数据集 84.0% +8.5pp）—— local-feature 纠正 same-species 这条 codex 明确"先例太近，主创新风险高"。
3. codex 主线诚实判：LM-ReID 6.5 现实最强 B 类，收尾投稿；Wildlife 即使工程增益也撞 WildFusion。

## ★最终决定（2026-06-28）

收 LM-ReID 6.5（codex 判现实最强 B 类）。Wildlife species-conditioning 死 + 真难点（同物种内细粒度）local-verifier upper-bound 有（ΔR1+6.67）但撞 WildFusion → **记 future work，不主推**。换量级三方向（AG/DG/Wildlife）cheap probe 全证伪/偏弱，LM-ReID 6.5 兜底。教训：local-verifier 脚本 false_top10 指标对 rerank 无意义（栽了指标设计坑），ΔR1 才是真信号 —— 差点被错误 verdict 埋没真信号。

exec
/bin/zsh -lc "sed -n '1,220p' experiments/exp366_active_evidence/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# exp366 Active Evidence Acquisition ReID（范式级方向 #1，2026-06-28）

## 动机

用户指令"放下 LM-ReID，找新 ReID 范式级创新"。codex 范式级调研 #1（7/10，最值得）：传统 ReID 给一张 query 就必须排序；**范式重定义=系统可花 1-3 次预算主动获取下一条视觉证据**（请求另一帧/另一 camera 视角/操作员二值 VQ）。先例 LLaVA-ReID/ChatReID/Inter-ReID 是**文本对话补全**，主动获取**视觉证据**（camera-view evidence acquisition）是空白；旧 human-in-loop 偏人工标注反馈，不是主动传感/证据预算。避所有探死方向（occluded/AG/DG/gallery/open-set/Wildlife/VI-ReID/lattice/SMPL/FM/test-time）。

## 核心假设

ReID 真实场景=多相机网络：query 在 camera A，系统可主动调 camera B 获取同人证据（预算受限）。**难 query 值得花预算获取第二证据，简单 query 不值；policy（预算分配给哪些 query）是真问题**。

## cheap kill-switch（零训练，cvpb_active_evidence_probe.py，frozen SOLIDER）

- baseline：single query mAP
- oracle-all：每 query + 同 ID 不同 camera 第二证据（multi-query mean）→ upper-bound
- **★policy**：只对 hard query（top1-top2 margin 小=不确定）花预算 20% 获取第二证据
- random：随机 20%（同 has_second 池公平对照）

**判定 GO**：policy gain / oracle-all gain ≥ 0.5 且 policy−random > 0.3 → 主动获取证据 policy 有真价值。
**DEAD**：policy ≈ random → trivial multi-query 无 policy 价值。

★**诚实设计**：避 codex 的 trivial oracle（multi-query 必涨 = upper-bound 不是创新），真验 policy（预算分配 vs random）。控 margin（top1-top2 = #false-in-topk 的代理）。自查抓到 2 个 bug（margins 长度 != len(qf) 退化 policy；policy hard 应只在 has_second 池选）已 fix。

## 预期

- GO → 设计轻量训练端 active-acquisition policy（学"选哪个 query/候选获取证据"），范式级第二 contribution。
- DEAD → 主动获取证据无 policy 价值（等 trivial multi-query），转 Generative Index（codex #2，6.5/10 真空白）。

## 状态

probe 跑中（3090，b984dv1y8，frozen SOLIDER exp260b 抽 Market query/gallery 特征 + camera split + oracle/policy/random）。

web search: 
web search: PersonViT person re-identification masked image modeling contrastive 2024 arxiv
web search: 
web search: "camera-aware" "person re-identification" "arXiv"
web search: 
web search: "person re-identification" "causal" "camera" arxiv
web search: 
web search: "person re-identification" "noisy labels" "CVPR" 2024
web search: 
web search: "noisy labels" "person re-identification"
web search: 
web search: "Smooth-AP" deep metric learning arxiv
web search: 
web search: "generative retrieval" "person re-identification"
web search: 
web search: "Differentiable Search Index" generative retrieval arxiv
web search: 
web search: "person re-identification" "hashing" "codebook" "product quantization"
web search: 
web search: "Deep hashing" "person re-identification"
web search: 
web search: "PartMix" "person re-identification"
web search: 
web search: "counterfactual" "person re-identification" "part"
web search: 
web search: "generative retrieval" "image retrieval" arxiv "identifier"
web search: 
web search: "person re-identification" "local matching" "LoFTR"
web search: 
web search: "BPBreID" "person re-identification"
web search: 
web search: "WildFusion" MegaDescriptor LoFTR wildlife re-identification arxiv
codex
结论先放前面：这些约束下，**没有 8-9 分训练侧方向**。能认真赌的最高也就 6.5-6.8；再往上基本要换数据/task。Top5 如下。

| 排名 | 方向 | 机制一句 | 为何不撞死区 | cheap kill-switch（必须控 `#false@topk`） | CCF-B分 | 先例风险 |
|---|---|---|---|---|---:|---|
| 1 | **Generative Identity-Code ReID** | 不再学一个 cosine descriptor，而是训练 query 生成目标 gallery 的 PQ/code-prefix，再用 code 候选精排。 | 不是 hubness/rank-regret scorer；不是 FM；不是 test-time marginalization；训练的是“检索索引算子”。但若你把所有 gallery/index 方向都禁掉，此项直接删。 | 冻结现有强特征，给 gallery 聚 PQ/hier code；训小 decoder 预测正样本 code。成功线：top-N code recall 明显高于 nearest-code/random-code，候选精排 mAP 正增；按 baseline `#false@10` 分桶报告 ΔAP，且固定候选数。 | **6.8** | general 侧已有 [DSI](https://arxiv.org/abs/2202.06991)、[NCI](https://arxiv.org/abs/2206.02743)；ReID 有老 hashing/[PDH](https://arxiv.org/abs/1705.02145)。风险是“生成式检索搬到 ReID”而非全新。 |
| 2 | **Counterfactual Part-Contradiction Training** | 用 pose-aligned part swap 造“局部像但整体身份矛盾”的 hard negative，训练模型学 identity 必须是多证据 conjunction。 | 不是 occlusion 补全/visibility；不是 FM；不是 gallery trick；不压 K 变体一致性。核心是“强局部证据不能单独决定身份”。 | 先 feature-level swap：torso/legs/bag 等从近邻异 ID 替换，测 donor/target false 是否上升；再训 5-10ep contradiction loss。成功线：同 `#false@10` 桶内 top false rank 下降、mAP 不掉。 | **6.5** | [PartMix](https://arxiv.org/abs/2304.01537) 已在 VI-ReID 做 part descriptor mixing；counterfactual/cloth-changing 也有先例，如 [IDNet](https://arxiv.org/abs/2403.08270)。claim 必须窄到“标准 RGB ReID 的身份矛盾负样本”。 |
| 3 | **Camera-Pose Transport Operators** | 不学 camera/pose invariant descriptor，而学低秩 transport，把 descriptor 从一个 camera-pose cell 映射到另一个 cell 后再比较。 | 避开 DG/frozen-prior；不做 invariance collapse；不碰 LM-ReID K 变体。它学的是 comparability operator，不是抹平 nuisance。 | 冻结特征，按 train ID 拟合 cam-pose pair Procrustes/ridge map；test ID 直接用。成功线：同 `#false@10` 分桶 mAP +0.5 以上，且 hubness/候选数量不变；camera-centering 作弱对照。 | **6.2** | camera-aware 风险高：[CamStyle](https://arxiv.org/abs/1711.10295) 很早；causal/camera ReID 也被综述覆盖，如 [causal video ReID survey](https://arxiv.org/abs/2505.20540)。新意只在“transport not invariant”。 |
| 4 | **Self-Local-Verifier Distillation** | 训练期用局部 correspondence/verifier 纠正 top-k hard false，再把 correction 蒸进单次 descriptor；测试不跑 verifier。 | 不用 DINO/LoFTR/MLLM 作 FM-import；不做 test-time rerank；不是 part pooling 小改，而是“pairwise 局部判别 → descriptor 蒸馏”。 | 冻结 part tokens，训轻量 pair verifier 看能否在 top-k 内把正样本提到 false 前；成功必须在同 `#false@10` 桶内成立。若只靠 rerank 有效但蒸馏无效，杀。 | **5.9** | 人体 part/local matching 先例密：[BPBreID](https://arxiv.org/abs/2211.03679)、[KPR](https://arxiv.org/abs/2407.18112)；local verifier 在 wildlife 有 [WildFusion](https://arxiv.org/abs/2408.12934)。主创新风险偏高。 |
| 5 | **False-Budgeted Listwise Training** | 训练 mini-gallery 时固定每个 query 的 false-positive budget，直接优化 SmoothAP/listwise top-k 排序。 | 这是训练 loss，不是 retrieval-side scorer；但它离 Rank-Regret/gallery 死区最近，只能当低优先级。 | 用缓存 hard pools 构造相同 `#false@10` 的 mini-gallery；SmoothAP loss 训 head/last stage。成功线：同 false-budget 内 ΔAP>0.5，shuffle negative identity control 无增益。 | **5.6** | listwise/AP loss 已很成熟：[SmoothAP](https://arxiv.org/abs/2007.12163)。ReID 里 triplet/ranking 老得很，所以只能卖“false-budgeted training protocol”。 |

**我会优先顺序：1 → 2 → 3。**  
4、5 只有在前三个 cheap kill-switch 全死后才值得碰；尤其 5 很容易被你们已有 gallery/rank-regret 死区吞掉。

明确不建议再开：JEPA/MIM continued-pretrain，已有 [PersonMAE](https://arxiv.org/abs/2311.04496)、[PersonViT](https://arxiv.org/abs/2408.05398)、[HAP](https://arxiv.org/abs/2310.20695)，而本仓 `PSC-JEPA` 已实测 catastrophic forgetting；uncertainty/evidential 也别主推，noisy-label ReID 已有 [CARE](https://arxiv.org/abs/2602.23133)，本仓 evidential 也中性。当前环境是只读，我没法把这轮调研落到 `experiments/` 文档。
tokens used
133,994
结论先放前面：这些约束下，**没有 8-9 分训练侧方向**。能认真赌的最高也就 6.5-6.8；再往上基本要换数据/task。Top5 如下。

| 排名 | 方向 | 机制一句 | 为何不撞死区 | cheap kill-switch（必须控 `#false@topk`） | CCF-B分 | 先例风险 |
|---|---|---|---|---|---:|---|
| 1 | **Generative Identity-Code ReID** | 不再学一个 cosine descriptor，而是训练 query 生成目标 gallery 的 PQ/code-prefix，再用 code 候选精排。 | 不是 hubness/rank-regret scorer；不是 FM；不是 test-time marginalization；训练的是“检索索引算子”。但若你把所有 gallery/index 方向都禁掉，此项直接删。 | 冻结现有强特征，给 gallery 聚 PQ/hier code；训小 decoder 预测正样本 code。成功线：top-N code recall 明显高于 nearest-code/random-code，候选精排 mAP 正增；按 baseline `#false@10` 分桶报告 ΔAP，且固定候选数。 | **6.8** | general 侧已有 [DSI](https://arxiv.org/abs/2202.06991)、[NCI](https://arxiv.org/abs/2206.02743)；ReID 有老 hashing/[PDH](https://arxiv.org/abs/1705.02145)。风险是“生成式检索搬到 ReID”而非全新。 |
| 2 | **Counterfactual Part-Contradiction Training** | 用 pose-aligned part swap 造“局部像但整体身份矛盾”的 hard negative，训练模型学 identity 必须是多证据 conjunction。 | 不是 occlusion 补全/visibility；不是 FM；不是 gallery trick；不压 K 变体一致性。核心是“强局部证据不能单独决定身份”。 | 先 feature-level swap：torso/legs/bag 等从近邻异 ID 替换，测 donor/target false 是否上升；再训 5-10ep contradiction loss。成功线：同 `#false@10` 桶内 top false rank 下降、mAP 不掉。 | **6.5** | [PartMix](https://arxiv.org/abs/2304.01537) 已在 VI-ReID 做 part descriptor mixing；counterfactual/cloth-changing 也有先例，如 [IDNet](https://arxiv.org/abs/2403.08270)。claim 必须窄到“标准 RGB ReID 的身份矛盾负样本”。 |
| 3 | **Camera-Pose Transport Operators** | 不学 camera/pose invariant descriptor，而学低秩 transport，把 descriptor 从一个 camera-pose cell 映射到另一个 cell 后再比较。 | 避开 DG/frozen-prior；不做 invariance collapse；不碰 LM-ReID K 变体。它学的是 comparability operator，不是抹平 nuisance。 | 冻结特征，按 train ID 拟合 cam-pose pair Procrustes/ridge map；test ID 直接用。成功线：同 `#false@10` 分桶 mAP +0.5 以上，且 hubness/候选数量不变；camera-centering 作弱对照。 | **6.2** | camera-aware 风险高：[CamStyle](https://arxiv.org/abs/1711.10295) 很早；causal/camera ReID 也被综述覆盖，如 [causal video ReID survey](https://arxiv.org/abs/2505.20540)。新意只在“transport not invariant”。 |
| 4 | **Self-Local-Verifier Distillation** | 训练期用局部 correspondence/verifier 纠正 top-k hard false，再把 correction 蒸进单次 descriptor；测试不跑 verifier。 | 不用 DINO/LoFTR/MLLM 作 FM-import；不做 test-time rerank；不是 part pooling 小改，而是“pairwise 局部判别 → descriptor 蒸馏”。 | 冻结 part tokens，训轻量 pair verifier 看能否在 top-k 内把正样本提到 false 前；成功必须在同 `#false@10` 桶内成立。若只靠 rerank 有效但蒸馏无效，杀。 | **5.9** | 人体 part/local matching 先例密：[BPBreID](https://arxiv.org/abs/2211.03679)、[KPR](https://arxiv.org/abs/2407.18112)；local verifier 在 wildlife 有 [WildFusion](https://arxiv.org/abs/2408.12934)。主创新风险偏高。 |
| 5 | **False-Budgeted Listwise Training** | 训练 mini-gallery 时固定每个 query 的 false-positive budget，直接优化 SmoothAP/listwise top-k 排序。 | 这是训练 loss，不是 retrieval-side scorer；但它离 Rank-Regret/gallery 死区最近，只能当低优先级。 | 用缓存 hard pools 构造相同 `#false@10` 的 mini-gallery；SmoothAP loss 训 head/last stage。成功线：同 false-budget 内 ΔAP>0.5，shuffle negative identity control 无增益。 | **5.6** | listwise/AP loss 已很成熟：[SmoothAP](https://arxiv.org/abs/2007.12163)。ReID 里 triplet/ranking 老得很，所以只能卖“false-budgeted training protocol”。 |

**我会优先顺序：1 → 2 → 3。**  
4、5 只有在前三个 cheap kill-switch 全死后才值得碰；尤其 5 很容易被你们已有 gallery/rank-regret 死区吞掉。

明确不建议再开：JEPA/MIM continued-pretrain，已有 [PersonMAE](https://arxiv.org/abs/2311.04496)、[PersonViT](https://arxiv.org/abs/2408.05398)、[HAP](https://arxiv.org/abs/2310.20695)，而本仓 `PSC-JEPA` 已实测 catastrophic forgetting；uncertainty/evidential 也别主推，noisy-label ReID 已有 [CARE](https://arxiv.org/abs/2602.23133)，本仓 evidential 也中性。当前环境是只读，我没法把这轮调研落到 `experiments/` 文档。
