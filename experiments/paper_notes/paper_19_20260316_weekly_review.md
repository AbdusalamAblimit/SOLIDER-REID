# Paper 19: 2026-03-16 周度方向复盘（KPR / ProFD / Pose2ID / DPEFormer / SSSC / FCFormer / TTPM）

**复盘日期**: 2026-03-16  
**目的**: 回答三个问题  
1. 当前 `PSG + GCN + PAA (+ROA)` 到底够不够支撑 B 类会议/期刊主线  
2. 近两年 occluded / pose-guided ReID 的主问题已经推进到哪里  
3. 接下来一周最值得做的方向是什么

---

## 本轮联网复盘涉及的主要论文/代码

1. **KPR — Keypoint Promptable Re-Identification**  
   - arXiv: https://arxiv.org/abs/2407.18112  
   - code: https://github.com/VlSomers/keypoint_promptable_reidentification

2. **ProFD — Prompt-Guided Feature Disentangling for Occluded Person Re-Identification**  
   - arXiv: https://arxiv.org/abs/2409.20081  
   - code: https://github.com/Cuixxx/ProFD

3. **Pose2ID — From Poses to Identity**  
   - arXiv: https://arxiv.org/abs/2503.00938  
   - code: https://github.com/yuanc3/Pose2ID

4. **DPEFormer — Dynamic Patch-aware Enrichment Transformer for Occluded Person Re-Identification**  
   - arXiv: https://arxiv.org/abs/2402.10435  
   - code: https://github.com/zhangxin06/DPEFormer

5. **SSSC-TransReID — Exploring Stronger Transformer Representation Learning for Occluded Person Re-Identification**  
   - arXiv: https://arxiv.org/abs/2410.15613

6. **FCFormer — Feature Completion Transformer for Occluded Person Re-identification**  
   - IEEE Xplore: https://ieeexplore.ieee.org/document/10476722/  
   - arXiv: https://arxiv.org/abs/2303.01656

7. **TTPM — Texture-aware Transformer with Pose-Patch Mapping for Occluded Person Re-Identification**  
   - Pattern Recognition abstract: https://www.sciencedirect.com/science/article/abs/pii/S0031320325010027

---

## 过去的“巨人们”到底干到了哪里

### 1. 问题定义已经不再停留在“再加一个 pose 模块”

- **KPR** 已把问题明确推进到 **multi-person ambiguity / target ambiguity**：
  图像里不只是“有没有遮挡”，还存在“你到底要认谁”的问题。
- **TTPM** 的摘要也直接把问题写成：
  **non-target pedestrian occlusion** 与 extreme occlusion 下的 pose misalignment。
- **FCFormer / FRT** 这条线把 occluded ReID 写成：
  **feature completion / retrieval-time recovery**，而不是单纯 backbone 提特征。
- **Pose2ID** 则说明：
  test-time 的 feature centralization / generation prior 也已经进入主叙事层。

### 2. 机制层面已经卷到 prompt / decoder / completion / retrieval reasoning

- **ProFD** 用文本 prompt + cross-attention decoder 解耦局部特征。
- **KPR** 把关键点 prompt 直接作为 backbone 输入条件，而不是只做后置 weighting。
- **DPEFormer** 用 dynamic token selection + feature blending + realistic occlusion augmentation。
- **FCFormer** 用 completion decoder + occlusion augmentation + paired feature consistency。

### 3. 纯数据增强或纯小模块很难再当主创新

- **ROA/OIA** 这一类真实遮挡增强在 **DPEFormer / FCFormer** 里已经出现。
- **quality weighting / common visible part weighting** 在 **KPR / QPM / BPBreID** 一类工作里也已经很成熟。
- 因此：
  - `ROA` 可以当有效 recipe
  - 但不能再当论文主贡献

---

## 对当前代码线最重要的结论

### 结论 1：当前结果还不够支撑 B 类主线

当前最强证据是：
- `exp066 PAA` = `61.6% / 74.2%`
- `exp067 PAA+ROA` = `62.0% / 73.7%`

但这离 **B 类主线** 还差三块关键东西：

1. **问题定义还不够强**
   - 目前主叙事仍接近“PSG 乘法注入 + PAA 加法注入”
   - 这更像“pose injection 更完整了”
   - 还没有把问题升级到 KPR/TTPM 那种明确的问题层：`target ambiguity`、`non-target pedestrian occlusion`、`pair comparability`

2. **机制虽然有效，但仍偏模块级**
   - `PAA` 本质上仍是零初始化 `1x1 conv` adapter
   - 它是一个好模块，但单独拿出来仍容易被审稿人归为“再加一个 adapter”

3. **证据链还不够闭环**
   - 当前 `PAA` 的最终结论还缺完整 multi-seed
   - 最强配置 `PAA+ROA` 目前是单 seed
   - 还缺“为什么有效”的 subset / case / failure mode 证据

### 结论 2：`exp070` 只否定了 naive target-only，不是否定 target ambiguity

`exp070` 的负结果是：
- 把 `PAA` 的热图输入从 scene 直接切到 target-only，会损失 scene context

它**不能**推出：
- “target-aware 路线没价值”
- “multi-person ambiguity 不是问题”

更准确的解释应是：
- **scene context 对 suppress 很重要**
- 但这不代表 target / distractor 区分不重要
- 只说明 `scene -> target` 的硬切换太粗暴

### 结论 3：下一步最该切的问题不是“再改 PAA”，而是“明确 target vs distractor”

当前代码资产已经具备：
- `exp033` target assignment
- `exp034` target-aware person reordering
- 多人 heatmaps / keypoints / scores 全都还在 dataloader 里

这意味着我们并不缺 target-aware 基础设施，缺的是：
- **把 target person 和 distractor person 的差异真正写进机制**

---

## 如果当前成果还不够，我们还能争什么

### 推荐主线：Target-Distractor Pose Conditioning（TDPC）

#### 问题定义
- Occluded-Duke 的多人图不只是“遮挡更难”，而是：
  **scene-level pose prior 会把 target 与 distractor 混在一起**
- `PSG/PAA` 目前默认使用 scene-level max-merge 热图
- 这对 suppress 背景有帮助，但对 target disambiguation 不够

#### 机制草案
保留当前最强主干：
- `PSG + GCN + PAA + 0.5x loss`

只改一个核心变量：
- 在 `PAA` 路径中引入 **target / distractor differential conditioning**

可行实现：
1. `H_scene = max(all persons)`
2. `H_target = heatmaps[:, 0]`
3. `H_distractor = max(persons[1:])`
4. 定义 ambiguity score `a`
   - 例如由 `num_persons`、target margin、non-target area ratio、`H_distractor` 强度共同构成
5. PAA 路径改为：
   - `x = x + Adapter(H_scene) + a * DeltaAdapter(H_target, H_distractor)`
   或
   - `x = x + Adapter([H_target, H_distractor])`

#### 为什么它比继续刷 PAA 变体更合理
1. **问题层面升级了**
   - 从“更好的 pose injection”
   - 升级到“如何在多人遮挡里区分 target 与 distractor”

2. **它没有被 exp070 否掉**
   - `exp070` 试的是 target-only hard switch
   - 这里是 scene + target/distractor differential

3. **它与 KPR/TTPM 对齐**
   - KPR: target ambiguity
   - TTPM: non-target pedestrian occlusion
   - 我们可以做出一个 **无需 prompt、无需 CLIP、直接用 pose 数据完成 disambiguation** 的版本

4. **一周内可落地**
   - 数据基础设施已经齐
   - 只需加 heatmap 准备、ambiguity score、adapter 分支和 subset 评估

---

## 备选主线：Pair-Specific Common-Support Recovery（仅作 Plan B）

如果 TDPC 2-3 天内单 seed 没有正信号，备选应转回：
- **retrieval-time common-support reasoning / feature recovery**

理由：
1. `cvk_hybrid` 已有稳定正 mAP 信号
2. `CSGT` 的失败说明 training mining 不是正确迁移方式
3. 更合理的路线应是：
   - 保留结构化 keypoint features 到检索阶段
   - 做 pair-specific support-aware correction / recovery

但这条线更偏 retrieval-time，论文表述要更谨慎，因此只作为 TDPC 的 fallback。

---

## 当前最终判断

### 1. 目前成果够不够 B 类？
- **还不够。**
- 当前更像：
  - 有效且强的 engineering/research baseline
  - 有一个值得重视的新模块 `PAA`
- 但还没有到：
  - 问题定义、机制、证据三者同时站稳的 B 类主线

### 2. 接下来一周最值得做什么？
- **不是继续做 PAA 小变体**
- **不是再做 generic decoder / attention / loss**
- **而是把主问题切到 `target ambiguity / target-distractor conditioning`**

### 3. 最推荐的下一实验
- 相对 `exp066` 或 `exp067`：
  - 只新增 `target-distractor pose conditioning`
  - 默认不动 batch size / backbone / GCN / global loss scale
- 单变量、问题更强、实现也最现实

