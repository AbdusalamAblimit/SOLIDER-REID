# 实验 exp341: CLIP-ReID 式可学习 ID prompt 移植到 SOLIDER（Step 1 — 找能涨的 CLIP 机制）

## 动机
- exp340 系列彻底证明：**固定 CLIP 文本部位原型 = 壳**（random 文本反而更好 +1.0 vs CLIP +0.6）。死路。
- 文献 + CLIP-ReID 证明：真能涨的 CLIP 机制是 **可学习 ID-level prompt（CoOp）+ 图文对比**，文本原型从数据学出 per-ID（不是固定词）。
- 用户两步路线：① 先找一个真涨点的 CLIP-ReID 机制（本实验）② 再像 LGPA 那样加姿态再涨（exp342）。

## 核心假设
**给 SOLIDER global 加一个 CLIP-ReID 式可学习 ID 文本原型分支（CoOp prompt + SupCon i2t/t2i 对比），让文本原型监督/正则化 global 特征 → global 涨点。**

## 技术方案（1-stage joint，最小可测）
- 新模块 `model/modules/clip_id_prompt.py`：
  - `CLIPIDPromptLearner`：用 open_clip ViT-B-32 的冻结文本组件（token_embedding/transformer/ln_final/text_projection）。模板 "A photo of a [X X X X] person."，`cls_ctx = nn.Parameter(num_classes, 4, 512)` per-ID 可学习；prefix/suffix 冻结 buffer。forward(label) → ID 文本原型。
  - 投影：SOLIDER global(768) → CLIP dim(512)，可学习 Linear。
- 损失（processor 加）：batch unique ID 的文本原型 vs 投影后的 global，`SupCon i2t + t2i`（CLIP-ReID stage1 损失），权重 `CLIP_ID_LOSS_WEIGHT`。加在现有 ID(FC)+triplet 之上。
- 测试描述子：仍用 SOLIDER global（不动 LGPA），看 i2t/t2i 正则有没有让 global 涨。

## 预期结果
- 理想：global > baseline（CLIP-ReID ID 原型对齐让特征更判别）。哪怕 +0.3 就算「找到能涨的 CLIP 机制」→ 进 Step 2 加姿态。
- 失败最可能原因：SOLIDER 已 73 很强，ID 文本原型正则边际为零；或 1-stage joint 不如 2-stage 稳（prompt 与特征互相追）。若 1-stage 平，再试 2-stage（stage1 冻特征学 prompt，stage2 冻 prompt 微调）。

## 对照组
- baseline：同 SOLIDER 配置但关 CLIP_ID_PROMPT（= 纯 global ID+triplet）。
- 单变量：仅多 CLIP-ReID ID prompt 分支 + i2t/t2i 损失。

## Step 2 预告（exp342）
若 exp341 涨：把姿态像 LGPA 那样注入——per-ID prompt 之外再加 pose-conditioned part prompt / pose-bias，让姿态在「能涨的 CLIP 机制」上再加一层。

## 审查修正（codex High）
- `GLOBAL_LOSS_SCALE` 0.5→**1.0**：exp341 无 part 分支，global 即描述子，须全权重训练（0.5 会砍半 CE+triplet 并相对放大 clip_id_loss）。
- 实际用 **ViT-L-14**（clip_dim 768），非 design 初稿的 B-32/512。
- **精确对照 exp341base**：= exp341 但 `POSE_CLIP_ID_PROMPT: False`，同 GLOBAL_LOSS_SCALE 1.0。判据：exp341 global > exp341base global = CLIP-ReID 机制真涨。

## ★★ 结果（e120, test.py global 同口径）—— Step 1 成功
| | global mAP |
|---|---|
| **exp341（CLIP-ID-prompt ON）** | **59.8** |
| **exp341base（prompt OFF，同 GLOBAL_LOSS_SCALE 1.0）** | **57.6** |
| **单变量增益（仅 prompt on/off）** | **+2.2** ✅ |

**CLIP-ReID 可学习 ID prompt（CoOp + i2t/t2i）在 SOLIDER 上真涨 +2.2**——这是死掉的固定文本（exp340 壳）做不到的。clip_id loss 8.7→2.83（prompt 学得好）。
注：1.0-scale 无 prompt baseline（57.6）比历史 0.5-scale（59.0）低，可能 1.0 scale 全权重 triplet 略降 global；exp341（59.8）≈ 0.5-scale baseline（59.0）即 +0.8。无论对哪个 baseline 都正向。**找到能涨的 CLIP 机制 = 达成。**

## → Step 2: exp342（姿态像 LGPA 那样注入这个能涨的 CLIP 机制上，再涨）
