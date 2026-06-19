# 实验 exp345 (Option C): K pose-localized 部位特征对齐 ID 原型

## 动机（用户路线，姿态融进 CLIP 机制 — 方式 C）
A 是单个 pose-guided 全局池化;C 是**部位级**:K=3 身体部位(头/躯干+臂/腿),每个用其关键点热图作 bias 池化 → K 个 pose-localized 部位特征,每个对齐 per-ID 原型。比 A 更细粒度。

## 核心假设
**K 个部位各自的 pose-localized 特征对齐同一 ID 原型 → 每个身体区都被逼成 ID-判别(姿态定位),backbone 学到部位级判别 → global 涨。**

## 技术方案
- `PoseGuidedPartPool`：K=3 可学习 query,每个 query 的 attention 加其部位关键点(PART_GROUPS: head[0-4]/torso[5-10]/legs[11-16])热图 amax 作 bias → (B, 3, C)。
- forward：`POSE_CLIP_ID_PART_GUIDED` 开 → `part_feats = part_pool(featmaps[-1], scene_heatmaps)`;clip_id_loss = mean_k [i2t/t2i(proj(part_feats[:,k]), ID原型)]。
- = exp341 config + `POSE_CLIP_ID_PART_GUIDED: True`。测试描述子 = global。
- scene_heatmaps None → 退回 A/exp341 路径(global)。RNG-preserve 已加。

## 预期结果
exp345 global **> exp341 global(59.8)**。部位级对齐若过强可能干扰 global,失败信号 = global < 59.8。

## 对照组
- **exp341(prompt 对齐 raw global, 59.8)vs exp345(K 部位对齐)**。单变量 = 仅 POSE_CLIP_ID_PART_GUIDED。
- baseline 57.6。

## 审查重点
PoseGuidedPartPool K 部位 shape/per-part pose bias(PART_GROUPS 正确)；per-part i2t/t2i 循环(proj 复用 in_planes→clip_dim、mean over K)；clip_id_loss 正确累加;RNG-preserve;scene_heatmaps None fallback;test 端 prompt train-only;单变量 vs exp341。
