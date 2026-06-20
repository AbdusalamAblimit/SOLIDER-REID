# 实验 exp347: 参数-free de-occluded 对齐(打破吸收陷阱)

## 动机
全夜探索:找 pose+CLIP 真涨的融合。3 整合式全负的根因 = **新通路有参数→吸收 i2t/t2i 梯度**(A/C)或 **global 编码姿态**(B)。外挂式(exp342)只 marginal(冗余)。
**核心突破点**:让姿态帮"对齐信号"更干净,但 (1) 对齐目标仍是纯 ID 原型(不编码姿态), (2) **池化无参数**(梯度直进 backbone,无通路可吸收)。

## 核心假设
**用参数-free 姿态可见性加权池化得到"去遮挡 global"(只看可见人体),对齐纯 ID 原型 → backbone 被逼成"可见人体特征即 ID-判别"(即使遮挡)→ raw GAP global 更干净 → 涨,尤其遮挡样本。**

## 技术方案
- `PoseWeightedPool`(无参数):`w = softmax(pose可见性 × temp); feat = Σ w·token`。无 query/k_proj → 无吸收。
- forward:`POSE_CLIP_ID_NOPARAM_POOL` 开 → `img_proj = clip_id_proj(pose_weighted_pool(featmaps[-1], scene_heatmaps))`;i2t/t2i 对齐**纯 ID 原型**(exp341 的,不改)。
- = exp341 + `POSE_CLIP_ID_NOPARAM_POOL True` + `POSE_CLIP_ID_POSE_TEMP 4.0`。描述子 = raw GAP global(POSE_TEST_FEAT global)。

## 关键区别 vs A(exp343=57.6)
A 的 PoseGuidedPool 有 learnable query/k_proj → 吸收对齐(global 没拿到)→ 57.6。
exp347 池化**零参数** → 对齐梯度只能流进 backbone → global 直接受益。这是 A 失败的针对性修复。

## 预期
exp347 global > exp341 59.8。失败可能:de-occluded global ≈ GAP(非遮挡时),增益被稀释(wash)。

## 对照
exp341(raw global 对齐, 59.8)vs exp347(de-occluded 对齐)。单变量 = POSE_CLIP_ID_NOPARAM_POOL。

## 审查重点
PoseWeightedPool 真无参数(无 nn.Parameter/Linear);梯度流进 backbone(featmaps[-1]);对齐目标是纯 ID 原型不变;descriptor=raw global;scene_heatmaps None fallback;test train-only;单变量。
