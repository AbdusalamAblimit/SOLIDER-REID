# 实验 exp225: GSPB + PADPQ Combined — Tiny

## 动机
- GSPB (exp220): equal_concat 62.9, **maxsim 64.6** (Tiny best maxsim)
- PADPQ K=4 (exp223): **equal_concat 63.7** (+0.5 mAP vs OA-SD), R1 74.5 (-0.9)
- 假设: GSPB 改善 backbone → 更好的 feature map; PADPQ 改善 sampling → 更好的 keypoint features
- 两者可能互补

## 核心假设
GSPB (5% Part→Backbone gradient) + PADPQ (deformable keypoint sampling) 在 Tiny 上可能产生叠加效果。

## 技术方案
- 基于 pose_psg_gcn_paa_roa.yml (含 ROA) + OA-SD + PLBOA
- POSE_PART_GRAD_SCALE=0.05 + POSE_DEFORMABLE_SAMPLE=True + POSE_DEFORMABLE_K=4

## 对照组
- exp191 OA-SD-only: 63.2/75.4 (eq), 64.2/77.1 (maxsim)
- exp220 GSPB: 62.9/74.3 (eq), 64.6/76.0 (maxsim)
- exp223 PADPQ K=4: 63.7/74.5 (eq), 63.9/74.8 (maxsim)

## 风险
GSPB 的 non-detached 梯度可能与 PADPQ 的 deformable offset 学习冲突。
如果 ep10 < 30%，立即终止。
