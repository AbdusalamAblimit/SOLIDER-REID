# 实验 exp349: Swin-Small 全 pose 系统 (exp255) + CLIP prompt — scale-up 组合模型

## 动机
最现实的组合模型 / 论文交付:用户最强 pose 系统 exp255(Swin-Small + 2-stage PSG + LGPA + GCN512 + OA-SD + PLBOA = 73.2/83.3)+ CLIP-ReID prompt(Swin-Tiny 上 +2.2)。问:CLIP 能否给这个强 pose 系统再加涨 → 一个 CLIP+pose 都有的更强模型。
注:Swin-Tiny 上整合式全负、外挂仅+0.2(冗余),但 Swin-Small 容量大、且这是"CLIP 加到强 pose 系统上"(反向归因),组合数字才是交付物。

## 核心假设
**CLIP prompt 对齐 global(纯ID语义,语言锚)与 pose 系统(部位/遮挡结构)互补 → exp349 equal_concat > exp255 的 73.2。**

## 技术方案
- = exp255 config(pose_psg_lgpa_gcn512_2stage_small.yml)**仅加** POSE_CLIP_ID_PROMPT True + CLIP 权重 + WEIGHT 1.0。
- clip_id_loss 经 LGPA 路径(line 733 注入 kp_data)回传;全系统走 LGPA+GCN dual 分支,clip_id 能流。
- 描述子 = equal_concat(global + 部位);global 0.5x(全系统 list-path,M1);CLIP i2t/t2i 全权重对齐 global。

## 预期
exp349 equal_concat > exp255 73.2。失败可能:CLIP 与 OA-SD 自蒸馏/PLBOA 多 loss 冲突;global 0.5x 稀释 CLIP 增益;强系统已饱和(冗余)。

## 对照
exp255(全 pose, 73.2)vs exp349(全 pose + CLIP)。单变量 = POSE_CLIP_ID_PROMPT。

## 审查重点
CLIP prompt 与全系统(PSG/LGPA/GCN/OA-SD/PLBOA)多分支多 loss 共存无冲突;clip_id_loss 经 dual 分支正确回传不重复;Swin-Small backbone 下 clip_id_proj 维度(in_planes)对;OA-SD 的 EMA teacher 不受 CLIP 干扰;单变量 vs exp255;Swin-Small 显存(+CLIP ViT-L)够。

## 变体 exp349b: un-detach (winner scale-up)
exp342b 证明 un-detach LGPA +0.9。exp255/exp349 本是 DETACH=True。exp349b = exp349 + POSE_LGPA_DETACH False → Swin-Small 全 pose 系统 + un-detach LGPA + CLIP = winner 方向的 scale-up。配置级 ablation(代码已审)。

## 本变体: un-detach LGPA (POSE_LGPA_DETACH False) on Swin-Small full system + CLIP = winner方向scale-up
配置级 ablation, 代码与父实验 exp349_small_full_clip 相同(已双审查通过), 仅改 config flag。
