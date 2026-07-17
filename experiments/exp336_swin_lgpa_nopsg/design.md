# 实验 exp336: Swin-Tiny + 纯 LGPA-D 隔离（关 PSG）

## 动机
- exp335(ViT 纯 LGPA-D)证:热图 bug 修复后(assign 0→7)仍只 +0.5、equalcat 不超 baseline。查 exp244 config 发现 **LGPA-D 从未单独跑过**——exp244/245g 全是 PSG+LGPA+OASD+parallel-aug+384+Swin 完整系统。
- 假设:LGPA 部位的价值来自 **PSG 把 pose 门控进 backbone 特征**;无 PSG 时部位从原始特征池化→与 global 冗余→不涨。
- 用户决策:**在 Swin 上隔离纯 LGPA**(关 PSG/OASD/aug),回答对 step2 最关键的问题——**CLIP 模块本身能否 standalone 涨点**。

## 核心假设
若纯-LGPA-on-Swin(无 PSG)也不涨(equalcat ≈ global) → 确认是 **PSG 驱动增益,非 CLIP 模块本身** → step2 的新 CLIP 接法必须 standalone-strong。
若涨(equalcat > global) → CLIP 模块能 standalone,exp335 的失败是 **ViT-specific**(ViT 单尺度 vs Swin 多尺度部位特征)。

## 技术方案（零新代码，纯 config + 原 pipeline）
- 基于 `pose_psg_lgpa_detach.yml`(exp244 原版),仅改:
  - `POSE_PSG_STAGES: []` → psg_modules_dict 空(已验证 size=0),无 PSG;`POSE_BACKBONE_PSG: True` 仅用于选 PoseBackboneModel。
  - `POSE_OA_SD: False`、`POSE_PARALLEL_AUG: False`、`POSE_LOWER_BODY_OCC: False`(关所有非-LGPA 组件)。
  - scene 热图(不设 POSE_USE_TARGET_HEATMAP→默认 False,忠实原版;exp335 的 bug 是设了 True)。
- Swin-Tiny,384×128,GLOBAL_LOSS_SCALE 0.5,LGPA-D detach,equal_concat。
- 模型已验证:PoseBackboneModel,psg_modules_dict=0,clip_part_head 存在,detach=True。

## 判据（同一 ckpt 两个描述子）
- `test.py POSE_TEST_FEAT=equal_concat` → LGPA 描述子 mAP。
- `test.py POSE_TEST_FEAT=global` → global-only mAP(LGPA detached → == 无-LGPA baseline)。
- **equalcat > global? 是→CLIP standalone 有效;否→PSG 驱动。**
- 训练 log 的 `lgpa_assign` 应非零(~7,scene 热图;对照 exp335 修复前=0)。

## 对照
- 同模型内 equal_concat vs global(detach 保证 global==baseline,干净单变量)。
- 对照 exp335(ViT 纯 LGPA):若 Swin 涨而 ViT 不涨 → ViT-specific;若都不涨 → PSG 驱动。

## 机器
lab-3090-d（swin_tiny.pth 在;CLIP 文本特征缓存在;原 train.py pipeline）。
