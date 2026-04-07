# exp249: Small LGPA-D + GCN 双分支 + OA-SD

## 动机
exp246b (Tiny) 证明 LGPA-D + GCN 双分支全面超越单 LGPA-D:
- equal_concat: +0.2/+1.5 (65.5/77.2 vs 65.3/75.7)
- MaxSim: +0.3/+1.3 (66.3/77.7 vs 66.0/76.4)
需要在 Small backbone 上验证泛化性，作为论文主结果表数据。

## 核心假设
LGPA-D + GCN 双分支在 Small backbone 上同样有效，
dual branch 增益与 Tiny 一致或更大。

## 技术方案
- 与 exp246b 相同方法，改用 Swin-Small backbone
- LGPA-D (detached) + GCN (detached) 双分支
- OA-SD + PLBOA(0.7) + PSG
- WITH_CP=True (gradient checkpointing, 3090 内存限制)
- 环境: solider-reid-pt2 conda env (PT2.5 + mmcv-full)
- LR=0.0004 (Small 标准)

## 代码修改
仅 config 参数:
- MODEL.TRANSFORMER_TYPE swin_small_patch4_window7_224
- MODEL.PRETRAIN_PATH pretrained/swin_small.pth
- MODEL.WITH_CP True
- MODEL.POSE_LGPA True + MODEL.POSE_SKELETON_GCN True + MODEL.POSE_LGPA_DETACH True
- SOLVER.BASE_LR 0.0004
- TEST.IMS_PER_BATCH 128

## 对照组
- exp245g (Small LGPA-D only): 70.2/80.1
- exp206r (Small GCN+PAA+OA-SD): 70.6/82.6
- exp246b (Tiny LGPA-D+GCN): 65.5/77.2

## 预期结果
- 成功: 71-72 mAP, 81-82 R1 (超过 exp245g 和 exp206r)
- 失败: ≈ exp245g (GCN 在 Small 上无额外贡献)
- 风险: OOM 或 WITH_CP 太慢 (exp245g 已验证 WITH_CP 可行)
