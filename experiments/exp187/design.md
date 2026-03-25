# 实验 exp187: Parallel Augmentation + SupCon (3-view training)

## 动机
- 之前 PCVT (exp148) 三视图训练早期 +2.4 mAP 但后期被追平
- 当时用的是 CE loss。现在有 SupCon
- SupCon 天然受益于更多样的 positive pairs（3 个 view = 更丰富的 contrastive signal）
- 目标：验证 parallel aug 能否在 SupCon 环境下普遍提高准确率
- 如果有效，后续在 Swin-Small/Base 上刷 SOTA

## 技术方案
- 基于 exp176 最佳配置 (triple + SupCon T=0.05 + PLBOA)
- 增加 POSE_PARALLEL_AUG: True
- 3 views: full (标准 RE) + ROA (物体遮挡) + heavy (强制 RE)
- 每个 view 各自 forward → 各自算 loss → 平均
- 显存约 3x，3090 24GB 应该够

## 预期
- 3 个 view 提供更丰富的 SupCon positive pairs → 可能改善 metric space
- 风险：3x 训练成本，可能过拟合

## 对照组
- exp176 (SupCon T=0.05, single view): 64.1/75.5
