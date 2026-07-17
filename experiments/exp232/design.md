# 实验 exp232: BT-PKD Cosine Decay on Small

## 动机
exp231 在 Tiny 上测试 BT-PKD cosine decay (w→0 by ep60)。
本实验在 Small 上验证同一创新。

exp230 (Small BT-PKD constant) 在 ep110 给出 70.8/81.9 (无 PARALLEL_AUG)，
然后 OOM crash。BT-PKD decay 应该减少后期内存压力（ep60 后 weight=0，
non-detached graph 不再需要 retain）。

## 核心假设
BT-PKD cosine decay 在 Small 上：
1. 保留早期加速 (ep10-30)
2. ep60 后完全关闭 → 无后期干扰 + 减少内存
3. Final 应该 ≥ baseline

## 技术方案
与 exp231 相同配置，仅换 backbone:
```
MODEL.POSE_BT_PKD True
MODEL.POSE_BT_PKD_WEIGHT 0.01
MODEL.POSE_BT_PKD_DECAY_EPOCH 60
```

无 PARALLEL_AUG (OOM with BT-PKD on Small)。
TEST.IMS_PER_BATCH 128 防止 eval OOM。

## 对照组
- exp230 (Small BT-PKD constant, no PAUG): ep110=70.8/81.9
- exp206r (Small OA-SD, PAUG): 70.6/82.6

## 早停
- ep10 < 30% → 终止
