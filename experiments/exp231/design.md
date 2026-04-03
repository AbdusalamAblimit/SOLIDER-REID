# 实验 exp231: BT-PKD with Cosine Decay on Tiny

## 动机

exp229 (BT-PKD constant w=0.01) 显示清晰的双阶段模式:
- ep10-30: mAP +3.5 加速 (BT-PKD distillation 引导 backbone 快速对齐)
- ep60-90: mAP -1.0~-1.5 干扰 (BT-PKD 梯度与 CE/triplet 在后期竞争)

这说明 BT-PKD 在训练早期有价值（帮助 backbone 快速学到 body-part 结构），
但在后期当 backbone 已收敛时变成噪声。

## 核心假设

用 cosine decay schedule 让 BT-PKD weight 从 0.01 衰减到 0:
- 前 60 epoch: 逐渐降低 BT-PKD 梯度
- ep60 后: BT-PKD 完全关闭，backbone 不受干扰地收敛

这应该保留 early acceleration (+3.5) 同时避免 late interference (-1.5)。

## 技术方案

仅修改一行配置:
```
MODEL.POSE_BT_PKD_DECAY_EPOCH 60
```

Weight schedule: `w = 0.01 * 0.5 * (1 + cos(π * epoch / 60))`
- ep0: w=0.01 (full)
- ep15: w=0.0085
- ep30: w=0.005 (half)
- ep45: w=0.0015
- ep60: w=0.0 (off)

## 对照组
- exp191: OA-SD only (63.2/75.4) — 无 BT-PKD baseline
- exp229: BT-PKD constant w=0.01 (进行中, 预计 ~62.0/75.0)

## 预期结果
- 如果假设成立: mAP 64.0+ (+0.8 vs baseline), R1 75.0+ (持平或正向)
- 如果失败: 与 exp229 类似 (BT-PKD 即使 decayed 也干扰)
- 最可能失败原因: decay_epoch=60 太早/太晚

## 早停
- ep10 < 25% → 终止
- ep60 < 57% → 终止 (此时 BT-PKD 已关闭, 应至少与 baseline 持平)
