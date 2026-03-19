# exp106 PISD (Pose-Informed Self-Distillation) 监控

## 配置
- 基线: exp066 (PSG+GCN+PAA) 61.6%/74.2%
- 改动: image-level pose masking + 二次 forward + cosine self-distillation
- 零新参数（纯训练范式改变）

## 启动确认
- Ep1, 7.9GB (warmup 期间无二次 forward), 57s/epoch
- PISD 激活后（Ep11+）速度预计翻倍（~114s/epoch）

## 评估记录

| Ep | mAP | R1 | pisd loss | vs exp066 |
|----|------|------|-----------|-----------|
| 10 | 38.4% | 51.8% | — (warmup) | 0.0%/0.0% |
| 20 | 46.9% | 59.2% | 0.026 | -0.8%/-0.8% |

PISD loss 极小 (~0.02-0.04): image-level masking 后的 L2-normalized GAP 特征与全图几乎相同。
**根本发现**: GAP 全局特征天然遮挡不变（cosine distance 几乎不受 body-part masking 影响）。

## 提前终止 (用户要求停止)
Ep28 时终止。PISD 大概率中性，原因同 PACD: 全局特征本身就对遮挡鲁棒。

## 关键教训
GAP 全局特征 inherently occlusion-invariant → self-distillation 无法提供有意义的学习信号。
真正受遮挡影响的是 per-keypoint 特征，不是全局特征。
这解释了为什么 SGCFR（per-keypoint recovery）有效而 PACD/PISD（global distillation）无效。
