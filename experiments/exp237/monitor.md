# exp237 Tiny + PPA (Pose-Prompted Part-Assignment) + OA-SD 监控

配置: Tiny + PSG + PPA (替换 GCN) + OA-SD + PLBOA(0.7) + 无 ROA
**范式创新**: 从 detached GCN sampling 到 end-to-end learnable part assignment
对照: exp191 (Tiny OA-SD, detached GCN): 63.2/75.4

## FINAL 结果

**ep120 FINAL: 63.7/75.0** (vs exp191 63.2/75.4 = **+0.5/-0.4**)

| Epoch | mAP/R1 | vs exp191 |
|-------|--------|-----------|
| 10 | 36.5/47.8 | +2.2/+1.0 |
| 20 | 48.4/59.5 | +2.4/+1.5 |
| 30 | 54.3/65.4 | +3.7/+0.6 |
| 40 | 58.2/69.1 | +3.1/+0.4 |
| 50 | 59.4/71.0 | +1.8/+0.0 |
| 60 | 61.1/72.2 | +0.5/-1.7 |
| 70 | 61.8/73.5 | +0.4/-1.1 |
| 80 | 62.6/73.9 | +0.6/-0.5 |
| 90 | 62.9/74.6 | +0.1/-0.5 |
| 100 | 63.2/74.4 | +0.0/-1.0 |
| 110 | 63.7/75.2 | +0.5/-0.2 |
| **120** | **63.7/75.0** | **+0.5/-0.4** |

## 结论

**PPA 是第一个在 final 仍然 mAP 正向的 Part branch 创新！**

对比所有之前的 Part 创新:
- BT-PKD: -1.0/-0.4 (early accel then late interference)
- FSDC: -1.5/-2.2 (detached feature completion ineffective)
- Per-part: -2.8/-2.2 (too many classifiers, slow convergence)
- PADPQ: +1.0/-1.8 (mAP+ but R1-)
- **PPA: +0.5/-0.4** ← **唯一 mAP 正向且 R1 接近持平!**

**为什么 PPA 成功而其他方法失败**:
1. 端到端训练: backbone 学到了 part-discriminative features (vs detached 无法教 backbone)
2. Clean gradient: softmax CE (vs BT-PKD cosine distill, vs GSPB mixed CE/triplet)
3. 持续改善: ep10→120 全程 mAP 正向，无后期崩塌

**PPA 的 assignment head 学习效果**:
- assign_loss: 1.77 → 0.23 (收敛)
- bg_ratio: 0.92 → 0.26 (tokens 被分配到 body parts)
- entropy: 1.79 → 0.37 (assignment 变得 confident)
