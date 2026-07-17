# exp248 PCFD — Pose-Conditioned Feature Differencing 监控

## 配置

Test-time 实验, 在已有 checkpoint (exp244 LGPA-D+OA-SD) 上验证。
不需要训练, 只需训练 MLP difference classifier。
脚本: scripts/eval_pcfd.py (已删除, 结果保留)

对照: exp244 equal_concat (cosine): 65.3/75.7

## 结果

### MLP Difference Classifier — 全面失败

| 方法 | mAP | R1 | delta vs cosine |
|------|-----|----|----|
| exp244 cosine baseline | 65.3 | 75.7 | — |
| PCFD alpha=0.1 | 52.1 | 70.5 | -13.2/-5.2 |
| PCFD alpha=0.3 | 46.8 | 68.1 | -18.5/-7.6 |
| Simple MaxSim (无学习) | 66.0 | 76.4 | +0.7/+0.7 |

**结论**:
1. MLP difference classifier 严重过拟合训练集 pairs, 完全不泛化到 test set
2. Learned pair-level matching 在 ReID 上不 work (与 MaxSim training exp152/153 结论一致)
3. 简单 MaxSim (无学习, max cosine) 反而有效 (+0.7%)
4. Feature-level cosine / MaxSim 是更好的选择
5. 此方向证伪, 不再继续
