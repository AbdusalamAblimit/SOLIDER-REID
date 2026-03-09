# 实验结果总表 — Phase 2: Pure Pose Heatmap

## 数据集: Occluded-Duke

### 无后处理（纯模型结果）

| ID | 方法 | mAP | R-1 | R-5 | R-10 | vs Baseline | 备注 |
|----|------|-----|-----|-----|------|-------------|------|
| 000 | Baseline (SOLIDER-Swin-Tiny, SW=0.2) | 56.6% | 66.5% | 79.4% | 83.4% | — | 120 epoch, 完美复现 |
| 001 | + Pose Part Pooling (sigmoid, 5 parts) | 57.1% | 66.7% | 78.4% | 83.0% | mAP+0.5%, R1+0.2% | Part 分类器收敛慢(id_part≈2.0 vs id_global≈0.2) |

### +NFC 结果（如适用）

| ID | 方法 | mAP | R-1 | R-5 | R-10 | 备注 |
|----|------|-----|-----|-----|------|------|
| — | — | — | — | — | — | 待测试 |

### +Re-ranking 结果（如适用）

| ID | 方法 | mAP | R-1 | R-5 | R-10 | 备注 |
|----|------|-----|-----|-----|------|------|
| — | — | — | — | — | — | 待测试 |

## 参考: Phase 1 最佳结果 (exp/003_offline_pose 分支)
- Baseline: mAP 56.6%, R1 66.5%
- GiLt+PCFC (exp012): mAP 58.0%, R1 68.0% (+1.4/+1.5 vs baseline)
- Full pipeline (NFC+Part): mAP 64.7%, R1 69.4%
