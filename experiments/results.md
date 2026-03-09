# 实验结果总表 — Phase 2: Pure Pose Heatmap

## 数据集: Occluded-Duke

### 无后处理（纯模型结果）

| ID | 方法 | mAP | R-1 | R-5 | R-10 | vs Baseline | 备注 |
|----|------|-----|-----|-----|------|-------------|------|
| 000 | Baseline (SOLIDER-Swin-Tiny, SW=0.2) | 56.6% | 66.5% | 79.4% | 83.4% | — | 120 epoch, 完美复现 |
| 001 | + Pose Part Pooling (sigmoid, 5 parts) | 57.1% | 66.7% | 78.4% | 83.0% | mAP+0.5%, R1+0.2% | Part 分类器收敛慢(id_part≈2.0 vs id_global≈0.2) |
| 001* | ↳ part-only 特征 | **57.5%** | **67.1%** | 79.1% | 83.5% | mAP+0.9%, R1+0.6% | Part 单独使用反而最好 |
| 002 | + Pose Part Pooling (spatial_softmax, T=1.0) | 57.2% | 66.4% | 79.2% | 83.5% | mAP+0.6%, R1-0.1% | 与 sigmoid 几乎相同 |
| 002* | ↳ part-only 特征 | **57.5%** | 66.8% | 79.6% | 84.0% | mAP+0.9%, R1+0.3% | 再次确认 part 单独使用更好 |
| 003 | Part-Dominant (67% part loss, part-only test) | 50.2% | 59.1% | 73.7% | 78.7% | mAP-6.4%, R1-7.4% | ❌ 降低 global weight 伤 backbone, ep60 终止 |
| 004 | + PFM (pose feature modulation, part-only test) | 57.5% | 66.3% | 79.6% | 84.1% | mAP+0.9%, R1-0.2% | 🟡 mAP 同 001*, R1 差 0.8%. PFM 加速收敛但不改善最终结果 |
| 004-g | ↳ global 特征 | 57.4% | 66.1% | 79.6% | 83.9% | mAP+0.8%, R1-0.4% | Global 略好于 001-global (57.1%), PFM 帮助 global |

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
