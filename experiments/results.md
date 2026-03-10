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
| 005 | Stage 2 Part Pooling (24×8, 384ch, part-only) | 37.0%* | 44.8%* | 59.1%* | 65.2%* | mAP-19.6% | ❌ ep40 数据, ep49 OOM 终止. Stage 2 语义不足 |
| 006 | L2-norm concat (exp001 model, test-only) | 57.4% | 66.9% | 78.9% | 83.5% | mAP+0.8% | 🟡 比 concat(57.2%) 好, 但不如 part-only(57.5%) |
| **007** | **Pose Spatial Gate in Backbone (PSG)** | **58.3%** | **67.9%** | **80.8%** | **84.9%** | **mAP+1.7%, R1+1.4%** | **✅ Phase 2 最佳！纯 global feat, 仅+102K params** |
| 008 | PSG + Part Pooling (part_only test) | 57.7% | 66.0% | 78.3% | 82.8% | mAP+1.1%, R1-0.5% | 🟡 组合不叠加, 低于 PSG-only. Part pooling 拖累全局特征 |
| 009 | Multi-stage PSG (Stage 2+3) | 58.3% | 67.2% | 81.2% | 85.2% | mAP+1.7%, R1+0.7% | 🟡 mAP 匹配 exp007, R1 略低(-0.7%), R5/R10 略优. 多 156K params 无显著收益 |
| 010 | PSG + Backbone Freeze 5ep | 12.5%* | 17.5%* | 30.4%* | 36.7%* | — | ❌ ep30 终止. 冻结 backbone 导致灾难性特征损坏 |
| 011 | PSG Stage 3 (200 epochs) | 58.3% | 67.6% | 81.1% | 85.3% | mAP+1.7%, R1+1.1% | 🟡 与 exp007(120ep) mAP 相同, 75% 更多训练时间无收益 |
| 012 | Pose Attention Bias (PAB, Stage 3) | 57.4% | 67.3% | 81.4% | 86.2% | mAP+0.8%, R1+0.8% | 🟡 有效但弱于 PSG. 仅 5.4K params. 证明 feature gate > attn bias |

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
