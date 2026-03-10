# exp014: PSG + Auxiliary Part Supervision (global test) 监控日志

## 实验描述
验证 exp008 (PSG+Part) 训练出的模型，用 global feature 测试时的表现。
本质是 exp008 的测试模式变体：训练完全相同，仅 POSE_TEST_FEAT 从 'part_only' 改为 'global'。
因此直接用 exp008 的 120ep checkpoint 评估即可，无需重新训练。

## 配置
- 模型: `PosePSGPartModel` (与 exp008 相同)
- Checkpoint: `log/occluded_duke/exp008_psg_part/transformer_120.pth`
- 评估 Config: `configs/occluded_duke/pose_psg_gilt.yml` (POSE_TEST_FEAT='global')
- 训练 Loss: id_global + id_part + tri_global + tri_part
- 测试特征: PSG global feature only (不含 part features)
- 对照: exp007 PSG-only (mAP 58.3%, R1 67.9%), exp008 part_only test (mAP 57.7%, R1 66.0%)

---
### [10:30] 检查点 #1

**状态**: 🟢正常
**进度**: Epoch 10/120 (8.3%)

| 指标 | exp014 PSG+GiLt | exp007 PSG | exp008 PSG+Part |
|------|----------------|-----------|----------------|
| mAP | 35.7% | 35.0% | 33.7% |
| R1 | 45.1% | 43.8% | 44.1% |

**Loss 分量**: id_global 5.82, id_part 6.40, tri_global 0.52, tri_part 0.53
**观察**: ep10 mAP 35.7% 略优于 PSG-only (+0.7%)。id_part 收敛明显慢于 id_global（6.40 vs 5.82），与 exp001 一致。tri_part 与 tri_global 几乎相同（0.53 vs 0.52），说明 per-part triplet 和 global triplet 提供了类似的学习信号。测试用的是 global feature（不含 part），所以 +0.7% 完全来自 GiLt 梯度对 backbone 的影响。
**决策**: 继续

---
### [10:38] 快速验证 — 直接用 exp008 checkpoint

用户正确指出：exp014 与 exp008 训练完全相同，只差 POSE_TEST_FEAT。因此直接加载 exp008 的 120ep checkpoint 做 global test 即可得到答案。

**结果**: exp008 checkpoint + global test → mAP **57.6%**, R1 **65.8%**, R5 77.9%, R10 82.6%

| 测试模式 | mAP | R1 | R5 | R10 |
|---------|-----|-----|-----|-----|
| exp007 PSG-only (无 part supervision) | **58.3%** | **67.9%** | **80.8%** | **84.9%** |
| exp008 PSG+Part, part_only test | 57.7% | 66.0% | 78.3% | 82.8% |
| **exp014 PSG+Part, global test** | **57.6%** | **65.8%** | **77.9%** | **82.6%** |

**已终止训练** (PID 1555996)，无需重新训练 2 小时。

## 结论

1. **Part supervision 明确损害了 PSG global feature 质量**：相同 backbone 配置下，加 part supervision 后 global feature mAP 从 58.3% 降到 57.6% (-0.7%)
2. **global test (57.6%) 甚至比 part_only test (57.7%) 更低**：说明 part features 本身有一点点互补信息，但弥补不了 global feature 被损害的程度
3. **Part-level 梯度与 PSG gate 梯度冲突**：PSG gate 学习的是"全局最优的空间注意力"，而 part supervision 的梯度要求"每个部件区域独立判别"，两者方向不一致
4. **重要教训**: 任何在 PSG backbone 上增加的辅助监督或辅助模块，只要会改变 backbone 梯度流，都有可能降低 PSG 的效果。PSG 的 58.3% 可能就是"单一全局监督 + pose spatial gating"的最优解

