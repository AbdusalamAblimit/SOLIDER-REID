# 实验 exp014: PSG + Auxiliary Part Supervision (训练时 part loss, 测试时 global)

## 动机
- PSG backbone injection 是当前最佳方法 (+1.7% mAP)
- 但训练信号仍是标准 ID+triplet loss，没有部件级监督
- exp008 (PSG+Part) 用 part_only 测试时 mAP 57.7%，低于 PSG-only 58.3%
- **核心问题**: exp008 失败是因为 part features 不够好，还是因为 part_only 测试丢弃了 PSG global feature？
- 如果只在训练时用 part supervision 提供辅助梯度，测试时保留 PSG global feature，能否两者兼得？

## 创新点 / 核心想法
- **假设**: 训练时的 part-level supervision（ID + triplet）能给 backbone 提供更细粒度的梯度，提升 PSG global feature 的质量
- 与 exp008 的唯一区别：`POSE_TEST_FEAT` 从 `part_only` 改为 `global`
- 本质是：exp008 相同训练，只改测试特征。验证"auxiliary part supervision 是否提升 global feature"

## 技术方案
- 使用 `PosePSGPartModel`（与 exp008 完全相同的模型）
- 训练时：id_global + id_part + tri_global + tri_part（4 个 loss 分量）
- 测试时：只用 PSG global feature（POSE_TEST_FEAT='global'）
- 新增参数：Part pooling head（~2.6M，测试时不参与推理）

### 已知局限
1. **part_valid mask 未进入 loss**: 无效 part 用 GAP fallback 回填后仍参与 loss，等效于重复 global supervision
2. **Scene-level merged heatmap**: part pooling 用的是多人合并热图，不是单人热图（与 PSG 一致）
3. **Part 定义**: head, upper_torso, arms, lower_torso, legs（见 `model/modules/pose_utils.py` PART_GROUPS）

### 关键超参数
- POSE_PART_WEIGHT: 1.0（id_part 与 id_global 等权）
- POSE_PART_TRI_WEIGHT: 1.0（tri_part 与 tri_global 等权）
- 其余与 exp007/exp008 完全一致

## 预期结果
- 如果 exp014 > exp007: part supervision 确实给 PSG global feature 带来正向梯度增益
- 如果 exp014 ≈ exp007: part supervision 梯度被 global supervision 淹没，无额外价值
- 如果 exp014 < exp007: part supervision 梯度与 PSG gate 梯度冲突

## 对照组
- Baseline 对照: exp007 (PSG-only, mAP 58.3%)
- 额外对照: exp008 (PSG+Part, part_only test, mAP 57.7%)
- 消融变量: POSE_TEST_FEAT 从 'part_only' (exp008) 改为 'global' (exp014)
