# exp070 审查报告

## 第一轮审查 (Opus 4.6)

| # | 严重程度 | 文件 | 问题 | 影响 exp070？ |
|---|----------|------|------|--------------|
| 1 | **CRITICAL** | `processor/processor.py:204` | `_prepare_pose` 返回值从 2→3，旧调用解包 2 值会 crash | 否（PAMC 未启用） |
| 2 | **CRITICAL** | `model/pose_psg_part_model.py:39` | 同上，继承了修改后的 `_prepare_pose` | 否（未使用该模型） |
| 3 | **HIGH** | `test_on_occluded_reid.py:199` | 同上 | 否（不同代码路径） |
| 4 | **MEDIUM** | `pose_backbone_model.py` | `person_mask[:, 0]=0` 时 target_heatmaps 全零，sigmoid 后为 0.5，训练后产生恒定 adapter 输出 | 否（极端边界） |
| 5 | **LOW** | `pose_backbone_model.py` | Stochastic Pose Dropout 不作用于 target_heatmaps | 否（dropout 未启用） |
| 6 | **LOW** | `pose_backbone_model.py` | target_heatmaps 在 S&C 未启用时也计算 | 否（开销可忽略） |

**结论**: 不通过 → 需修复 #1, #2, #3

## 修复内容

1. `processor/processor.py:204`: `pamc_scene_hm, _, _ = _m._prepare_pose(pose_dict)`
2. `model/pose_psg_part_model.py:39`: `scene_heatmaps, scene_scores, _ = self._prepare_pose(pose_dict)`
3. `test_on_occluded_reid.py:199`: `scene_heatmaps, scene_scores, _ = model._prepare_pose(pose_dict)`

## 第二轮审查 (Opus 4.6)

所有 3 个修复已验证正确。全项目 `_prepare_pose` 调用点无遗漏。

**结论**: ✅ 审查通过，可以开始训练。
