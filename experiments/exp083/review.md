# exp083 PGFI 审查报告

## 审查范围
- `experiments/exp083/design.md`
- `experiments/exp083/monitor.md`
- `configs/occluded_duke/pose_psg_gcn_paa_pgfi.yml`
- `model/modules/pose_feature_inpainter.py`
- `model/pose_backbone_model.py`
- `log/occluded_duke/exp083_pgfi/train_log.txt`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | pose_feature_inpainter.py | `vis_mask = sigmoid(hm).max(...)` 直接作用在近似非负的 pose heatmap 上。基于 200 个真实训练样本统计，`12x4` 上 `vis_mask` 的最小值平均约 `0.5001`，均值约 `0.5551`，几乎不存在真正“低可见/高遮挡”区域，PGFI 实际会在整张特征图上施加残差更新 | 未修复 |
| 2 | HIGH | design.md vs pose_feature_inpainter.py | 设计写的是 `feat_visible + (1-vis) * inpainted`，即“遮挡区域替换”；代码却是 `feat_map + occ_mask * inpainted`，原始遮挡特征并没有被替换，只是叠加了一个全图软残差 | 未修复 |
| 3 | MEDIUM | design.md | 设计把 PGFI 定义成“只修改遮挡区域的 inpainting”，但当前实现更接近“用 pose 条件调制的 feature-map residual adapter” | 未修复 |

## 审查通过项

- PGFI 接线位置正确：在 Stage 3 输出、GAP 之前生效
- 输出卷积做了 zero-init，训练初期等价于恒等映射
- 默认开关安全，关闭 `POSE_FEATURE_INPAINTER` 不会影响 baseline
- 日志完整，120 epoch 跑完

## 结论

❌ **不通过**

`exp083` 当前没有真正实现“遮挡区域特征补全”这个设计假设。现有结果不能作为 PGFI 有效性的证据。
