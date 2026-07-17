# exp062 LKU 审查报告

## 审查范围
- `experiments/exp062/design.md`
- `configs/occluded_duke/pose_psg_gcn_lku.yml`
- `config/defaults.py`
- `model/pose_backbone_model.py`
- `model/modules/skeleton_gcn.py`
- `loss/make_loss.py`
- `log/occluded_duke/exp062_lku/train_log.txt`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | design.md vs skeleton_gcn.py | 设计写的是“用 learned uncertainty 加权 keypoint pairwise distance，并在 test-time 距离计算中替代 confidence weighting”；实际实现只是把 `reliability = 1 - uncertainty` 乘到 keypoint pooling 权重上，再输出单个 pooled skeleton feature。实验没有真正实现“distance-level uncertainty weighting”这个核心假设 | 未修复 |
| 2 | MEDIUM | design.md vs pose_backbone_model.py | 默认评估模式是 `equal_concat`，推理时只拼接 global 与 pooled part features；`kp_uncertainty` 不会作为显式 test-time 匹配信号进入距离计算 | 未修复 |

## 审查通过项

- uncertainty head 本身接线正确，输出范围也被 `sigmoid` 限制在 `[0,1]`
- uncertainty regularization 会进入总损失，能防止全部塌到高不确定性
- 默认关闭 `POSE_KP_UNCERTAINTY` 时不会影响 baseline
- 日志显示训练路径稳定，无 NaN / Traceback

## 结论

❌ **不通过**

`exp062` 当前实现并没有验证 design.md 声称的问题。它测到的其实是“learned uncertainty 调制 pooled skeleton feature 权重”，而不是“learned uncertainty 改进 retrieval-time keypoint distance”。如果后续要让 Claude 重做，这个实验应优先重做，因为现有负结果不能直接用于否定设计稿里的原始想法。
