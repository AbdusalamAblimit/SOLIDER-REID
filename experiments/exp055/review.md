# exp055 PGAM 阈值消融审查报告

## 审查范围
- `experiments/exp055/design.md`
- `configs/occluded_duke/pose_psg_gcn_pgam_t05.yml`
- `config/defaults.py`
- `model/modules/pose_attn_mask.py`
- `model/pose_backbone_model.py`
- `log/occluded_duke/exp055_pgam_t05/train_log.txt`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | LOW | design.md | 本实验是纯 config 消融，没有独立代码改动；review 结论依赖 exp054 共用的 PGAM 实现 | 接受 |

## 审查通过项

- `POSE_ATTN_MASK_THRESHOLD: 0.5` 已正确接到 `PoseAttnMask(threshold=...)`
- 阈值实际作用位置正确：先对 raw heatmap 做 `sigmoid`，再做二值化
- PGAM stage 默认为最后一层；本实验未误改 PSG stage
- 默认配置仍为 `POSE_ATTN_MASK=False`，不会污染 baseline
- 日志中训练与评估均正常，无 NaN / Traceback

## 结论

✅ **通过**

这是一个干净的阈值消融实验。当前代码确实只把 PGAM 的 body/non-body 阈值从 `0.3` 改到 `0.5`，没有发现实现错误。
