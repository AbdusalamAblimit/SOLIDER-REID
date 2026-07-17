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
| 1 | HIGH | pose_attn_mask.py / config | `threshold=0.5` 并没有形成“更严格”的 body mask。基于真实训练 heatmap 抽样统计，`12x4` 与 `24x8` 上 body mask 覆盖率仍是 **100%**，本实验与 exp054 在有效计算图上没有差别 | 未修复 |
| 2 | MEDIUM | design.md | 设计把本实验表述为“阈值敏感性消融”，但在当前实现与当前数据分布下，这个消融实际上没有发生 | 未修复 |

## 审查通过项

- `POSE_ATTN_MASK_THRESHOLD: 0.5` 已正确接到 `PoseAttnMask(threshold=...)`
- 阈值实际作用位置正确：先对 raw heatmap 做 `sigmoid`，再做二值化
- PGAM stage 默认为最后一层；本实验未误改 PSG stage
- 默认配置仍为 `POSE_ATTN_MASK=False`，不会污染 baseline
- 日志中训练与评估均正常，无 NaN / Traceback

## 结论

❌ **不通过**

从 config 角度它改了阈值，但从真实执行效果看，PGAM mask 仍然是全 1，因此 `exp055` 不是一个有效的阈值消融实验。
