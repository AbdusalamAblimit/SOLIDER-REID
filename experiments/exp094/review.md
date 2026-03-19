# exp094 PCQA 审查报告

## 审查范围
- `experiments/exp094/design.md`
- `experiments/exp094/monitor.md`
- `configs/occluded_duke/pose_psg_gcn_paa_pcqa.yml`
- `model/modules/pose_translation.py`
- `model/pose_backbone_model.py`
- `processor/processor.py`
- `log/occluded_duke/exp094_pcqa/train_log.txt`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | design.md vs repo | 设计要求测试时“把 query feature 翻译到 gallery pose 再比较”，但仓库里没有任何 PTM 推理脚本，也没有把 PTM 接入模型 inference | 未修复 |
| 2 | HIGH | pose_translation.py / processor.py | `compute_training_loss()` 内部会先 `detach()` global features，PTM loss 只训练 PTM 本身，不会反向塑造 backbone 表征；因此它不会实现 design 里想验证的“让特征变得更 pose-comparable” | 未修复 |
| 3 | MEDIUM | train_log.txt | 现有训练日志只到 epoch 78，没有形成完整实验结果 | 未修复 |

## 审查通过项

- PTM 模块本身已注册进模型，参数会被优化器正确更新
- pose descriptor 从关键点坐标 + scores 构建，数值路径合理

## 结论

❌ **不通过**

`exp094` 当前只是“训练一个不参与检索的 PTM 辅助模块”。无论从训练影响还是测试使用，都还没有落到 design.md 声称的方法上。
