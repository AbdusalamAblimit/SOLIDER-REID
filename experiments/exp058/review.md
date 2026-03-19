# exp058 ROA 审查报告

## 审查范围
- `experiments/exp058/design.md`
- `configs/occluded_duke/pose_psg_gcn_roa.yml`
- `config/defaults.py`
- `datasets/occlusion_augmentation.py`
- `datasets/make_dataloader.py`
- `datasets/pose_dataset.py`
- `log/occluded_duke/exp058_roa/train_log.txt`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | LOW | datasets/make_dataloader.py | 未对 `POSE_ROA_PATH` 做存在性检查；VOC 路径缺失会在启动时直接报错 | 接受 |
| 2 | LOW | datasets/pose_dataset.py | ROA 后仍可能叠加 RE，样本会出现“双重遮挡”；这是设计上明确接受的行为，不是实现 bug | 接受 |

## 审查通过项

- ROA 只在训练集启用，验证集不会被污染
- occluders 只在 dataloader 启动时加载一次，不会每个 batch 重复读盘
- ROA 应用时序正确：在 pad/crop 之后、tensor 化之前
- 图像增强后 pose 数据保持不变，符合 design.md 的实验定义
- 默认配置下 ROA 关闭，不影响 baseline
- 日志完整，无异常中断

## 结论

✅ **通过**

ROA 的数据增强接线是正确的，训练/验证隔离也正确。唯一需要记住的是它对外部 VOC 路径有硬依赖，但这属于运行前提，不是实验实现错误。
