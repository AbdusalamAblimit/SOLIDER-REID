# exp060 PA-ROA 审查报告

## 审查范围
- `experiments/exp060/design.md`
- `configs/occluded_duke/pose_psg_gcn_paroa.yml`
- `config/defaults.py`
- `datasets/occlusion_augmentation.py`
- `datasets/make_dataloader.py`
- `datasets/pose_dataset.py`
- `log/occluded_duke/exp060_paroa/train_log.txt`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | LOW | datasets/occlusion_augmentation.py | `jitter_std=30.0` 硬编码在函数签名里，不能通过 config 消融 | 接受 |
| 2 | LOW | datasets/occlusion_augmentation.py | 当没有有效关键点时会退回 random ROA；这会轻微稀释“纯 pose-aware”设定，但属于合理 fallback | 接受 |

## 审查通过项

- `POSE_ROA_POSE_AWARE=True` 已正确传到训练集 dataset
- Pose-aware 分支确实使用 `persons[0]` 的 keypoints 和 scores 采样遮挡中心
- 关键点位置是在 resize / flip / crop 之后读取，坐标系一致
- 训练集之外不会启用 PA-ROA
- 日志显示训练完整结束，无异常

## 结论

✅ **通过**

PA-ROA 的实现与设计基本一致。当前代码真实测到的是“基于可见关键点采样遮挡中心的 ROA”，不是随机 ROA 的伪装版本。
