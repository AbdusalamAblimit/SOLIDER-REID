# exp086 PA-PAT 审查报告

## 审查范围
- `experiments/exp086/design.md`
- `experiments/exp086/monitor.md`
- `configs/occluded_duke/pose_psg_gcn_paa_parallel.yml`
- `datasets/make_dataloader.py`
- `datasets/pose_dataset.py`
- `processor/processor.py`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | MEDIUM | design.md vs pose_dataset.py | design.md 把第三路增强写成 pose-guided erasing / body-part erasing，但当前并行增强路径实际调用的是 `forced random erasing` | 未修复 |
| 2 | MEDIUM | design.md vs 实现 | 设计前半部分仍保留了 PADE 风格的 `crop` 叙事；当前真正实现并运行的是 `full + ROA + heavy RE`，实验定义需要依赖 monitor 才能说清 | 未修复 |

## 审查通过项

- `make_dataloader.py` 会正确设置 `train_set.parallel_aug = True`
- `PoseImageDataset` 会返回三路视图，`pose_train_collate_fn` 也能正确堆叠成 list
- `processor.py` 的三次 forward + 平均 loss 路径已修好，不再有 list shape / `_loss_details` 丢失问题
- 日志与 monitor 说明当前真正跑到的是“full + ROA + forced_RE”这一版

## 结论

🟡 **实现正确，但设计文档有漂移**

如果把 `exp086` 定义为“并行三路增强 = full + ROA + heavy RE”，那代码是通的；但它不是 design.md 最初写的那版 PAT，后续写论文前必须先统一叙事。
