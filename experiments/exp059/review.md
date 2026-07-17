# exp059 ROA + PGAM 组合审查报告

## 审查范围
- `experiments/exp059/design.md`
- `configs/occluded_duke/pose_psg_gcn_roa_pgam.yml`
- `datasets/occlusion_augmentation.py`
- `datasets/pose_dataset.py`
- `model/modules/pose_attn_mask.py`
- `model/pose_backbone_model.py`
- `model/backbones/swin_transformer.py`
- `log/occluded_duke/exp059_roa_pgam/train_log.txt`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | PGAM shared path | `exp059` 虽然 config 同时打开了 ROA 和 PGAM，但 PGAM 本身在当前数据分布下是 no-op，因此这个组合实验实际退化成了 `ROA only` | 未修复 |

## 审查通过项

- config 同时启用了 `POSE_ROA=True` 和 `POSE_ATTN_MASK=True`
- PGAM 的 `pose_bias_map` 会进入 `ShiftWindowMSA -> WindowMSA(extra_attn_bias)`，不是“配了但没用”
- ROA 与 PGAM 分别作用于 dataloader 和 backbone，代码路径互不覆盖
- 默认配置安全，关闭任一开关即可回到单模块实验
- 训练日志无报错，说明组合路径能稳定运行

## 结论

❌ **不通过**

`exp059` 不是一个有效的组合实验。当前实现真正新增的只有 ROA，PGAM 不产生有效 mask，因此不能据此讨论 PGAM 与 ROA 是否正交。
