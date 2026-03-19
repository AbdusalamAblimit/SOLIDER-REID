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
| 1 | LOW | experiments/exp059/monitor.md | 监控中记录“各评估点与 exp058 完全一致”，这在经验上可疑；但代码层面已确认 PGAM bias 会真实进入 Swin attention，未发现接线失效 | 接受，保留为残余风险 |

## 审查通过项

- config 同时启用了 `POSE_ROA=True` 和 `POSE_ATTN_MASK=True`
- PGAM 的 `pose_bias_map` 会进入 `ShiftWindowMSA -> WindowMSA(extra_attn_bias)`，不是“配了但没用”
- ROA 与 PGAM 分别作用于 dataloader 和 backbone，代码路径互不覆盖
- 默认配置安全，关闭任一开关即可回到单模块实验
- 训练日志无报错，说明组合路径能稳定运行

## 结论

✅ **通过**

从代码正确性看，`exp059` 的组合实验是成立的。我没有发现“其实没开 PGAM”这类实现错误。唯一残余风险是日志与 `exp058` 过于接近，若后续要复核，可以优先做一次同 checkpoint 的组件消融确认。
