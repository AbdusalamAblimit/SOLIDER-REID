# exp061 GKD 审查报告

## 审查范围
- `experiments/exp061/design.md`
- `configs/occluded_duke/pose_psg_gcn_gkd.yml`
- `config/defaults.py`
- `model/pose_backbone_model.py`
- `model/modules/skeleton_gcn.py`
- `processor/processor.py`
- `log/occluded_duke/exp061_gkd/train_log.txt`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | MEDIUM | config / code path | `GKD` 没有独立开关，而是复用 `POSE_SGMKC=True` + `POSE_SGMKC_WEIGHT=0.0`。功能上实现了“mask 输入、不优化重建”，但实验隔离不干净，后续读代码很容易把它误认为 SGMKC 变体 | 未修复 |
| 2 | LOW | processor.py | 即使权重为 0，processor 仍会构建并记录 `sgmkc_loss`，带来少量无效计算 | 接受 |

## 审查通过项

- `skeleton_gcn.py` 中的关键点 mask 确实会在训练时生效
- `POSE_SGMKC_WEIGHT=0.0` 确保 reconstruction loss 不参与优化，主损失仍只有 ID + triplet
- dropout 只影响 GCN branch 输入，不会改 backbone 默认路径
- 实验日志完整，无异常中断

## 结论

🟡 **功能正确，但实验隔离不够干净**

`exp061` 目前从优化意义上等价于“仅做关键点 dropout，不做 reconstruction loss”，所以结果可以参考；但从代码审查角度，它不是一个独立、可维护的 GKD 实现。若后续要重做或写论文，建议补一个真正的 `POSE_GKD` 开关，把它与 `SGMKC` 完全拆开。
