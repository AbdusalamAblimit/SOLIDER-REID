# exp054 PGAM 审查报告

## 审查范围
- `experiments/exp054/design.md`
- `configs/occluded_duke/pose_psg_gcn_pgam.yml`
- `model/modules/pose_attn_mask.py`
- `model/pose_backbone_model.py`
- `log/occluded_duke/exp054_pgam/train_log.txt`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | pose_attn_mask.py | PGAM 假设输入是 raw logits，并先做 `sigmoid` 再 threshold；但当前数据管线里的 pose heatmap 已近似非负。实测抽样 200 个训练样本后，在 `12x4` 上 `threshold=0.3` 的 body mask 覆盖率是 **100%**，PGAM 在真实数据上是确定性的 no-op | 未修复 |
| 2 | HIGH | design.md vs data reality | 设计文档把 PGAM 叙述成“硬掩码阻断非人体 token”，但当前实现不会产生非 body token，因此该实验没有真正验证设计假设 | 未修复 |

## 审查通过项

- PGAM 接线本身是通的，`pose_bias_map` 会进入 Swin attention
- 默认配置安全，关闭 `POSE_ATTN_MASK` 即可回到 baseline
- 训练日志完整，无运行时错误

## 结论

❌ **不通过**

`exp054` 的主要问题不在“代码没连上”，而在“连上以后 mask 永远全 1”。现有结果不能作为 PGAM 有效性的证据。
