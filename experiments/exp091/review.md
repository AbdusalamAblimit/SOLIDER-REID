# exp091 TTSFR 审查报告

## 审查范围
- `experiments/exp091/design.md`
- `configs/occluded_duke/pose_psg_gcn_paa_ttsfr.yml`
- `model/modules/skeleton_gcn.py`
- `log/occluded_duke/exp091_ttsfr/train_log.txt`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | MEDIUM | design.md vs 实现 | 设计写了“测试时用 SGCFR 做 gallery-based recovery”，但 `exp091` 代码和日志只实现了训练期的 batch 内恢复，测试仍是普通 `equal_concat` | 未修复 |
| 2 | LOW | skeleton_gcn.py | donor keypoint feature 会 `detach()` 后再写回恢复样本，因此恢复过程是单向的；这不是 bug，但比完全耦合的联合优化更保守 | 接受 |

## 审查通过项

- TTSFR 在 `SkeletonGCNHead.forward()` 中发生在 GCN 之前，符合设计
- 恢复后的 `kp_feats` 会继续进入 GCN、池化和主 ID+Triplet loss，确实影响主特征
- 配置单变量清晰，只打开 `POSE_TTSFR`
- 日志完整，120 epoch 结束

## 结论

🟡 **训练侧实现正确，但完整方法未做完**

如果只是想回答“训练时做 batch 内 skeleton recovery 是否成立”，`exp091` 可以参考；如果要对应 design.md 里的完整方法，还需要把 test-time SGCFR 那一半补上。
