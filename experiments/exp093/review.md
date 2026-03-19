# exp093 PGTM 审查报告

## 审查范围
- `experiments/exp093/design.md`
- `configs/occluded_duke/pose_psg_gcn_paa_pgtm.yml`
- `model/modules/pose_token_merge.py`
- `model/pose_backbone_model.py`
- `log/occluded_duke/exp093_pgtm/train_log.txt`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | design.md vs pose_backbone_model.py | 设计把 PGTM 叙述成“在 Stage 3 内部把 48 个 spatial token 合成 5 个 body-part token 再做 attention”；实际实现是在原始 Swin block 之后额外加一个 merge-attend-expand 残差模块，Stage 3 的空间 self-attention 并没有被替换，也没有减少 token 数量 | 未修复 |
| 2 | MEDIUM | pose_token_merge.py | 基于真实训练 heatmap 抽样统计，PGTM 的 part weight 仍然很分散：48 个 token 上平均熵约 `3.848`，接近均匀分布的 `3.871`，说明它没有形成 design 里那种强语义 body-part tokenization | 未修复 |

## 审查通过项

- `POSE_TOKEN_MERGE` 已正确接到 backbone
- 模块有 zero-init gate，训练初期不会破坏 baseline
- 日志完整，120 epoch 结束

## 结论

❌ **不通过**

`exp093` 现在不是 design.md 宣称的“backbone 内部范式替换”，而是一个后置的 residual token adapter。现有结果不能支撑 PGTM 的原始叙事。
