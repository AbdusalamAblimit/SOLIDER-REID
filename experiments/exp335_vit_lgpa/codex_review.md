# Codex Review — exp335 ViT-base + LGPA-D 复现

**Verdict: approve**（复审已修代码）

复核确认 2 个 High 全修：
- **损失路径**：`[score_a]+cls_scores` = `[global, pooled, p1..p5]`(7)，`loss_func` 调一次 → make_loss list-branch `0.5·global + 0.5·mean(pooled+5parts)` ID + 每部位 triplet + `0.5·assign`，**与 pose_backbone_model.py 原 LGPA 返回结构一致**。
- **pose dataloader parity**：两臂 POSE_ENABLED=True，`--use_lgpa` 只 gate head；train 总解 5-tuple、eval 总解 7-tuple；off 臂传 heatmaps=None、返回 appearance-only。

reconfirm：`_tokens`→(B,768,16,8) 布局正确、LGPA-D 用 feat_map.detach()、set_seed 在 model init 后、AMP/scheduler/optimizer/R1_mAP_eval 全对、融合 L2(cls)+alpha·L2(pooled) concat 后 metric 归一。

**No blocking findings.** 配合 Claude review（PASS after fix）+ 两臂 smoke 端到端跑通，可训练。

## 结论
codex 审查通过。
