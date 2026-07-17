# Codex Review — exp330 Compositional Occluder Generalization + group-DRO

**Verdict**: approve
**Date**: 2026-06-17
**Review round**: 第 2 轮（第 1 轮 needs-attention → 修复后复审）

## Findings（第 1 轮 → 修复 → 第 2 轮复核）

### 第 1 轮 needs-attention findings + 修复
- **Critical — `scheduler.get_last_lr()` 不存在于 timm scheduler，e1 后崩溃**。已修：改 `optimizer.param_groups[0]["lr"]`。第 2 轮 verified fixed (exp330_train.py:266)。
- **High — make_model 签名/返回值不符**。**FALSE ALARM**：Codex 第 1 轮（本地 read-only）读的是**本地 SOLIDER-REID** `model/make_model.py`（Swin + semantic_weight + 3 返回值）。脚本实际 `--repo /hy-tmp/transreid` + `sys.path.insert(0, repo)` 导入的是**vanilla TransReID** make_model（已验证：(cfg,num_class,camera_num,view_num) → train 返回 (cls_score, global_feat)、eval 返回单 tensor；burstiness_probe 已在该树成功用过）。第 2 轮已明确告知，Codex 不再 re-flag。
- **High — eval 遮挡未 seed，ERM/DRO 评在不同随机遮挡上不公平**。已修：MarketSet.__getitem__ 按 `(i, cell)` 稳定 seed（`s=(i*101+ALL_CELLS.index(cell))%2**31`，跨进程稳定）再 apply_cell → ERM/DRO 同一 query 遮挡。第 2 轮 verified fixed (exp330_train.py:177)。
- **High — 空 occluder 类静默 no-op**。已修：`if occ.empty: raise RuntimeError`。第 2 轮 verified fixed (exp330_train.py:105)。
- **Medium — DRO 变体非字面 Sagawa**。已在 design.md + claude_review.md 文档化为刻意的 present-group-renormalized CE-scale-matched 变体。
- **Medium — design.md 副判据 冲突**。已修 design.md。

### 第 2 轮 remaining finding
- **Medium（文档，非 runtime blocker）**：design.md 的 group-DRO 公式描述未显式写出实现的 7-group 重归一变体。**已修**：design.md「group-DRO 目标」节已补完整实现细节。

### Codex Checked OK（两轮）
CE/triplet shapes 正确；torch 2.9 AMP 用法正确；L2-norm 平方欧氏 = cosine 排序等价；train collate 正确；gallery cache 每 eval 清；SIE off 时 train/eval 对称；dummy center criterion 无害。

## Novelty verdict（Codex web search）
likely novel as benchmark/mechanism combo：无 ReID 先例做 2-D (occluder-class × body-region) 组合 held-out + group-DRO。最近相关：occluded-ReID survey、AACN（compositional part attention）、ADP（合成遮挡泛化）、OGFR（occlusion-pattern, 1-D 类型 held-out）、Sagawa group-DRO。

## 结论
codex 审查通过（verdict: approve）。无 Critical/High runtime blocker；make_model 路径确认为第 1 轮误读本地 repo。可启动训练。
