# Codex Review — exp340b

**Verdict**: approve（with Medium scientific risk）
**Date**: 2026-06-19

## Findings（codex --search exec）
- **Medium（耦合变量）**: 非严格单标量——line-by-line diff 确认仅 `POSE_LGPA_DETACH True→False`、`GLOBAL_LOSS_SCALE 0.5→0.1`、`OUTPUT_DIR` 三处行为变化（+1 行注释）。config 隔离干净，但科学上是「regime 翻转」的耦合改动。**若赢，follow-up 拆 `undetach@0.5` 或 `detach@0.1` 归因。**
- **Medium（主风险）**: `DETACH=False` 让**固定 canonical 的 assign/part 梯度首次回传 backbone**（`pose_backbone_model.py:637` 用 `featmaps[-1]` 不 detach）。canonical 是 image-independent 布局，可能把 backbone 推向固定空间先验 → 遮挡/裁剪数据上的主要 destabilize/underfit 风险。历史 exp320 的 `DETACH=False` 曾灾难性（但 regime 不同，是警告证据非 blocker）。
- **Low**: `GLOBAL_LOSS_SCALE=0.1` → 有效 global 权重 0.05，part ID/triplet 仍 0.5（不受 GLOBAL_LOSS_SCALE 影响）；un-detached 下 part 损失仍监督 backbone。风险主要是 global 描述子退化拖累 equal_concat，而非 ID 信号缺失。
- **Low**: 无新代码路径/无新 key——`POSE_LGPA_DETACH`(defaults:222) 与 `GLOBAL_LOSS_SCALE`(defaults:141) 均已存在、已被现有 LGPA/loss 代码消费；assign loss(processor:1027) 不受 GLOBAL_LOSS_SCALE 影响。

## 结论
**Verdict: approve。codex 审查通过。** Allow launch，需密切早期监控 `lgpa_assign`、`id_part`、`id_global`、global eval、e10/20 underfit 信号。两个 Medium 是科学风险（即本实验要测的假说），非阻断；mitigation = 监控 + fallback（global 回调 0.2–0.3 或降 assign 权重）。
