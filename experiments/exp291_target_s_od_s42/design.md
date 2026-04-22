# exp291_target_s_od_s42 — target-heatmap 机制在 Occluded-Duke 上的消融

## 动机

exp290 在 Occ-PTrack 上测 POSE_USE_TARGET_HEATMAP (追 KPR-with-prompt 82.3/92.3)。本实验在 **Occluded-Duke** 上同机制消融, 验证 target-heatmap 对 single-person/少多人场景是否回归。

### 预期

OD dataset 多为 single-person 图 (target_person_idx = 0, person_mask 通常 only [:, 0] = 1), `target_heatmaps = heatmaps[:, 0]` 严格等于 `merge_person_heatmaps([p0])` (max over 1 person = p0)。故 **target-heatmap swap 在 OD 上 ≈ no-op**, 预期结果持平 exp262 / exp285b 73.8/83.8。

若 OD 上有显著提升 → OD 也包含多人样本, target-heatmap 对多人场景普遍有效。若持平 → 机制专门解决 OP 多人歧义, 论文叙事更清晰。

## 核心假设

target-heatmap swap 的增益 scales with 多人场景比例:
- OP (多人密度高): 显著增益 (exp290 验证)
- **OD (本实验, 少多人)**: 近似持平 (no-op on single-person majority)
- Market (几乎全 single-person): 等价于 no-op

## 技术方案

**代码零改动** — 直接 reuse exp290 合入的 `POSE_USE_TARGET_HEATMAP` flag。详见 `experiments/exp290_target_s_op_s42/design.md`, `claude_review.md`, `codex_review.md` (flag 默认 False 保 byte-identical backward compat)。

### 实验配置

- **Backbone**: Swin-Small (Full Scaffold)
- **Dataset**: Occluded-Duke
- **Config**: `configs/occluded_duke/prcv_best_small.yml` + `MODEL.POSE_USE_TARGET_HEATMAP True`
- **Scaffold**: LGPA + GCN512 + OA-SD + ParAug + LOWER_BODY_OCC + 2-stage PSG `[-2,-1]` (同 exp262/exp285b)
- **Seed**: 42 (对齐 exp262/exp285b)
- **Epochs**: 120
- **机器**: lab4090 (idle, mmpose-abu env, OD pose_data 已齐全 2026-04-20 确认)
- **Speed**: ~176s/epoch × 120 ≈ 5h52min (Small Full Scaffold on lab4090, 参考 exp285b)
- **FINAL ETA**: 启动后 ~6h

## 对照组

- **主对照**: exp285b lab4090 same-device 73.8/83.8 (gold-standard, scene_heatmap default path)
- **辅对照**: exp262 srvA 73.8/83.1
- **Phase 3-B Small 2×2** (all lab4090): exp282-285b

## 预期结果

| scenario | mAP 预期 | 解读 |
|----------|---------|------|
| OD 上 target ≈ scene | 73.6-73.9 (≈ exp285b 73.8) | 机制在 single-person 上是 no-op, 支持"机制专为多人设计"叙事 |
| OD 上有意外增益 +0.5+ | 74.3+ | OD 也有多人样本, target-heatmap 普遍有效 |
| OD 上回归 -0.5+ | 73.3- | target-only 丢失 scene 上下文, 需扩大上下文 window |

## 风险

- **OD 回归不会影响其他 exp**: 本实验独立, flag 默认 off, 不影响其他配置
- **训练时间**: 6h, lab4090 可承受 (之前 exp285b 6h30)
- **重复实验**: 和 exp285b 仅 flag 差别, 是直接对照, 跨 eval 结果可信

## auto-chain

无 chain 需求 (Market pose_data 需先重提取, exp292 留作后续)。lab4090 训练期间可并行重提取 Market pose_data (CPU-heavy 不用 GPU, 但 mmpose extractor 用 GPU, 需在训练外协调)。
