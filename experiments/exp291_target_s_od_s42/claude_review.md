# Claude Broad Review — exp291_target_s_od_s42

**Review round**: 1 (referential to exp290)
**Date**: 2026-04-22 12:06 CST
**Reviewer**: claude-opus-4-7 (main agent, referential review)

## Scope

exp291 reuses the POSE_USE_TARGET_HEATMAP flag implemented for exp290. Zero new code changes. The experiment differs from exp290 only in:
- Dataset: Occluded-Duke instead of Occluded-PoseTrack-ReID
- Config: `configs/occluded_duke/prcv_best_small.yml` instead of `.../occluded_posetrack/...`
- Machine: lab4090 instead of srvB

All code changes (`config/defaults.py` + `model/pose_backbone_model.py`) are identical to exp290 and already reviewed in `experiments/exp290_target_s_op_s42/claude_review.md` with verdict 审查通过.

## Findings

### Critical
None. No code delta beyond exp290.

### High
None.

### Medium
1. **OD dataset 多为 single-person, target-heatmap swap 预期 ≈ no-op**: 实际需验证 — 若 Occluded-Duke 包含多人样本, target-heatmap 也可能有效。design.md 已讨论此风险, 预期结果明确 (持平 / 小增益 / 意外回归)。
2. **lab4090 OD pose_data 完整性**: 需确认 pose_data/train/index.json 有 target_person_idx 字段且正确 reorder。已通过 memory 中 2026-04-20 快照确认 "visibility + target_person_idx 全部补齐"。

### Low
- design.md 写明 "代码零改动", 明确 refer 到 exp290 审查文件, 审计轨迹清晰。

## Backward compatibility 验证

继承 exp290 review 结论: `POSE_USE_TARGET_HEATMAP=False` 时字节级等价于 HEAD。exp291 命令行打开 flag, 但 exp262/exp285b 等 default False 不受影响。

## Data flow / single-person no-op

对 single-person 样本 (person_mask[:, 0] = 1, person_mask[:, 1:] = 0):
- `merge_person_heatmaps` = `max([p0], pad_zeros)` = `p0`
- `target_heatmaps` = `heatmaps[:, 0] * person_mask[:, 0]` = `p0`
- `scene_heatmaps` 和 `target_heatmaps` 数值完全相同 → flag on 时 `scene_heatmaps = target_heatmaps` 是 no-op swap

预期 OD 上因多数 single-person 样本, 整体训练行为 ≈ 持平 exp285b。

## 训练健康度

- 机器: lab4090 (idle, Phase 3 都已完成), mmpose-abu conda env
- Speed: 预期 ~176s/epoch × 120 = 5h52min (参考 exp285b lab4090)
- Data: OD pose_data 完整, visibility + target_person_idx 齐全
- GPU: 4090 24G, Small Full Scaffold ~12GB 峰值, 余量充足

## 边界条件

承接 exp290 审查结论:
- pose_dict=None: 无 swap 发生, 原路径
- 零 target heatmap: 合理 fallback 行为 (PSG sigmoid near 1.0, LGPA 背景特征)
- OA-SD teacher/student: deepcopy 保持 flag 一致
- Flip test: 脚本会翻转所有 person 的 heatmap, target 仍在 index 0, swap 后对齐

## Verdict

**审查通过**

exp291 为 exp290 的 dataset 消融, 代码零改动, 继承 exp290 审查结论, 无新引入风险。启动条件满足。建议 lab4090 启动 exp291 后顺便保留 Market pose extraction 作为 exp292 前置。
