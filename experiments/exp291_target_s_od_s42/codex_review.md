# Codex Review — exp291_target_s_od_s42

**Verdict**: approve
**Date**: 2026-04-22 12:06 CST
**Review round**: 1 (referential to exp290)

## Summary

exp291 runs POSE_USE_TARGET_HEATMAP=True on Occluded-Duke instead of Occluded-PoseTrack. Zero code delta beyond what exp290 committed and reviewed. Codex review for exp290 already approved (see `experiments/exp290_target_s_op_s42/codex_review.md`).

## Findings

### Low — consistent with exp290 codex review
- Flag default False preserves all existing OD training runs (exp262/exp282-285b unchanged when re-run).
- OD pose_data on lab4090 confirmed populated with target_person_idx (per memory 2026-04-20 snapshot). No blocker.
- Shape / dtype / device of target_heatmaps identical to scene_heatmaps (both (B, 17, H, W) from same `_prepare_pose` call).

### No new findings

No code changes beyond exp290. All structural concerns (backward compat, downstream module shapes, flip test, OA-SD deepcopy, pose dropout) have been verified in exp290 codex review and remain valid.

## Single-person no-op verification

For OD single-person majority (Occluded-Duke dataset statistic):
```
scene_heatmaps = merge([p0, 0, 0, 0, 0, 0], mask=[1,0,0,0,0,0]).max(dim=1)[0] = p0
target_heatmaps = heatmaps[:, 0] * mask[:, 0] = p0
```
Mathematical identity on single-person samples → swap has ZERO effect on such samples.

For multi-person OD samples (if any):
```
scene_heatmaps = max([p0, p1, ...]) includes distractor keypoints
target_heatmaps = p0 only, excludes distractor
```
This is the disambiguation improvement — expected no regression on OD overall.

## Conclusion

codex 审查通过

No additional fixes required. Ready to launch on lab4090 with standard CLI overrides.
