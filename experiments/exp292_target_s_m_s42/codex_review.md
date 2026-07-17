# Codex Review — exp292_target_s_m_s42

**Verdict**: approve
**Date**: 2026-04-22 12:50 CST
**Review round**: 1 (referential to exp290)

## Summary

Zero code delta from exp290. Dataset-only change: Market-1501 instead of Occluded-PoseTrack. Launching on lab3090 (solider-reid env, 24GB 3090, Market pose_data present at 4.3GB/46635 npz).

## Findings

### Low
- **Code delivery via scp not git**: lab3090 can't reach github.com (network timeout). Files scp'd from local HEAD (`d80f5be`). Integrity verified via grep (flag present in both files). For traceability, log the local commit hash in monitor.md post-FINAL.
- **Legacy pose_data fields**: Market npz missing visibility, index.json missing target_person_idx. Both handled gracefully by `datasets/pose_dataset.py` backward-compat fallbacks. No runtime failures expected.

### No new code-level findings

Code = exp290 approved diff. All structural concerns (shape, dtype, backward compat, OA-SD, flip test, pose dropout) resolved in exp290 codex review.

## Single-person mathematical no-op

Market `num_persons=1` dominance:
```
scene = max([p0_hm]) = p0_hm
target = heatmaps[:, 0] * mask[:, 0] = p0_hm
```
swap is exact identity for Market samples. Expected exp292 ≈ exp268 (94.3/97.3).

## Conclusion

codex 审查通过

Ready to launch. No blocking issues.
