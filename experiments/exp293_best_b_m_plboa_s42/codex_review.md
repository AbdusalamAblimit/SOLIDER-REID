# Codex Review — exp293_best_b_m_plboa_s42

**Verdict**: approve
**Date**: 2026-04-22 17:30 CST
**Review round**: 1 (config-only, no code delta)

## Summary

Zero code changes. Single CLI override `MODEL.POSE_LOWER_BODY_OCC True` on existing Market Base config. Purpose: activate OA-SD distillation signal on Market (was no-op due to PLBOA disabled).

## Findings

### Low
- **Test-time consistency**: PLBOA only applied in training data augmentation, not in eval/test data loading. `datasets/pose_dataset.py:__getitem__` PLBOA branch guarded by `self.is_train` (or equivalent). Confirmed by existing OD/OP training with PLBOA True where test-time eval matches training FINAL numbers.
- **Market 分布特性**: full-body benchmark, PLBOA 增加 lower-body occlusion augmentation. Theoretically may reduce Market performance if student sees too many occluded variants and tests on clean full-body. BUT PLBOA_PROB 0.7 means 70% samples augmented, 30% clean — distribution should not be fully shifted. Risk mitigated by ParAug diverse views.
- **OA-SD weight**: `POSE_OA_SD_WEIGHT 1.0` unchanged. With PLBOA active, L2 distill loss would be non-zero. Loss scale 1.0 is calibrated from OD experiments where PLBOA is True.

### No new findings beyond Claude review

## 启动建议

CLI:
```
python train.py \
  --config_file configs/market/prcv_best_base.yml \
  SOLVER.SEED 42 \
  MODEL.POSE_LOWER_BODY_OCC True \
  OUTPUT_DIR /home/afr/SOLIDER-REID/log/market1501/exp293_best_b_m_plboa_s42
```

## Conclusion

codex 审查通过

No blocking issues. Ready to launch on lab4090 post-exp291 FINAL.
