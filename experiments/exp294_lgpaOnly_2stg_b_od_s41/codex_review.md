# Codex Review — exp294

**Verdict**: approve
**Date**: 2026-04-23 18:00
**Review round**: 1

## Findings

### Scope

exp294 启动前 git diff: 仅新增 `experiments/exp294_lgpaOnly_2stg_b_od_s41/design.md` + `claude_review.md` + `codex_review.md` (本文件)。**零代码改动**。

所有模块 (`model/pose_backbone_model.py`, `model/make_model.py`, `datasets/pose_dataset.py`, `processor/processor.py`, `config/defaults.py`) 均未修改。

### Risk Analysis

**CLI override only — 完全 runtime 行为**:
- `SOLVER.SEED 41` (与 exp263d 一致, 已 FINAL 74.1/83.3)
- `MODEL.POSE_SKELETON_GCN False` (与 exp286/287/288/289 一致, 4/4 FINAL)
- `TEST.IMS_PER_BATCH 64` (与 exp263b/exp293 restart 一致, OOM-safe)
- `OUTPUT_DIR` 新建目录

无 config schema 冲突, 无新依赖。

### Code Path 等价性

POSE_SKELETON_GCN=False 分支在 `model/pose_backbone_model.py` 中已多次经过 Phase 3-C 验证:
- exp286 Tiny 1-stg: 480 iter × 120 epoch = 57600 iter FINAL 无异常
- exp287 Tiny 2-stg: 57600 iter FINAL 无异常
- exp288 Small 1-stg: 57600 iter FINAL 无异常
- exp289 Small 2-stg: 57600 iter FINAL 无异常

本 exp 仅 backbone 从 Small → Base, PSG/LGPA/OA-SD/ParAug/PLBOA/数据 pipeline 完全一致。

### Reproducibility

- Config 文件 commit hash: 当前 HEAD (4a1cc1e)
- Seed 41 固定, 与 exp263d 可严格对照
- Base config `configs/occluded_duke/prcv_best_base.yml` 已多次验证 (exp263/263b/263d)

## 结论

零代码 diff, 完全基于已验证 Phase 3-C + Base config 的配置级消融。

**codex 审查通过**
