# Claude Broad Review — exp293_best_b_m_plboa_s42

**Review round**: 1 (config-only change, no code delta)
**Date**: 2026-04-22 17:30 CST
**Reviewer**: claude-opus-4-7 (main agent)

## Scope

exp293 makes ZERO code changes. Only CLI override to existing `configs/market/prcv_best_base.yml`:
- `MODEL.POSE_LOWER_BODY_OCC True` (was False in default config)

All other scaffold preserved: LGPA, GCN512, OA-SD, ParAug, 2-stage PSG. Swin-Base. Seed 42.

## Findings

### Critical
None. Flag toggle only.

### High
1. **PLBOA enables OA-SD 蒸馏**: 这是预期效果, 也是本实验动机。之前 Market 所有实验 OA-SD 蒸馏信号 ≈ 0, 本实验让 OA-SD 真正工作。

### Medium
1. **分布偏差风险**: Market 是 full-body benchmark, PLBOA 人工遮挡下半身可能产生 train-test mismatch。**eval 时不应用 PLBOA** (PLBOA 只在 training data augmentation, 在 pose_dataset.py:327+ 的 OA-SD 分支, 不影响 test set 加载)。验证:
   - `datasets/pose_dataset.py:_load_pose_data_with_oa_sd` — 只在 training 路径触发 PLBOA
   - `test.py` 和 flip-test 走 standard pose loading, 无 PLBOA → 无 train-test inference mismatch ✓

### Low
1. **SEED 42 对齐 exp269** 便于直接对比, 无新 seed 风险
2. **auto-chain** 用 tools/queue_on_ckpt.sh 已验证 (Phase 1/3 多次成功), 低风险
3. **Pose data legacy 字段**: /mnt1/afrdata Market pose_data 缺 visibility + target_person_idx, backward compat 已在 pose_dataset.py 处理

## 验证代码路径 (OA-SD + PLBOA 交互)

- `processor/processor.py:471-474`: OA-SD EMA teacher 创建
- `processor/processor.py:534-539`: parallel_oa_sd 组合模式 (4 views: 3 student + 1 teacher)
- `datasets/pose_dataset.py` PLBOA: 对 student img 应用 lower body occlusion, teacher img 保持 clean
- `processor/processor.py:776-841`: OA-SD 蒸馏 loss 计算 (student feat vs teacher feat L2 距离)
- PLBOA True → student 和 teacher 差异显著 → 蒸馏 signal 非零 ✓

## 运行前确认清单

- [x] Market dataset ready on lab4090 (`data/market1501` symlink)
- [x] Pretrained `swin_base.pth` exists on lab4090
- [x] Config 无需修改, 仅 CLI override
- [x] auto-chain daemon 模板齐备
- [x] Seed 42 对齐 exp269 baseline

## Verdict

**审查通过**

零代码改动, 单 flag CLI override, backward compat 完整。预期 ~6h lab4090 训练后 FINAL ≈ 94.4 ± 0.3 (取决于 PLBOA 是助力还是干扰)。建议 launch 后观察 e10 eval 和 exp268/269 对照轨迹。
