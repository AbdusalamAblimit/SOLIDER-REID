# exp273_psg3_t_od_s42 — Phase 3-A: 3-stage PSG (Tiny + Occ-Duke)

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-A)。

## 本 exp 变量

- 相对 exp272: `POSE_PSG_STAGES=[-3,-2,-1]` (增加 stage 1 注入,即所有后 3 个 stage)
- 其他 pose 模块仍关(LGPA/GCN/OA-SD/PLBOA/ParAug)

## 核心假设

相对 exp272 (2-stage PSG),本 run 增加 stage 1 PSG 注入。历史数据 (exp000 系列旧协议) 显示 3-stage 不如 2-stage,预期 mAP ≈ exp272 或**略低 0.3-0.8**。如果 3-stage 显著低于 2-stage → 支持"**2-stage 为最优 instantiation**"的论点,用于消融表。

## CLI 配置

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_tiny.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp273_psg3_t_od_s42 \
  MODEL.POSE_BACKBONE_PSG True \
  MODEL.POSE_PSG_STAGES "[-3,-2,-1]" \
  MODEL.POSE_LGPA False \
  MODEL.POSE_SKELETON_GCN False \
  MODEL.POSE_OA_SD False \
  MODEL.POSE_LOWER_BODY_OCC False \
  MODEL.POSE_PARALLEL_AUG False \
  MODEL.POSE_TEST_FEAT 'global'
```

## 预期结果

| 指标 | 目标 | 意义 |
|------|------|------|
| mAP | ≈ 59.5-60.5 | stage 1 PSG 可能拖累深层特征 |
| R1 | ≈ 69-70 | 稍低于 exp272 |

## 对照组

- exp270 (no PSG): 59.2/68.4
- exp271 (1-stage): 60.2/69.5
- exp272 (2-stage): 进行中
- **exp273 (3-stage, 本)**: 关键比较点

## 输出

- 机器: srvB (auto-chain from exp272)
- 预计时长: ~3.1h (3-stage 比 2-stage 慢 2-3%)
