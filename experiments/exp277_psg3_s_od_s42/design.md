# exp277_psg3_s_od_s42 — Phase 3-A: 3-stage PSG Small (Occ-Duke)

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-A)。

## 本 exp 变量

- Backbone: Swin-Small
- Dataset: Occluded-Duke
- Seed: 42
- `POSE_BACKBONE_PSG=True`, `POSE_PSG_STAGES=[-3,-2,-1]` (stage 1+2+3)
- 其他 pose 模块关

## 核心假设

相对 exp276 (2-stage),本 run 加 stage 1 PSG。历史数据 (exp000 系列旧协议) 显示 3-stage 不如 2-stage,预期 Small 3-stage ≈ Small 2-stage 或**略低 0.3-0.8**。如果 3-stage 显著低于 2-stage → 完整消融支持 2-stage 最优。

## CLI 配置

```bash
python train.py --config_file configs/occluded_duke/prcv_best_small.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR ./log/occluded_duke/exp277_psg3_s_od_s42 \
  MODEL.POSE_BACKBONE_PSG True \
  MODEL.POSE_PSG_STAGES "[-3,-2,-1]" \
  MODEL.POSE_LGPA False \
  MODEL.POSE_SKELETON_GCN False \
  MODEL.POSE_OA_SD False \
  MODEL.POSE_LOWER_BODY_OCC False \
  MODEL.POSE_PARALLEL_AUG False \
  MODEL.POSE_TEST_FEAT global
```

## 对照

- exp276 (Small 2-stage)
- exp273 (Tiny 3-stage) 作 backbone 对照

## 输出

- 机器: lab4090 (auto-chain from exp276)
- 预计时长: ~3.1h
