# exp276_psg2_s_od_s42 — Phase 3-A: 2-stage PSG Small (Occ-Duke)

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-A)。

## 本 exp 变量

- Backbone: Swin-Small
- Dataset: Occluded-Duke
- Seed: 42
- `POSE_BACKBONE_PSG=True`, `POSE_PSG_STAGES=[-2,-1]` (stage 2+3)
- 其他 pose 模块关

## 核心假设

相对 exp275 (1-stage),本 run 加 stage 2 PSG 注入。和 exp272 (Tiny 2-stage) 形成 backbone 缩放对照。如果 Small 2-stage 显著高于 Small 1-stage → 支持"**2-stage 为最优 instantiation**"的论点,用于 PRCV 主表消融。

## CLI 配置

```bash
python train.py --config_file configs/occluded_duke/prcv_best_small.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR ./log/occluded_duke/exp276_psg2_s_od_s42 \
  MODEL.POSE_BACKBONE_PSG True \
  MODEL.POSE_PSG_STAGES "[-2,-1]" \
  MODEL.POSE_LGPA False \
  MODEL.POSE_SKELETON_GCN False \
  MODEL.POSE_OA_SD False \
  MODEL.POSE_LOWER_BODY_OCC False \
  MODEL.POSE_PARALLEL_AUG False \
  MODEL.POSE_TEST_FEAT global
```

## 对照

- exp274 (Small no-PSG)
- exp275 (Small 1-stage)
- exp272 (Tiny 2-stage) 作 backbone 对照

## 输出

- 机器: lab4090 (auto-chain from exp275)
- 预计时长: ~3h
