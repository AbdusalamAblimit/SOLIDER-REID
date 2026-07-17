# exp275_psg1_s_od_s42 — Phase 3-A: 1-stage PSG Small (Occ-Duke)

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-A)。

## 本 exp 变量

- Backbone: Swin-Small
- Dataset: Occluded-Duke
- Seed: 42
- `POSE_BACKBONE_PSG=True`, `POSE_PSG_STAGES=[-1]` (stage 3 only)
- 其他 pose 模块关(LGPA/GCN/OA-SD/PLBOA/ParAug)

## 核心假设

相对 exp274 (Small no-PSG),本 run 加 1-stage PSG。历史 exp007(Tiny + 1-stage)vs exp000(Tiny no-PSG) = +1.7 mAP。预期 Small + 1-stage 相对 Small no-PSG ≈ +1~2 mAP。和 exp271 (Tiny 1-stage FINAL 60.2) 形成 backbone 缩放对照。

## CLI 配置

```bash
python train.py --config_file configs/occluded_duke/prcv_best_small.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR ./log/occluded_duke/exp275_psg1_s_od_s42 \
  MODEL.POSE_BACKBONE_PSG True \
  MODEL.POSE_PSG_STAGES "[-1]" \
  MODEL.POSE_LGPA False \
  MODEL.POSE_SKELETON_GCN False \
  MODEL.POSE_OA_SD False \
  MODEL.POSE_LOWER_BODY_OCC False \
  MODEL.POSE_PARALLEL_AUG False \
  MODEL.POSE_TEST_FEAT global
```

## 预期

- mAP: ~66-68
- R1: ~75-77

## 对照

- exp274 (Small no-PSG)
- exp271 (Tiny 1-stage FINAL 60.2/69.5)

## 输出

- 机器: lab4090 (auto-chain from exp274)
- 预计时长: ~3h
