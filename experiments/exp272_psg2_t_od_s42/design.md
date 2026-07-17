# exp272_psg2_t_od_s42 — Phase 3-A: 2-stage PSG (Tiny + Occ-Duke)

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-A)。

## 本 exp 变量

- 相对 exp271: `POSE_PSG_STAGES=[-2,-1]` (增加 stage 2 注入)
- 其他 pose 模块仍关(LGPA/GCN/OA-SD/PLBOA/ParAug)

## 核心假设

相对 exp271 (1-stage PSG),本 run 增加 stage 2 PSG。期望 mAP ≈ 或略超 exp271。历史 exp009 (Tiny + Stage2+3 PSG, 旧协议): 58.3/67.2, 与 exp007 (stage 3 only) 58.3/67.9 基本持平。本 run 新协议预期也是和 exp271 持平或 +0.5 内。

## CLI 配置

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_tiny.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp272_psg2_t_od_s42 \
  MODEL.POSE_BACKBONE_PSG True \
  MODEL.POSE_PSG_STAGES "[-2,-1]" \
  MODEL.POSE_LGPA False \
  MODEL.POSE_SKELETON_GCN False \
  MODEL.POSE_OA_SD False \
  MODEL.POSE_LOWER_BODY_OCC False \
  MODEL.POSE_PARALLEL_AUG False \
  MODEL.POSE_TEST_FEAT 'global'
```

## 输出

- 机器: srvB (exp271 后 auto-chain)
- 预计时长: ~3h
