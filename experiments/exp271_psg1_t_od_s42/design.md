# exp271_psg1_t_od_s42 — Phase 3-A: 1-stage PSG (Tiny + Occ-Duke)

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-A)。

## 本 exp 变量

- Backbone: Swin-Tiny
- Dataset: Occluded-Duke
- Seed: 42
- **PSG: 开,只 1 stage (stage 3)**,`POSE_PSG_STAGES=[-1]`
- LGPA / GCN / OA-SD / PLBOA / Parallel-Aug: **全部关闭**(纯 PSG scaffold,隔离 PSG 单独贡献)
- TEST_FEAT: `global`
- BS=64, LR=8e-4, 120 epoch, WARMUP=20 cosine

## 核心假设

相对 exp270 (no-PSG baseline),本 run 只新增 PSG stage 3 injection。期望 mAP +1-2 over baseline(exp270 e90=58.7,预期本 run ≥60)。

## CLI 配置

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_tiny.yml \
  MODEL.POSE_BACKBONE_PSG True \
  MODEL.POSE_PSG_STAGES "[-1]" \
  MODEL.POSE_LGPA False \
  MODEL.POSE_SKELETON_GCN False \
  MODEL.POSE_OA_SD False \
  MODEL.POSE_LOWER_BODY_OCC False \
  MODEL.POSE_PARALLEL_AUG False \
  MODEL.POSE_TEST_FEAT 'global' \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp271_psg1_t_od_s42
```

## 对照

- **exp270 (无 PSG) vs 本 exp271 (1-stage PSG)**: 单变量差异 = 是否启用 PSG
- 历史 exp007 (PSG only Tiny, 120 epoch, no default flip) = 58.3/67.9
- 本 run 新协议 default flip-test 预期 ~59-60/68-69

## 输出

- 机器: srvB (exp270 完成后自动?——否,需要 manual launch)
- Log: /hy-tmp/log/occluded_duke/exp271_psg1_t_od_s42/train_log.txt
- 预计时长: ~2h40m (Tiny + PSG 加轻量模块,略比 exp270 慢 5-10%)

## 下一步

完成后按 Phase 3-A 矩阵继续 exp272 (2-stage PSG)。
