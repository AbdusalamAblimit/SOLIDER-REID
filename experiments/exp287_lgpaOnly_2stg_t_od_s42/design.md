# exp287_lgpaOnly_2stg_t_od_s42 — Phase 3-C: LGPA-only + 2-stage PSG (Tiny + Occ-Duke)

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-C)

## 本 exp 变量

- 相对 exp261 (Tiny Full Scaffold GCN512 + 2-stage FINAL 65.9/77.4) 单变量:
  - `POSE_SKELETON_GCN` True→False (关闭结构分支 GCN, PSG_STAGES 保持 `[-2,-1]`)
- 保持 LGPA True, OA-SD True, ParAug True, LOWER_BODY_OCC True

## 核心假设

对照:
- exp287 vs exp286: 同 semantic branch (LGPA + OA-SD) 下 2-stage vs 1-stage PSG
- exp287 vs exp261: 移除 GCN 后 2-stage 是否仍优于 1-stage?若仍优 → 2-stage 对 semantic branch 也有效

**核心问题答案**:
- 若 exp287 > exp286 的差距 ≈ exp261 > exp280 → 2-stage PSG 是普适的,不仅仅是 GCN 的配套
- 若 exp287 ≈ exp286 → 2-stage PSG 主要通过 GCN 增益

## CLI 配置

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_tiny.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp287_lgpaOnly_2stg_t_od_s42 \
  MODEL.POSE_SKELETON_GCN False
```

## 输出

- 机器: srvC (exp286 后 auto-chain)
- 预计时长: ~3h30min
