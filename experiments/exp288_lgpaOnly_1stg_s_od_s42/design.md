# exp288_lgpaOnly_1stg_s_od_s42 — Phase 3-C Small LGPA-only + 1-stage PSG

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-C)

## 本 exp 变量

- 相对 exp262 Small Full Scaffold (GCN512+2stg FINAL 73.8/83.1):
  - `POSE_SKELETON_GCN True → False` (关 GCN)
  - `POSE_PSG_STAGES [-2,-1] → [-1]` (1-stage PSG)
- 保持 LGPA/OA-SD/ParAug/LOWER_BODY_OCC/SEED 42

## 对照

Phase 3-C Small 对应 Tiny:
- exp286 Tiny LGPA-only 1stg: 66.0/76.6
- exp287 Tiny LGPA-only 2stg: 65.9/77.0
- **exp288 (本)** Small LGPA-only 1stg: pending
- exp289 Small LGPA-only 2stg: pending (auto-chain)

## CLI 配置

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_small.yml \
  SOLVER.SEED 42 \
  TEST.IMS_PER_BATCH 128 \
  MODEL.POSE_SKELETON_GCN False \
  MODEL.POSE_PSG_STAGES "[-1]" \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp288_lgpaOnly_1stg_s_od_s42
```

## 输出

- 机器: srvC (exp287 FINAL 后空闲, 立即接)
- 预计时长: ~10-13h (srvC 5060Ti + Small Full-GCN)
- ETA: 20:49 启动 → tmr 07:00-10:00 CST FINAL
- daemon 挂 exp288 → exp289 (2-stg 对照)
