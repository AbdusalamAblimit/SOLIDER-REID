# exp289_lgpaOnly_2stg_s_od_s42 — Phase 3-C Small LGPA-only + 2-stage PSG

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-C)

## 本 exp 变量

- 相对 exp262 Small Full Scaffold GCN512+2stg (73.8/83.1) 单变量:
  - `POSE_SKELETON_GCN True → False` (关 GCN, PSG stages 保持 `[-2,-1]`)

## 对照

- **exp288 Small LGPA-only 1stg**: 本 exp 的 1stg 对照 (预计 tmr 07-10 FINAL)
- exp262 Full Scaffold GCN512+2stg 73.8/83.1: 是否有 GCN 下降多少
- exp287 Tiny LGPA-only 2stg: 65.9/77.0 (Tiny 缩放参考)

## CLI 配置

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_small.yml \
  SOLVER.SEED 42 \
  TEST.IMS_PER_BATCH 128 \
  MODEL.POSE_SKELETON_GCN False \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp289_lgpaOnly_2stg_s_od_s42
```
(PSG_STAGES 保持 yml default `[-2,-1]`)

## 输出

- 机器: srvC (daemon 76271 auto-chain after exp288)
- 预计时长: ~10-13h
- ETA: exp288 FINAL 后 → tmr 17-22 CST FINAL
