# exp286_lgpaOnly_1stg_t_od_s42 — Phase 3-C: LGPA-only + 1-stage PSG (Tiny + Occ-Duke)

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-C)

## 本 exp 变量

- 相对 exp261 (Tiny Full Scaffold GCN512 + 2-stage FINAL 65.9/77.4):
  - `POSE_SKELETON_GCN` True→False (关闭结构分支 GCN)
  - `POSE_PSG_STAGES` `[-2,-1]`→`[-1]` (1-stage PSG)
- 保持 LGPA True, OA-SD True, ParAug True, LOWER_BODY_OCC True

## 核心假设

Phase 3-C 回答"2-stage PSG 的收益是偏 structural branch 还是 semantic branch 也吃?"

对照 (Phase 3-B 中 GCN on 的结果 vs Phase 3-C 中 GCN off 的结果):
- exp286 vs exp280 (Tiny GCN512 + 1-stage): 移除 GCN 的损失
- exp286 vs exp271 (Tiny pure PSG 1-stage 60.2/69.5): 加入 LGPA + OA-SD + ParAug 对 PSG 1-stage 的增益
- exp286 vs exp287 (LGPA-only + 2-stage): 同 semantic branch 下 PSG stage 数的影响

若 exp286/287 差距 ≈ exp280/261 差距 → 2-stage PSG 的收益是独立于 structural branch (GCN) 的,semantic branch 也吃
若 exp286/287 差距 << exp280/261 差距 → 2-stage PSG 主要通过 GCN 产生收益

## CLI 配置

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_tiny.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp286_lgpaOnly_1stg_t_od_s42 \
  MODEL.POSE_SKELETON_GCN False \
  MODEL.POSE_PSG_STAGES "[-1]"
```

## 输出

- 机器: srvC (exp266 silent exit 后空闲)
- 预计时长: ~3h30min (srvC 5060 Ti Tiny 速度参考 exp264 Tiny OP)
