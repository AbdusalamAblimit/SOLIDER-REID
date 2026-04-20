# exp278_gcn256_1stg_t_od_s42 — Phase 3-B: GCN256 + 1-stage PSG (Tiny + Occ-Duke)

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-B)

## 本 exp 变量

- 相对 exp261 (Tiny Full Scaffold = GCN512 + 2-stage PSG) 两变量同时改:
  - `POSE_GCN_HIDDEN` 512→256 (降低 GCN 容量)
  - `POSE_PSG_STAGES` `[-2,-1]`→`[-1]` (减少 PSG stage 到 1)
- 其他 pose 模块保持 full scaffold (LGPA True, GCN True, OA-SD True, ParAug True, LOWER_BODY_OCC True)

## 核心假设

GCN256 + 1-stage PSG 是 Tiny 下最精简 full-scaffold 组合。对照 exp261 (GCN512 + 2-stage Tiny FINAL 65.9/77.4):
- 若 Δ ≈ -2 → GCN cap + PSG stage 有交互增益,Phase 3-B 设计意图站得住
- 若 Δ < 1 → 容量/stage 都不关键,论文需重新措辞"high-capacity structural branch 必要性"

## CLI 配置

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_tiny.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp278_gcn256_1stg_t_od_s42 \
  MODEL.POSE_GCN_HIDDEN 256 \
  MODEL.POSE_PSG_STAGES "[-1]"
```

## 输出

- 机器: srvB (exp273 FINAL 后 auto-chain)
- 预计时长: ~3h20min (srvB 99s/epoch × 120)
