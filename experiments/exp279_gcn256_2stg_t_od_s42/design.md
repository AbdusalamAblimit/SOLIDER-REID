# exp279_gcn256_2stg_t_od_s42 — Phase 3-B: GCN256 + 2-stage PSG (Tiny + Occ-Duke)

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-B)

## 本 exp 变量

- 相对 exp261 (Tiny Full Scaffold GCN512 + 2-stage) 单变量改:
  - `POSE_GCN_HIDDEN` 512→256 (仅降低 GCN 容量,PSG stage 保持 `[-2,-1]`)
- 其他 pose 模块保持 full scaffold (LGPA True, GCN True, OA-SD True, ParAug True, LOWER_BODY_OCC True)

## 核心假设

对照 exp261 (Tiny GCN512 + 2-stage FINAL 65.9/77.4) 和 exp278 (Tiny GCN256 + 1-stage):
- exp279 vs exp278 的 Δ 就是"GCN256 下多 1 stage PSG 的边际收益"
- exp279 vs exp261 的 Δ 就是"2-stage PSG 下 GCN cap 的边际收益"
- 若 exp279 ≈ exp261 → GCN 容量过剩;若 exp279 << exp261 → GCN 512 是 2-stage 的必要配套

## CLI 配置

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_tiny.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp279_gcn256_2stg_t_od_s42 \
  MODEL.POSE_GCN_HIDDEN 256
```

## 输出

- 机器: srvB (exp278 后 auto-chain)
- 预计时长: ~3h20min
