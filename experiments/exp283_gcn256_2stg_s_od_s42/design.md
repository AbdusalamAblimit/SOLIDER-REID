# exp283_gcn256_2stg_s_od_s42 — Phase 3-B: GCN256 + 2-stage PSG (Small + Occ-Duke)

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-B)

## 本 exp 变量

- 相对 exp262 (Small Full Scaffold GCN512 + 2-stage FINAL 73.8/83.1) 单变量改:
  - `POSE_GCN_HIDDEN` 512→256 (仅 GCN 容量)
- 其他 pose 模块保持 full scaffold (PSG stage 保持 `[-2,-1]`)

## 核心假设

对照 exp282 (Small GCN256 + 1-stage) 和 exp262 (Small GCN512 + 2-stage):
- exp283 vs exp282 = "GCN256 下多 1 stage PSG 的边际收益" (Small 版)
- exp283 vs exp262 = "2-stage PSG 下 GCN cap 的边际收益" (Small 版)
- 和 Tiny exp279 vs exp278/exp261 对比 → "容量×stage 交互是否跨 backbone 一致"

## CLI 配置

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_small.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR ./log/occluded_duke/exp283_gcn256_2stg_s_od_s42 \
  MODEL.POSE_GCN_HIDDEN 256
```

## 输出

- 机器: lab4090 (exp282 后 auto-chain)
- 预计时长: ~1h42min
