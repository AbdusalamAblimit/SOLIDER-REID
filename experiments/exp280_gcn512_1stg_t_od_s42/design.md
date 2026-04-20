# exp280_gcn512_1stg_t_od_s42 — Phase 3-B: GCN512 + 1-stage PSG (Tiny + Occ-Duke)

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-B)

## 本 exp 变量

- 相对 exp261 (Tiny Full Scaffold GCN512 + 2-stage) 单变量改:
  - `POSE_PSG_STAGES` `[-2,-1]`→`[-1]` (仅减少 PSG stage 到 1,GCN cap 保持 512)
- 其他 pose 模块保持 full scaffold

## 核心假设

本 run 是 **Phase 3-B 的核心最小闭环** (phase3_design.md L153):
- 对照 exp261 Tiny GCN512 + 2-stage FINAL 65.9/77.4,本 run 是"GCN512 下 1-stage PSG"
- Δ 回答: `exp255 vs exp255b` 观察到的"高容量 GCN 下 2-stage PSG 更优"是否跨 backbone 稳定
- 若 exp280 << exp261 → "2-stage PSG 是高容量 GCN 的必要配套",论文写法明确
- 若 exp280 ≈ exp261 → 2-stage PSG 降级为 "default setting",不作 scalable extension

## CLI 配置

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_tiny.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp280_gcn512_1stg_t_od_s42 \
  MODEL.POSE_PSG_STAGES "[-1]"
```

## 输出

- 机器: srvB (exp279 后 auto-chain)
- 预计时长: ~3h20min
