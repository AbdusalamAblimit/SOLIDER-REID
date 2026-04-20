# exp284_gcn512_1stg_s_od_s42 — Phase 3-B: GCN512 + 1-stage PSG (Small + Occ-Duke)

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-B)

## 本 exp 变量

- 相对 exp262 (Small Full Scaffold GCN512 + 2-stage FINAL 73.8/83.1) 单变量改:
  - `POSE_PSG_STAGES` `[-2,-1]`→`[-1]` (仅减少 PSG stage,GCN cap 保持 512)
- 其他 pose 模块保持 full scaffold

## 核心假设

本 run 是 **Phase 3-B 的 Small 核心最小闭环** (phase3_design.md L153):
- 对照 exp262 Small GCN512 + 2-stage FINAL 73.8/83.1
- Δ 回答: 2-stage PSG 在 Small + GCN512 下是否仍比 1-stage 好
- 和 Tiny exp280 vs exp261 结果对照 → 跨 backbone 结论一致性

## CLI 配置

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_small.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR ./log/occluded_duke/exp284_gcn512_1stg_s_od_s42 \
  MODEL.POSE_PSG_STAGES "[-1]"
```

## 输出

- 机器: lab4090 (exp283 后 auto-chain)
- 预计时长: ~1h42min
