# exp282_gcn256_1stg_s_od_s42 — Phase 3-B: GCN256 + 1-stage PSG (Small + Occ-Duke)

**共享设计见** `experiments/prcv_2026_psg/phase3_design.md` (Phase 3-B)

## 本 exp 变量

- 相对 exp262 (Small Full Scaffold GCN512 + 2-stage FINAL 73.8/83.1) 两变量同时改:
  - `POSE_GCN_HIDDEN` 512→256
  - `POSE_PSG_STAGES` `[-2,-1]`→`[-1]`
- 其他 pose 模块保持 full scaffold

## 核心假设

Small 上的 GCN256 + 1-stage 是最精简 scaffold。对照 exp262 (73.8/83.1):
- 若 Δ 类似 Tiny (exp278 vs exp261) → "容量×stage 交互" 跨 backbone 稳定
- 若 Small 上 Δ 显著缩小 → "Small backbone 容量已够,pose scaffold 边际收益小"

## CLI 配置

```bash
python3 train.py --config_file configs/occluded_duke/prcv_best_small.yml \
  SOLVER.SEED 42 \
  OUTPUT_DIR ./log/occluded_duke/exp282_gcn256_1stg_s_od_s42 \
  MODEL.POSE_GCN_HIDDEN 256 \
  MODEL.POSE_PSG_STAGES "[-1]"
```

## 输出

- 机器: lab4090 (exp277 FINAL 后 auto-chain)
- 预计时长: ~1h42min (lab4090 51s/epoch × 120)
