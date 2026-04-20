# exp275 monitor — Phase 3-A Small 1-stage PSG (Occ-Duke, seed 42)

- 机器: lab4090
- 启动: 2026-04-20 21:38 CST (daemon python3 bug 修复后手动启动)
- Log: `/tmp/exp275.log` → `/home/afr/SOLIDER-REID/log/occluded_duke/exp275_psg1_s_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_small.yml` + CLI override (PSG_STAGES=`[-1]`)
- Scaffold: Swin-Small + PSG 1-stage (LGPA/GCN/OA-SD/PLBOA/ParAug 全关)
- Python: `/usr/local/anaconda3/envs/mmpose-abu/bin/python`
- Speed: ~50s/epoch (预估),120 epoch ETA ~23:18 CST

## 启动背景

exp274 FINAL 后原 daemon 3580255 crash (系统 python3 无 torch),详情见 decisions.md `2026-04-20 21:35` 条目。脚本已修(PYTHON env var 支持),新 daemon 链已重启。

## 对照(Phase 3-A 矩阵)

| Exp | Backbone | PSG stages | FINAL mAP/R1 |
|-----|---------|-----------|-------------|
| exp270 | Tiny | 无 | 59.2 / 68.4 |
| exp271 | Tiny | `[-1]` | **60.2 / 69.5** ← Tiny 1-stage 参考 |
| exp272 | Tiny | `[-2,-1]` | 60.5 / 69.7 |
| exp273 | Tiny | `[-3,-2,-1]` | 进行中 |
| exp274 | Small | 无 | **68.1 / 76.8** ← Small baseline |
| **exp275 (本)** | **Small** | **`[-1]`** | **pending** |
| exp276 | Small | `[-2,-1]` | queued (daemon 275→276_v2) |
| exp277 | Small | `[-3,-2,-1]` | queued (daemon 276→277_v2) |

## 核心假设

vs exp274 Small no-PSG (68.1/76.8): 加入 PSG stage 3 (`[-1]`) 是否能在 Small backbone 上延续 Tiny 的 +1.0/+1.1 收益?
- 若 Δ ≈ +1.0/+1.0 (Tiny 持平):backbone 容量不影响 PSG 有效性
- 若 Δ < +0.5:Small 容量已接近饱和,PSG 收益被 backbone 吃掉
- 若 Δ > +1.5:Small 的注意力能更好地 leverage PSG prior

## 预期

- mAP: 68.5-70.0 (对标 Tiny +1.0 = 69.1,加上 Small noise 可能到 70)
- R1: 77.0-78.5
