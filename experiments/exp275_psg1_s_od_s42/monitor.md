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

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 | 备注 vs exp274 同期 |
|-------|-----|----|----|----|---------------------|
| 10 | 44.5 | 53.8 | 68.1 | 73.1 | -1.2/-1.7 |
| 20 | 50.6 | 58.5 | 71.9 | 76.8 | -0.2/-0.6 |
| 30 | 58.1 | 66.7 | 79.0 | 84.0 | -1.7/-1.8 |
| 40 | 62.5 | 70.4 | 83.2 | 87.0 | -0.3/-1.8 |
| 50 | 62.6 | 71.0 | 82.8 | 86.4 | **-2.4/-2.9** ⬇️ |
| 60 | 66.5 | 74.8 | 85.4 | 89.1 | **+0.9/+0.3** ⬆️ 反超 |
| 70 | 66.3 | 74.7 | 86.3 | 89.4 | -0.6/-0.9 |
| 80 | 67.7 | 75.7 | 86.5 | 89.7 | 0/-0.8 |
| 90 | 68.2 | 76.2 | 86.6 | 89.7 | +0.2/-1.1 |
| 100 | 68.5 | 76.2 | 87.2 | 90.2 | +0.2/-1.1 |
| 110 | 68.7 | 76.7 | 87.1 | 90.5 | **+0.6/-0.1** ⬆️ |
| **120 FINAL** | **68.8** | **76.8** | **87.2** | **90.4** | **+0.7/0** ⬆️ |

## FINAL (2026-04-20 23:37 CST)

- **mAP: 68.8%**, **Rank-1: 76.8%**, Rank-5: 87.2%, Rank-10: 90.4%
- **vs exp274 Small no-PSG FINAL 68.1/76.8**: **Δ=+0.7 / 0** (mAP 明显涨, R1 完全持平)
- **vs Tiny exp271 (1-stage FINAL 60.2/69.5)**: Small backbone +8.6/+7.3 增益 (符合 Tiny→Small 容量差)
- **vs Phase 3-A Tiny 1-stage 增益 (+1.0/+1.1)**: Small 1-stage 增益更小 (+0.7/0), 可能接近 Small 容量饱和
- Ckpt: `/home/afr/SOLIDER-REID/log/occluded_duke/exp275_psg1_s_od_s42/transformer_120.pth` (199MB)

## 结论

1. **PSG 1-stage 在 Small 上仍然有效** (mAP +0.7),但相比 Tiny (+1.0) 增益缩小
2. R1 完全持平 (76.8 = 76.8) — **Small + PSG 1-stage 的提升主要体现在 mAP (排序整体稳健性),不改变 top-1 匹配能力**
3. 训练中期波动大 (e50 -2.4, e60 反超 +0.9), 尾部 LR 低才稳定
4. 为 Phase 3-A Small 矩阵填坑,下一步 exp276 (2-stage) / exp277 (3-stage) 验证 stage 递增收益
5. 已自动 auto-chain 到 **exp276** via v2 daemon 3654950

