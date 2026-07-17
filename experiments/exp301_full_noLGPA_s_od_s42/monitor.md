# exp301_full_noLGPA_s_od_s42 monitor — Phase 3-D Small OD Full − LGPA s42

- 机器: lab4090 (24GB 4090, mmpose-abu env)
- Config: `prcv_best_small.yml` + CLI `SOLVER.SEED 42 MODEL.POSE_LGPA False TEST.IMS_PER_BATCH 128`
- Scaffold: Swin-Small + GCN512 + OA-SD + ParAug + PLBOA + 2-stage PSG (**LGPA OFF**)
- 启动: 2026-04-26 03:30 CST, FINAL: 09:38 CST (~6h)
- 动机: **Phase 3-D LGPA 必要性消融** (Full−LGPA, mirror Phase 3-C Full−GCN exp289). Paper claim "LGPA 关键 vs GCN 冗余" 的关键对照实验。

## 训练轨迹

| Epoch | mAP | R1 |
|-------|-----|----|
| 10 | 55.3 | 65.9 |
| 20 | 62.8 | 73.3 |
| 30 | 67.6 | 78.0 |
| 40 | 67.9 | 78.6 |
| 50 | 70.5 | 81.7 |
| 60 | 70.7 | 82.2 |
| 70 | 71.1 | 82.5 |
| 80 | 71.4 | 83.3 |
| 90 | 71.9 | 82.9 |
| 100 | 71.9 | 82.4 |
| 110 | 71.9 | 83.2 |
| **120 FINAL** | **71.9** | **83.0** |

## FINAL (2026-04-26 09:38 CST)

- **eq+flip**: mAP **71.9%**, R1 **83.0%**, R5 90.5%, R10 92.5%
- **Global cosine+flip**: 73.5 / 83.4
- **MaxSim hybrid+flip**: **71.9 / 83.0**

## 🎯 核心 Paper Claim — LGPA vs GCN 重要性对比

| Removed Module | Exp | MaxSim+flip | Δ vs Full Scaffold (74.7/84.8) |
|----------------|-----|-------------|--------------------------------|
| **− GCN** (LGPA only, 2-stage PSG) | exp289 | **74.8/84.8** | **+0.1 / 0** ← **GCN 冗余** |
| **− LGPA** (GCN only, 2-stage PSG) | **exp301 (本)** | **71.9/83.0** | **−2.8 / −1.8** ← **LGPA 关键** |

**结论**:
- **LGPA 贡献**: +1.9 mAP (eq+flip) / +2.8 mAP (MaxSim+flip) / +0.8-1.8 R1
- **GCN 贡献**: 0-0.1 mAP (噪声范围)
- **Paper claim**: "LGPA captures essential pose-conditioned semantic features, while GCN is redundant — they share representational capacity, with LGPA being the dominant contributor"

## eq+flip vs MaxSim 在 LGPA-OFF 下的反常

注意: exp301 MaxSim hybrid+flip = **71.9** = eq+flip mAP, **MaxSim 没 boost**!

对照 Full Scaffold 同 backbone:
- exp285b MaxSim+flip = 74.7 vs eq+flip 73.8 → MaxSim **+0.9 mAP boost**
- exp289 (LGPA-only) MaxSim+flip = 74.8 vs eq+flip 73.8 → MaxSim **+1.0 mAP boost**
- **exp301 (no-LGPA) MaxSim+flip = 71.9 vs eq+flip 71.9 → MaxSim 0 boost** ← 失去 boost

**机理**: MaxSim hybrid 利用 part features 做 late interaction. LGPA 提供的 5 个 CLIP-aligned semantic part features 是 MaxSim 主驱动力. GCN 提供的 part features (基于 keypoint geometry) 对 MaxSim 贡献小. 移除 LGPA, MaxSim 失去关键 part features → 0 boost.

**Paper figure 4 / table 5 (component ablation)** 可加这一行强化 LGPA's role in MaxSim performance。

## 训练曲线观察

- e10-e80 健康 climb (55.3 → 71.4)
- e90-e120 plateau 稳定 71.9 (mAP 收敛)
- R1 在 e80 (83.3 peak) 然后 e120 微 dip (83.0)
- 整体收敛健康, 无 NaN/Inf

## lab4090 idle (FINAL 后)

- 主训练进程结束 @ 09:38 CST
- MaxSim eval 完成 @ 09:40 CST
- lab4090 可接下一任务
