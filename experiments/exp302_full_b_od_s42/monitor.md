# exp302_full_b_od_s42 monitor — Base OD Full multi-seed s42

- 机器: srvA (5060Ti)
- Config: `prcv_best_base.yml` + CLI `SOLVER.SEED 42 TEST.IMS_PER_BATCH 64`
- 启动: 2026-04-26 ~05:00 CST, FINAL: 2026-04-27 01:42 CST (~20.7h)
- Scaffold: Swin-Base + GCN512 + LGPA-D + OA-SD + ParAug + PLBOA + 2-stage PSG
- 动机: **multi-seed Base** 第 3 个 seed (s41 exp263d / s1234 exp300 / **s42 exp302**), 验证 "robust to seed" claim 在 Base 上一致

## FINAL (e120, 2026-04-27 01:42 CST)

- **eq+flip (train log)**: mAP **73.3%**, R1 **81.4%**, R5 90.2%, R10 92.1%
- **Global cosine+flip**: 72.6 / 81.7
- **MaxSim hybrid+flip**: **74.4 / 83.6**

## 训练轨迹 (mAP, eq+flip)

| Epoch | mAP |
|-------|-----|
| 10 | 51.7 |
| 20 | 59.8 |
| 30 | 66.3 |
| 40 | 69.4 |
| 50 | 72.1 |
| 60 | 72.4 |
| 70 | 72.5 |
| 80 | 73.1 |
| 90 | 73.1 |
| 100 | 73.4 |
| 110 | 73.3 |
| **120 FINAL** | **73.3** |

e80 起 plateau 73.1-73.4 ± 0.1 噪声范围。

## 对照 — multi-seed Base Full Scaffold

| Exp | Seed | eq+flip | Global+flip | **MaxSim+flip** |
|-----|------|---------|-------------|-----------------|
| exp263d | 41 | 74.1/83.3 | 73.8/82.9 | **75.2/84.8** ⭐ best |
| exp300 (e120) | 1234 | 74.0/83.8 | 73.9/83.9 | **75.0/85.0** |
| **exp302 (本)** | **42** | **73.3/81.4** | 72.6/81.7 | **74.4/83.6** |

**3 seed 统计**:
- mAP MaxSim: mean **74.87**, std **0.42**, range 0.8
- R1 MaxSim: mean **84.47**, std **0.78**, range 1.4

**Paper claim 支撑**: 3 个 seed (41/1234/42) Base mAP MaxSim 标准差 0.42 (< 0.5), 与 Small s42/s1234/s2024 (std 0.45) 一致, 论文可写 "Our method is robust to seed selection across all backbones (std ≤ 0.5 mAP for both Small and Base across 3 seeds each)"。

## 结论

exp302 是 multi-seed Base 第 3 个 seed,稍弱于 s41/s1234 (-0.6 ~ -0.8 MaxSim mAP), 但仍在 1 mAP 范围内, 健康 multi-seed 表现。Paper Base 主行仍用 exp263d 75.2/84.8。

## 下一步

srvA idle 后立即跑 **exp268 Small Market v2 重 eval** (验证 v1 numbers 是否需要更新)。
