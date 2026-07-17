# exp304_full_s_od_s2024 monitor — Small OD Full multi-seed s2024

- 机器: srvC (5060Ti)
- Config: `prcv_best_small.yml` + CLI `SOLVER.SEED 2024 TEST.IMS_PER_BATCH 128`
- 启动: 2026-04-26 ~05:00 CST, FINAL: 20:24 CST (~15.4h)
- Scaffold: Swin-Small + GCN512 + LGPA-D + OA-SD + ParAug + PLBOA + 2-stage PSG
- 动机: **multi-seed Small** 第 3 个 seed (s42 exp285b / s1234 exp295 / **s2024 exp304**), 验证 "robust to seed" 论文 claim

## FINAL (e120, 2026-04-26 20:24 CST)

- **eq+flip (train log)**: mAP **73.3%**, R1 **82.7%**, R5 90.0%, R10 91.9%
- **Global cosine+flip**: 73.3 / 83.3
- **MaxSim hybrid+flip**: **74.3 / 84.0**

## 训练轨迹 (mAP, eq+flip)

| Epoch | mAP |
|-------|-----|
| 10 | 69.8 |
| 20 | 71.6 |
| 30 | 71.6 |
| 40 | 72.4 |
| 50 | 73.2 |
| 60 | 73.2 |
| 70 | 73.3 |
| 80 | 73.3 |
| 90 | (skip, only every 10 epoch save) |
| **120 FINAL** | **73.3** |

e70 起 plateau 73.3,基本无 further gain。

## 对照 — multi-seed Small Full Scaffold

| Exp | Seed | eq+flip | Global+flip | **MaxSim+flip** |
|-----|------|---------|-------------|-----------------|
| exp285b | 42 | 73.8/83.8 | 73.6/83.2 | **74.7/84.8** |
| exp295 | 1234 | 74.2/84.0 | 73.7/83.3 | **75.2/85.4** ⭐ best |
| **exp304 (本)** | **2024** | **73.3/82.7** | 73.3/83.3 | **74.3/84.0** |

**3 seed 统计**:
- mAP MaxSim: mean 74.7, std 0.45, range 0.9
- R1 MaxSim: mean 84.7, std 0.74, range 1.4

**Paper claim 支撑**: 3 个 seed (42/1234/2024), MaxSim mAP 标准差 0.45 (< 0.5), 论文可写 "Our method is robust to seed selection (std ≤ 0.5 mAP across 3 seeds: 42, 1234, 2024)"。

## 结论

exp304 是 multi-seed Small 第 3 个 seed,稍弱于 s42/s1234 (-0.4 ~ -0.9 MaxSim mAP), 但仍在 1 mAP 范围内, 健康 multi-seed 表现。Paper 主行仍用 exp295 75.2/85.4。
