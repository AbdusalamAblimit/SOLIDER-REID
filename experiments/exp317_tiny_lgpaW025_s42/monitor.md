# exp317_tiny_lgpaW025_s42 — Tiny OD Full + POSE_LGPA_ASSIGN_WEIGHT 0.25

- 机器: lab3090 docker (`18fbbab202e1`)
- Config: `prcv_best_tiny.yml` + CLI `SOLVER.SEED 42 MODEL.POSE_LGPA_ASSIGN_WEIGHT 0.25 TEST.IMS_PER_BATCH 64`
- 启动: 2026-04-28 02:37 server, FINAL: 09:57 (~7.3h)
- 动机: LGPA aux 减半 (0.5 → 0.25), 测降低 LGPA 监督是否给 backbone 更自由的优化空间

## FINAL (e120)

- **eq+flip**: mAP **66.2%**, R1 **77.4%**, R5 87.2%, R10 89.8%
- **Global cosine+flip**: 66.0 / 76.3
- **MaxSim hybrid+flip**: **67.4 / 78.6** ⭐

## 训练轨迹 (eq+flip)

| Epoch | mAP | R1 |
|-------|-----|-----|
| 10 | 44.2 | 56.9 |
| 20 | 54.5 | 67.6 |
| 30 | 59.3 | 72.0 |
| 40 | 61.7 | 73.5 |
| 50 | 63.3 | 74.6 |
| 60 | 64.0 | 75.7 |
| 70 | 64.8 | 75.7 |
| 80 | 65.1 | 76.7 |
| 90 | 66.1 | 77.9 |
| 100 | 66.0 | 77.5 |
| 110 | 66.1 | 77.4 |
| **120 FINAL** | **66.2** | **77.4** |

## 对照 vs exp261 baseline (default lgpaW=0.5)

| 指标 | exp317 (lgpaW=0.25) | exp261 baseline | Δ |
|------|---------------------|------------------|----|
| eq+flip | 66.2/77.4 | 65.9/77.4 | **+0.3/0** |
| Global+flip | 66.0/76.3 | 65.8/76.0 | +0.2/+0.3 |
| **MaxSim** | **67.4/78.6** | **67.2/78.6** | **+0.2/0** ⭐ |

## 结论

**LGPA_ASSIGN_WEIGHT 0.25 (减半 default) net positive +0.2 mAP MaxSim** on Tiny。第一个 sweep 中 MaxSim 超 baseline 的结果。

**论文意义**: 微小 (+0.2 mAP) 可能在 multi-seed std (0.42-0.45) 内, 不能 strong claim。但如果 Small/Base 上重现, 可写为 "LGPA assignment supervision 不需要太强, 0.25× weight 给 backbone 更多 freedom"。

## 后续验证

需要在 Small s1234 上验证 (exp321) 才能确定 +0.2 是真实 improvement 还是 seed noise。
