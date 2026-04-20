# exp270 monitor — Phase 3-A no-PSG baseline (Swin-Tiny @ Occ-Duke, seed 42)

- 机器: srvB (exp263 OOM 后空闲,srvB 现重用)
- 启动: 2026-04-20 09:30 (main PID 46956)
- Log: /hy-tmp/log/occluded_duke/exp270_psg0_t_od_s42/train_log.txt
- Config: configs/occluded_duke/prcv_best_tiny.yml + CLI override 关所有 pose 模块
- Scaffold: **pure Swin-Tiny** (no PSG / LGPA / GCN / OA-SD / PLBOA / Parallel-Aug)

## 对照(Phase 3-A 矩阵)

Phase 3-A 矩阵:
- exp270 (本) — no PSG
- exp271 — 1-stage PSG
- exp272 — 2-stage PSG
- exp273 — 3-stage PSG

目标: 用 exp270 vs exp271-273 量化 PSG 本体贡献,回答 "PSG 在 pure backbone setting 下是否稳定正增益"。

## 预期

- 历史 exp000 baseline (Tiny, SW=0.2, no flip, 旧协议) = 56.6/66.5
- 本 run 加 default flip-test 期望 +0.5-0.9 → **~57/67**

## 中间 eval

| Epoch | mAP | R-1 | 备注 |
|-------|-----|-----|------|
| e1 | 待出 | — | 启动中 |

## 时间

- 单 epoch ~4min (Tiny 无 pose)
- 120 epoch ≈ 8h → 预计 2026-04-20 ~17:30 完成
- 完成后 srvB 接 Phase 3-A 下个: **exp271 (1-stage PSG Tiny OD)**

## 下一步计划

exp270 完成后 → exp271/272/273 顺序起(同 Tiny OD + 只改 PSG_STAGES),srvB 大约跑到 2026-04-21 早晨完成 Phase 3-A Tiny 全 4 格。
