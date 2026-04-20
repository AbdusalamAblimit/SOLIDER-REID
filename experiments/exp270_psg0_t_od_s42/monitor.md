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

## FINAL (e120) — 2026-04-20 12:29:59 srvB

- **mAP: 59.2%**
- **CMC Rank-1: 68.4%**
- CMC Rank-5: 82.2%
- CMC Rank-10: 85.8%
- ckpt: `/hy-tmp/log/occluded_duke/exp270_psg0_t_od_s42/transformer_120.pth`

### 轨迹

| Epoch | mAP | R-1 |
|-------|-----|-----|
| 10 | 36.2 | 45.5 |
| 20 | 41.9 | 50.0 |
| 30 | 47.3 | 56.5 |
| 40 | 53.2 | 62.9 |
| 50 | 55.4 | 64.3 |
| 60 | 57.7 | 66.7 |
| 70 | 58.2 | 67.7 |
| 80 | 58.3 | 67.9 |
| 90 | 58.7 | 68.1 |
| 100 | 59.1 | 69.0 |
| 110 | 59.0 | 68.4 |
| **120** | **59.2** | **68.4** |

最后 3 次 eval (e100/110/120) mAP 平台 59.0-59.2,R1 68.4-69.0。收敛稳定。

### 对照

- **exp000 SOLIDER-Tiny 旧协议 (无 default flip)**: 56.6 / 66.5 → 本 run 加 default flip **+2.6 / +1.9**,新协议默认 flip-test 的 baseline 加成验证
- 本 run 是 Phase 3-A "no PSG" 基线,用于对比 exp271 (1-stage PSG)、exp272 (2-stage)、exp273 (3-stage)

### 结论

- Swin-Tiny (pure baseline, no pose module, default flip-test) @ Occluded-Duke = **59.2 / 68.4**
- 这个数字是 Phase 3-A "无 PSG" 对照基线,等 exp271-273 出完,就能量化 PSG 各 stage 配置的独立贡献

### 后续

- queue_on_ckpt.sh daemon 52096 于 12:30:xx 自动起 **exp271 (1-stage PSG Tiny OD, PID 53178)**,预计 2026-04-20 ~14:30 完成(Tiny+PSG 略慢于 pure Tiny)
