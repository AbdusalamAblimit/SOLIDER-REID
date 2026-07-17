# exp273 monitor — Phase 3-A 3-stage PSG (Swin-Tiny @ Occ-Duke, seed 42)

- 机器: srvB
- 启动: 2026-04-20 20:19 CST (auto-chain from exp272 via queue_on_ckpt daemon 62026)
- Log: `/hy-tmp/log/occluded_duke/exp273_psg3_t_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_tiny.yml` + CLI override (PSG_STAGES=`[-3,-2,-1]`)
- Scaffold: Swin-Tiny + PSG 3-stage (LGPA/GCN/OA-SD/PLBOA/ParAug 全关)
- Speed: ~99s/epoch, 120 epoch 总时长 3h46min (20:19→00:05)

## 对照(Phase 3-A 矩阵)

| Exp | PSG stages | FINAL mAP/R1 |
|-----|-----------|-------------|
| exp270 | 无 | 59.2 / 68.4 |
| exp271 | `[-1]` | 60.2 / 69.5 |
| exp272 | `[-2,-1]` | 60.5 / 69.7 |
| **exp273 (本)** | `[-3,-2,-1]` | **60.5 / 69.9** |

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 | 备注 vs exp272 同期 |
|-------|-----|----|----|----|---------------------|
| 30 | 50.0 | 58.6 | 72.4 | 78.1 | ~-1.8 (vs exp272 e30 预估 51.8) |
| 40 | 53.6 | 63.4 | 76.5 | 80.9 | - |
| 50 | 56.5 | 65.8 | 80.0 | 84.4 | - |
| 60 | 56.7 | 66.4 | 79.2 | 84.4 | - |
| 70 | 58.4 | 68.2 | 80.5 | 85.4 | +0.6/+0.7 超 exp272 |
| 80 | 59.3 | 69.0 | 81.8 | 86.2 | +0.2/+0.9 超 |
| 90 | 60.4 | 69.8 | 83.3 | 87.3 | **+0.5/+1.3** ⬆️ |
| 100 | 60.7 | 69.9 | 82.9 | 86.7 | +0.3/0 |
| 110 | 60.4 | 69.8 | 82.6 | 86.7 | +0.1/+0.2 |
| **120 FINAL** | **60.5** | **69.9** | **82.8** | **87.0** | **+0/+0.2** 持平 mAP,R1 微涨 |

## FINAL (2026-04-21 00:05 CST)

- **mAP: 60.5%**, **Rank-1: 69.9%**, Rank-5: 82.8%, Rank-10: 87.0%
- 对照:
  - exp270 no-PSG: 59.2/68.4 → exp273 Δ=**+1.3/+1.5** (3-stage 累计贡献)
  - exp271 1-stage: 60.2/69.5 → exp273 Δ=+0.3/+0.4
  - **exp272 2-stage: 60.5/69.7 → exp273 Δ=0/+0.2** (stage 1 边际贡献接近 0 mAP, R1 微涨)
- Ckpt: `/hy-tmp/log/occluded_duke/exp273_psg3_t_od_s42/transformer_120.pth` (113MB)

## 结论

### Phase 3-A Tiny 矩阵完整结果

| PSG stages | mAP | R1 | vs no-PSG |
|-----------|-----|----|-----------|
| 无 | 59.2 | 68.4 | baseline |
| `[-1]` | 60.2 | 69.5 | +1.0/+1.1 |
| `[-2,-1]` | 60.5 | 69.7 | +1.3/+1.3 |
| `[-3,-2,-1]` | **60.5** | **69.9** | **+1.3/+1.5** |

**边际收益递减**: 无→1 (+1.0), 1→2 (+0.3), 2→3 (**0 mAP, +0.2 R1**)

**论文论述 (Table 2)**:
- "2-stage PSG 达到 mAP peak, 3-stage 不再提供额外 mAP 收益"
- "3-stage 在 R1/R10 上微有提升 (69.7→69.9 R1, 86.2→87.0 R10), 但代价是多一层注入复杂度"
- "**Default 选 2-stage**: 足够抓住 PSG 主要收益, 保持实现简洁"

### 与历史 exp009 (Tiny 2+3 stage 旧协议 58.3/67.2) 对比

新协议下 PSG 增益模式变化: **3-stage 不再劣于 2-stage** (旧协议下 3-stage ≤ 2-stage), 新协议下 3-stage ≈ 2-stage。可能因为:
1. flip-test + default 协议差异带来的整体 baseline 提升
2. seed 42 特定轨迹

### 下一步

- Phase 3-A Tiny **全部 4 runs FINAL**, 等 Phase 3-A Small 4 runs (exp274/275 FINAL, exp276/277 running/queued) 完成构建完整 Table 2
- srvB slot auto-chain → exp278 (Phase 3-B Tiny 首 run, GCN256 + 1-stage)
- daemon 70447 已 detect ckpt @ 00:04, 等 previous run 全 worker 退出后启动 exp278
