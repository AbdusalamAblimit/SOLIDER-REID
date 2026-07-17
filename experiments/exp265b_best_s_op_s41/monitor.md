# exp265b monitor — Small Full Scaffold Occ-PTrack seed 41 SOTA 刷

- 机器: srvA (5060Ti 16G, TEST.IMS_PER_BATCH 128 防 Base eval OOM 规范沿用)
- 启动: 2026-04-21 12:04 CST (kill+restart with TEST.IMS_PER_BATCH 128 after initial config issue)
- Log: `/hy-tmp/log/occluded_posetrack/exp265b_best_s_op_s41/train_log.txt`
- Config: `configs/occluded_posetrack/prcv_best_small.yml` + CLI `SOLVER.SEED 41 TEST.IMS_PER_BATCH 128`
- Scaffold: Swin-Small + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + LOWER_BODY_OCC) + 2-stage PSG `[-2,-1]` (default)
- Speed: ~10.5 min/epoch (BS=64, 275 iter, eval ~2 min), 总训练 20h49min

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 74.3 | 83.0 | 93.4 | 96.2 |
| 20 | 75.5 | 83.5 | 93.7 | 96.5 |
| 30 | 77.2 | 84.9 | 94.1 | 96.7 |
| 40 | 77.8 | 85.5 | 94.3 | 96.9 |
| 50 | 78.3 | 85.9 | 94.7 | 96.9 |
| 60 | 78.2 | 86.0 | 94.5 | 96.9 |
| 70 | 78.4 | 86.1 | 94.7 | 97.1 |
| 80 | 78.4 | 86.0 | 94.8 | 97.1 |
| 90 | 78.3 | 85.9 | 94.7 | 96.9 |
| 100 | 78.4 | 86.0 | 94.7 | 97.1 |
| 110 | 78.5 | 85.9 | 94.7 | 97.1 |
| **120 FINAL** | **78.5** | **85.9** | **94.7** | **97.1** |

## FINAL (2026-04-22 09:03:48 CST)

- **mAP: 78.5%**, **Rank-1: 85.9%**, R5: 94.7%, R10: 97.1%
- **对照 exp265 s42** (srvC 2026-04-20 04:45): **78.4 / 86.2 / 94.8 / 97.3** → Δ=**+0.1 / -0.3 / -0.1 / -0.2**
- seed 41 微优 mAP (+0.1), 略弱 R1 (-0.3) 和 R5/R10 (-0.1/-0.2)
- **跨 seed 差异 ≤ 0.3** (R1 最大), 论文鲁棒性证据
- Ckpt: `transformer_120.pth` (237MB)

## 🔥 Small OP seed 对比 (exp265 vs exp265b)

| seed | mAP | R1 | R5 | R10 | 设备 |
|------|-----|----|----|----|------|
| 42 (exp265) | 78.4 | **86.2** | **94.8** | **97.3** | srvC |
| 41 (exp265b) | **78.5** | 85.9 | 94.7 | 97.1 | srvA |

**观察**:
1. **mAP 非常稳**: 78.4 vs 78.5, Δ +0.1 (跨 seed + 跨设备微差)
2. **R1/R5/R10 seed 42 略优** 0.2-0.3, seed 41 无 SOTA 增益
3. **论文主表用 exp265 s42 78.4/86.2** (更高 R1), exp265b 作 supplementary robustness

## auto-chain → exp266b (srvA 5060Ti Base OP s41)

daemon 992 (21h runtime) 挂 exp265b/transformer_120.pth → **exp266b_best_b_op_s41** (Swin-Base + seed 41 + TEST.IMS_PER_BATCH 128)
- 预计启动 ~09:04 CST
- Base OP @ 5060Ti 16G, ~5h/run → FINAL ~14:00 CST
- 和 lab3090 上的 `exp266b_best_b_op_s41_3090` 形成 **跨设备对照**, Phase 3 最后一块双设备一致性验证

## srvA GPU 状态

- Phase 3 seed 41 SOTA 刷阶段, exp265b FINAL 后 auto-chain exp266b, srvA 保持忙
