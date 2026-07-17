# exp271 monitor — Phase 3-A 1-stage PSG (Swin-Tiny @ Occ-Duke, seed 42)

- 机器: srvB
- 启动: 2026-04-20 12:30:xx (auto-chained from exp270 via queue_on_ckpt daemon 52096,main PID 53178)
- Log: /hy-tmp/log/occluded_duke/exp271_psg1_t_od_s42/train_log.txt
- Config: configs/occluded_duke/prcv_best_tiny.yml + CLI override
- Scaffold: **Swin-Tiny + PSG stage 3 only** (LGPA/GCN/OA-SD/PLBOA/ParAug 全部关)

## 对照(Phase 3-A 矩阵)

- exp270 (no PSG) FINAL: 59.2 / 68.4 ← 基线
- **exp271 (本,1-stage PSG)**: 预期 60-61 (+0.8-1.5)
- exp272 (2-stage PSG): 将来跑
- exp273 (3-stage PSG): 将来跑

核心: exp270 vs exp271 的 mAP 差就是 PSG stage 3 的独立贡献。

## 历史参考

- exp007 (Tiny + PSG stage 3, 旧协议 no flip): 58.3 / 67.9
- 本 run 新协议 default flip-test 预期 +0.8-1.5 → **~59-60**
- 相对 exp270 基线 59.2/68.4 预期 **+1-2 mAP**

## 中间 eval

| Epoch | mAP | R-1 | 备注 |
|-------|-----|-----|------|
| e1 | 冷启动 | — | 刚 launch |

## FINAL (2026-04-20 16:36:44)

**e120 eval 结果 (eq_concat + default flip-test)**:

| Metric | 值 |
|--------|-----|
| mAP | **60.2%** |
| Rank-1 | **69.5%** |
| Rank-5 | 81.8% |
| Rank-10 | 85.9% |

Epoch 98-120 速度: ~95-98s/epoch (预期吻合)。3h 完整跑完。

## 与对照组对比

| Exp | PSG stages | mAP | R1 | Δ vs no-PSG |
|-----|-----------|-----|-----|-----|
| exp270 | 无 (baseline) | 59.2 | 68.4 | — |
| **exp271** (本) | `[-1]` (1-stage) | **60.2** | **69.5** | **+1.0 / +1.1** |

## 结论

PSG 单 stage 注入 (stage 3) 在 Tiny + Occ-Duke 纯骨架下确认有正向贡献 **+1.0 mAP / +1.1 R1**,与历史 exp007 (旧协议 58.3/67.9) + default flip 贡献(+2.0 mAP) = 预期 60.3 基本一致。

下一步 exp272 (2-stage PSG) 将验证 stage 2+3 vs 单 stage 3 是否进一步增益。

## 自动化状态

- queue_on_ckpt.sh daemon 52096 已完成 exp270→exp271 转场使命,已退出
- exp272 auto-chained via queue_on_ckpt daemon 53447 (launched 16:37:25, main PID 60400)
- exp272→exp273 daemon 待挂
