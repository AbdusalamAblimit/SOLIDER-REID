# exp272 monitor — Phase 3-A 2-stage PSG (Swin-Tiny @ Occ-Duke, seed 42)

- 机器: srvB (auto-chain from exp271 @ 2026-04-20 16:37 via queue_on_ckpt daemon)
- PID 60400, NUM_WORKERS=8 (workers 跟随主进程退出)
- Log: `/hy-tmp/log/occluded_duke/exp272_psg2_t_od_s42/train_log.txt`
- Config: `configs/occluded_duke/prcv_best_tiny.yml` + CLI override (PSG_STAGES=`[-2,-1]`)
- Scaffold: Swin-Tiny + PSG 2-stage (LGPA/GCN/OA-SD/PLBOA/ParAug 全关)
- 速度: ~98s/epoch,120 epoch ~3h18min

## 训练轨迹 (flip-test,eq_concat global)

| Epoch | mAP | R1 | R5 | R10 | 备注 |
|-------|-----|----|----|-----|------|
| 70 | 59.0 | 67.5 | 80.5 | 85.8 | 小落后 exp271 同期 |
| 80 | 59.1 | 68.1 | 80.8 | 85.4 | 持平 |
| 90 | 59.9 | 68.5 | 81.9 | 86.2 | 追平 |
| 100 | 60.4 | **69.9** | 82.4 | 86.2 | peak R1,超 exp271 (60.2/69.5) |
| 110 | 60.3 | 69.6 | 82.6 | 86.0 | 微跌 |
| **120 FINAL** | **60.5** | **69.7** | **82.6** | **86.2** | e100 R1 之后轻微回撤但 mAP 继续涨 |

## FINAL (20:19:49 CST)

- **mAP: 60.5%**, **Rank-1: 69.7%**, Rank-5: 82.6%, Rank-10: 86.2%
- 对照:
  - exp270 no-PSG: 59.2/68.4 → exp272 Δ=+1.3/+1.3 (PSG 2-stage 累计贡献)
  - exp271 1-stage: 60.2/69.5 → exp272 Δ=+0.3/+0.2 (加入 stage 2 的边际贡献)
- Ckpt: `/hy-tmp/log/occluded_duke/exp272_psg2_t_od_s42/transformer_120.pth` (113MB)

## 结论

- **2-stage ≥ 1-stage**,但边际收益仅 +0.3 mAP,**符合历史 exp007 vs exp009 旧协议轨迹** (~持平)
- Peak R1 在 e100 达 69.9,e120 回撤到 69.7,体现尾部 LR 低训练噪声
- Phase 3-A Tiny 矩阵剩 exp273 (3-stage) 验证:是否 3-stage 会继续升 or plateau
- 已自动 auto-chain 到 **exp273**,queue_on_ckpt daemon 62026 已触发 @ ~20:19,当前 e29/120 ETA ~23:45
