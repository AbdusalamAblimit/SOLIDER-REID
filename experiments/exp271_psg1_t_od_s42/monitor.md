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

## 自动化状态

- queue_on_ckpt.sh daemon 52096 已完成 exp270→exp271 转场使命,即将退出
- srvB 后续无 daemon,exp271 完成后需人工挂新 daemon 接 exp272

## 预期 ETA

- Tiny + PSG 单模块 ~88-90s/epoch (比 pure Tiny 80s 慢 10%)
- 120 epoch ≈ 3h
- 预计 2026-04-20 ~15:30 完成
