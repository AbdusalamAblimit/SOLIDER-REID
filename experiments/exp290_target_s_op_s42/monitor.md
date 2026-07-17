# exp290 monitor — Small Full Scaffold + target-heatmap on Occ-PTrack (seed 42)

- 机器: srvB (5060Ti 16G, TEST.IMS_PER_BATCH 128)
- 启动: 2026-04-22 12:00 CST
- Log: `/tmp/exp290.log` + `/hy-tmp/log/occluded_posetrack/exp290_target_s_op_s42/train_log.txt`
- Config: `configs/occluded_posetrack/prcv_best_small.yml` + CLI `SOLVER.SEED 42 MODEL.POSE_USE_TARGET_HEATMAP True TEST.IMS_PER_BATCH 128`
- Scaffold: Swin-Small + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + PLBOA) + 2-stage PSG `[-2,-1]` + **target-heatmap swap**
- Speed: ~10.5 min/epoch, 总训练 21h22min

## 训练轨迹 (flip-test, eq_concat global)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 74.7 | 82.9 | 93.8 | 96.6 |
| 20 | 76.2 | 84.4 | 94.2 | 96.7 |
| 30 | 77.1 | 85.0 | 94.3 | 97.1 |
| 40 | 77.7 | 85.6 | 94.6 | 97.0 |
| 50 | 78.0 | 86.1 | 94.8 | 97.2 |
| 60 | 78.1 | 86.0 | 94.7 | 97.4 |
| 70 | 78.3 | 86.1 | 95.0 | 97.5 |
| 80 | 78.3 | 85.9 | 95.0 | 97.4 |
| 90 | 78.3 | 86.1 | 94.9 | 97.4 |
| 100 | 78.4 | 86.1 | 94.9 | 97.4 |
| 110 | 78.4 | 86.2 | 94.9 | 97.4 |
| **120 FINAL** | **78.4** | **86.2** | **94.8** | **97.4** |

## FINAL (2026-04-23 09:22:01 CST)

- **mAP: 78.4%**, **Rank-1: 86.2%**, R5: 94.8%, R10: 97.4%
- 🔥 **对照 exp265 scene baseline s42 FINAL**: 78.4 / 86.2 / 94.8 / 97.3 → Δ **0 / 0 / 0 / +0.1**
- **target-heatmap 严格持平 scene baseline** on OP!
- 对照 exp265b s41 FINAL (scene): 78.5/85.9/94.7/97.1 → Δ -0.1/+0.3/+0.1/+0.3
- Ckpt: `transformer_120.pth` (~237MB)

## 🎯 target-heatmap 3 数据集完整收尾

| Dataset | target mAP/R1 | scene baseline mAP/R1 | Δ mAP / Δ R1 |
|---------|---------------|----------------------|--------------|
| Occ-Duke (exp291) | 73.5/82.9 | exp285b 73.8/83.8 | -0.3 / -0.9 |
| **Occ-PTrack (exp290)** | **78.4/86.2** | exp265 78.4/86.2 | **0 / 0** ✓ 严格持平 |
| Market (exp292 e90 eff) | 94.2/97.1 | exp268 94.3/97.3 | -0.1 / -0.2 |

**target-heatmap 机制最终定位**:
- 3 dataset 跨 multi-person / mixed / all-single 均 **near no-op** (|Δ| ≤ 0.3 mAP, |Δ| ≤ 0.9 R1)
- 原假设 "OP 多人 SOTA 推动" **未实现** (尽管理论上合理 — KPR-with-prompt 82.3 表明可以, 但我们机制不足以达到)
- **论文定位**: supplementary 消融, 证明机制 backward-compat + 语义正确性, 不 claim 主创新
- 主表 Small OP 数字仍用 **exp265 scene baseline 78.4/86.2** (= exp290 数字, 两者等价)

## auto-chain → exp266c (Base OP s42 full 120 restart)

daemon 109773 detected `transformer_120.pth` @ 09:21 CST, 等 exp290 进程退出。预期 launch @ ~09:25。
- config: `configs/occluded_posetrack/prcv_best_base.yml`
- SEED 42, TEST.IMS_PER_BATCH 64
- 预期 FINAL ~15:00 CST today
