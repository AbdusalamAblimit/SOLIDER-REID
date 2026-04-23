# exp293 monitor — Base Full Scaffold + PLBOA on Market-1501 (seed 42)

- 机器: lab4090 (24GB 4090, mmpose-abu env)
- 启动: 2026-04-22 18:14 CST (auto-chain from exp291 FINAL via daemon 706372)
- Log: `/tmp/exp293.log` + `/home/afr/SOLIDER-REID/log/market1501/exp293_best_b_m_plboa_s42/train_log.txt`
- Config: `configs/market/prcv_best_base.yml` + CLI `SOLVER.SEED 42 MODEL.POSE_LOWER_BODY_OCC True`
- Scaffold: Swin-Base + Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + **PLBOA 启用** + 2-stage PSG) — 动机: Market 历史实验都关 PLBOA, OA-SD 蒸馏无差异 teacher/student; 本实验激活 PLBOA 看是否推 Market Base SOTA
- Speed: ~208-297s/epoch (fluctuating)

## 训练轨迹 (原 First Run, OOM 截断 @ e80)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 85.3 | 93.3 | - | 98.8 |
| 20 | 89.7 | 95.4 | - | 99.0 |
| 30 | 90.9 | 96.0 | - | 99.1 |
| 40 | 92.6 | 96.9 | - | 99.3 |
| 50 | 93.0 | 96.9 | - | 99.3 |
| 60 | 93.5 | 97.3 | - | 99.5 |
| 70 | 93.9 | 97.1 | - | 99.4 |
| 80 (OOM eff) | **94.1** | **96.9** | - | - |

## 训练轨迹 (Restart Full 120 Epoch, TEST.IMS_PER_BATCH 64)

| Epoch | mAP | R1 | R5 | R10 |
|-------|-----|----|----|----|
| 10 | 70.4 | 85.6 | 94.8 | 96.9 |
| 20 | 88.6 | 95.3 | 98.1 | 99.0 |
| 30 | 91.0 | 96.1 | 98.4 | 99.2 |
| 40 | 92.2 | 96.9 | 98.5 | 99.1 |
| 50 | 92.1 | 96.6 | 98.6 | 99.2 |
| 60 | 92.9 | 96.9 | 98.8 | 99.5 |
| 70 | 93.4 | 97.1 | 99.0 | 99.5 |
| 80 | 93.6 | 97.2 | 98.9 | 99.4 |
| 90 | 93.7 | 97.4 | 98.9 | 99.5 |
| 100 | 93.8 | 97.3 | 99.0 | 99.5 |
| 110 | 93.8 | 97.3 | 98.9 | 99.5 |
| **120 FINAL** | **93.8** | **97.2** | **98.9** | **99.5** |

## FINAL (Restart full 120, 2026-04-23 08:24:32 CST)

- **mAP: 93.8%**, **Rank-1: 97.2%**, R5: 98.9%, R10: 99.5%
- **对照 exp269 Base Market PLBOA OFF e80 eff**: 94.4/97.0 → Δ **-0.6/+0.2** (mAP 损 0.6, R1 +0.2)
- **对照 exp293 first run e80 eff** (同 PLBOA ON): 94.1/96.9 → Δ **-0.3/+0.3** (同 config 跨 restart 方差 0.3 mAP)
- **对照 exp268 Small Market**: 94.3/97.3 → Δ -0.5/-0.1

**Cross-domain Market → Occluded-ReID (见 Plan A/B 后续 eval)**:
- exp293 e80 eff ckpt (first run): Global 69.4/73.6, MaxSim **72.4/76.7** — PLBOA 灾难
- 后续可 eval 新 e120 ckpt 看是否缓解 (训练完整后泛化可能更差)

## OOM Crash @ e80 flip-test eval (2026-04-22 23:07 CST)

训练 e79 正常 (Loss 3.69 Acc 0.98), e80 iter 完整 (186/186), 进入 flip-test eval 时 CUDA OOM:

```
Epoch[80] Iter[180/186] Loss: 3.686, Acc: 0.981
Epoch 80 done. Time per epoch: 297.240[s]
Traceback:
File "processor/processor.py", line 67, in _extract_feat_flip
File "model/backbones/swin_transformer.py", line 702, attn = (q @ k.transpose(-2, -1))
torch.cuda.OutOfMemoryError: Tried 658 MiB, 373 MiB free, 12.61 GiB reserved PyTorch
```

**诊断**: 80 epoch 累积内存碎片 + TEST.IMS_PER_BATCH 256 默认过大。Market flip-test 有 22k+ images (19k gallery + 3k query), 两遍 forward (flip + original) × 256 batch 峰值超内存。

**修复**: `transformer_80.pth` ckpt 完整保存, 用 `scripts/eval_fliptest_maxsim.py` + `TEST.IMS_PER_BATCH 64` 独立重跑 e80 eval, 成功出数字。

## e80 eff FINAL eval (独立 test.py 跑, 2026-04-22 23:16 CST)

- **Global cosine+flip**: mAP **94.1%**, R1 **96.9%**
- **MaxSim hybrid+flip**: mAP **94.1%**, R1 **97.2%**

## 对照 exp269 Base Market PLBOA OFF

| | exp293 PLBOA ON (e80 eff) | exp269 PLBOA OFF (e80 eff FINAL) | Δ |
|---|---------------------------|------------------------------------|---|
| mAP (Global+flip) | 94.1 | 94.4 | -0.3 |
| R1 (Global+flip) | 96.9 | 97.0 | -0.1 |
| mAP (MaxSim+flip) | 94.1 | (未跑) | — |
| R1 (MaxSim+flip) | 97.2 | — | — |

**PLBOA 在 Market 上 net effect: -0.3 mAP / -0.1 R1 (Global+flip) — 轻微 net negative**

假设验证: 
- ✗ 不是 **>94.5** 正向增益 (OA-SD 收益 < 分布偏差)
- ✓ 是 **持平 94.2±0.3** 情景 (两力相抵, 微 net 负)
- ✗ 不是 **<94.0** 大回撤

## 论文定位

- main_results Table 1 Base Market 主数字仍用 **exp269 94.4/97.0** (PLBOA OFF, 完整 80 epoch eff)
- exp293 作 **supplementary ablation**: "PLBOA on Market 分布不匹配, 不建议默认开启"
- **OA-SD 依赖 PLBOA 问题** 已识别并记录在 decisions.md

## 训练轨迹分析

PLBOA ON 相比 PLBOA OFF (exp269):
- e10-e30: -2.3 → -1.7 → -1.8 mAP (严重 gap)
- e40: -0.9 mAP, R1 首次反超 (+0.1)
- e50-70: gap 稳在 -0.5 ~ -0.9 mAP, R1 持平或微优
- e80: -0.3 mAP, R1 -0.1 (Global) / +0.2 (MaxSim)

**OA-SD 蒸馏中后期 partial 抵消 PLBOA 分布偏差, 但未能完全抵消**。

## lab4090 状态

- e80 OOM crash 后 lab4090 idle (22:30 CST)
- 不 restart (接受 e80 eff FINAL)
- 下一任务待用户指示
