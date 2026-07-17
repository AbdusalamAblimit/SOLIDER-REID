# exp324f 监控 / 结果：DINO 姿态部位对应 ⊕ exp255 Swin 融合（eval-only）

## 运行环境与流程（两阶段，npz 桥接）

lab-3090-d 无单一 python 同时有 mmengine（SOLIDER swin）与 transformers（DINO），故拆两阶段：
- **Stage 1**（`solider-reid` conda env, torch1.13）：`scripts/exp324f_swin_distmat.py` 跑 exp255 Swin（`exp255_small_gcn512_2stage/transformer_120.pth`），flip-test + MaxSim，dump `experiments/exp324f/swin_distmat.npz`（d_swin + q/g 文件名/pid/camid）。
- **Stage 2**（系统 python3, torch2.7）：`scripts/exp324f_fuse.py` load npz + exp324b 缓存部位特征 + `head_60.pth` 出 d_dino，按文件名对齐、归一化、扫 w 融合、eval。

## Sanity 校验（全部通过）

- **Swin MaxSim ALONE: mAP=75.16 R1=85.57 R5=91.86 R10=93.17** —— 与 exp255 主线 75.2 一致 ✓
- **DINO part-MaxSim ALONE: mAP=14.61 R1=21.99** —— 与 exp324b e60 完全一致 ✓
- **对齐**：2210 query / 17661 gallery 按文件名 join，pid 全等校验通过；camid 偏移恒为 +1（SOLIDER 0-indexed vs exp324 1-indexed，同相机），eval 全程用 Swin（0-indexed）camid ✓
- **w=0 = 纯 Swin**：ALL 75.16 / HEAVY 72.57，与单独完全一致 ✓
- 重遮挡子集：989/2210 query（pose vis≤8）

## 融合扫描结果（d = (1-w)·d_swin_norm + w·d_dino_norm）

### zscore 归一化
| w | ALL mAP | ALL R1 | HEAVY mAP | HEAVY R1 | ALL Δ | HEAVY Δ |
|---|---------|--------|-----------|----------|-------|---------|
| 0.00 | 75.16 | 85.57 | 72.57 | 84.23 | +0.00 | +0.00 |
| 0.10 | 75.14 | 84.93 | 72.43 | 83.32 | -0.02 | **-0.14** |
| 0.20 | 74.87 | 84.84 | 72.03 | 82.71 | -0.29 | -0.54 |
| 0.30 | 74.23 | 84.16 | 71.21 | 82.00 | -0.93 | -1.36 |
| 0.40 | 72.87 | 82.62 | 69.51 | 79.47 | -2.29 | -3.06 |
| 0.50 | 70.41 | 79.86 | 66.47 | 75.83 | -4.75 | -6.10 |

### minmax 归一化（稳健性对照，结论相同）
| w | ALL mAP | HEAVY mAP | HEAVY Δ |
|---|---------|-----------|---------|
| 0.00 | 75.16 | 72.57 | +0.00 |
| 0.10 | 75.13 | 72.42 | -0.15 |
| 0.50 | 70.22 | 66.22 | -6.35 |

## 结论（NEGATIVE，clean）

**融合在重遮挡子集上没有 > Swin 单独，反而从 w=0.1 起就单调变差**（HEAVY -0.14 @ w=0.1 → -6.10 @ w=0.5），ALL 同向单调下降。两种归一化结论一致。

判断：
- exp255 Swin 重遮挡子集已经很强（72.57 mAP，本身用 PSG/LGPA/GCN 处理遮挡）。
- 冻结 DINO part-MaxSim 信号（14.61 全部 / 8.65 重遮挡，量级远低于 Swin）**噪声太大**，对 SOTA Swin 是**严格冗余且有害**的距离源，任何正权重都拖垮排序。
- "DINO 对应给 SOTA 模型补遮挡鲁棒性"这条**晚融合 / score-level fusion 路线证伪**。冻结 DINO 距离矩阵与 Swin 不互补（至少在 score-level 简单加权下）。

止损：score-level 融合死路。若 DINO 这条还想救，只能在**特征/表征端**而非距离端注入（成本远高，且 exp324b 已显示冻结特征天花板 14），优先级低。下一步看 exp325（更强冻结 backbone 能否抬天花板）的天花板结论再定 DINO 线整体去留。

## 附：rank-disagreement oracle（planner #1，0-GPU，`scripts/exp324f_oracle.py`）

为坐实"为什么融合死"，跑了 oracle（与并行 agent exp324g 独立各跑一次，数字逐位一致）：

重遮挡子集（989/2210，全有效，top-10）：
| 量 | 值 | 解读 |
|----|----|------|
| Swin-alone mAP / R1 | 72.57 / 84.23 | baseline |
| DINO-alone mAP / R1 | 8.65 / 11.73 | 太弱 |
| top-10 retrieved-PID Jaccard | 0.088 | 低，但**虚假正交** |
| **P_dino_rescue**（Swin r1 错、DINO r1 对）| **0.20%**（2/989） | DINO 几乎救不了 Swin |
| P_swin_rescue（DINO r1 错、Swin r1 对）| **72.70%** | Swin 大量救 DINO（极不对称）|
| both-right / both-wrong | 11.5% / 15.6% | — |
| **ORACLE best-of-both mAP** | **72.69（Δ +0.12）** | 每 query 取更优 AP 的上界只 +0.12 |

ALL queries：oracle Δ +0.12，P_dino_rescue 0.63%，Jaccard 0.143，P_swin_rescue 64.21%。

**KILL-SWITCH 双中**（P_dino_rescue 0.20% < 2% 且 oracle gain +0.12 < +1 mAP）→ **整条"DINO⊕Swin"家族止损**。机理：Jaccard 0.06–0.09 是冻结 DINO 整体太弱（差 64 mAP）导致 top-10 基本噪声的**虚假正交**，不重叠≠命中；决定性证据是 **per-query best-of-both 上界都只 +0.12** + P_swin/P_dino rescue 极度不对称（72.7% vs 0.20%）——**Swin 失败的 query 上 DINO 也失败**，信息上限不存在。融合涨不动不是融合方式问题。planner #2（遮挡门控 re-rank）依赖 #1 正向 → 已砍；#3/#5 OT 线（只改聚合）同理降到底。
