# exp324g monitor — rank-disagreement oracle（0-训练诊断）

## 运行环境
- 机器：lab-3090-d（`/root/work/SOLIDER-REID`），系统 python3（无 mmcv，纯 torch+numpy）。
- 与 exp324d LoRA 共享 GPU：本诊断**无 DINO 前向**（只用缓存部位特征 + head_60.pth 做投影 + numpy），几乎不吃 GPU，安全共享。
- 资产复用：
  - Swin distmat 直接复用 exp324f 已 dump 的 `experiments/exp324f/swin_distmat.npz`（exp255 Swin MaxSim+flip，**mAP=75.16 R1=85.57**，未再跑 Swin 前向）。
  - DINO distmat 由 `exp324f_fuse.get_dino_distmat` 从 exp324b 缓存 + `head_60.pth` 算出。
  - align / heavy mask 全部复用 `exp324f_fuse`。

## 对齐 sanity（先确认再算 oracle）
- 两个 distmat 形状均 **(2210, 17661)**，按文件名对齐到 Swin 顺序；pid 逐一校验通过，camid 偏移恒定（DINO 1-indexed vs Swin 0-indexed，offset=1，一致）。
- DINO part-MaxSim ALONE（全 query）：mAP=14.61 R1=21.99（与 exp324f 报的 DINO-only 数一致，确认 head/缓存加载正确）。
- 重遮挡子集：**989/2210** query（pose vis_binary.sum() ≤ 8）。989 个全部有有效真值（n_valid=989）。

## 三个核心量（重遮挡子集，top-10）

| 量 | 值 |
|----|----|
| (a) top-10 Jaccard（Swin vs DINO，排 same-cam 后） | **0.0619** |
| (b) P_dino_only（DINO r1 命中真值 ∧ Swin r1 没命中） | **0.20%**（2/989） |
| — Swin-only r1 命中 ∧ DINO 没命中 | 72.70%（719/989） |
| — both r1 命中 | 114；neither r1 命中 154 |
| (c) Swin-only heavy mAP | **72.57** |
| (c) DINO-only heavy mAP | **8.65** |
| (c) ORACLE(max per-query) heavy mAP | **72.69** |
| — oracle gain over Swin | **+0.12 mAP** |

## 判定（明确）
**STOP-LOSS：DINO⊕Swin 在重遮挡上没有 Swin 漏掉的独立正确信息。整条"DINO 补 Swin"家族止损。**

触发条件（两条都中，任一即止损）：
- P_dino_only = 0.20% < 2%
- oracle gain = +0.12 mAP < +1 mAP

## 机理解读（为什么 Jaccard 0.06 ≠ 正交有用）
- Jaccard 极低（0.06）看似"正交"，但这是**虚假正交**：DINO part-MaxSim 整体判别力太弱
  （DINO-only heavy mAP 仅 8.65 vs Swin 72.57，差 ~64 mAP），它的 top-10 基本是噪声，
  和 Swin 自然不重叠——但不重叠不代表"对"。
- 真正的 kill 证据是 **per-query oracle 上界**：即便允许每个 query 取 Swin/DINO 中更好的那个 AP，
  heavy mAP 也只从 72.57 升到 72.69（+0.12）。说明在 **Swin 失败的 query 上 DINO 几乎也失败**
  （DINO-only r1 命中而 Swin 漏的只有 2 个 query）。DINO 没有补充信息可言。
- 结论：late-fusion（exp324f）涨不动，不是融合方式不对，而是**信息上限本身不存在**。
  LoRA 微调（exp324d）若不能把 DINO part 判别力从 8.65 大幅拉高到接近 Swin，这条线无救。

## 产物
- `experiments/exp324g/swin_distmat.npz`、`dino_distmat.npz`（self-contained 对齐副本，远程）
- `experiments/exp324g/oracle_summary.json`（已拉回本地）
- 脚本：`scripts/exp324g_oracle.py`（复用 exp324f 管线，新增三个 oracle 量 + 单 query AP，逐行对齐 eval_func）
