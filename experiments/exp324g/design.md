# 实验 exp324g: rank-disagreement oracle 诊断（0-训练）

## 动机
- "DINO 补 Swin" 整条创新家族（frozen DINOv2 pose-part-MaxSim 给 75-mAP SOTA Swin 补遮挡信息）需要一个 gate：
  **在重遮挡 query 上，DINO part-MaxSim 是否携带 Swin 漏掉的独立正确信息？**
- 前序 exp324f 的 late-fusion sweep（w∈{0,…,0.5}）只能回答"线性融合是否涨点"，
  无法区分"DINO 没有独立信息" vs "有但融合方式不对"。本实验用 **per-query oracle 上界**
  直接测信息上限：如果连 oracle（每 query 取 max AP）都不涨，整条家族止损。
- 0-训练、纯诊断，复用已有资产，不开新训练、不 commit。

## 核心假设
- 一句话：若 DINO 与 Swin 正交（Jaccard 低）且 oracle mAP 显著高于 Swin-only，
  说明 DINO 在重遮挡上有 Swin 拿不到的独立正确信息，值得做遮挡门控 re-rank。

## 技术方案
- 新增 `scripts/exp324g_oracle.py`，**复用 exp324f 的全部管线**：
  - `exp324f_fuse.get_dino_distmat`：从 exp324b 缓存部位特征 + head_60.pth 算 DINO part-MaxSim distmat（无 DINO 前向，几乎不吃 GPU）。
  - `exp324f_fuse.align_dino_to_swin`：按文件名把 DINO distmat permute 到 Swin 顺序，校验 pid/camid。
  - `exp324f_fuse.compute_heavy_mask`：query pose vis_binary.sum()<=8 的重遮挡 mask。
  - Swin distmat **直接复用** 已 dump 的 `experiments/exp324f/swin_distmat.npz`（exp255 MaxSim+flip，mAP=75.16），不再重跑 Swin 前向。
- 数据流：load Swin npz → 算 DINO distmat → align → dump 两份 npz 到 experiments/exp324g/ →
  在 heavy 子集上 numpy 算三个量：
  a. **top-10 Jaccard**：Swin top-10 vs DINO top-10 gallery 索引（排 same-pid-same-cam 后），均值。
  b. **P_dino_only**：heavy query 里 "DINO rank-1 命中真值(同 pid 非同 cam) 且 Swin rank-1 没命中" 的比例。
  c. **per-query oracle 上界**：每个 heavy query 取 max(Swin AP, DINO AP)，均值 = oracle mAP；
     对比 Swin-only / DINO-only heavy mAP。per-query AP 与 `eval_func` 内层逐行一致（同排除、同 junk 跳过）。
- 关键超参：topk=10（Jaccard），HEAVY_OCC_THR=8（沿用 exp324）。

## 预期结果
- 若 DINO 有独立信息：Jaccard 低（<0.5）、P_dino_only 较高、oracle mAP 明显高于 Swin-only。
- 若失败（最可能）：oracle 几乎不涨（DINO 在 Swin 已对的 query 上也对、在 Swin 错的 query 上也错），
  说明 DINO 部位特征判别力太弱（exp324b DINO-only 绝对 mAP 极低）。

## 对照组
- 对照 baseline：Swin-only heavy mAP（exp255 MaxSim distmat 在 heavy 子集上的 mAP）。
- 消融变量：只加 "DINO 信息是否可用" 这一维，融合策略不参与（用 oracle 上界绕过融合方式问题）。

## 判定（明确写出）
- P_dino_only<2% 或 oracle-gain<+1mAP → "DINO⊕Swin 无独立信息，整条家族止损"。
- oracle +3~5 且 Jaccard<0.5 → "正交性坐实，值得做 #2 遮挡门控 re-rank"。
- 其余 → 模糊区，按幅度判断。
