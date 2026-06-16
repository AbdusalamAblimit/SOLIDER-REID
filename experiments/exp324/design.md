# 实验 exp324: DINOv2 Emergent Correspondence — pose-anchored part-MaxSim 廉价首验

> **来源**：post-PRCV「搬范式」路线。`paradigm-import-survey` 排名 #2（并行/退路，最稳）。
> **性质**：**training-free 廉价首验**（frozen DINOv2，零训练）。与 exp323 同周并行、不抢 GPU。
> **机器**：lab-3090-d（DINOv2-B via hf-mirror + pose data 现成 + RTX 3090 idle）。

## 动机

- **搬范式**：DINOv2/v3 是当下最热的 SSL 基础模型，其 **emergent dense correspondence**（无监督涌现的跨图密集对应）是核心卖点。
- **绿地 niche**：PersonViT 等只把 DINO 当"更好的全局 backbone"——**没人用 DINO 的 emergent correspondence 做跨图 part 对齐**。换 backbone = me-too；用 correspondence = 新机制。zero-shot 纯 DINOv2 在 ReID 仅 0.3-4.7 mAP，说明"天生对应"无人显式接 occluded part 匹配。
- **我们的咬合**：MaxSim 天然吃 DINO dense tokens（late-interaction，非全局池化）；pose+5-part 把 DINO 在低分辨率人物上 noisy 的 correspondence 约束到 body-part 语义上降噪；只比 mutually-visible part 对症遮挡。24G 最稳、不依赖大模型、最复用现成 MaxSim 全套。

## 核心假设

frozen DINOv2-B dense patch tokens，按 pose 锚定成 5 个 body-part 表征（带 per-part visibility），跨图只比 mutually-visible part 的 part-MaxSim，在**重遮挡子集**上**超过 holistic 基线**（DINO 全局 cosine 以及/或 exp255 holistic）——全程 training-free。

## 技术方案（training-free）

1. **DINO 特征**：frozen DINOv2-B（facebook/dinov2-base）对 query+gallery 图抽 dense patch tokens（patch 14×14 grid，对 224 输入 = 16×16 tokens；ReID 图 resize 到 DINO 友好分辨率，如 224×224 或 keep-aspect 224×112 padded）。
2. **pose 锚定 5-part**：用 `pose_data/{query,gallery}/<img>.npz` 的 keypoints(17,2)+visibility，把 17 COCO keypoint 投到 DINO patch grid，按 LGPA 5-part 分组（head / upper-torso+arms / lower-torso / legs / feet 之类，复用 `model/modules/pose_part_pooling.py` 的分组）池化成 5 个 part 向量 + per-part visibility mask（该 part 有无可见 keypoint）。
3. **跨图 part-MaxSim**：每个 (query,gallery) pair，只在 **mutually-visible** 的 part 上做 part-level cosine + MaxSim 聚合（复用 `utils/metrics.py` 的 MaxSim 逻辑思路；present-part 归一化）。
4. **eval**：mAP/R1 在 (a) 全 query，(b) 重遮挡子集（visibility_binary.sum 低，同 exp323 口径）。
5. **对照基线**：① DINO 全局 cls/mean-pool cosine（holistic DINO）；② exp255 holistic（equal_concat）；③ DINO part-MaxSim **不带 pose 锚定**（均匀网格 part，证 pose 锚定的必要）。

## 预期结果

- 假设成立：DINO pose-anchored part-MaxSim 在重遮挡子集超 holistic DINO + 接近/超 exp255，且 pose 锚定 > 均匀网格 → emergent correspondence + pose 降噪有效 → 上轻量 part-projection 头训练。
- 失败最可能原因：(1) DINO correspondence 在 128×256 低分辨率脏 crop 上漂移严重，pose 降噪不够；(2) part-MaxSim 不带训练直接 eval 太弱（DINO 特征非 ReID-judiciable）；(3) 只在整体涨、重遮挡组不涨。任一即降级。

## 对照组

- baseline = holistic DINO 全局 cosine（无 part、无 pose）。treatment = pose-anchored part-MaxSim。
- 关键消融：pose-anchored vs 均匀网格 part（隔离 pose 锚定贡献）；mutually-visible-only vs 全 part（隔离遮挡处理贡献）。

## Kill-switch / 下一步

- 重遮挡组超 holistic 且 pose 锚定有效 → exp324b：轻量 part-projection 头 / LoRA 微调 DINO，全量评测 vs KPR。
- 否则 → 降级（或退到 DIFT/SD 特征对应，survey 候选 5）。
