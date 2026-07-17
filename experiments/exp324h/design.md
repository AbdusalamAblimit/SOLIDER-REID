# 实验 exp324h: adapted-DINO (LoRA) 是否对 SOTA Swin 有独立信息（ORACLE 探针）

## 动机
- exp324g oracle 已证：**冻结 DINO** 对 75-mAP Swin 无独立信息
  （P_dino_only=0.20%（2/989）、oracle 上界 +0.12 mAP、Jaccard 0.062、DINO-only heavy mAP 8.65）
  → 冻结融合死路。
- **但 exp324d 让 DINO 经 LoRA adaptation 变判别了**：base-r16 重遮挡 part-MaxSim mAP
  8.65（frozen e20）→ 36.78（LoRA e10）。
- 不同 backbone（DINO vs Swin）→ 不同错误模式 → adapted-DINO **可能反而互补 Swin**
  （冻结时不互补只是因为太弱）。这是没测过的方法角度，便宜且高价值。

## 核心假设
- 一句话：LoRA-adapted DINO 的 part-MaxSim 检索在重遮挡 query 上携带 Swin 漏掉的独立正确信息
  （oracle 上界 >> +0.12、P_dino_only >> 0.2%）。若成立，融合可能 beat Swin 75 = 真方法结果。

## 技术方案
- **不训练、不 commit**。复用 exp324d/f/g 全部 plumbing：
  - `build_lora_dino` 的 base + `PeftModel.from_pretrained(base, lora_10/)` 加载 e10 LoRA 权重
  - `head_10.pth` 加载 PartHead（与 LoRA 同一 e10）
  - `prepare_split` 复用已缓存的 query/gallery pooling 矩阵（_cache 已有 n2210 / n17661）
  - `encode_split`（exp324d）前向 DINO(LoRA)+head → L2-norm 5-part 向量
  - `part_maxsim_distmat`（exp324_dino）→ adapted-DINO part-MaxSim distmat（Q×G）
- 对齐：`align_dino_to_swin`（exp324f）按 filename 把 DINO 排列到 Swin 顺序，校验 pid/camid。
- **oracle 数学逐行复用 exp324g**：`topk_excluded`（top-10 Jaccard）、`per_query_ap`
  （per-query AP 镜像 eval_func，给 P_dino_only 与 oracle 上界）。
- heavy mask：exp324g 用 `compute_heavy_mask(swin q_names)`（vis≤8），与 exp324d 同源同阈值，已核对等价。
- **(若 oracle 正)** 顺手 fusion sweep（z-score / min-max，w∈{0..0.5}）+ k-reciprocal re-rank
  （`utils/reranking.py` re_ranking with local_distmat），看重遮挡/全部能否 > Swin 75 单独。

## 数据流
image → DINO(LoRA e10) patch grid → 可微 pose part pooling（缓存矩阵）→ PartHead(e10)
→ L2-norm 5-part → part-MaxSim distmat（adapted-DINO）→ 对齐 Swin 顺序 → oracle 对照。

## 关键超参/选择依据
- 用 e10 checkpoint（当前已落盘的最新；base-r16 仍在 309591 跑，**不 kill，并行用 ckpt**）。
- topk=10（与 exp324g 一致，可直接对照冻结基线）。

## 预期结果
- **若 adapted-DINO 互补**：oracle 上界 >> +0.12（如 +2~+5），P_dino_only >> 0.2%，
  Jaccard 仍偏低（错误模式不同）→ 融合/re-rank 有望 beat 75。
- **若仍冗余**：oracle ~+0.1、P_only ~0.2% → adapted-DINO 也被 Swin 包含，确认 analysis 结论。
- 失败最可能原因：adapted-DINO 虽变强（36 mAP）但其"对的"样本仍是 Swin 也对的简单样本，
  重遮漏检集合高度重叠 → oracle 上界不动。

## 对照组
- 对照 baseline：**exp324g 冻结 DINO**（oracle 上界 +0.12、P_only 0.20%、Jaccard 0.062）。
- 单变量：唯一变化 = DINO frozen → LoRA-adapted（head/几何/eval/heavy mask/Swin distmat 全不变）。
