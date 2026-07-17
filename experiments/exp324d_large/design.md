# 实验 exp324d_large: LoRA-解冻 DINOv2-large + pose-anchored part-MaxSim

## 动机
- FM（foundation model）方向唯一活口的彻底测试。冻结 DINO 部位匹配天花板低：
  重遮挡 mAP **8.65** / 全部 **14.61**（exp324b，冻结 DINOv2-base + 433K 轻量 head，e20 触顶）。
  Swin baseline 72.57。换特征源全死（DIFT 0.73、registers +0.29）。只剩 **LoRA 解冻**。
- 核心问题：FM-adaptation（让 backbone 经 ReID 目标自适应）能否让 DINO 特征判别化、
  把重遮挡 mAP 明显推过 8.65 往 competitive（几十分）走？
- 本变体测**最强 FM**：DINOv2-large（hidden 1024，比 base 容量大）+ LoRA。

## 核心假设
- 冻结天花板低是因为 backbone 从未见过 ReID 目标；用 LoRA 让更大容量的 large backbone
  在 q/v attention 上低秩自适应，dense token 判别性会显著提升，突破 14/8.65 天花板。

## 技术方案
- 复用 lab-3090-d 的 `scripts/exp324d_lora.py`（dual-reviewed），rsync 到 hyy。
  - 可微部位池化：预计算 per-image sparse pool matrix `pool_w (NPARTS, 512)`（pose+3x3 窗口
    选 cell，行归一），loop 内 `parts = bmm(pool_w, patch)` 可微 → 梯度到 LoRA。等价 exp324
    build_part_pose 的 mean-over-cells，只是写成 matmul。
  - LoRA 注入 DINOv2 attention 的 query/value，主权重冻结，gradient_checkpointing(use_reentrant=False)。
  - 损失/采样/eval 全部与 exp324b 相同：global ID CE + soft-margin batch-hard triplet +
    part_weight 0.5 per-part ID CE；PK sampler P16×K4=64；eval = global cosine + 互见 part-MaxSim，
    ALL + heavy-occ(vis<=8) 两片。
- **本次改动**（最小，已 review）：
  - `build_lora_dino` 加 `model_name` 参数，返回 `(model, hidden)`，从 `base.config.hidden_size` 读 hidden。
  - `main` 加 `--dino_model` flag，`PartHead(in_dim=hidden)` 用真实 hidden（large=1024，不是模块常量 768）。
  - `ROOT` 改为读 `EXP324_ROOT` 环境变量（hyy 路径 `/hy-tmp/reid-clean/SOLIDER-REID`）。
  - `dino_patch_forward`/`pool_parts_diff` HIDDEN-agnostic（slice + bmm），无需改。

## 数据 / 缓存复用
- pool matrix 缓存（geometry-only，patch-grid 16×32 空间，base/large 通用，均 patch14@224×448）
  从 lab-3090-d rsync：train_pool_n15618 / query_pool_n2210 / gallery_pool_n17661。
- 图像在 hyy `/hy-tmp/reid-clean/data/Occluded_Duke`，symlink 进 repo data/occluded_duke。
- query/gallery pose_data 在 hyy（heavy_mask 用）；train pose 缺失但有 train pool 缓存 → 不调 find_pose。

## 超参
- DINOv2-large, hidden 1024, LoRA rank16/alpha16, lora_lr 1e-4, head_lr 3.5e-4。
- micro_bs 16（峰值 5.34G，远低于 16G），grad_ckpt on，有效 BS=64（全 batch triplet 不变）。
- 30 epoch，eval_period 5，cosine LR。

## 预期结果
- 假设成立：重遮挡 mAP 从 8.65 明显上升（>20 即有信号，>40 才算 competitive）。
- 失败最可能原因：pose-anchored 5-part 表征上限就低（part-MaxSim 信息瓶颈），
  或 LoRA 容量不足以让 frozen DINO 判别化；冻结天花板是表征结构问题而非 adaptation 问题。

## 对照组
- 冻结基线 exp324b：重遮挡 8.65 / 全部 14.61。
- 平行变体 exp324d_r32（base + LoRA rank32，GPU1）；base-rank16 在 lab-3090-d 跑着。
- 消融变量：backbone 容量（large vs base）+ LoRA 是否能突破冻结。
