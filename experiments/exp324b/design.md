# 实验 exp324b: 冻结 DINOv2 + 轻量投影头 + 姿态部位匹配

> **来源**：exp324 frozen 首验通过 kill-switch（姿态部位匹配在重遮挡上 ×3.4 mAP 超整图，且均匀网格不涨→涨点全来自姿态锚定）。本实验验证"训一个轻量头能否把冻结 DINO 部位特征拉到可用/可打 KPR"。
> **性质**：**训练实验**。开训前必须过 Claude broad review + Codex review（hook 阻断）。
> **机器**：lab-3090-d（DINOv2-base 已下、exp324 已缓存部位特征、RTX 3090 idle）。

## 动机

- exp324（frozen，无训练）已证：用姿态把 DINOv2 dense token 锚定到 5 个身体部位、跨图只比双方可见部位（part-MaxSim），在 Occluded-Duke 重遮挡 query 上 mAP 1.86 vs 整图 0.55（×3.4），且**均匀网格对照只 0.67（几乎不涨）→ 涨点几乎全来自"姿态锚定"机制本身**（单变量隔离干净）。
- 但绝对分低（1.86 mAP），符合 DINO 零样本 ReID 文献（0.3-4.7）。training-free 不可用。
- 与 exp323 对照：同样 frozen + 同样 pose，**MLLM-reasoning 那条无信号、DINO dense-correspondence 这条有信号**——差别在"特征表示端"而非"大模型推理端"。下一步应在特征端使劲：用最小训练把冻结特征投到 ReID 判别空间。

## 核心假设

冻结 DINOv2-base、**仅训一个轻量 per-part 投影头**（先线性，必要时小 MLP），用 ID 分类 + triplet loss 训练、用 mutually-visible part-MaxSim 匹配，能把重遮挡 mAP 从 1.86 **大幅拉高**（先看能否进入"几十分"量级证明冻结特征有救），同时保留"姿态锚定 + 只比可见部位"的机制优势。

## 技术方案

- **数据流**：图 → 冻结 DINOv2-base dense tokens → 姿态锚定 5 部位池化 + per-part visibility（exp324 已实现）→ **轻量共享线性投影头** 768→512 + BNNeck → 全局特征 = 可见部位投影向量的 masked mean；测试用 mutually-visible part-MaxSim 出 distmat。
- **损失（最终实现，见双审查修正）**：全局 ID CE（BNNeck 特征）+ 全局 batch-hard soft-margin triplet（pre-BN 全局特征）+ **per-part 共享 ID CE**（每个可见部位投影向量，权重 `--part_weight` 默认 0.5，=0 为 global-only 消融臂）。
- **冻结边界**：DINO **不反传**（不解冻 backbone，避免大成本 + 防过拟合 15K 数据）；只训投影头 + 两个分类器。**在缓存特征上训** → 每 step 无 backbone forward，飞快。
- **修改文件**：新增独立训练脚本 `scripts/exp324b_train_head.py`（不碰 model/ 核心、不碰 train.py）：train/query/gallery 部位特征**全部经同一 DINO 流程补抽并缓存到 `experiments/exp324b/_cache_train/`**（exp324 缓存的是 raw patch grid 非池化部位，不能直接复用）→ 训头 → 评测。
- **关键超参（先最小）**：投影头 = 线性（per-part 独立 vs 共享，先共享）；投影维 D=512；ID loss + triplet（权重 1:1）；优化器 SGD/Adam，LR 小、epoch 少（缓存特征训得快，先 30-60 epoch 看趋势）；**BS=64（项目硬约束，不可改）**；mutually-visible part-MaxSim 同 exp324。
- **评测**：`test.py` 风格（永不用 train.py 评估）。mAP/R1，全量 + 重遮挡子集（vis≤8）。

## 预期结果

- 假设成立：重遮挡 mAP 从 1.86 显著上升（若进入几十分量级 → 冻结特征有救、方向成立 → 再加容量/解冻部分 DINO 冲 SOTA）。
- 失败最可能：冻结 DINO 特征 ReID 判别力有天花板，轻量头拉不动（停在个位数）→ 则下一步换 LoRA 解冻部分 DINO，或换更强 DINO（large/v3）；若仍不动则这条线天花板低、止损。

## 对照组

- **baseline**：exp324 frozen 无训练（重遮挡 1.86 / 全量 3.21）。
- **消融**：投影头 on 姿态部位 vs on 均匀网格（训练后再证姿态锚定贡献仍在）；线性头 vs 小 MLP（容量敏感性）；ID-only vs ID+triplet。
- **对标**：最终对 KPR（Occluded-Duke）；与项目 Swin 主线（75 mAP）做"frontier-independent 新表征"定位（不必正面超，讲机制新颖 + 参数/范式角度）。

## 双审查修正（2026-06-16）

**两个审查（Claude broad review + Codex）独立收敛到同一个 High 阻断项**（也是设计时自己存疑的点）：

> **训练/测试空间不一致**：原代码只用"全局 masked-mean 特征"监督投影头（ID CE + triplet 都作用在全局），但主指标 part-MaxSim 用的是"逐部位 L2-归一化向量的余弦"。只优化平均向量 → 逐部位向量可能仍弱判别 → part-MaxSim 训不上来、甚至低于 frozen baseline → **会被误判成"冻结特征天花板低"而错误止损**。

**修复（已应用）**：加 **per-part 共享 ID 分类头**，对每个可见部位的投影向量直接做 CE（监督落在 part-MaxSim 测试的同一空间），权重 `--part_weight`（默认 0.5，**设 0 即 global-only 消融臂** → 可做 global-only vs +part-loss 对照，正是审查建议的隔离实验）。最终损失 = 全局 ID CE + 全局 soft-margin triplet + part_weight × per-part ID CE。
**顺带修的非阻断项**：加 cosine LR 衰减（原无 scheduler）；BN 参数排除 weight decay（repo 惯例）；PKSampler num_batches ≥1 守卫；移除 dead `CACHE_QG`。
**成功口径**：主看 **part-MaxSim 重遮挡子集** vs exp324 frozen(1.86)；全局 cosine 同时报告作对照。审查确认 triplet/采样/eval/同cam排除/边界 全部正确。

## 协议待办（开训前硬性）

1. [x] exp324 frozen 首验通过 kill-switch
2. [x] 写训练脚本 `scripts/exp324b_train_head.py`（+ 补抽 train 部位特征）
3. [x] **Claude broad review** → 需修改(H1) → 已修
4. [x] **Codex review** → needs-attention(同 H1) → 已修
5. [ ] 修复后**重跑双审查至 approve** → `claude_review.md` + `codex_review.md`
6. [ ] 双审查 approve 后才开训（hook 阻断）
