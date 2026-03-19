# 实验 exp106: PISD (Pose-Informed Self-Distillation at Image Level)

## 动机
- 为什么要做？
  - PACD v3/v4 证明: feature map 级 masking 在 12×4 上无效（GAP 太鲁棒）
  - ROA (+1.27%) 证明: IMAGE 级遮挡增强有效（backbone 被迫从部分输入学习）
  - **PISD = ROA 的自蒸馏升级版**: 不仅用遮挡图训练 ID loss，还要求遮挡图特征 ≈ 全图特征
- 基于哪些前序实验？
  - exp104 PACD: feature map masking 太弱 → 必须在 IMAGE 级别做
  - exp067 ROA: image-level 遮挡增强有效 → PISD 在此基础上加自蒸馏

## 创新点
- 核心假设：**在图像级别用 pose heatmap 遮蔽身体部位 + 要求 backbone 在遮挡图上产出与全图相似的特征，可以训练出遮挡不变的表示**
- 与 ROA 的区别：ROA 粘贴 VOC 物体（随机遮挡），PISD 精确遮蔽身体部位（pose-guided）+ 自蒸馏
- 与 PACD 的区别：PACD 在 feature map 级别 mask（太弱），PISD 在 IMAGE 级别 mask（强信号）

## 技术方案
- 修改 `processor/processor.py`:
  1. 正常 forward → feat_full (teacher, detached)
  2. 用 pose heatmap 在 INPUT IMAGE 上遮蔽随机身体部位
  3. 遮蔽图 → backbone forward → feat_partial (student)
  4. Loss: cosine(feat_partial, feat_full)
- 开销：2 次 forward pass（但只有 student 需要 backward）
- 关键超参数：mask_ratio=0.4, weight=0.3, warmup=10

## 预期结果
- 如果成立：mAP +0.5~1.5%（backbone 学到遮挡不变表示）
- 风险：2 次 forward 可能导致 GPU OOM 或训练时间翻倍

## 对照组
- exp066 PAA: 61.6%/74.2%
- exp067 ROA: 62.0%/73.7%
