# 实验 exp063: Pose-Token Distillation (PTD)

## 动机
- 当前 PSG+GCN 需要 **推理时也提供 pose heatmap**（ViTPose 推理开销 ~40ms/image）
- 如果能在训练时用 pose 热图监督可学习的 part tokens，推理时就不需要 pose → 降低部署开销
- PAFormer (arXiv 2024) 和 SapiensID (CVPR 2025) 展示了类似思路的可行性
- 这是一个**范式转移**：从 "pose as input" 到 "pose as supervision"

## 创新点 / 核心想法
- **核心假设**: K 个可学习 part tokens 通过 cross-attention 从 backbone 特征中提取 part-level 信息。训练时用 pose heatmap MSE 监督 attention maps 使 tokens 定位到正确的身体部位。推理时 tokens 自行定位（不需要 pose）。
- **与 GCN 的区别**: GCN 需要 pose 做 bilinear sampling（runtime pose）；PTD 的 part tokens 是 learned（no runtime pose）
- **与 XCAD (exp053) 的区别**: XCAD 用 pose 做 query input；PTD 用 pose 做 attention supervision（训练 loss，非 forward path）
- **论文 story**: "Pose-Free Occluded Person ReID via Pose-Distilled Part Tokens"

## 技术方案
- 新模块 `PoseTokenDecoder`:
  - K=5 learnable part tokens (head, upper, lower, left, right)
  - 2-layer cross-attention: Q=part_tokens, K/V=Stage3_features
  - attention map supervision: MSE(attn_map, heatmap_part_gt) during training
  - 输出: K×768-d part features
- 训练时: PSG + PoseTokenDecoder + heatmap supervision
  - ID loss: 对 part tokens 的 pooled feature 做 classification
  - Triplet loss: 对 part features
  - Heatmap loss: MSE between attention weights and grouped heatmap
- 推理时（关键）: PSG + PoseTokenDecoder（无 pose 输入）
  - Part tokens 自行 attend to features
  - 与 global feature concat 做匹配

## 关键超参数
- K = 5 (body part groups)
- Cross-attention dim: 256
- Heatmap loss weight: 1.0
- Decoder layers: 2

## 预期结果
- 如果成功: PTD 无 pose ≈ GCN 有 pose → 大幅降低推理开销，论文 story 极强
- 如果失败: part tokens 无法仅靠 heatmap supervision 学会定位 → 退化为 random tokens
- 需要对照: PTD with pose (upper bound) vs PTD without pose (contribution) vs GCN (baseline)

## 对照组
- exp030a (PSG+GCN, 有 pose): 60.73%/72.57%
- exp063 (PSG+PTD, 无 pose 推理): 预期 59-61%
