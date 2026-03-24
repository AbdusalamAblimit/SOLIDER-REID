# 实验 exp167: Dual-Path Token Learning (DPTL)

## 动机
- exp166 per-token classification 改善了 R1（+1.2 over V1 mean-pool），通过强制每个 token 独立有判别力
- 但当前 6 个 structural tokens 完全独立，永远不与彼此交互
- 身体部位不是独立的：头部、躯干、手臂之间有结构约束
- 对遮挡图像：可见部位之间的上下文信息可以帮助提高判别力
- **核心矛盾**：self-attention 让 tokens 交互 → 可能降低 diversity → 伤害 per-token CE

## 核心假设
用 dual-path 设计解决 diversity vs coherence 矛盾：
- **CE path (diversity)**：在 raw tokens 上计算 CE loss → 保持每个 token 独立判别力
- **Triplet path (coherence)**：在 self-attention refined tokens 上计算 triplet → 学习 inter-part 结构
- **Test**：使用 refined tokens → 更丰富的匹配特征

## 技术方案

### 架构变化
```
Backbone → STD-PR cross-attention → raw_tokens (B, 6, C)
                                          ↓
                      ┌───────────────────┴───────────────────┐
                      ↓                                       ↓
              Per-token CE loss                    Self-Attention (1 layer)
              (on raw_tokens)                            ↓
              → maintains diversity              refined_tokens (B, 6, C)
                                                     ↓
                                             Per-token Triplet loss
                                             → learns inter-part structure
                                                     ↓
                                             Test: equal_concat(refined)
```

### 修改文件
1. `model/modules/structural_routing.py`: 添加 1-layer self-attention (MHA + FFN)，返回 raw_tokens 和 refined_tokens
2. `model/pose_backbone_model.py`: CE 用 raw_tokens，triplet 用 refined_tokens
3. `config/defaults.py`: 新增 `POSE_STR_SELF_ATTN` 开关

### Self-attention 参数
- MHA: 768-d, 8 heads (~2.4M)
- FFN: 768 → 1536 → 768 (~2.4M)
- 新增参数: ~4.7M
- 总模型增加: ~16%
- 零初始化: out_proj 和 FFN 最后一层权重初始化为 0，确保初始时 self-attn 是恒等映射

### 重要发现：pooled test feature 优于 per-token concat
- exp166 训练时每个 token 独立 CE/triplet
- 但 test 时 confidence-weighted pooling (1536-d) 比 per-token concat (5376-d) 好 +1.3% mAP / +1.4% R1
- 结论：per-token 是 TRAINING 技巧（强制多样性），test 时 pooling 更好
- 本实验使用 pooled test feature（与 exp166 一致）

### 关键超参数
- 与 exp166 完全相同，只加了 `POSE_STR_SELF_ATTN: True`
- 单变量实验，对照组 = exp166

### 诊断日志
- `sa_cos`: raw vs refined tokens 的 cosine similarity（太低=改变太多，太高=self-attn 没效果）
- 目标: 0.85-0.95 范围，说明 self-attn 在有意义地精化但不破坏原始特征

## 预期结果
- 如果假设成立：mAP/R1 > exp166 (63.1/73.9)，因为 inter-part context 改善匹配
- 如果失败：self-attention 让 tokens 过于相似，R1 下降。最可能原因：diversity loss

## 对照组
- exp166 (STD-PR per-token + PLBOA, 无 self-attention): 63.1/73.9
- 消融变量：仅增加 POSE_STR_SELF_ATTN
