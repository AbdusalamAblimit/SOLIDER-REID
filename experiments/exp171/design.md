# 实验 exp171: Pose-Augmented Patch Embedding (PAPE)

## 动机
- PSG 在 Stage 3 注入 pose → +1.7% mAP。但 Stage 0-2 完全是 pose-blind
- 170 个实验的核心规律：成功的创新都在 **新的层面** 引入信息
- Stage 3 pose injection 已验证有效；**输入层** pose injection 是全新的、未被触及的层面
- DPTL (exp167) 失败的根因是 zero-init + SGD 梯度太弱；PAPE 不同——它从 **global CE + global triplet** 获得强梯度

## 核心假设
在 Swin PatchEmbed 之后注入 pose 信息（并行 patch embedding），让整个 backbone 从第一层就知道人体结构。

## 技术方案

### 架构
```
RGB (B,3,384,128) → PatchEmbed → rgb_tokens (B, 3072, 96)
                                        ↓
Heatmaps (B,17,96,32) → Conv2d(17,96,1×1) → pose_tokens (B, 3072, 96)
                                        ↓
                            tokens = rgb_tokens + pose_tokens
                                        ↓
                            Swin Stage 0-3 (all stages now pose-aware)
                                        ↓
                            Stage 3 + PSG (existing, dual injection)
                                        ↓
                            STD-PR per-token + PLBOA (existing)
```

### 关键设计
1. **Conv2d(17, 96, 1×1)**: heatmaps (96×32) 恰好匹配 PatchEmbed 输出空间维度
2. **零初始化**: 模型从预训练行为开始，逐渐学习使用 pose
3. **仅 1,728 参数**: 极度轻量
4. **双层 pose injection**: PAPE(输入层) + PSG(Stage 3) = 全程 pose-aware

### 与 DPTL 失败原因的区别
| | DPTL (exp167) | PAPE (exp171) |
|---|---|---|
| 新增参数 | 4.7M | **1.7K** |
| 梯度来源 | 仅 part triplet（弱） | **global CE + global triplet（强）** |
| 位置 | Stage 3 之后（post-hoc） | **输入层（影响整个 backbone）** |
| 预期激活速度 | 120ep 未激活 | 应在 20-30ep 内激活 |

## 预期结果
- 如果假设成立：mAP/R1 > exp166 (63.1/73.9)
- 双层 pose injection 可能在 Stage 0-2 产生更好的局部特征
- 失败原因：pose 信息在早期阶段可能是噪声（backbone 还没学会利用）

## 对照组
- exp166 (per-token + PLBOA, PSG only): 63.1/73.9
- 消融变量：仅增加 POSE_PATCH_EMBED: True
