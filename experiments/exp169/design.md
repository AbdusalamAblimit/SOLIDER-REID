# 实验 exp169: Part-Level Token Dropout (PLTD)

## 动机
- PLBOA 通过图像级遮挡增强提供 +2.8% mAP — 数据增强是最有力的改进来源
- 但 PLBOA 只在像素层操作，structural tokens 层没有类似的正则化
- Per-token CE 强制多样性，但每个 token 在训练时始终存在——没有训练"部分 token 缺失"的情况
- 测试时遮挡 query/gallery 的某些 body parts 时，pooled feature 中包含了从被遮挡区域提取的低质量 tokens

## 核心假设
在训练时随机丢弃部分 structural tokens（语义级 dropout），迫使模型用更少的 part 信息也能保持判别力。

## 技术方案
- 训练时：每个 structural token 独立以 p=0.3 的概率被 zero-out
- 保证每个样本至少保留 2 个 tokens（避免极端退化）
- Zero-out 同时应用于 CE path (raw_tokens) 和 triplet path (structural_tokens)
- Pooled feature 自然变弱 → BN 归一化 → loss 迫使存活 tokens 更有判别力
- 测试时：使用全部 tokens（与标准 dropout 行为一致）

### 与已有技术的区别
| 技术 | 操作层 | 粒度 | 语义性 |
|------|--------|------|--------|
| Standard Dropout | neuron | 随机 neuron | 无 |
| Random Erasing | pixel | 随机矩形 | 无 |
| PLBOA | pixel | body region | 有 |
| **PLTD** | **feature token** | **semantic body part** | **有** |

### 零额外参数，零速度损耗
- 不增加任何参数
- Training 时仅多一次 mask 计算（忽略不计）
- Test 时无任何额外操作

## 预期结果
- 如果假设成立：mAP/R1 > exp166 (63.1/73.9)
- PLTD + PLBOA 双层遮挡正则化可能产生协同效应
- 失败原因：p=0.3 太高导致训练不稳定，或 pooled feature 在 token dropout 下退化

## 对照组
- exp166 (per-token + PLBOA, 无 PLTD): 63.1/73.9
- 消融变量：仅增加 POSE_STR_PART_DROP: 0.3
