# Paper 1: KPR - Keypoint Promptable Re-Identification
**来源**: ECCV 2024
**仓库**: https://github.com/VlSomers/keypoint_promptable_reidentification
**核心**: 支持可选关键点 prompt 的 Part-Based ReID

## 代码架构概览
- 核心模型: `torchreid/models/kpr.py` (~800行)
- 损失函数: `torchreid/losses/GiLt_loss.py`
- 距离计算: `torchreid/metrics/distance.py`
- Prompt处理: `torchreid/models/promptable_transformer_backbone.py`

## 可拆解模块清单

### M1: Visibility-Weighted Distance Computation
- 文件: `torchreid/metrics/distance.py` L182-220
- 功能: 推理时只比较 query/gallery 相互可见的部件
- 权重公式: w_ij^k = sqrt(v_i^k * v_j^k) (几何平均)
- **移植可行性**: 高 | **显存**: 0 | **预期收益**: mAP +2-3%

### M2: GiLt Loss (Global-identity Local-triplet)
- 文件: `torchreid/losses/GiLt_loss.py` L20-93
- 策略: 全局特征→ID loss, 部件特征→Triplet loss
- 支持 visibility 加权: binary 或 continuous
- **移植可行性**: 高 | **显存**: 0 | **预期收益**: +1-2%

### M3: Keypoint Prompt Embedding
- 文件: `torchreid/models/promptable_transformer_backbone.py` L106-148
- 两种策略: embed_heatmaps_patches / spatialize_part_tokens
- Prompt 可选: 推理时可禁用
- **移植可行性**: 中-高 | **显存**: +0.2G

### M4: Body Part Attention Loss
- 文件: `torchreid/losses/body_part_attention_loss.py`
- 像素级部件分类监督 (CE/Focal/Dice)
- **移植可行性**: 高 | **显存**: <0.1G

## 与我们 ViTPose Visibility 的对比
| 维度 | KPR | 我们的 ViTPose |
|------|-----|---------------|
| 来源 | 注意力图最大激活 | BCELoss 监督 |
| 精确性 | 可能混淆推断/真实可见 | 更精确(有标签) |

## 关键启发
1. 可见性加权距离计算是"零成本"的重要改进
2. GiLt loss 的分离策略(全局ID+部件Triplet)比统一策略更优
3. ViTPose visibility 可替换 KPR 的注意力可见性，效果应更好
