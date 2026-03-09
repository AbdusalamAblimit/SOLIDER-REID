# Paper 4: PGDS - Pose Guidance by Deep Supervision
**来源**: AVSS 2024
**仓库**: https://github.com/huyquoctrinh/PGDS
**核心**: 多层姿态知识蒸馏到 ReID 模型

## 代码架构概览
- 模型: `model/make_model.py` L306-468 (build_transformer + TTK)
- 蒸馏: `processor/custom_processor.py` L40-91
- 姿态编码器: OpenPose (冻结)

## 可拆解模块清单

### M1: TTK (Transformer Token Kernel) - 多层特征提取
- 文件: `model/make_model.py` L156-188
- 功能: 从 Swin 不同 stage 提取 local 特征
- stage1 [96,56,56] → stage2 [192,28,28] → stage3 [384,14,14] → 各自独立 TTK → [N,768]
- **移植可行性**: 中 | **显存**: ~0.3G

### M2: KL Divergence 多层蒸馏
- 文件: `processor/custom_processor.py` L85-91
- 姿态特征的 softmax 分布 → 各层 local 特征的 softmax 分布
- 权重递减: 0.5, 0.3, 0.2
- 总权重: 0.8*main_loss + 0.2*distill_loss
- **移植可行性**: 高 | **显存**: <0.1G

## 关键洞察
1. 推理时不需要姿态模型(全蒸馏到 ReID 网络)，推理速度不变
2. 蒸馏的是分类分布而非几何特征，可能丢失细粒度姿态信息
3. **改进方向**: Visibility-Guided Distillation — 遮挡部位降低蒸馏权重
4. 可与我们的 ViTPose 直接结合: ViTPose 冻结 → KL 蒸馏到 Swin 多层
