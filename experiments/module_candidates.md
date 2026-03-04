# 候选模块总表

## 已有实验基线

| 方法 | Occ-Duke mAP | Occ-Duke R-1 | 备注 |
|------|-------------|-------------|------|
| Baseline (Swin-Tiny, no pose) | 55.2% | 65.5% | E0 |
| Pose-Swin 双分支 (concat) | 59.0% | 68.6% | E1, 主要来自参数翻倍 |
| PAMS v8 (epoch 30/120) | 45.8% (global) | 55.5% | 未完成训练 |

## 候选模块

| 序号 | 模块名称 | 来源论文 | 类型 | 与 Swin-Tiny 兼容性 | 额外显存估算 | 预期增益 | 实现难度 | 优先级 |
|------|----------|----------|------|---------------------|-------------|----------|----------|--------|
| M01 | Pose2ID NFC (邻域特征中心化) | Pose2ID (CVPR25) | 后处理 | 高（零参数） | <0.2G | 中 | 低 | ⭐⭐⭐ |
| M02 | PartClassifier + BPA | KPR/PAMS | 可学习部件分割 | 高（已实现） | ~0.3G | 高 | 已完成 | ⭐⭐⭐ |
| M03 | Visibility-Aware Distance | KPR (ECCV24) | 评估距离 | 高（已部分实现） | 0 | 中 | 低 | ⭐⭐⭐ |
| M04 | PGDS TTK 模块 | PGDS (AVSS24) | 多尺度知识蒸馏 | 高（匹配Swin 4 stage） | ~0.5G | 中 | 中 | ⭐⭐ |
| M05 | PFD Matching Head | PFD (AAAI22) | 姿态-特征对齐 | 中（需HRNet） | ~1.5G | 中 | 高 | ⭐⭐ |
| M06 | PGFA Heatmap Masking | PGFA (ICCV19) | 姿态引导池化 | 高（简单乘法） | ~0.3G | 低 | 低 | ⭐ |
| M07 | TransReID JPM | TransReID (ICCV21) | 局部特征 | 低（为ViT设计） | ~0.3G | 低 | 中 | ⭐ |
| M08 | GiLt Loss 策略 | KPR (ECCV24) | 损失函数 | 高（配置即可） | 0 | 中 | 低 | ⭐⭐⭐ |
| M09 | BPA 软标签 | 改进 | 标签生成 | 高 | 0 | 中 | 低 | ⭐⭐⭐ |
| M10 | Part Feature Centralization | Pose2ID+PAMS | 特征后处理 | 高 | 0 | 高 | 中 | ⭐⭐⭐ |

## 推荐实验路线

### Phase 2a: 完善 PAMS 并跑完 baseline (1-2 experiments)
1. **exp001**: PAMS v9 完整 120 epoch 训练 — 确认 PAMS 的最终性能
2. **exp002**: Baseline (no pose) 完整训练 — 确认 baseline 数字

### Phase 2b: 围绕创新点做核心验证 (3-5 experiments)
3. **exp003**: PAMS + BPA 软标签 (用概率分布替代硬 argmax)
4. **exp004**: PAMS + NFC 后处理 (Pose2ID 的邻域特征中心化)
5. **exp005**: PAMS + 遮挡感知对比学习 (新 loss)
6. **exp006**: PAMS + PGDS-style 多尺度知识蒸馏
7. **exp007**: PAMS + Part-aware Re-ranking

### Phase 2c: 消融与组合 (3-5 experiments)
8. 消融 MSF 分辨率: (24,8) vs (12,4) vs (48,16)
9. 消融 N_PARTS: 3 vs 5 vs 7
10. 消融 BPA 权重: 0.5 vs 1.0 vs 2.0
11. 最优组合实验
