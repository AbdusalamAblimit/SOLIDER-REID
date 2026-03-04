# 候选模块总表

| 序号 | 模块名称 | 来源论文 | 类型 | Swin-Tiny兼容性 | 显存估算 | 预期增益 | 实现难度 | 优先级 |
|------|----------|----------|------|-----------------|----------|----------|----------|--------|
| M01 | Visibility-Weighted Distance | KPR/BPBreID | 推理优化 | 高 | 0 | +2-3% mAP | 低 | P0 |
| M02 | NFC (邻域特征中心化) | Pose2ID | 推理后处理 | 高 | 0 | +2-5% mAP | 低 | P0 |
| M03 | GiLt Loss 分离策略 | KPR/BPBreID | 损失函数 | 高 | 0 | +1-2% mAP | 低 | P0 |
| M04 | SIE (相机嵌入) | TransReID | 输入增强 | 高 | <0.1G | +0.5-1% | 低 | P1 |
| M05 | PFA (热图特征对齐) | PFD | 特征对齐 | 高 | <0.1G | +1-2% | 中 | P1 |
| M06 | KL 多层蒸馏 | PGDS | 知识蒸馏 | 高 | <0.1G | +1-2% | 中 | P1 |
| M07 | Push Loss (遮挡分离) | PFD | 损失函数 | 高 | 0 | +0.5-1% | 低 | P1 |
| M08 | JPM (局部特征) | TransReID | 局部特征 | 中 | ~0.5G | +1-3% | 中 | P2 |
| M09 | Keypoint Prompt Embedding | KPR | 输入融合 | 中-高 | +0.2G | +1-2% | 中 | P2 |
| M10 | Part Attention Loss (像素监督) | BPBreID | 监督信号 | 高 | <0.1G | +0.5% | 低 | P2 |

## 推荐实验路线

### Phase 2a: 验证 VPReID baseline 有效性
1. exp001: Swin-Tiny baseline (无 pose)
2. exp002: VPReID v1 (当前实现)

### Phase 2b: 零成本推理优化
3. exp007: +Visibility-Weighted Distance (M01)
4. exp008: +NFC 后处理 (M02)
5. exp009: +M01+M02 组合

### Phase 2c: 训练优化
6. exp010: GiLt Loss 替换 (M03)
7. exp011: +SIE (M04)
8. exp012: 最佳组合
