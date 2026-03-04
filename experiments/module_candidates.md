# 候选模块总表

基于对 12 个论文仓库的代码分析，以下是可移植到我们 SOLIDER-REID (Swin-Tiny + with_cp) 框架的候选模块。

## 当前 Baseline 状态
- **Backbone**: Swin-Tiny (768-dim output), WITH_CP=True
- **Input**: [384, 128]
- **数据集**: Occluded-Duke (15618 train, 2210 query, 17661 gallery)
- **最佳结果**: E1 = 59.0% mAP, 68.6% Rank-1 (concat, pose dual-branch)
- **无姿态 Baseline**: E0 = 55.2% mAP, 65.5% Rank-1

## 候选模块总表

| 序号 | 模块名称 | 来源论文 | 类型 | 与 Swin-Tiny 兼容性 | 额外显存估算 | 预期增益 | 实现难度 | 优先级 |
|------|----------|----------|------|---------------------|-------------|----------|----------|--------|
| M01 | NFC (Neighbor Feature Centralization) | Pose2ID (CVPR25) | 测试时后处理 | 高 | 0 (CPU) | +1-2% mAP | 低 | P0 |
| M02 | PixelToPartClassifier + GWAP | KPR (ECCV24) | Part特征提取 | 高 | <50MB | +2-3% mAP | 中 | P0 |
| M03 | Body Part Attention Loss | KPR (ECCV24) | 损失函数 | 高 | ~10KB | +1% mAP | 低 | P0 |
| M04 | GiLt Loss Strategy | KPR (ECCV24) | 损失策略 | 高 | 0 | +0.5-1% mAP | 低 | P0 |
| M05 | Part-Averaged Triplet Loss | KPR (ECCV24) | 损失函数 | 高 | 0 | 已有(PAMS) | 低 | — |
| M06 | Prompt Tokenizer (零初始化) | KPR (ECCV24) | 输入增强 | 高 | ~50MB | +1-2% mAP | 中 | P1 |
| M07 | PFA (Pose Feature Alignment) | PFD (AAAI22) | 特征对齐 | 中 | ~20MB | +1-2% mAP | 中 | P1 |
| M08 | PVM (Pose Visibility Matching) | PFD (AAAI22) | 特征匹配 | 中 | ~10MB | +0.5-1% mAP | 中 | P2 |
| M09 | Push Loss (可见/遮挡分离) | PFD (AAAI22) | 损失函数 | 高 | 0 | +0.5% mAP | 低 | P1 |
| M10 | TTK多层姿态蒸馏 | PGDS (AVSS24) | 知识蒸馏 | 高(同Swin-Tiny) | ~0.5G训练 / 0推理 | +1-2% mAP | 中 | P1 |
| M11 | 姿态热图空间注意力 | PGFA (ICCV19) | 特征加权 | 高 | ~0.1G | +1% mAP | 低 | P1 |
| M12 | Shared-Region Distance | PGFA (ICCV19) | 评估策略 | 高 | 0 | 已有(类似) | 低 | — |
| M13 | SIE (Side Information Embedding) | TransReID (ICCV21) | 输入增强 | 高 | ~1MB | +0.3-0.5% mAP | 低 | P2 |
| M14 | JPM (Jigsaw Patch Module) | TransReID (ICCV21) | Part特征提取 | 中 | ~100MB | +1% mAP | 中 | P2 |
| M15 | Part Token Learning | PAT (CVPR21) | Part发现 | 中 | ~100MB | +1% mAP | 高 | P3 |
| M16 | Diverse Part Discovery | PAT (CVPR21) | 正则化 | 高 | 0 | +0.5% mAP | 中 | P2 |
| M17 | 联合语义分割+ReID | ISP (ECCV20) | 多任务学习 | 低 | ~0.5G | +1% mAP | 高 | P3 |
| M18 | 文本Prompt引导特征 | CLIP-ReID (AAAI23) | 多模态 | 低 | ~200MB | 不确定 | 高 | P3 |
| M19 | Semantic-Appearance解耦控制 | SOLIDER (CVPR23) | 特征调节 | 高 | 0 | 已集成 | — | — |
| M20 | 人体部件Parsing+Pooling | BPBreID (WACV23) | Part特征提取 | 高 | <50MB | +1-2% mAP | 中 | P1 |

## 推荐实验路线

### Phase 2a: 零成本改进（立即可做，不需要姿态权重）
1. **exp002_nfc_test**: NFC 测试时后处理 — 仅修改评估代码，零训练开销
2. **exp003_gilt_loss**: 将 PAMS 的 loss 策略改为 GiLt（Global-ID + Local-Triplet）

### Phase 2b: 轻量级姿态模块（需要 ViTPose 权重）
3. **exp004_pixel_part_cls**: PixelToPartClassifier + Body Part Attention Loss + GWAP
   - 从 KPR 移植，替代/增强 PAMS 的 BPA 机制
   - 用 ViTPose 的 visibility 向量作为 GT 监督 part attention
4. **exp005_pose_spatial_attn**: PGFA 式姿态热图空间注意力
   - 离线热图 × backbone 特征图 → 部件级特征
5. **exp006_push_loss**: 可见/遮挡部件 Push Loss
   - 从 PFD 移植，利用 visibility 向量分离可见/遮挡特征

### Phase 2c: 中等复杂度模块
6. **exp007_prompt_tokenizer**: KPR Prompt Tokenizer (零初始化)
   - 将关键点热图编码为 prompt tokens 加到 Swin patch tokens 上
7. **exp008_pfa_align**: PFA 姿态特征对齐
   - 离线关键点特征与 Swin part 特征做余弦对齐
8. **exp009_ttk_distill**: PGDS 多层姿态蒸馏
   - 离线姿态 embedding 通过 KL 散度蒸馏到 Swin 中间层

### Phase 2d: 组合实验
9. **exp010_combine_best2**: 最佳两个模块组合
10. **exp011_combine_best3**: 最佳三个模块组合
11. **exp012_full_system**: 全系统整合 + 超参数精调

## 关键设计决策

### 1. 姿态信息获取策略
- **离线提取**: 使用用户提供的 ViTPose (VisPredictHead) 提取 17 关键点 + visibility
- **格式**: 每张图保存 (keypoints_xy [17,2], confidence [17], visibility [17])
- **热图生成**: 按需在线从关键点生成 Gaussian heatmap，参考 KPR 的 `keypoints_to_masks.py`

### 2. Part 分组策略
- 参考 KPR 的 cck5/cck6/cck8 分组：
  - cck5: head, upper_body, lower_body, left_arm, right_arm (5 parts)
  - cck6: head, torso, left_arm, right_arm, left_leg, right_leg (6 parts)
- 与 PAMS 现有的 5-part 设计兼容

### 3. 显存预算
- Swin-Tiny + with_cp 约占 7-8G
- AMP 开启后可省 ~1G
- 可用预算: ~3-4G (假设 24G GPU)
- 所有新模块总显存开销需控制在 <2G

## 模块依赖关系
```
M01 (NFC) ← 独立，立即可做
M02 (PixelPartCls) ← 需要离线关键点作为GT
M03 (BPA Loss) ← 需要 M02
M04 (GiLt) ← 独立，修改 loss 配置即可
M06 (Prompt Tokenizer) ← 需要离线关键点
M07 (PFA) ← 需要离线关键点特征
M10 (TTK蒸馏) ← 需要离线姿态 embedding
M11 (PGFA注意力) ← 需要离线热图
```
