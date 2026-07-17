# exp245 Claude Review: LGPA-D on Small (Agent Review)

## 审查范围
1. design.md — 泛化性验证设计
2. pose_psg_lgpa_detach.yml — 基础配置
3. pose_backbone_model.py — LGPA Small 兼容性
4. clip_part_head.py — feat_dim 自适应
5. defaults.py — 配置安全性

## 审查结论

### 架构兼容性
- Small Stage 3 output = 768 dim (同 Tiny), CLIPPartHead 无需修改
- PSG 自动为 Small 的 18 个 Stage 3 blocks 创建模块 (vs Tiny 6个)
- head_dim = 768/8 = 96, 整除, 无问题

### 内存评估
- Small + PSG(18 blocks) + LGPA-D + OA-SD on 5060Ti 16GB: 风险中等
- exp206r (Small+GCN+OA-SD) 在类似配置下成功运行
- LGPA-D 是 detached, 无额外反向传播图, 内存与 GCN 类似
- TEST.IMS_PER_BATCH=128 提供安全余量

### PPA 灾难对比
- exp242 (PPA+GCN non-detach on Small) 灾难性失败 (-9.7 mAP)
- 原因: non-detached 梯度干扰 Small 的 18-block Stage 3
- exp245 LGPA-D 使用 detach (line 501), 梯度不传到 backbone, 安全

### Config 验证
- 命令行覆盖模式: YACS merge_from_list, 正确
- TRANSFORMER_TYPE, PRETRAIN_PATH, BASE_LR, TEST.IMS_PER_BATCH 均正确
- CLIP text features cache (pretrained/clip_part_text_features.pt) 已存在

### 单变量原则
- vs exp244: 仅换 backbone (Tiny→Small), 其他完全相同

## 结论

审查通过。纯 backbone 切换, 架构自适应, 无代码变更风险。
