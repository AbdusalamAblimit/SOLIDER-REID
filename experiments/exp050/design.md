# 实验 exp050: PAMC (Pose-Aware Masking Consistency)

## 动机
- 48 个实验后，所有在 PSG/GCN 基础上的训练端"加模块"改进均失败
- 文献调研发现一个**未被占领的 gap**：没有现有方法同时结合"姿态引导的身体部位遮挡增强" + "自监督一致性 loss" 用于**有监督 ReID 训练**
  - HAP (NeurIPS 2023): 姿态引导 masking，但仅用于**预训练**阶段，不参与有监督微调
  - SSSC-TransReID: SimSiam 风格一致性 loss 用于 ReID 训练，但使用**随机矩形遮挡**，不感知身体结构
  - DPEFormer: ROA 真实遮挡增强，但无一致性 loss
  - **我们的 PAMC 填补这个交叉空白**

## 创新点 / 核心想法
**核心假设**: 如果我们用姿态信息识别可见身体部位的空间范围，在训练时随机遮挡其中一个部位作为增强视图，然后用 stop-gradient 一致性 loss 鼓励模型在两个视图间输出一致的特征，那么模型将学会从可见部位更鲁棒地推断身份信息。

与 baseline / 前序实验相比，改了什么：
- 新增模块：PAMC consistency branch（projector MLP + 一致性 loss）
- 新增数据增强：Pose-Aware Body Masking（基于热图响应强度识别身体部位空间区域）
- 训练方式：每个 batch 做两次前向（原图正常训练 + masked 图 eval 模式 no_grad 前向），额外显存约 1.2GB

## 技术方案

### 数据流（非对称设计）
```
原始图像 img → model forward (train mode, with grad) → global_feat_orig
                                                           ↓
                                            projector(global_feat_orig)
                                                           ↓
                                            loss = -cos(projected, masked_target)
                                                           ↑
Masked 图像 img_masked → model forward (eval mode, no grad) → masked_target (detached)
```

关键设计：
- **非对称**：只有原始视图的特征通过 projector 并产生梯度
- **masked 视图用 eval 模式**：禁用 DropPath/StochasticDepth，BN 使用 running stats，产生确定性目标特征
- **no_grad**：masked 视图前向不保存激活值，显存开销极低（~1.2GB vs 完整梯度 ~8GB）
- **不修改 pose_dict**：masked 视图使用原始 pose 热图（信息更完整）

### 身体部位 Masking 策略
1. 从 pose_dict 中取 scene heatmaps (17, hm_H, hm_W)
2. 按 COCO 6 部位组（head/left_arm/right_arm/torso/left_leg/right_leg）分组
3. 对每组的关键点热图取 max → 该部位的空间响应图
4. 筛选"有效部位"：peak response > 0.1
5. 对有效部位的响应区域（> 30% peak）计算 bounding box，并扩展 50%
6. 随机选择 1 个部位的 bbox 区域填零（normalized pixel mean = 0）
7. 如果无有效部位（热图无响应），退化为随机矩形 mask

### Projector MLP
- 输入: global_feat (768-d, pre-BN)
- 结构: Linear(768, 2048) → BN1d → ReLU → Linear(2048, 768)
- 输出: 768-d projection
- 参数量: ~3.15M

### Consistency Loss（非对称）
- `L = -cos(projector(orig_feat), masked_feat)`
- masked_feat 已 detached（来自 no_grad 块）
- 只有 projector 和 orig_feat 接收梯度
- 与 BYOL 的 online/target branch 设计类似

### 修改的文件
1. `config/defaults.py`: 新增 POSE_PAMC / POSE_PAMC_WEIGHT / POSE_PAMC_WARMUP / POSE_PAMC_PROJ_DIM
2. `model/modules/pamc.py`: 新增 PoseBodyMasker, PAMCProjector, pamc_consistency_loss
3. `model/pose_backbone_model.py`: 新增 pamc_projector / pamc_masker 作为子模块
4. `processor/processor.py`: 新增 PAMC 训练逻辑（masking + eval-mode no_grad forward + consistency loss）

### 关键超参数
- `POSE_PAMC = True`: 开关
- `POSE_PAMC_WEIGHT = 0.5`: 一致性 loss 权重（相对于标准 ReID loss）
- `POSE_PAMC_WARMUP = 10`: 前 N epoch 不启用 PAMC（让模型先学基本 ID）
- `POSE_PAMC_PROJ_DIM = 2048`: projector 隐藏维度

### 显存估算（实测）
- 正常 PSG+GCN forward (batch=64): ~8.4GB
- PAMC masked-view forward (eval, no_grad): 额外 ~1.2GB
- 总计 forward: ~9.6GB
- Backward: 估计 ~12-14GB
- 3090 (24GB) 充分可行

## 预期结果
- 如果假设成立，预期 mAP +0.5~2.0%（基于 SSSC-TransReID 报告的 +0.5~1.0% on Occluded-Duke）
- PAMC 使用姿态引导的 body-aware masking，应比随机矩形 masking 更有效
- 如果失败，最可能原因：
  1. Consistency loss 与 ID loss 梯度冲突（类似 exp048 SGMKC）
  2. 姿态引导的 masking 选区不够准确（覆盖面积太小）
  3. 非对称单方向 loss 信号不足

## 对照组
- Baseline 对照：exp030a (PSG+GCN, equal_concat) = 60.73% mAP (3-seed mean)
- 消融变量：PAMC on/off（单变量；config 与 exp030a 完全相同，仅增加 PAMC 相关配置）
- 后续消融（如 PAMC 有效）：pose-aware masking vs random masking（验证姿态引导的必要性）

## 风险评估
- **显存**: 实测仅增加 ~1.2GB（eval+no_grad），完全可行
- **训练时间**: 额外一次 backbone forward（~30-40% 增加），可接受
- **梯度冲突**: consistency loss 可能干扰 ID loss → 使用 warmup + 适当权重控制
- **与 exp048 SGMKC 关键区别**: SGMKC 让同一个 GCN 学两个矛盾任务（重建+分类）；PAMC 在 global branch 上加独立 projector 的 loss，不改变现有模块的学习目标
