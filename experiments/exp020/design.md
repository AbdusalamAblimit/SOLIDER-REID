# 实验 exp020: PSG + Pose Reconstruction Auxiliary (PRA)

## 动机
- PSG (exp007, +1.7%) 是目前唯一有效的训练端 pose 利用方式
- 18 个实验尝试了各种组合（PAB, Part Pooling, PCG, PGE, PXA），全部无法超越 PSG 或与 PSG 叠加
- **关键观察**: 所有方法都是 pose→features 方向（注入/门控/调制），没有尝试 features→pose 方向
- **核心假设**: 如果 backbone features 被迫保持能够重建 pose structure 的信息，它们将更具判别性
  - PSG 注入 pose 信息 → 特征被调制 (forward path)
  - PRA 要求特征能预测 pose → 特征必须编码结构 (backward path)
  - 双向的 pose-feature 交互可能比单向更有效

## 创新点 / 核心想法
- **双向 pose-feature 交互**: PSG (pose→feature) + PRA (feature→pose) 构成闭环
- 与之前所有组合的区别: PRA 不修改 PSG 的前向传播（不改 feature map，不加模块到 backbone 内部）
  - PSG + PAB (exp013): PAB 修改了 attention → 干扰 PSG 输入
  - PSG + Part Pooling (exp008): Part 分支改变 loss → 干扰 PSG 梯度
  - PSG + PCG (exp017): PCG 修改了 GAP 后 feature → 通道级干扰
  - PSG + PRA: PRA 只**读取** feature map → 不修改 PSG 的任何计算
- PRA 的梯度是 MSE 回归（非分类），与 ID/triplet 梯度**性质不同**，可能提供互补信号

## 技术方案

### 新增模块: PoseReconstructionHead (model/modules/pose_reconstruction_head.py)

```
输入: feature_map (B, C=768, H=12, W=4) — Stage 3 输出（PSG 后）
1. Conv2d(768, 128, 3, padding=1) → BN → ReLU
2. Conv2d(128, 17, 1)               [无激活函数，输出 raw logits]
输出: predicted_heatmaps (B, 17, 12, 4)
```

### 损失函数
- GT heatmaps: resize scene_heatmaps to (12, 4), apply sigmoid
- Loss: MSE(predicted, GT_sigmoid) × λ
- λ = 0.1（起始值，保持辅助 loss 较小）

### 数据流
```
图片 → Swin Stages 0-2 → Stage 3 + PSG → feature_map (768, 12, 4)
                                              ├→ GAP → BN → ID/Triplet Loss (主 loss)
                                              └→ PRA Head → predicted_heatmap → MSE Loss (辅助 loss)
```

### 训练时 forward 返回
- 增加一个返回值: `(cls_score, global_feat, featmaps, pose_recon_loss)`
- pose_recon_loss 在模型内部计算并返回（避免修改 processor 的 loss_fn 接口）

### 关键超参数
- λ = 0.1: PRA loss 权重（需要足够小不干扰主 loss，足够大提供信号）
- 3×3 conv 第一层: 提供空间上下文（vs 1×1）
- BN + ReLU: 标准归一化 + 非线性

### 参数量估算
- Conv2d(768, 128, 3×3): 768×128×9 + 128 = 884,864
- BN(128): 256
- Conv2d(128, 17, 1×1): 128×17×1 + 17 = 2,193
- 合计: ~887K
- 加上 PSG 的 102K: 总计 ~989K

注: PRA 参数较多 (887K) 但只用于训练，测试时可移除（辅助任务头）

### 修改文件
1. 新增: `model/modules/pose_reconstruction_head.py`
2. 修改: `model/pose_backbone_model.py` — 添加 PRA 选项
3. 修改: `processor/processor.py` — 处理额外的 recon loss
4. 修改: `config/defaults.py` — 添加 POSE_RECON 开关
5. 新增: `configs/occluded_duke/pose_psg_recon.yml`

### Config 开关
```yaml
POSE_BACKBONE_PSG: True          # 使用 PSG
POSE_RECON_HEAD: True             # 启用 PRA
POSE_RECON_WEIGHT: 0.1            # λ 权重
```

## 预期结果
- **乐观**: mAP 59%+ — 双向交互突破 PSG 的 58.3% 上限
- **中性**: mAP 58.0-58.3% — PRA 不帮助也不伤害（PSG 已充分）
- **悲观**: mAP < 57% — PRA 梯度干扰 PSG 学习
- 失败最可能原因: backbone features 已经隐式编码了足够的结构信息（PSG 保证了这一点），显式重建是冗余的

## 对照组
- Baseline: exp000 (mAP 56.6%)
- 核心对比: exp007 PSG-only (mAP 58.3%)
- 消融变量: 仅添加 PRA head + MSE loss，PSG 配置完全相同
