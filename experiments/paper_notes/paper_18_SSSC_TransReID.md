# Paper 18: SSSC-TransReID — Exploring Stronger Transformer Representation Learning for Occluded Person Re-Identification

**来源**: arXiv 2024（2410.15613），submitted October 2024
**仓库**: **代码未找到公开仓库**
**arXiv**: https://arxiv.org/abs/2410.15613
**arXiv 摘要**: 提出 SSSC-TransReID，将自监督对比学习分支与标准有监督 ReID 训练联合，使用新颖的随机矩形遮挡增强（Random Rectangle Mask）模拟真实遮挡，通过 Joint Training Loss 同时优化两个目标。

---

## 代码架构概览

**注意：代码未公开**，以下内容基于 arXiv HTML（2410.15613）描述重建。

- **Backbone**: Transformer（基于 TransReID 的标准 ViT）
- **核心创新**: 双分支训练（有监督 + 自监督），Random Rectangle Mask 数据增强
- **参考框架**: TransReID（JPM + SIE），在此基础上加入自监督分支

---

## 可拆解模块清单

### 模块 A: Random Rectangle Mask（随机矩形遮挡增强）

**功能**: 比简单的随机擦除更贴近真实遮挡的增强策略，生成多个不重叠的矩形遮挡区域

**算法**:
1. 设定目标遮挡比例 r（最优值 r=0.5）
2. 设定最大遮挡块大小 M（对 256×128 输入，M=128×128）
3. 循环迭代：每次随机放置一个矩形，确保不与已有矩形重叠
4. 继续直到达到目标遮挡比例 r

**对比标准方法的优势**:
- 比 Random Erasing 更真实（多个小块 vs 单个大块）
- 比 Hide-and-Seek 更灵活（任意大小 vs 固定网格）
- 报告比 Hide-and-Seek +0.6% Rank-1

**移植可行性**: **高** — 纯数据增强，可以直接加入我们的训练 pipeline
**额外开销**: 几乎没有（CPU 端数据增强）

### 模块 B: 自监督对比学习分支（SimSiam 风格）

**功能**: 对强增强图像提取特征，通过 SimSiam 风格的 stop-gradient 对比损失鼓励模型学习遮挡不变的特征表示

**架构**:
- **正常增强分支**（normal aug）：standard ReID pipeline（flip + crop + erasing）→ Transformer (l-1 layers) → 全局特征 + K 个局部特征（JPM）
- **强增强分支**（strong aug）：Random Rectangle Mask + Gaussian blur + color jitter + solarization → 共享 Transformer (l-1 layers) → Projector + Predictor

**Projector**: 3层 MLP（768 → 768 → 256，带 ReLU 和 BatchNorm）
**Predictor**: 2层 MLP（4096 → 256，带 ReLU）

**损失函数**:
```
L_contrast = 0.5 * D(p1, stop_grad(z2)) + 0.5 * D(p2, stop_grad(z1))
```
其中 D 是 negative cosine similarity，p 是 predictor 输出，z 是 (l-1)层的 Transformer 输出

**关键设计**：stop_gradient 在 z 上，而不是 p 上（与 SimSiam 相同），防止 collapse

**移植可行性**: **中** — 需要双前向传播（增加一倍计算量）；在 Swin-Tiny 上可能受显存限制

### 模块 C: Patch Projection 层冻结（训练稳定性 trick）

**功能**: 在 Transformer 开始训练时，冻结 patch embedding 的 projection 层参数（随机初始化后不更新）

**效果**: 报告改善约 +0.7% mAP/Rank-1
**原理**: 防止 patch embedding 层在两个分支不同目标的梯度信号下不稳定更新

**移植可行性**: **高** — 一行代码，可以在我们框架里试验（在 PSG 引导的 backbone 中冻结早期层）

---

## 损失函数

### 有监督分支（正常增强）
```
L_TS = L_ID(fg) + L_T(fg) + (1/K) * sum_k(L_ID(fl^k) + L_T(fl^k))
```
- L_ID：CE loss（无 label smoothing，这与多数工作不同）
- L_T：soft-margin triplet loss = log(1 + exp(||fa - fp||² - ||fa - fn||²))
- fg：全局特征，fl^k：第 k 个局部特征

### 自监督分支（强增强）
```
L_contrast = 0.5 * D(p1, stop_grad(z2)) + 0.5 * D(p2, stop_grad(z1))
```
D = negative cosine similarity

### 联合损失
```
L = λ * L_TS + (1 - λ) * L_contrast
```
λ = 0.95（有监督主导）

---

## 训练 Tricks

- 输入: 256×128
- Batch size: 100（4 images × 25 identities）
- Optimizer: SGD（momentum=0.9，weight_decay=1e-4）
- Base LR: 0.0125，cosine decay
- Patch projection 层冻结（关键！）
- λ=0.95（强调有监督，自监督为辅）

## 性能（报告值）

- Occluded-Market: 71.7% mAP, 83.3% R1
- Market-1501: 89.8% mAP, 95.8% R1
- Occluded-Duke: 结果未在摘要中明确提及

---

## 该工作的局限性 / 未解决的问题

1. **双分支增加计算量**：强增强分支多做一次前向传播，训练时间约增加 40-60%
2. **在已有强 baseline 上的增益不明确**：论文里的 baseline 是 TransReID，我们 baseline 是 SOLIDER-Swin，直接移植不确定效益
3. **自监督信号来自遮挡增强**：本质上是让模型学习"被遮挡的图像和原始图像应该有相似特征"，但这个信号在 SOLIDER 预训练后可能已经部分具备
4. **Projector/Predictor 参数量**：增加了较多参数和计算量（MLP 层），但可能在 Swin-Tiny 上显存允许范围内

---

## 对我们框架的改进建议

1. **Random Rectangle Mask 作为数据增强直接加入**:
   - 比现有的 random erasing 更模拟真实遮挡
   - 可以在 training dataloader 中作为一个可选增强策略
   - 计划在现有实验中试验（不需要修改模型，纯数据增强）

2. **Stop-gradient 自监督对比分支 → 轻量版**:
   - 完整的双分支可能过于重量级
   - 轻量替代：只在强增强样本上加一个 consistency loss（目标：强增强特征 ≈ 原始特征 in normalized space）
   - 等效于：f_strong → projector → p，f_normal → stop_grad → z，minimize D(p, z)
   - 这比完整 SimSiam 轻约一半，但可能保留主要效益

3. **Patch projection 冻结值得尝试**:
   - 在 Swin-Tiny 中等价于冻结 patch embedding 层（第一个 PatchEmbed 模块）
   - 成本极低，可能对训练稳定性有帮助

4. **核心 gap**:
   - SSSC-TransReID 的自监督 branch 用的是 random mask 增强，没有利用任何姿态信息
   - **若将自监督分支的"遮挡模拟"改为基于姿态热图的 body-aware masking**（只遮挡低置信度的关键点区域），可能比随机矩形更精准
   - 这是一个与 SSSC-TransReID 的明确差异化方向：**姿态引导的遮挡感知一致性训练**
