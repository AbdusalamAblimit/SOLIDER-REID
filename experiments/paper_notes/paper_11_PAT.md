# Paper 11: PAT (Part-Aware Transformer)
**来源**: ICCV 2023（注：README 标注为 ICCV 2023，而非早期预印本所标的 CVPR 2021）
**仓库**: https://github.com/liyuke65535/Part-Aware-Transformer.git
**arXiv 摘要**: 提出 Part-Aware Transformer (PAT) 用于 Domain Generalization Person ReID。通过在 ViT 中插入可学习的 part token 并设计受限注意力掩码(Part Attention Mask)，使 part token 只关注特定空间区域，自动发现语义部件。结合 Cross-ID Similarity Learning (CSL) 代理任务和 Part-guided Self-Distillation (PSD) 策略，实现跨域泛化能力。

## 代码架构概览
- **核心文件**: `model/backbones/vit_pytorch.py` — 包含 `part_Attention_ViT` 类（核心模型）和 `part_Attention`/`part_Attention_Block` 类
- **模型入口**: `model/make_model.py` 中的 `build_part_attention_vit` 类
- **损失函数**: `loss/myloss.py`（Pedal 损失 = CSL 聚类损失）、`loss/softmax_loss.py`（带 soft label 的 CE 损失）、`loss/smooth.py`（PatchMemory 记忆库）
- **训练流程**: `processor/part_attention_vit_processor.py` — PAT 专用训练循环
- **基础框架**: 基于 TransReID，ViT-Base 为默认 backbone

### 关键代码路径
```
train.py → make_model('part_attention_vit') → build_part_attention_vit
         → build_loss() → loss_func (ID + Triplet)
         → PatchMemory → Pedal loss (CSL)
         → part_attention_vit_do_train_with_amp()
```

## 可拆解模块清单

### 模块 A: Part Token 机制 + Part Attention Mask
- **文件位置**: `model/backbones/vit_pytorch.py` L564-L738 (`part_Attention_ViT`)
- **功能**: 在 ViT 中额外添加 3 个可学习的 part token（与 cls token 并列），通过注意力掩码约束每个 part token 只能关注图像的特定空间区域（上/中/下三部分），从而自动发现人体部件特征
- **核心设计**:
  - **Part Token 初始化**: 3 个独立的 `nn.Parameter`（`part_token1/2/3`），shape 为 `(1, 1, embed_dim)`，用 truncated normal 初始化（std=0.02）。加载预训练权重时，用 cls_token 的权重初始化所有 3 个 part token（见 L718-L721）
  - **Token 拼接顺序**: `[cls, part1, part2, part3, patch_tokens]`，位置编码长度为 `num_patches + 4`
  - **注意力掩码生成**（`attn_mask_generate`, L627-L636）:
    - 生成 `(N, N)` 的 bool 矩阵，N = num_patches + 4
    - 默认所有 patch token 之间、cls 与所有 token 之间可以互相注意
    - part token 之间**互相不可见**（L629: `mask[1:4, 0] = 0`，初始化时 part 位置为 0）
    - 每个 part token 通过 `generate_2d_mask` 只能看到特定空间区域：
      - part1: 上半部分 `(0, 0, W, H/2)` — 大致对应头部+肩部
      - part2: 中间部分 `(0, H/4, W, H/2)` — 大致对应躯干
      - part3: 下半部分 `(0, H/2, W, H/2)` — 大致对应腿部
    - `generate_2d_mask` 中有**随机采样**机制：在指定区域内随机选取一个子矩形（L92-93: `random.sample`），增加训练时的多样性
  - **Part Attention**: `part_Attention` 类（L202-L232）与标准 `Attention` 相同，但在计算注意力分数后用 mask 屏蔽不可见位置（L224: `masked_fill(-1e3)`），并且 softmax 后再乘 mask（L226: `torch.mul(attn, mask)`）做双重约束
- **输入**: 图像 `[B, 3, H, W]`
- **输出**: `layerwise_tokens` — 所有层的 token 列表，每个元素 shape `[B, N, C]`。最终：
  - `layerwise_tokens[-1][:, 0]` = cls token (global feature)
  - `layerwise_tokens[-1][:, 1:4]` = 3 个 part token features
- **依赖**: 无外部依赖，纯 Transformer 模块
- **移植到我们框架的可行性**: **低-中**
  - 核心问题：PAT 是为 ViT 设计的，part token 直接参与 self-attention。我们的 backbone 是 Swin-Tiny，使用窗口注意力（window attention），不存在全局 token 的概念。直接添加 part token 需要大幅修改 Swin 的注意力机制
  - 间接方案：可以在 Swin 输出的 stage3 特征图上追加一个轻量 Transformer decoder，用 part query token + cross-attention 的方式实现类似效果（类似 DETR 的 object query 思路）
- **额外显存开销估算**:
  - 原始方式（修改 ViT）: 几乎无额外开销（仅增加 3 个 token 的参数）
  - 间接方式（额外 decoder）: ~0.3-0.5G（取决于 decoder 层数）
- **移植方案**: 在 Swin-Tiny 输出后接一个 1-2 层的 cross-attention decoder，用 3-5 个 learnable part query 作为 Q，backbone 空间特征作为 K/V，实现 part feature 提取。掩码约束可通过空间先验（上/中/下区域 mask）施加

### 模块 B: Cross-ID Similarity Learning (CSL) — 核心代理任务
- **文件位置**: `loss/smooth.py` L5-L55 (`PatchMemory`) + `loss/myloss.py` L9-L52 (`Pedal`)
- **功能**: 维护全局 part feature 记忆库，通过聚类学习跨 ID 的局部视觉相似性
- **详细机制**:
  1. **PatchMemory**（`loss/smooth.py`）:
     - 为训练集中每张图片维护一个特征中心（3 个 part token 的特征向量）
     - 使用动量更新：`agent = agent * (1 - momentum) + momentum * current_feat`
     - 输出 `(agent, position)`：agent 是所有样本的 part 中心矩阵，position 是当前 batch 样本在记忆库中的索引
  2. **Pedal 损失**（`loss/myloss.py`）:
     - 对每个 part (共 3 个)，计算当前 batch 特征与所有记忆库中心的距离
     - 排除自身后，找到最近的 k 个邻居（`k=10`）
     - 损失函数形式：log-sum-exp 对比损失 — `loss = (-log(exp_topk) + log(exp_all)).sum() / batch_size`
     - 额外过滤：排除相同摄像头的样本（camera-aware filtering）
     - 返回 top-k 邻居的身份标签 `all_posvid`，用于 soft label
  3. **Soft Label 交叉熵**（`loss/softmax_loss.py`）:
     - 使用 CSL 发现的跨 ID 相似样本构建软标签
     - `soft_target = (1 - soft_lambda) * hard_target + soft_lambda * similarity_dist`
     - 最终 CE 损失 = `(1 - soft_weight) * hard_CE + soft_weight * soft_CE`
- **输入**:
  - `feature`: `[3, B, C]` — 3 个 part 的 batch 特征
  - `centers`: `[3, N_all, C]` — 记忆库中所有样本的 part 中心
  - `position`: `[B]` — 当前 batch 在记忆库中的位置索引
- **输出**: `loss` (标量) + `all_posvid` (top-k 邻居的身份 ID 列表)
- **依赖**: 需要在训练前初始化记忆库（第一个 epoch 的前向传播不计算梯度，仅填充记忆库）
- **移植可行性**: **中**
  - PatchMemory 机制与 backbone 无关，可直接复用
  - 但需要有 part 特征输出——目前我们的 PosePartHead 已能提供 5 个 part 特征，可以直接使用
  - 计算开销：记忆库存储全训练集特征，Market1501 约 12000 张图 × 5 part × 768 维 ≈ 180MB，可接受
  - Pedal 损失中的距离计算在记忆库较大时可能较慢
- **额外显存开销估算**: ~0.2G（主要是记忆库，存在 CPU 上，仅 forward 时搬到 GPU）
- **移植方案**:
  - 利用 PosePartHead 的 `part_feats [B, K, C]` 作为 CSL 的输入
  - 修改 PatchMemory 支持 5 个 part（原来是 3 个）
  - 在训练循环中加入记忆库初始化和 Pedal 损失计算

### 模块 C: Part-guided Self-Distillation (PSD)
- **文件位置**: `processor/part_attention_vit_processor.py` L100-L121 + `model/make_model.py` L287-L298
- **功能**: 利用 part token 的信息来指导全局 cls token 的学习（隐式蒸馏）
- **实现方式**:
  - 模型返回所有层的 cls token (`layerwise_cls_tokens`) 和 part token (`layerwise_part_tokens`)
  - 但代码中实际的 PSD 主要通过 CSL 的 soft label 间接实现：CSL 从 part 特征中发现的跨 ID 相似性被编码为 soft label，这些 soft label 用于监督全局 cls token 的分类器
  - 这形成了一条信息流：part 特征 → CSL 发现相似性 → soft label → 指导全局分类
- **移植可行性**: **高** — 概念简单，只需将 soft label 传递给全局分类损失
- **额外显存开销估算**: 几乎为零

## 损失函数

### 1. ID Loss（label smoothing CE）
- 标准 label smoothing（epsilon=0.1）
- 当启用 soft label 时，增加 CSL 发现的跨 ID 相似性作为软监督信号
- 公式：`L_id = (1 - w_soft) * L_ce_hard + w_soft * L_ce_soft`

### 2. Triplet Loss
- soft margin triplet loss（无显式 margin，使用 SoftMarginLoss）
- 仅对全局 cls token 特征计算

### 3. Pedal Loss（CSL 聚类损失）
- 基于 log-sum-exp 的对比损失，鼓励 part 特征与最近 k 个邻居聚集
- 公式：`L_pedal = sum_p [(-log(sum_topk exp(-s*d_i)) + log(sum_all exp(-s*d_i))) / B]`
- 超参：scale=0.02, k=10
- 权重：PC_LR=1.0（与 reid loss 等权）

### 4. Ipfl Loss（Cycle Ranking Triplet，代码中定义但未使用）
- 基于循环排序的改进 triplet loss，使用 center-to-center 的距离
- 当前 PAT 配置未启用

## 训练 Tricks

1. **冻结 Patch Embed**（`train.py` L75-78）: MoCo v3 的 trick，冻结 patch embedding 的 Conv 层权重，提升训练稳定性
2. **记忆库动量更新**: momentum=0.1，保证特征中心的平滑演进
3. **注意力掩码随机化**: `generate_2d_mask` 中在指定区域内随机选取子矩形，每次 forward 都不同，起到数据增强效果
4. **SGD 优化器**: lr=0.001, weight_decay=1e-4, momentum=0.9（非常规 Adam，因为 DG-ReID 场景下 SGD 泛化更好）
5. **无 Random Erasing**: REA.ENABLED=False（可能因为 DG 场景下 REA 引入分布偏移）
6. **输入归一化**: mean/std=[0.5,0.5,0.5]（而非 ImageNet 标准值）
7. **Part token 初始化**: 用预训练的 cls token 权重初始化 3 个 part token，保证起点良好
8. **Camera-aware 过滤**: CSL 中排除同摄像头样本，避免学到摄像头偏置

## 该工作的局限性 / 未解决的问题

### 1. Part 划分基于固定空间先验
- 3 个 part 对应上/中/下三个区域，通过**硬编码的空间掩码**实现
- 无法适应姿态变化：弯腰、侧身、坐姿等非标准姿态下，固定的上/中/下划分与实际人体部件严重不匹配
- 虽然有随机化增强，但本质上仍依赖于"人体大致直立"的假设

### 2. 不适用于遮挡场景
- PAT 的设计目标是 DG-ReID（跨域泛化），完全没有考虑遮挡问题
- 当人体被部分遮挡时，固定区域的 part token 会编码背景/遮挡物的特征
- 没有可见性判断机制——无法区分 part 特征是否来自被遮挡区域

### 3. CSL 记忆库效率问题
- PatchMemory 使用 Python list + 线性搜索（`self.name.index(key)`），O(N) 复杂度
- 大规模数据集上（>10万张图）会显著影响训练速度
- 距离计算在 GPU 上进行，但记忆库主体在 CPU 上，频繁的设备间数据传输增加开销

### 4. 仅限 ViT 架构
- Part token 直接参与全局 self-attention，这只在 ViT 的全局注意力机制下可行
- 无法直接移植到 CNN 或窗口注意力（如 Swin）backbone

### 5. Part 数量固定为 3
- 代码中硬编码 3 个 part token 和 3 个空间区域
- 无法灵活调整部件粒度（如需要手/脚级别的细粒度特征）

### 6. Soft Label 仅用于分类器
- CSL 发现的跨 ID 相似性只通过 soft label 传递给 CE loss
- 没有直接影响特征空间的学习（如直接约束特征空间中的相似性）
- Triplet loss 仍然仅使用硬标签

## 对我们框架的改进建议

### 1. 姿态引导的 Part Token vs 固定空间先验
**核心对比**：PAT 用固定的上/中/下空间掩码来约束 part token 的注意力区域，而我们的 PosePartHead 用姿态热图做 soft attention pooling。

**关键差异**：
- PAT：part token 从第一层就参与注意力计算，逐层refinement，有深度特征交互；但空间先验固定，不适应姿态变化
- PosePartHead：直接在最终特征图上做 pose-guided pooling，准确但没有深度交互；且有 visibility 信息用于遮挡处理

**互补方案**：可以将 PAT 的 learnable part query 思路与我们的 pose guidance 结合：
- 用 pose keypoint 坐标初始化 part query 的位置先验（而非固定上/中/下），使 part attention 的空间约束自适应人体姿态
- 保留 visibility 信息，对被遮挡部位的 part query 输出降权
- 这解决了 PAT 的两个核心局限：固定空间先验 + 无遮挡感知

### 2. CSL 思路的改造
PAT 的 CSL 发现了一个有趣的视角：不同 ID 的相似局部（如"黑色书包"）共享视觉特征。这个思路可以增强我们的 part 特征学习：
- 在我们的 5-part 特征空间中实现类似的跨 ID 相似性挖掘
- 但利用 visibility 信息过滤被遮挡 part 的贡献，避免噪声

### 3. Learnable Part Query + Cross-Attention Decoder
在 Swin-Tiny 输出上接一个轻量 cross-attention decoder（1-2 层）：
- K 个 learnable part query（K=5 对应 5 个身体部位）
- 用 pose heatmap 生成空间偏置加到 cross-attention 的注意力分数上（代替 PAT 的硬掩码）
- 这结合了 PAT 的深度特征交互优势和我们的姿态精确引导优势
- 显存开销可控：1 层 cross-attention ≈ 0.1G

### 4. Part-guided Self-Distillation 可直接借用
PAT 的 PSD 思想（part 知识指导全局特征）与我们框架天然兼容：
- 用 part 特征的信息来构建全局特征的学习目标
- 具体实现：在 part triplet loss 中找到的 hard positive/negative 信息，作为全局 triplet 的额外监督

### 5. 注意力掩码随机化 → 训练时随机遮挡
PAT 中注意力掩码的随机子区域选择本质上是一种训练时增强。类似思路可以应用于我们的 pose-guided pooling：
- 训练时随机 mask 掉某些 part 的 attention（模拟遮挡），增强鲁棒性
- 这与 visibility-aware 训练形成天然配合
