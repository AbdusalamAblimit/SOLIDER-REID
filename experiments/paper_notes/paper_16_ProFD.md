# Paper 16: ProFD — Prompt-Guided Feature Disentangling for Occluded Person Re-Identification

**来源**: ACM Multimedia 2024
**仓库**: https://github.com/Cuixxx/ProFD
**arXiv**: https://arxiv.org/abs/2409.20081
**arXiv 摘要**: 利用 CLIP 视觉-语言模型，通过部位特定的文本 prompt 引导特征解耦，结合混合注意力 Decoder 和自蒸馏策略处理遮挡人员重识别。

---

## 代码架构概览

- **Backbone**: CLIP ViT-B/16（视觉编码器）
- **核心文件**: `torchreid/models/profd.py`（928行）
- **训练引擎**: `torchreid/engine/image/part_based_engine.py`
- **损失函数**: `torchreid/losses/GiLt_loss.py`, `body_part_attention_loss.py`, `dissimilar_loss.py`
- **框架基础**: 基于 BPBreID (VlSomers/bpbreid) 的 torchreid 框架扩展

---

## 可拆解模块清单

### 模块 A: PromptLearnerPose（姿态 prompt 学习器）
- **文件位置**: `torchreid/models/profd.py` L797-L882
- **功能**: 将人体部位名称（head/torso/arms/legs/feet等）通过 CLIP 文本编码器编码为可学习的 prompt 向量
- **输入**: 无（forward 只输出已学习的 prompt embeddings）
- **输出**: `[K, 77, 512]`（K个部位的 token 序列）
- **部位配置**: 支持 eight/five_v/two_v/three_v/four_v 等多种粒度划分
- **关键设计**: 在 CLIP 文本 token 序列中，前缀是 learnable context tokens（X X X...），后面是部位名称的固定 embedding
- **移植可行性**: **中** — 我们框架没有 CLIP，但可以改为用简单的可学习 body-part query tokens（不需要语言模型）
- **额外显存**: ~100MB（CLIP 文本编码器保留在 CPU 加载后 forward 一次生成固定 prompt，不占 GPU 显存）

### 模块 B: PartFeatureDecoder（部位特征解码器，核心创新）
- **文件位置**: `torchreid/models/profd.py` L249-L295
- **功能**: 以文本 prompt embedding 为 Query，以视觉 spatial tokens 为 Key/Value，通过 cross-attention 解码出每个人体部位的特征
- **内部结构**: 2层 `SemiAttentionDecoderLayer`
- **输入**: text `[N, K+2, 512]`, visual spatial tokens `[N, HW, 512]`
- **输出**: part embeddings `[N, K+2, 512]` + attention maps（用于 softmax 生成 soft part masks）
- **SemiAttentionDecoderLayer 创新点** (L192-L248):
  - proxy-to-token cross attn：文本 query 从视觉中吸收信息
  - token-to-proxy cross attn：视觉 key 从文本中吸收信息（**双向交互**）
  - distill attn：额外的单头注意力（用于 distillation 信号）
  - final cross attn：最终融合
- **移植可行性**: **高** — 可以将文本 prompt 替换为 pose-guided learnable queries（如用热图引导的 K 个查询）
- **额外显存**: ~200MB（2层 cross-attention，batch=64，K=6）

### 模块 C: Dissimilar Loss（部位多样性损失）
- **文件位置**: `torchreid/losses/dissimilar_loss.py`
- **功能**: 鼓励不同 part embeddings 之间相互不同（最大化 cosine 距离），防止部位特征collapse
- **公式**: 计算 [N, K, K] part-pair cosine similarity 矩阵的上三角均值（并用 softmax 加权使相似度高的 pair 权重更大）
- **移植可行性**: **高** — 只需 part embeddings 即可，可直接用
- **额外显存**: negligible

### 模块 D: ClusterMemoryAMP（聚类对比记忆库）
- **文件位置**: `torchreid/losses/ClusterMemoryAMP.py`
- **功能**: 维护每个 identity 的特征中心（centroid），用对比学习（InfoNCE 变体）拉近同类特征、推开异类
- **使用方式**: 先用全部训练图片提取特征、聚类到每个 identity，作为 memory bank 初始化；每次 forward_backward 都更新 memory（momentum=0.2）
- **计算代价**: 需要在训练开始前跑一次全量特征提取（增加启动时间约 1-2 分钟），但训练中 memory loss 本身很轻量
- **移植可行性**: **中** — 我们框架里可以加类似的 prototype 对比损失

### 模块 E: GiLt Loss（分层身份+三元组组合损失）
- **文件位置**: `torchreid/losses/GiLt_loss.py`
- **功能**: 全局特征用 ID 损失（CE），part 特征用 triplet 损失；visibility-weighted
- **关键**: 使用 DAI（Dynamic Attention-Inverse）做 visibility-weighted softmax 对 part CE 损失加权
- **移植可行性**: **高** — 已在 BPBreID 中验证，和我们的框架思路高度吻合

### 模块 F: BodyPartAttentionLoss（部位分割监督损失）
- **文件位置**: `torchreid/losses/body_part_attention_loss.py`
- **功能**: 对模型预测的 pixel-level part classification scores 和 ground-truth parsing labels 之间做 CE 损失
- **依赖**: 需要 external human parsing labels（PifPaf 或 HRNet 生成的 mask）
- **移植可行性**: **中** — 需要 parsing 数据；我们有热图，可以考虑用热图作为 soft supervision

---

## 损失函数完整组合

训练时同时使用以下损失（`part_based_engine.py` L148-L221）：

1. **GiLt Loss**（ID CE + Triplet）：主要 ReID 损失
2. **Body Part Attention Loss**（BPA）：监督 pixel-level part 分类（weight=0.35）
3. **Attention Consistency Loss**（attn_scores vs target parsing，KL 散度形式）
4. **ClusterMemoryAMP（PCL）**：全局 + 拼接 part 特征的对比学习
5. **Dissimilar Loss**：part 特征多样性
6. **Visibility Focal Loss**：监督 part visibility 预测（weight=10.0）

---

## 训练 Tricks

- 输入分辨率：384×128
- Batch size：64
- Backbone：CLIP ViT-B/16（stride_size 可配置）
- 特征维度：512（CLIP space）
- Memory bank：每个 epoch 开始前提取全量特征聚类初始化

---

## 该工作的局限性 / 未解决的问题

1. **依赖 CLIP**：整个框架绑定 ViT-B/16 CLIP，无法直接换成 Swin-Tiny
2. **依赖外部 parsing labels**：需要 PifPaf/HRNet 生成的部位分割图，多了一个离线预处理步骤
3. **部位语义模糊**：虽然用了 CLIP text prompt，但实际上 PartFeatureDecoder 的输出部位对应关系仍然是通过 soft attention 学出来的，没有显式的部位定位精度保证
4. **推理时多分支 concatenation**：测试时用 `bn_foreg + parts` 拼接，需要调节比例；不够简洁
5. **Part collapse 问题**：需要 Dissimilar Loss 才能防止 part 特征退化，说明 PartFeatureDecoder 本身有 collapse 倾向

---

## 对我们框架的改进建议

1. **PartFeatureDecoder 思路可移植**：
   - 将文本 prompt 替换为 pose-heatmap-guided learnable queries（K个体部位 query）
   - 用 heatmap 注意力初始化 query，然后让 cross-attention 从 Swin 特征中提取部位信息
   - 这样可以借用 ProFD 的双向 SemiAttentionDecoder 结构，但不依赖 CLIP

2. **Dissimilar Loss 直接复用**：
   - 我们的 GCN/KPP branch 已经有多个 part embeddings，加 Dissimilar Loss 防止 collapse 很合理
   - 计算成本很低，可以直接添加

3. **ClusterMemoryAMP 值得尝试**：
   - 作为 global branch 的额外对比信号（不是主创新，但可能带来稳定的 +0.5~1% 提升）
   - 缺点是需要 warmup 期先提取特征

4. **关键可移植 gap**：
   - ProFD 中 PartFeatureDecoder 的 query 是文本 prompt（语义驱动）
   - 如果换成**姿态热图驱动的 query**（spatial 驱动）会更加 body-aligned
   - 这是一个值得研究的方向差异：**语义对齐 vs 空间对齐**

5. **与 exp030a-PSG 的结合**：
   - PSG 已经在 backbone 内注入了热图信息（spatial 层面）
   - 在 backbone 输出端加一个 lite PartFeatureDecoder（pose-query 版本）做 part 解码
   - 可能比 PSG + attention-pooling 更强（因为 cross-attention 比 elementwise 乘法更有表达力）
