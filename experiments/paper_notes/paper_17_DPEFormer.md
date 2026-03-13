# Paper 17: DPEFormer — Dynamic Patch-aware Enrichment Transformer for Occluded Person Re-Identification

**来源**: arXiv 2024（2402.10435）
**仓库**: **代码尚未开源**（"code will be made publicly available"）
**arXiv**: https://arxiv.org/abs/2402.10435
**arXiv 摘要**: 提出 DPEFormer，通过动态 patch token 选择模块（DPSM）自动过滤遮挡 token，结合特征混合模块（FBM）融合全局与局部信息，以及基于 SAM 的真实遮挡数据增强（ROA），无需外部检测器或精确对齐即可处理遮挡 ReID。

---

## 代码架构概览

**注意：该论文代码尚未开源**，以下内容基于 arXiv HTML 版本（2402.10435）的架构描述重建。

- **Backbone**: Vision Transformer (ViT-Base)
- **核心模块**: DPSM（token 选择）+ FBM（特征混合）+ ROA（数据增强）
- **特征维度**: 768（ViT-Base hidden dim）→ 最终描述子 3072（4个 part 拼接）

---

## 可拆解模块清单

### 模块 A: DPSM — Dynamic Patch Token Selection Module（动态 patch token 选择）

**功能**: 无需外部检测器，自动识别并选择人体 token，过滤遮挡/背景 token

**算法步骤**:
1. **Proxy Token 识别**: 找到与 global [CLS] token 余弦相似度最高的 patch token → 这是最强的人体区域指示器
2. **相似度评分**: 计算 proxy token 与所有其他 patch tokens 的点积相似度，得到评分集合
3. **动态阈值检测**: 将评分从高到低排序，计算相邻分数的一阶差分（梯度），差分最大的位置 = 人体 token 和遮挡/背景 token 的边界
4. **最小 token 保护**: 设置 k_min=48，防止过度过滤（确保至少保留 48 个 token 的信息）

**输入**: ViT 的所有 patch tokens `[N, L, D]`，CLS token `[N, 1, D]`
**输出**: 选中的"干净" patch tokens `[N, k, D]`（k 由动态阈值决定，约 48-196 个）

**移植可行性**: **高** — 不依赖任何外部模型，纯基于特征相似度的无监督选择
**额外显存**: ~50MB（额外的相似度矩阵计算）
**移植方案**:
- 在 Swin-Tiny 的 Stage 4 输出（`[N, 768, 12, 4]` → reshape 为 `[N, 48, 768]`）上应用
- Swin 没有 CLS token，可以用 GAP 后的全局特征代替 CLS token
- 或者用 PSG 已经知道的姿态热图直接做 token 重要性评分（更精确）

**关键洞察**: DPSM 的本质是**用全局特征做局部 token 的置信度评分**，然后动态截断。这比固定的 visibility threshold 更灵活。

### 模块 B: FBM — Feature Blending Module（特征混合模块）

**功能**: 用修改后的 multi-head self-attention 将全局上下文注入到局部 part 特征中

**算法步骤**:
1. 将 DPSM 选出的干净 tokens 分为 4 个 part（顺序划分，12 个 token 一组）
2. 对每个 part 做 1×1 conv 降维
3. **Cross-attention**: Query = part 特征，Key/Value = 全局特征 → 让每个局部特征吸收全局上下文
4. Layer Norm + FFN
5. 4个增强后的 part 特征拼接 → 3072维最终描述子

**输入**: 全局特征 `[N, D]`，part tokens `[N, 4, D/4]`
**输出**: 增强后的 part 特征 `[N, 4*D]`（本文为 `[N, 3072]`）

**移植可行性**: **高** — 标准 cross-attention，可以直接复用思路
**移植方案**: 用全局 GAP 特征作为 Key/Value，用 PSG 注入后的 Swin tokens 作为 Query

### 模块 C: ROA — Realistic Occlusion Augmentation（真实遮挡数据增强）

**功能**: 用 SAM 生成的真实物体 mask 贴到 ReID 图像上模拟真实遮挡，改善对比学习的难例样本质量

**实现步骤**:
1. 预先从 SA-1B 数据集的 100 张图中提取 9913 个物体 mask
2. 训练时随机选取一个 mask，缩放到占行人 bbox 面积的 50-75%
3. 将 mask 贴在图像的三个角位置（左上、左下、右下）+ 随机水平翻转
4. 生成的 ROA 图像作为额外的对比学习负样本（权重 0.3）

**移植可行性**: **高** — 纯数据增强，不需要修改模型
**依赖**: SA-1B 数据集 mask（可从 segment-anything 下载），或者用任意真实物体 mask 库
**额外开销**: 只是数据增强，训练速度影响可忽略

---

## 损失函数

1. **Identity Loss (CE)**：对全局特征、选中 part 特征、增强后 part 特征、最终拼接描述子各自独立计算 CE
2. **Contrastive Loss (InfoNCE 变体)**：
   - 用 backbone features（不是 classifier 输出）做 memory bank 对比学习
   - Identity centroids 作为 memory
   - ROA 增强样本权重 0.3，原始样本权重 1.0
3. **Total Loss**: CE loss（多个分支求和）+ InfoNCE loss

---

## 训练 Tricks

- Backbone: ViT-Base（不是 Swin）
- Batch size: 64（16 identities × 4 images）
- Optimizer: Adam，lr=1e-4，weight_decay=1e-4
- Schedule: cosine decay，120 epochs
- Memory bank momentum: μ=0.2
- k_min=48（最少保留 48 个 token）

---

## 该工作的局限性 / 未解决的问题

1. **DPSM 假设 CLS token 必然与人体区域高度相似**：当图像中遮挡物比人体更大时，proxy token 可能指向遮挡物而非人体
2. **顺序划分 part（不是语义划分）**：FBM 中 4 个 part 是按 token 顺序切分的（不是头/躯干/腿），可能不保证语义一致性
3. **对 ViT 的强依赖**：DPSM 依赖 ViT 的 patch token 独立性；Swin 有局部窗口注意力，token 独立性不如 ViT，直接移植需要调整
4. **ROA 需要外部 mask 库**：虽然只需预处理一次，但仍有依赖
5. **代码未开源**：无法直接验证实现细节

---

## 对我们框架的改进建议

1. **DPSM 的思路 → Pose-Guided Token Selection**:
   - DPSM 用 CLS 相似度来动态选 token，是完全无监督的
   - **我们有热图！** 可以直接用热图的响应强度做 token importance score
   - 将 17 通道热图在 spatial 维度 max-reduce → `[N, 12, 4]` 的重要性图
   - 用热图响应 > threshold 的 token 作为"可靠 token"
   - 相比 DPSM 更精确（有明确的姿态语义），且已经在我们框架里验证过热图的有效性

2. **FBM 的思路 → Pose-Part Cross-Attention**:
   - DPSM 选出来的 token 按顺序切分为 4 个 part → 语义不精确
   - 替换为：用 17 个 keypoint 热图作为 attention mask，从 Swin 特征中提取 17 个 pose-guided part features
   - 再用 FBM 风格的 cross-attention（Query=pose-guided part, Key/Value=全局特征）做特征增强
   - 这是把 DPSM+FBM 从"无监督 token 选择 + 顺序切分"升级为"姿态引导 token 聚焦 + 语义切分"

3. **ROA 值得加入我们数据增强**:
   - 我们目前的数据增强里没有真实遮挡模拟
   - SA-1B mask 可以一次性下载几百张图的 mask（9913 个 mask 够用了）
   - 贴 mask + 0.3 权重对比学习，成本很低，效果可能显著

4. **核心 gap 发现**:
   - DPSM 本质上是在解决"哪些 token 是人体的"这个问题，但用的是无监督方法（全局相似度）
   - 我们可以用姿态热图做更精确的有监督版本
   - **这个方向值得作为实验方向：pose-guided token importance scoring vs DPSM 风格的无监督 scoring**
