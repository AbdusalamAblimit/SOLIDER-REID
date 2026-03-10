# 创新点头脑风暴 — Phase 2: Pure Pose Heatmap

## 前轮教训 (Phase 1, 33 experiments)
- ViTPose visibility 向量不够可靠（AP 相关性仅 0.237）
- 中间层 visibility modulation 有害（破坏预训练空间结构），但这是 visibility 特有问题
- PCFC alpha suppression 是该框架特有的脆弱平衡点
- NFC test-time 方法有效但不算训练端创新
- **关键结论**: 不要用 visibility 向量，用原始 pose 热图

## 本轮方向: 纯 Pose Heatmap

### 核心优势
1. 热图是连续值（0~1），比二值 visibility 更 rich
2. 热图自然编码遮挡信息（遮挡区域响应低）
3. 热图保留空间结构（17个关键点的2D位置）
4. mmpose 提供多种鲁棒模型（RTMPose、HRNet-W48、ViTPose-Large）

### 候选创新方向

#### 方向 A: Pose Heatmap Attention Injection
- 核心想法: 将 pose heatmap 下采样到与 Swin feature map 相同大小，作为 spatial attention bias 注入 self-attention
- 技术路线: heatmap → Conv downsample → attention bias (加到 QK^T 上)
- 与前轮的区别: 前轮用 visibility 做 token scaling，这次用热图做 attention routing
- 注入位置: 可以在每个 Swin stage 都注入（不同分辨率）
- 预期: backbone 在每一层都知道人体结构 → 更好的遮挡感知

#### 方向 B: Pose-Guided Part Pooling (改进版)
- 核心想法: 用热图代替固定 Gaussian，做 soft part pooling
- 与前轮 PCFC 的区别: PCFC 用 Gaussian(keypoint, sigma) 作为 attention map，这次用 pose model 直接预测的热图
- 热图本身就是"该关键点存在的概率分布"，比 Gaussian 更准确
- 特别是对非正面/非标准姿势的人，热图比 Gaussian 更贴合实际

#### 方向 C: Pose Structure Graph Network
- 核心想法: 将 17 个关键点构建为图结构（骨骼连接），用 GCN 学习结构化 part representations
- 从 Swin feature map 在每个关键点位置提取特征 → GCN propagation → 结构化 part features
- 创新点: 利用人体骨骼连接关系做特征传播（可见部位 → 遮挡部位）

#### 方向 D: Multi-Resolution Pose Injection
- 核心想法: 在 Swin 的不同 stage 注入不同分辨率的热图
- Stage 1 (48x16, 96ch): 高分辨率热图，细粒度空间引导
- Stage 2 (24x8, 192ch): 中分辨率热图，部位级引导
- Stage 3 (12x4, 384ch): 低分辨率热图，全局结构引导
- 每个 stage 的注入方式可以不同（add/concat/cross-attention）

#### 方向 E: Pose-Conditioned Feature Masking
- 核心想法: 训练时根据热图概率 mask 部分 spatial tokens，迫使模型从可见 tokens 推断遮挡区域
- 类似 MAE (Masked Autoencoder) 但用 pose 引导 mask 策略
- 与前轮 PGFC 的区别: PGFC 用 visibility 做 hard replacement，这次用热图做 soft probabilistic masking

## 推荐主攻方向
**方向 A (Pose Heatmap Attention Injection)** — 最直接、最有创新性
- 直接修改 Swin attention 机制，是架构级创新
- 每个 stage 都注入 pose 信息，是"深层 pose conditioning"
- 论文 story 清晰: "将人体结构先验注入 Transformer 注意力机制"

**备选**: 方向 D (Multi-Resolution) 如果 A 效果好，可以扩展为 multi-resolution 版本

## 重要约束
- **训练侧创新为核心**: NFC/RR 等 test-time 方法不算公平对比（所有 SOTA 都可以用），训练侧才是论文贡献
- **Pose 模型可以参与训练**: 不限于离线热图，可以将冻结的 pose 模型作为在线特征提取器
- **可以大胆修改中间层**: 与 Phase 1 不同，本轮使用更可靠的热图（非 visibility），中间层修改值得重新探索

## 新增方向

#### 方向 F: Pose Feature Cross-Attention (在线)
- 核心想法: 使用冻结的 RTMPose 模型提取 pose feature maps，通过 cross-attention 将 pose 信息注入 Swin backbone
- 具体: Swin feature map (Q) × Pose feature map (K,V) → pose-aware features
- 优势: RTMPose 的中间特征比 17 个关键点热图更丰富（包含人体结构上下文）
- 关键: pose 模型冻结不训练，只提供 spatial structure prior
- 论文 story: "将 pose 估计网络的结构化知识蒸馏到 ReID backbone 中"

#### 方向 G: Dual-Task Pose-ReID (联合训练)
- 核心想法: 在 Swin backbone 上同时做 ReID + pose estimation，共享底层特征
- Swin stage 1-2 共享 → stage 3 分为 ReID head 和 pose head
- Pose head 的梯度帮助 backbone 学习更好的空间结构
- 与前轮 VPReID (exp002) 的区别: 前轮用独立的 ViTPose backbone，本次共享底层特征
- 风险: 多任务平衡（前轮 exp002 就是因此失败）

---

## Phase 2 实验反馈总结 (exp001-exp015)

### 已证实的核心发现

**1. Backbone Injection > Post-hoc Pooling**
- PSG (特征形成阶段注入 pose) +1.7% mAP，仅 102K params
- Part Pooling (特征形成后用 pose 选择) +0.9% mAP，~2.6M params
- PFM (后置调制) 中性效果
- **结论**: 让 backbone 在特征提取过程中知道人体结构，比事后选择更有效

**2. PSG 对干扰极度敏感**
- PSG + PAB Combo: ❌ (-0.7% vs PSG-only)
- PSG + Part Pooling: ❌ (-0.6% vs PSG-only)
- PSG + Part Supervision (global test): ❌ (-0.7% mAP, -2.1% R1)
- PSG + Spatial Conv (3×3 DWConv): 🟡 持平 (mAP 58.3% = exp007, R1 -0.8%)
- **结论**: 任何改变 PSG 梯度流的额外模块/损失都会降低效果。PSG 58.3% 是该框架的理论上限。

**3. PSG 内部结构已达最优**
- 1×1 conv gate 足够，3×3 depthwise conv 冗余
- Multi-stage (S2+S3) 无额外收益
- 200 epochs 无额外收益
- **结论**: PSG 的瓶颈不在感受野、不在训练长度、不在注入位置

### 战略反思

**当前困境**: PSG 58.3% 是 +1.7% mAP 的稳定改进，但对论文来说幅度不够。且无法通过组合/扩展进一步提升。

**需要的突破方向**: 必须从 PSG 之外找到 orthogonal 的提升。两条路：
1. **训练策略改进** — 不改模型架构，改训练方式（数据增强、损失函数设计）
2. **全新模型设计** — 跳出 PSG 的 feature gating 范式，探索其他利用 pose 的方式

### 新增候选方向

#### 方向 H: Pose-Guided Erasing (PGE)
- **核心想法**: 用 pose 热图引导数据增强，替代 Random Erasing。训练时随机选择 1-2 个身体部件组（头部、上身、下身），基于热图 mask 对应区域，迫使模型从部分身体学习身份
- **与 Random Erasing 的区别**: RE 随机放置矩形框，PGE 基于语义部件区域做结构化擦除
- **优势**: (1) 训练时 zero 额外计算 (2) 完全正交于 PSG (3) 模拟真实遮挡
- **论文 story**: "Random Erasing 不能模拟真实遮挡（人的遮挡是整个部件被挡住），PGE 通过 pose 引导的结构化擦除训练模型应对真实遮挡"
- **风险**: 如果 RE 已经足够模拟遮挡，PGE 可能无额外收益
- **论文位置**: 消融表格（PGE vs RE）、主实验表（PSG + PGE 的联合效果）

#### 方向 I: Pose-Conditioned Normalization (PCN)
- **核心想法**: 用 pose 热图调制 LayerNorm/BatchNorm 的 scale 和 bias 参数，类似 SPADE (image generation)
- **实现**: 对每个空间位置，根据 pose 热图生成 γ(pose) 和 β(pose)，替代标准 LN 的可学习参数
- **与 PSG 的区别**: PSG 是 feature gating（乘法），PCN 是 normalization modulation（影响分布）
- **风险**: 可能与 PSG 类似——都是空间调制，可能不叠加

#### 方向 J: 关键点引导的对比学习 (Keypoint-Conditioned Contrastive Learning)
- **核心想法**: 修改 triplet loss，使其考虑 pose 相似度。Pose 相似的正样本对应更高 margin
- **动机**: 两张同一人但 pose 完全不同的图片（一张正面站立、一张侧面行走）比两张 pose 相似的图片更难匹配
- **实现**: margin = base_margin * (1 + λ * pose_dissimilarity)
- **风险**: 前轮 GiLt 已用 part triplet，这个方向可能与之冲突
