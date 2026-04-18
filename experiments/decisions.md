# 决策记录 — Phase 2: Pure Pose Heatmap

### [2026-03-07 19:30] 决策 #1

**上下文**: Phase 1 (33 个实验) 已充分探索了基于 ViTPose visibility 向量的所有方向。最佳训练端改进仅 +1.4%（GiLt+PCFC）。用户指示放弃 visibility 方向，转向纯 pose heatmap + mmpose 更鲁棒模型。

**选择**: 从 SOLIDER 作者原始代码重新开始，纯 pose heatmap 方向
**理由**:
1. Visibility 向量不够可靠（与 AP 相关性仅 0.237）
2. PCFC alpha suppression 限制了所有后续改进
3. 纯 pose 热图更原始、更可靠、更多论文验证过有效性
4. 从干净代码开始避免 33 个实验的代码污染

**执行计划**:
1. 先跑 baseline 确认可复现（预期 mAP ~56.6%）
2. 选择并部署 mmpose 模型进行 pose heatmap 提取
3. 设计新的 heatmap 利用方式

### [2026-03-07 19:22] 决策 #2

**上下文**: 用户补充了多条重要指导：
1. 姿态模型可以参与训练（不限于离线热图）
2. 尽可能做训练侧创新（NFC/RR 等 test-time 方法不够公平）
3. 可以大胆修改 backbone 中间层
4. 数据必须从 log 文件读取，不能凭记忆

**选择**: 以训练侧创新为核心方向
**理由**:
1. NFC/Re-ranking 等 test-time 方法所有 SOTA 都可以用，不算公平的对比
2. 训练侧创新才是论文的核心贡献
3. 用户允许 pose 模型参与训练 + 修改中间层 → 创新空间大大增加
4. 可以考虑：在线 pose 特征注入、pose-guided attention、pose 结构约束等

**潜在技术方向**:
- 冻结的 pose 模型提取特征 → 通过 cross-attention 注入 Swin 中间层
- Pose heatmap 作为 spatial attention bias 直接修改 window attention
- Pose 骨骼结构约束 part features 之间的关系

### [2026-03-09 11:55] 决策 #3

**上下文**: exp001 (Pose Part Pooling with sigmoid) 完成。结果：mAP 57.1% (+0.5%), R1 66.7% (+0.2%)。有效但提升有限。关键发现：id_part 收敛极慢（最终仍在 2.0 vs id_global 0.2），说明 sigmoid 热图在 12×4 分辨率的 soft attention pooling 不够 discriminative。

**选项**:
  A. 改进 part pooling：使用 spatial softmax 代替 sigmoid，增强热图峰值对比度
  B. 放弃 part pooling，转向 pose heatmap 作为 attention bias 注入 Swin backbone
  C. 使用更高分辨率的中间层特征（stage2: 24×8 而非 stage3: 12×4）

**选择**: 先试 A（spatial softmax 改进），如果 id_part 收敛改善但最终结果仍有限，再转 B
**理由**:
1. exp001 证明 part pooling 方向有效（+0.5% mAP），但 id_part 是瓶颈
2. Spatial softmax 是最小改动，只改一行代码就能验证 "热图对比度不够" 的假设
3. 如果 id_part 收敛问题解决，part pooling 可能有更大提升空间
4. 如果 A 验证后仍不够，则 B 是完全不同的方向，有更大的创新性

**执行结果**: exp002 结果 mAP 57.2% vs exp001 57.1%，几乎无差异。id_part 训练中期收敛更快但最终效果相同。**结论：归一化方式不是瓶颈，转向方案 B。**

### [2026-03-09 14:13] 决策 #4

**上下文**: exp001 和 exp002 结果对比完成。两种归一化方式（sigmoid vs spatial_softmax）效果几乎一致。特征模式消融发现 part-only > concat > global，说明 part 特征有效但融合方式有问题。

**关键发现**:
1. 两种 normalization 最终 part-only mAP 都是 57.5%（+0.9% vs baseline）
2. Concat 融合反而比 part-only 差（1/N scaling 稀释信号）
3. id_part 收敛慢不是 normalization 的问题，而是 12×4 分辨率下 part 区分度本身有限

**选项**:
  A. 改进 part pooling 的融合方式（如 learnable weights, attention-based fusion）
  B. 转向 pose heatmap attention bias 注入 Swin backbone 中间层
  C. 提高 part 特征图分辨率（使用 stage2 特征 24×8）

**选择**: A — 改进特征融合方式。Part 特征已被证明有效（+0.9% mAP），但融合方式拖累了整体效果。这是最直接的改进方向。

**理由**:
1. Part-only 已经超 baseline 0.9%，说明 part 学到了有用信息
2. 当前 concat 的 1/N scaling 太朴素，直接稀释了 part 信号
3. 改进融合方式是低风险高回报：不需要改 backbone，只需修改测试时的特征组合
4. 如果简单的融合改进有效，可以作为消融实验的重要证据
5. B 和 C 是更大的改动，作为备选

**具体方案**: exp003 — 移除 1/N scaling，等权拼接 global + parts；或测试只用 part-only 作为最终特征

**执行结果**: exp003 在 ep60 终止，mAP 50.2%（-6.4% vs baseline）。降低 global loss weight 严重伤害 backbone 特征质量。Part 分类器虽学得更快（id_part 2.08 vs exp001 ~3.3），但池化的 backbone 特征变差了。**结论：global 和 part 是共生关系，不能通过削弱 global 来强化 part。**

### [2026-03-09 15:32] 决策 #5

**上下文**: exp001-003 完成。核心发现：
1. Part pooling 有效（+0.9% mAP with part-only feature）
2. 归一化方式（sigmoid vs spatial_softmax）无差异
3. 融合方式：part-only > concat（1/N scaling 有害）
4. 降低 global weight 反而伤害 part（因为 backbone 质量下降）
5. id_part 收敛极慢是核心瓶颈（id_part≈2.0 vs id_global≈0.2）

**关键问题**: 如何在不削弱 backbone 的前提下增强 part 学习？

**选项**:
  A. 独立 Part BN+分类器 + 更高 LR（加速 part 收敛，不改 global loss weight）
  B. 转向 Direction B：Pose heatmap 作为 attention bias 注入 Swin backbone 中间层（全新方向）
  C. Part feature 使用更高分辨率特征图（stage2: 24×8 而非 stage3: 12×4）
  D. 在 Part head 加入额外的 self-attention 层增强 part 特征表达
  E. 改进 Part 学习信号：per-part triplet loss (GiLt) + part-specific augmentation

**选择**: E — Per-part triplet loss (GiLt)
**理由**:
1. Phase 1 中 GiLt 已证明有效（+0.5% on top of PCFC），这次在 pure heatmap 框架下重试
2. 当前 part triplet 是"所有 part 特征拼起来做一个 triplet"，每个 part 没有独立的 hard positive/negative mining
3. Per-part triplet 让每个 part 独立学习判别性特征，直接解决 id_part 收敛慢的问题
4. 最小改动：只改 loss 计算方式，不改 backbone 或 part pooling 模块
5. 如果 GiLt 有效，可以作为消融实验的重要证据（"per-part triplet vs global triplet"）

**方案 B 作为备选**：如果 GiLt + part pooling 组合无法超过 +1.5% mAP，则转向全新的 backbone attention 方向。

**执行结果**: exp004 PFM 是中性结果。mAP 与 exp001 part-only 相同（57.5%），R1 反而下降 0.8%。PFM 加速收敛但不改善最终表征。**结论：不要在同一处重复使用 pose 信息（PFM+part pooling 是冗余的）。**

### [2026-03-09 17:52] 决策 #6

**上下文**: exp001-004 已探索了当前 part pooling 架构的多个变体：
- exp001/002: 不同热图归一化（sigmoid vs spatial_softmax）→ 无差异
- exp003: 改变 loss 权重 → 负面
- exp004: 加 PFM feature modulation → 中性

当前最佳：mAP 57.5% (part-only), R1 67.1% (+0.9%/+0.6% vs baseline)

**核心瓶颈**: id_part ≈ 2.0 无法进一步降低，part 特征质量受限于 12×4 分辨率。

**选项**:
  A. 使用 stage 2 特征 (24×8, 384ch) 做 part pooling — 4× spatial resolution
  B. Part diversity loss — 惩罚 part 特征间的相似度
  C. 转向 backbone attention 注入 — 修改 Swin 中间层
  D. Part-specific data augmentation — 基于 pose 的部位级数据增强
  E. Adaptive global-part fusion — 学习动态融合权重

**选择**: A — 使用 stage 2 高分辨率特征做 part pooling

**理由**:
1. 当前 12×4 分辨率对 5 个 part 来说太粗（每个 part 只能覆盖 2-3 个 spatial position）
2. Stage 2 (24×8 = 192 positions) 提供 4× 空间分辨率，pose heatmap attention 可以更精确
3. 384 channels 虽然比 768 少，但仍然有丰富的语义信息
4. 实验简单：只需改一下 part pooling 使用的特征图来源
5. 如果分辨率是瓶颈，这个实验会看到 id_part 明显改善

**风险**: Stage 2 特征可能不够 semantic（还没经过 stage 3 的进一步抽象）

**执行结果**: exp005 明确负面。ep40 mAP 仅 37.0%（baseline 56.6%）。id_part 到 ep49 才降到 4.70（exp001 同期 ~2.0）。**确认：Stage 2 特征语义不足以支撑 part-level identity 分类。** 更高空间分辨率无法补偿语义信息的缺失。

### [2026-03-09 19:10] 决策 #7

**上下文**: exp005 证明浅层（stage 2）特征不够 semantic。exp001-005 总结：
- Part pooling 方向的上限在 +0.9% mAP（part-only mode）
- Fusion 方式（concat_scaled vs equal_concat）对最终结果影响约 0.4%
- PFM 是冗余的；stage 2 太浅
- id_part 收敛始终慢于 id_global

**核心问题**：如何突破 +0.9% 的瓶颈？

**选项**:
  A. 改进 test-time 融合（L2-norm concat）— 可能挤出 0.2-0.5%，但不需要训练
  B. 转向 backbone attention 注入 — pose 信息参与特征形成过程
  C. 多尺度 part pooling — stage 2 spatial + stage 3 semantic 的融合
  D. Part feature diversity loss — 惩罚 part 间的冗余
  E. Pose-guided token selection — 剔除无关 token 提高效率

**选择**: 先做 A（test-time L2-norm fusion，零成本验证），然后转 B（backbone attention）

**理由**:
1. A 不需要训练，5 分钟可验证，如果能把 +0.9% 提升到 +1.2%，对论文有价值
2. B 是全新方向，改变特征形成过程本身，可能突破当前 part pooling 的上限
3. C-E 仍在 part pooling 框架内优化，上限有限
4. B 的创新性更好（"pose-conditioned attention" vs "better pooling"），更适合论文

**执行结果**:
- exp006 (A): L2-norm concat 57.4% vs concat 57.2%，小改进但仍不如 part-only (57.5%)。融合方向上限已到。
- **exp007 (B): PSG backbone injection → mAP 58.3%, R1 67.9%。Phase 2 最佳结果！+1.7% mAP, +1.4% R1。超过 Phase 1 最佳 (58.0%/68.0%)。**

### [2026-03-09 21:25] 决策 #8

**上下文**: exp007 PSG 取得突破性结果 (58.3%/67.9%)。关键发现：
1. Backbone-level pose injection (+1.7%) 远优于 post-hoc part pooling (+0.9%)
2. 纯 global feature，无需 part branch，架构极简
3. 额外参数仅 102K（两个 PSG 模块），几乎不增加计算量
4. 已超过 Phase 1 最佳

**下一步方向**:
  A. PSG + Part Pooling 组合 — 让 backbone 和 part branch 同时利用 pose
  B. PSG 消融实验 — 证明 PSG 每个组件的必要性
  C. PSG 在不同 stage 注入 — Stage 2 vs Stage 3 vs 全部 stages
  D. PSG 超参数分析 — hidden_dim, 是否 sigmoid, etc.

**选择**: A — PSG + Part Pooling 组合

**理由**:
1. PSG global feat (58.3%) 和 part-only feat (57.5%) 都有各自的优势
2. PSG 改善了 backbone 特征质量 → part features 也应该受益
3. 组合后可能进一步提升（PSG backbone + enhanced part features）
4. 如果组合有效，这就是完整的方法（backbone injection + part pooling = 全方位 pose 利用）

**执行结果**: exp008 mAP 57.7%, R1 66.0%。**组合不叠加**，低于 PSG-only (58.3%/67.9%)。Part pooling 的 part_only 测试模式丢弃了 PSG 增强的 global feature，而 part features 本身无法匹配 PSG-global 的质量。**结论：backbone-level injection 是更有效的 pose 利用方式，post-hoc pooling 在 PSG 基础上没有增量价值。**

### [2026-03-09 23:35] 决策 #9

**上下文**: exp007 (PSG) 和 exp008 (PSG+Part) 的对比揭示了重要规律：
1. PSG backbone injection: mAP 58.3% (+1.7%) — 全局特征，无 part branch
2. PSG + Part Pooling: mAP 57.7% (+1.1%) — part_only 测试，丢弃 global
3. Part Pooling alone: mAP 57.5% (+0.9%) — exp001

**核心洞察**:
- PSG 的增益主要来自改善全局特征质量，而 part pooling 依赖的是局部特征
- 两种方法的增益来源有重叠：都利用 pose heatmap 做 spatial attention
- 在 part_only 测试模式下，PSG 增强的 global 特征被浪费了

**选项**:
  A. PSG + concat 融合 — 保留 PSG global + part features，不丢弃全局特征
  B. 多 stage PSG — 在 Stage 2 也注入 PSG，更早引入 pose 先验
  C. PSG 改进 — 更强的 gate 机制（如 channel attention, multi-head gate）
  D. Backbone freeze warmup — 冻结 backbone 前 5 epochs，防止随机初始化模块破坏预训练

**选择**: 先做 B（多 stage PSG），这是架构级改进，有更大创新潜力

**理由**:
1. 当前 PSG 只在 Stage 3（2 个 block）注入，信息利用有限
2. 多 stage 注入可以让 pose 信息更早参与特征形成（Stage 2 的 24×8 分辨率对 pose heatmap 更有利）
3. exp005 证明 Stage 2 特征不足以直接做 identity classification，但这不代表 Stage 2 不适合做 spatial attention（PSG 不做分类，只做 spatial gating）
4. 多 stage PSG 是论文中可以画出更好架构图的设计
5. 如果多 stage 有效，这构成了一个"层次化姿态注入"的创新点

**执行结果**: exp009 mAP 58.3%, R1 67.2%, R5 81.2%, R10 85.2%。Multi-stage PSG (Stage 2+3) 与 single-stage (Stage 3 only) mAP 持平，R1 略低（-0.7%），R5/R10 略优（+0.4%/+0.3%），但增加了 156K 额外参数。**结论：Stage 2 PSG 无显著收益，pose spatial gating 在 Stage 3 已足够。后续聚焦于改进 PSG 机制本身，而非扩展注入范围。**

### [2026-03-10 01:45] 决策 #10

**上下文**: Phase 2 九个实验的系统总结：
1. Post-hoc part pooling 上限: +0.9% mAP (exp001)
2. PSG backbone injection 最佳: +1.7% mAP (exp007)
3. PSG + Part Pooling 组合: 不叠加 (exp008)
4. Multi-stage PSG: 无额外收益 (exp009)

**核心发现**: PSG Stage 3 (2 blocks, 102K params) 是当前最优配置。进一步改进需要改变 PSG 的内部机制或训练策略。

**选项**:
  A. PSG Channel Attention — 在 spatial gate 基础上加 channel-wise attention (SE-style)
  B. Backbone Freeze Warmup — 冻结 backbone 前 N epochs，让 PSG 先学稳定的 gate 模式
  C. PSG + Global-Part Concat — concat PSG-global + part features (不是 part_only 测试)
  D. PSG 超参数搜索 — hidden_dim, sigmoid vs tanh, gate 初始化方式
  E. Pose-Conditioned Attention (PCA) — 替代简单的 channel gate，用 pose 调制 self-attention QKV

**选择**: B — Backbone Freeze Warmup

**理由**:
1. 用户曾建议冻结 backbone 前 5 epochs，这个想法值得验证
2. 当前 PSG 零初始化，但训练初期 backbone 的梯度（来自 ID loss 和 triplet loss）会同时更新 backbone 和 PSG，可能让 PSG 来不及学到好的 gate pattern 就被 backbone 适应掉了
3. 冻结 backbone warmup 让 PSG 先在"固定"的特征空间上学习 pose-to-gate mapping，之后解冻 backbone 时 PSG 已有良好初始化
4. 实现简单：只需修改训练循环，前 N epochs 冻结 backbone 参数
5. 如果有效，这个训练策略也是论文素材（"stage-wise training for pose injection"）

**执行结果**: exp010 彻底失败。ep30 mAP 仅 12.5%（exp007 ep30 ~49%），提前终止。冻结 backbone 导致：(1) 解冻后特征空间剧变使 PSG/classifier 失效 (2) PSG 学到错误的 gate pattern 产生负面干扰 (3) 训练指标看似正常但测试表现灾难性。**教训：PSG 必须和 backbone 端到端同时训练。**

### [2026-03-10 02:20] 决策 #11

**上下文**:
- exp007 PSG Stage 3: mAP 58.3% (+1.7%) — 当前最佳
- exp008 PSG+Part: 不叠加
- exp009 Multi-stage PSG: 无额外收益
- exp010 Freeze warmup: 灾难性失败
- **PSG 的基本配置（Stage 3, zero-init, 102K params）已被证明是最优的**
- 需要从不同角度改进

**选项**:
  A. PSG + Channel Attention (SE-style) — 增加 channel-wise gating 维度
  B. Pose-Conditioned Self-Attention — 让 pose 调制 Swin 的 self-attention (QKV)
  C. PSG + Global-Part Concat — 保留 PSG-global + part features 的拼接测试
  D. 更强的 gate 网络 — 增加 3x3 conv, 多层, 或加 spatial conv
  E. 训练策略改进 — 更长训练(160/200 epochs)、不同 LR schedule、不同 optimizer

**选择**: E — 更长训练 (200 epochs)

**理由**:
1. 所有 PSG 实验都在 ep100-120 还在上升，曲线没有完全收敛
2. exp007 的 mAP 轨迹：ep80 56%, ep100 58.2%, ep110 58.2%, ep120 58.3% — 后段仍有微弱上升
3. exp009 也是后发优势型：ep50 才超过 exp007 同期
4. 200 epochs 可以测试 PSG 是否有更大的后段潜力
5. 这是零风险实验：同样的模型，只是训练更久
6. 如果 120→200 epochs 带来 0.5-1% 的额外提升，说明 PSG 确实需要更长训练
7. 对论文也有价值：可以报告"our method continues to improve with longer training"

**执行结果**: exp011 最终 mAP 58.3%, R1 67.6%。与 exp007 (120ep) 完全相同的 mAP，75% 更多训练时间无收益。**结论：PSG 的性能上限由架构决定（~58.3% mAP），120ep 已足够。需要架构创新来突破。**

### [2026-03-10 05:48] 决策 #12

**上下文**: Phase 2 实验全面总结（11 个实验）：
- **有效**: PSG Stage 3 (+1.7%), Part Pooling (+0.9%)
- **无效/中性**: Multi-stage PSG (=), PFM (=), Stage 2 Parts (❌), Part-Dominant (❌)
- **有害**: Freeze Warmup (❌❌)
- **无收益**: 200ep extended training (=)

PSG Stage 3 (mAP 58.3%) 是确认的性能上限。所有尝试过的改进方向都无法突破这个上限。需要全新的方法。

**核心反思**: 当前 PSG 是一个简单的 spatial gate (17→64→768 的 1×1 conv)。它做的是"根据 pose heatmap 在空间维度上调制特征"。这个方法的局限性：
1. 1×1 conv 没有空间感受野 — 每个位置独立处理，不考虑邻域关系
2. 纯 channel-wise output — gate 对每个 channel 产生相同的 spatial weight
3. 不影响 self-attention — PSG 在 SwinBlock 之后作用，不改变 attention 计算
4. 单向信息流 — pose → feature，没有 feature → pose 的反馈

**选项**:
  A. Pose-Conditioned Self-Attention (PCA) — 将 pose 信息注入 Swin 的 self-attention 中（如 attention bias, KV 修改）
  B. PSG 3x3 conv — 在 PSG 中加 3×3 conv（depthwise），让 gate 有空间感受野
  C. Channel Attention PSG — 在 spatial gate 基础上加 channel attention (SE-style)
  D. PSG + Global-Part Concat Test — 测试时 concat PSG-global + part features（非 part_only）
  E. Label Smoothing / Stronger Augmentation — 训练技巧改进

**选择**: A — Pose-Conditioned Self-Attention (PCA)

**理由**:
1. **最高创新潜力**: 修改 self-attention 是比 post-block gate 更深层的干预，直接改变信息交互模式
2. **论文故事**: "从 post-hoc pooling → post-block gating → attention-level injection" 构成清晰的技术递进
3. 实现方式：将 pose heatmap 编码为 attention bias，加到 Swin 的 window attention 分数上
4. 这改变了 token 之间的 attention 权重，比只改变 token 值（PSG 的做法）更根本
5. 如果 PCA > PSG，这是一个强有力的消融证据

**执行结果**: exp012 最终 mAP 57.4%, R1 67.3%。PAB 有效但弱于 PSG（-0.9% mAP, -0.6% R1）。尽管只有 5.4K 参数，attention bias 的调制效果不如 feature gate。**结论：在 Swin 的 window attention 中，additive bias decomposition (bias(i,j) = val[i] + val[j]) 的表达能力有限，softmax 压缩了 bias 的效果。Feature-level gating (PSG) 仍是更有效的 pose 注入方式。**

### [2026-03-10 08:05] 决策 #13

**上下文**: 12 个实验已完成。PSG 仍是最佳方法 (mAP 58.3%)。已验证：
- Post-hoc part pooling: +0.9% (exp001)
- Backbone feature gating (PSG): +1.7% (exp007) ← BEST
- Attention-level bias (PAB): +0.8% (exp012)
- Multi-stage PSG: 无额外收益 (exp009)
- PSG+Part 组合: 不叠加 (exp008)
- 200ep: 无额外收益 (exp011)
- Freeze warmup: 灾难性 (exp010)

**关键排序**: Feature gating (PSG) > Post-hoc pooling > Attention bias (PAB)

**核心问题**: 如何突破 58.3% 的性能上限？

**选项**:
  A. PSG + PAB 组合 — 同时做 feature gating 和 attention bias，双重 pose 注入
  B. PSG + 3×3 Depthwise Conv — 给 PSG 加空间感受野
  C. Cross-Attention Pose Injection — 用 pose token 和 feature token 做 cross-attention
  D. Stronger PSG Gate — 增大 hidden_dim 或加深 gate 网络
  E. Dual-Stream PSG — 分离 body-part gate 和 structure gate

**选择**: A — PSG + PAB 组合

**理由**:
1. PSG 和 PAB 作用在不同层面：PSG 调制 feature values, PAB 调制 attention patterns
2. 两者理论上互补：PAB 让 attention 关注正确的位置，PSG 增强这些位置的特征
3. 两者加起来只有 ~108K 参数，计算开销极低
4. 如果组合有效，这构成了"多层次姿态注入"的创新点（attention + feature + pooling 三个层面）
5. 如果组合无效（像 exp008 一样），也提供了重要消融证据

**执行结果**: exp013 最终 mAP 57.6%, R1 67.2%。**组合未能超越 PSG-only (-0.7% mAP, -0.7% R1)**。与 exp008 类似的规律：两种 pose 注入方式在同一层（Stage 3）互相干扰。PAB 修改了 attention 分布导致 PSG 基于的特征分布发生变化，PSG 的 gate 不再最优。**结论：单一高质量 pose 注入点（PSG）比多个中等质量注入点更好。**

### [2026-03-10 10:15] 决策 #14

**上下文**: 13 个实验完成。Phase 2 的 backbone injection 方向已充分探索：
- PSG Stage 3 only: +1.7% mAP (exp007) ← BEST
- Multi-stage PSG: 无额外收益 (exp009)
- PAB: +0.8% mAP (exp012)
- PSG + PAB combo: +1.0% mAP (exp013，不如 PSG alone)
- PSG + Part Pooling: +1.1% mAP (exp008，不如 PSG alone)
- Extended training: 无收益 (exp011)

**核心认知**: PSG 的 58.3% 性能上限是由**pose heatmap 本身的信息量**和**Stage 3 的 2 个 block**共同决定的。在 backbone 内部叠加更多 pose 模块只会互相干扰。

**选项**:
  A. 转向训练策略创新 — 新的 loss function（如 pose-guided contrastive loss）
  B. 转向推理端改进 — pose-guided re-ranking 或 NFC
  C. PSG 内部机制改进 — 3×3 depthwise conv, channel attention
  D. 全新方向 — 如 Pose-Guided Token Pruning, Pose-aware MixUp
  E. PSG + 不同 loss 组合 — 如 per-part triplet loss (GiLt) 在 PSG backbone 上

**选择**: E — PSG + per-part triplet loss (GiLt)

**理由**:
1. PSG 只改善了特征质量，但训练信号仍然是标准的 ID+triplet loss
2. Per-part triplet loss 可以给 backbone 提供更细粒度的梯度信号，可能进一步提升 PSG 增强的特征
3. Phase 1 中 GiLt 在 PCFC 基础上额外 +0.5%，PSG 基础上可能也有类似增益
4. 这是正交方向的改进（module 级 vs loss 级），不会像 PAB 那样干扰 PSG
5. 实现简单：在 PSG 增强的特征图上做 pose-guided part pooling，然后每个 part 独立 triplet loss
6. 论文价值：训练信号层面的 pose 利用，与 backbone 层面的 PSG 互补

**执行结果**: 用户指出 exp014 与 exp008 训练完全相同，直接加载 exp008 checkpoint 做 global test 即可。结果：mAP 57.6%, R1 65.8%。**比 PSG-only 低 0.7% mAP, 2.1% R1。Part supervision 确实损害了 PSG global feature。** "loss 级正交改进"的假设不成立——part supervision 的梯度回传到 backbone 时会干扰 PSG gate 的学习。

### [2026-03-10 10:40] 决策 #15

**上下文**: 14 个实验完成。完整总结：
- **有效**: PSG Stage 3 only (+1.7% mAP) — 唯一突破
- **弱正**: PAB (+0.8%), Part Pooling (+0.9%)
- **所有组合尝试都失败**: PSG+Part, PSG+PAB, PSG+Part(global test)
- **训练策略**: Multi-stage 无效, Freeze 灾难, 200ep 无效

**关键规律**: 任何修改 PSG 训练时梯度流的改动都会降低 PSG 效果。PSG 的 58.3% 是"全局 ID+triplet + pose spatial gating"的最优解。

**战略性反思**: 继续在 PSG 基础上小修小补已经穷尽了选项。需要跳出"在 backbone 里加东西"的思路。

**选项**:
  A. PSG 机制改进 — 3×3 depthwise conv, channel attention (仍在 backbone 内部)
  B. Test-time 改进 — NFC/re-ranking (不是训练端创新)
  C. PSG + Pose-Guided Data Augmentation — 训练数据层面的 pose 利用
  D. 全新方向: Pose-Guided Feature Disentangling — 用 PSG feature 做 pose/appearance 解耦
  E. 全新方向: Pose-Conditioned Contrastive Learning — pose 相似的样本对应更严格的判别要求

**选择**: A — PSG 内部机制改进（3×3 depthwise conv），而非 C

**修正**: 重新考虑后选择 A。理由：
1. 这是对 PSG gate 本身的改进，不增加新模块或新 loss，不会像 exp008-014 那样干扰梯度流
2. 当前 1×1 conv 每个位置独立计算 gate，没有空间连贯性
3. 3×3 depthwise conv 只增加 576 参数，让 gate 考虑邻域
4. 人体部件是连续区域，相邻位置应有相似 gate 值
5. 如果无效，再转向 C (Pose-Aware Data Augmentation)

**执行结果**: exp015 训练完成。mAP 58.3% 与 exp007 完全持平, R1 67.1% 低 0.8%。3×3 depthwise conv 是冗余的，1×1 gate 已是最优。训练过程中波动极大（差距从 -2.2% 到 +2.3%），但最终收敛到相同水平。PSG 的瓶颈不在感受野。

### [2026-03-10 12:49] 决策 #16

**上下文**: exp015（PSG 空间卷积改进）完成，与原始 PSG 持平。至此：
- PSG 内部结构改进（depthwise conv）：无效
- PSG 外部组合（PAB, Part Pooling, Part Supervision）：全部有害
- PSG 训练策略（freeze, 200ep, multi-stage）：无效

**已穷尽的方向**: 在"PSG + 全局 ID/Triplet Loss"框架内的所有优化都已探索完毕。PSG 58.3% mAP 是该框架的理论上限。

**需要根本性的方向转变**: 不再在 PSG 上修修补补，需要全新的利用 pose heatmap 的方式。

**选项**:
  A. Pose-Guided Feature Disentangling — 用 pose 热图将特征解耦为 part-specific 子空间
  B. Pose-Conditioned Contrastive Learning — 基于 pose 相似度的对比学习
  C. Pose-Guided Data Augmentation — 基于 pose 的数据增强
  D. Adaptive PSG — 根据遮挡程度动态调节 PSG 强度
  E. Deformable PSG — 可变形卷积替代固定网格，对齐到关键点位置

**选择**: A → H (Pose-Guided Erasing)

**执行结果**: exp016 完成。PGE 严重有害（mAP 54.8% vs exp007 58.3%，-3.5%）。身体部件级擦除过强+削弱 PSG 输入。数据增强层面的 pose 利用方向失败。

### [2026-03-10 15:10] 决策 #17

**上下文**: exp016 PGE 失败后，16 个实验的完整总结：
- PSG 58.3% 是唯一有效方法
- 所有扩展/组合/增强/增强均失败
- 数据增强方向（PGE）也失败

**关键反思**: 已经把 PSG 本身和所有"加法"都试遍了。需要完全不同的思路。

**选项**:
  A. Pose-Conditioned Normalization (PCN) — 用 pose 调制 LayerNorm 参数
  B. 全新模型: Deformable PSG — 可变形卷积对齐关键点
  C. 全新方向: Pose Structure Token — 在 Swin 输入层添加 pose 编码
  D. 测试端优化: 基于 PSG 的 NFC/Re-ranking
  E. 跨步思考: 不再改 PSG，而是替换整个 Part Pooling 方案

**选择**: 先尝试 Pose-Conditioned Channel Gate (PCG)——在 GAP 后做通道级 pose gating，与 PSG 的空间级正交

**执行结果**: exp017 完成。PCG 与 exp007 基本持平（mAP 58.0% vs 58.3%，-0.3%）。证实了通道级正交操作不干扰 PSG（不像之前的 combo 实验那样降性能），但 GAP 后的 pose 信息太弱无法提供额外收益。

### [2026-03-10 17:25] 决策 #18

**上下文**: exp017 PCG 中性结果。17 个实验的总结：
- PSG 58.3% 仍是最佳，所有扩展都无法超越
- **新发现**: 正交维度操作不干扰 PSG（PCG 仅 -0.3%），但空间级操作干扰 PSG（PAB combo -0.7%, Part pooling -0.6%）
- 这说明 PSG 的空间 gating 格局已定，但通道维度仍有空间

**关键洞察**: PCG 失败是因为 GAP 压缩了空间信息。如果在 GAP **之前** 做通道调制（保留空间信息），可能有效。

**选项**:
  A. SE-style Pose Channel Attention (在 GAP 前) — 用 pose 热图生成通道权重，应用到 feature map 上（空间信息还在）
  B. PCG-only (无 PSG) — 验证 PCG 单独效果，排除 PSG 已经做了足够 conditioning 的可能
  C. 完全新方向：跳出 PSG 框架

**选择**: B — PCG-only 消融实验

**执行结果**: exp018 完成。**PCG-only mAP 57.8%, R1 67.7%（+1.2%/+1.2% vs baseline）。** 出乎意料，PCG 有独立效果！推翻了"PCG 无效"的初步判断。但仍低于 PSG (58.3%/-0.5%)，且 PSG+PCG (exp017) 不叠加。结论：PCG 和 PSG 各自有效但捕获相似的 pose 信号，组合不互补。

---

### [2026-03-10 22:08] 决策 #19

**上下文**: exp019 PXA (Pose Cross-Attention) 训练完成。最终 mAP 57.3%, R1 66.9%。超过 baseline (+0.7%) 但落后 PSG (-1.0%)。

**分析**:
- PXA 证明了 cross-attention 方式注入 pose 信息有效（>baseline），但效果不如 PSG 的简单空间门控
- 原因：cross-attention 表达能力过强导致过拟合（训练 acc 99.5%，loss 0.10），PSG 的逐元素乘法有更强归纳偏置
- PXA 和 PSG 不太可能组合使用（都在 Stage 3 blocks 中间注入，已有先例显示 PSG+其他会互相干扰）

**核心结论**: 三种 pose 注入机制的排序确认：PSG(空间门控) > PXA(跨注意力) > PCG(通道门控, GAP后)。简单的空间逐元素操作始终优于复杂机制。

**下一步**: 启动 exp020 (PSG + Pose Reconstruction Auxiliary)，测试**正交方向**——不修改 PSG 前向传播，仅通过辅助重建任务提供额外梯度信号。这是突破 58.3% 上限的希望所在。

---

### [2026-03-11 00:14] 决策 #20

**上下文**: exp020 PRA 完成。mAP 57.8%, R1 67.3%。中性结果，不如 PSG (58.3%)。至此，20 个实验全部无法超越 PSG (exp007)。

**总结过去尝试突破 PSG 的失败**:
- exp008 PSG+Part Pooling: 57.7% ❌ 组合干扰
- exp009 Multi-stage PSG: 58.3% 🟡 匹配但不超越
- exp012 PAB: 57.4% ❌ 弱于 PSG
- exp013 PSG+PAB: 57.6% ❌ 互相干扰
- exp015 PSG Spatial 3×3: 58.3% 🟡 匹配但不超越
- exp017 PSG+PCG: 58.0% ❌ 不叠加
- exp019 PXA: 57.3% ❌ 过拟合
- exp020 PSG+PRA: 57.8% ❌ 梯度干扰

**核心问题**: 所有**添加额外模块/任务**到 PSG 的尝试都失败。PSG 似乎已经是"局部最优"——它简单有效，但不接受增强。

**新思路**: 不添加模块到 PSG，而是**改进 PSG 本身**。核心问题：PSG 的门控是**静态的**——给定相同的 heatmap，不同图像得到相同的 gate。如果让 gate 同时依赖 pose 和当前特征内容（Content-Adaptive PSG / CAPSG），可能打破这个限制。

**选项**:
A. Content-Adaptive PSG (CAPSG): gate = f(pose, features) 而非 gate = f(pose)
B. PSG + 超参调优 (weight decay, dropout, label smoothing)
C. 接受 PSG 58.3% 作为最终方法，开始写论文

**选择**: A — CAPSG

**理由**:
1. CAPSG 是对 PSG 机制本身的改进，而非外挂模块，避免了"组合干扰"问题
2. 与 PXA (cross-attention) 不同，CAPSG 保持了 PSG 的逐元素乘法范式，只是让乘法因子变成 content-dependent
3. 额外参数很少 (~50K/gate = ~100K total)
4. 零初始化保证初始行为等同 PSG，只有学到有用的 content-feature 交互才会偏离

**执行结果**: exp021 完成。**CAPSG mAP 57.2%, R1 66.0%（-1.1% vs PSG）。** Content-adaptive gate 不如静态 pose-only gate。CAPSG 前期慢启动（ep20 落后 -4.2%），虽多次追近但从未在后段超越 PSG。关键洞察：PSG 的静态 pose-only gating 不是局限而是优势——ReID 需要的是一致的空间先验，不是动态调制。

---

### [2026-03-11 02:30] 决策 #21

**上下文**: 21 个实验完成，PSG (exp007) 仍是唯一超过 baseline 1.5%+ 的方法。已尝试的所有突破方向均失败。

**21 个实验的完整教训总结**:
- **有效**: PSG (+1.7%), Part Pooling (+0.9%), PCG-only (+1.2%), PXA (+0.7%), PRA (+1.2%)
- **最佳**: PSG 58.3% — 简单空间门控
- **组合全部失败**: PSG+Part, PSG+PAB, PSG+PCG, PSG+PRA — 要么中性要么有害
- **改进 PSG 也失败**: Multi-stage PSG, PSG Spatial, CAPSG — 匹配或弱于 PSG
- **复杂机制全部不如简单门控**: PXA < PSG, CAPSG < PSG

**核心洞察**: 对于 Swin-Tiny backbone + pose heatmap 的组合，简单的逐元素空间门控（PSG）已经是最优解。所有增加复杂度的尝试都是负面的。

**下一步方向选择**:
A. 接受 PSG 作为最终方法，转向完善论文（跨数据集实验、可视化、效率分析）
B. 尝试完全不同的正则化方向（DropPath增大、Label Smoothing 等超参数）
C. 探索 PSG 在更长训练、不同 LR 下的潜力
D. 在不同数据集（Market-1501）上验证 PSG 泛化性

**选择**: D — Market-1501 上验证 PSG 泛化性（已由用户在 4090 完成）

**执行结果**: 用户已在 4090 上完成所有跨数据集实验。PSG 在所有配置上均有效（Occluded-Duke Swin-Small +2.0%, Market-1501 Swin-Tiny +0.8%, Market-1501 Swin-Small +0.6%）。

---

### [2026-03-11 06:30] 决策 #22

**上下文**: 用户指示不要继续做跨数据集实验（已在 4090 完成），要求探索新的训练侧创新。用户确认：
1. 可以添加大模块（ResNet、GCN、Decoder 等），Swin-Tiny 仅是效率约束
2. 可以完全放弃 PSG 框架
3. 可以加新分支、双流架构等

经 Web 搜索调研，识别出以下高潜力方向：
- PDS (双分支架构): 共享 Stage 1-2，独立 Stage 3 for Global/Part
- KP-RPE (关键点相对位置编码): CVPR24 人脸识别，与 PSG 正交
- Skeleton GCN: 沿骨架传播特征做遮挡补全

**选项**:
A. exp022: PDS 双分支 — 解决梯度干扰核心瓶颈，~6M 额外 params
B. exp022: KP-RPE — 最轻量，~10K params，快速验证
C. exp022: Skeleton GCN — 后置模块，~3M params

**选择**: A — PDS 双分支

**理由**:
1. 21 个实验证明梯度干扰是 PSG 组合失败的根本原因。PDS 从架构层面解决这个问题。
2. 用户明确说"可以加大模块、新分支"，PDS 正是这个方向
3. PDS 的 Part 分支为后续集成 GCN、Feature Completion 等模块提供了载体
4. 如果 PDS 有效，结合 PSG 在 Global 分支 + 结构化 Part 分支，论文故事非常完整
5. KP-RPE 虽轻量但受 Swin window attention 限制风险较大，可作为后续补充

**执行结果**: PDS 训练完成。global-only mAP 57.9% (vs PSG-only 58.3%)，Stage 3 解耦有效但共享 Stage 0-2 仍有轻微干扰。Part 分支独立效果一般 (55.2%)。PDS 未超过 PSG-only。

### [2026-03-11 09:30] 决策 #22

**上下文**: exp022 PDS 结果分析 — global-only 57.9% (接近但未超过 PSG-only 58.3%)。Part 分支 ID loss (2.02) 远未收敛到 Global (0.17) 水平，Part 特征质量不足。

**核心问题**: PDS 证明了 Stage 3 权重解耦可以保护 Global 分支（57.9% vs exp008 57.7%），但 Part 分支不够强以贡献额外增量。关键原因：
1. Part 分支共享 Stage 0-2 仍向 Global 方向传梯度（微小干扰 -0.4%）
2. Part ID loss 2.02 远未收敛，5 个 part 分类器学习难度远大于 1 个 global 分类器
3. equal_concat 的 5:1 维度比严重稀释 Global

**选项**:
A. exp023: Part 分支 stop_gradient — 完全阻断 Part→Stage0-2 梯度，保护 Global 分支
B. exp023: Part 分支延迟启动 — 先让 Global 收敛再引入 Part，减少早期干扰
C. exp023: 换方向 — 放弃 dual-stream，回到单分支 + 更好的 test-time fusion

**选择**: A — stop_gradient

**执行结果**: 🎉 **突破性成功！** exp023 global-only mAP 59.5% (+1.2% vs PSG-only 58.3%, +2.9% vs baseline)。
stop_gradient 不仅消除了 Part 干扰，还通过改善共享特征质量间接提升了两个分支。
Part-only 也从 55.2% 提升到 56.7%，证明更好的共享特征反哺 Part 分支。

### [2026-03-11 11:55] 决策 #23

**上下文**: exp023 PDS+StopGrad 取得全实验最佳结果 (mAP 59.5%)。需要决定下一步方向。

**关键发现**:
1. PDS+StopGrad global-only (59.5%) > PSG-only (58.3%) > PDS global (57.9%)
2. concat_scaled (59.1%) 接近 global-only，说明 Part 有正贡献
3. Part 分支独立效果 (56.7%) 超过 baseline (56.6%)
4. 但 Part ID loss 仍高 (2.06)，Part 特征尚有提升空间

**选项**:
A. 对 exp023 结果做完整消融：单独评估 PSG 贡献、Part 贡献、stop_grad 贡献
B. 优化 test-time fusion：尝试更好的 global+part 融合策略
C. 在 Market-1501 上验证 PDS+StopGrad 的泛化性

**选择**: 先完成文档，然后按 A → C 的顺序执行

### [2026-03-11 13:18] 决策 #24

**上下文**: exp024 (PDS+StopGrad 无 PSG 消融) 在 epoch 60 达到 equal_concat mAP 53.9%（vs exp023 54.4%，-0.5%），趋势明确但被误杀了 DataLoader worker 导致终止。

用户提出关键质疑：**exp023 的 +1.2% 提升（vs exp007 PSG-only）在理论上说不通**——如果 Part 梯度被阻断，Global 分支应该和 PSG-only 完全等价。分析后认为 +1.2% 很可能是训练随机性差异（不同模型类的初始化消耗不同随机状态），而非架构真实贡献。

**用户建议**: "Delayed StopGrad"——前 30 轮阻断 Part 梯度保护预训练权重，之后释放让 Part 梯度也优化共享层。这样：
1. 论文故事更好写（不需要解释"永久阻断"的不合理提升）
2. 有明确的技术动机（保护预训练权重免受随机初始化 Part 分支的干扰）
3. 如果有效，可能超过 exp023 的 59.5%

**选项**:
A. 先重跑 exp024 (消融)，再跑 exp025 (Delayed StopGrad)
B. 直接启动 exp025，exp024 可以后续补跑

**选择**: B — 直接启动 exp025

**理由**: exp025 是用户指定的高优先级实验，且 exp024 的趋势数据已足够（6 个 epoch 评估点都确认了无 PSG 的影响）

**执行结果**: exp025 完成。Delayed StopGrad (前 30 轮阻断→释放) 最终结果：
- global-only: mAP 58.9% (vs exp023 59.5%, -0.6%; vs exp022 57.9%, +1.0%)
- concat_scaled: mAP 58.6% (vs exp023 59.1%, -0.5%)
- **结论**: Delayed StopGrad < Permanent StopGrad。释放 Part 梯度后虽然没有灾难性崩溃，但造成了轻微干扰，最终收敛到 exp022 和 exp023 之间。永久 StopGrad 仍是最优策略。但本实验提供了有价值的消融证据：梯度隔离的必要性。

### [2026-03-11 15:45] 决策 #25

**上下文**: exp025 (Delayed StopGrad) 完成。关键发现：
1. 永久 StopGrad (exp023: 59.5%) > 延迟 StopGrad (exp025: 58.9%) > 无 StopGrad (exp022: 57.9%)
2. 即使 Part 分支预热 30 轮后释放梯度，仍有轻微干扰
3. 这为论文提供了有力的消融证据：梯度隔离不仅有效，而且是**必要**的

**exp023 的 +1.2% 提升的新解释**:
- 原始假设：随机状态差异导致训练方差
- 新证据：exp025 global 58.9% > exp007 PSG-only 58.3% (+0.6%)，说明 Part 分支确实有正面影响
- 但完全隔离 (59.5%) > 延迟释放 (58.9%)，说明 Part 梯度的负面干扰 > 正面贡献
- **修正后解释**: exp023 的 +1.2% 包含两部分：(1) ~0.6% 来自 Part 分支辅助训练的正面效应 (2) ~0.6% 来自梯度隔离避免的干扰。两者叠加 = 1.2%

**论文价值排序**:
1. exp023 (PDS+StopGrad) — 主结果，+2.9% mAP
2. exp022 (PDS) — 消融：证明无隔离时 Part 梯度干扰 Global
3. exp025 (Delayed StopGrad) — 消融：证明即使预热后释放也不如永久隔离
4. exp024 (No PSG) — 消融：证明 PSG 在 Global 分支中的贡献

---

### [2026-03-11 18:10] 决策 #26

**上下文**: PDS 系列实验 (exp022-025) 完成后，需要决定下一步方向。用户建议三个选项：(a) 更长梯度隔离 (b) 更早分支分离 (c) 全新创新点。启动了 Opus 研究子代理分析创新方向。

**选项**:
  A. 更早分支分离（share Stage 0-1, split Stage 2+3）— 消融实验，~10M 额外参数
  B. 更长梯度隔离（60 epochs）— 低价值，exp025 已证明延迟释放不如永久隔离
  C. Stochastic Pose Dropout (SPD) — 全新方向，在 PSG 基础上做正则化
  D. Pose-Contrastive Representation Alignment (PCRA) — 全新方向，pose-aware triplet 距离
  E. Pose-Guided Variance Regularization (PVR) — 全新方向，特征空间分布正则化

**选择**: C (SPD)
**理由**:
1. 最简单实现（~5 行代码），最低风险（最差情况 = PSG baseline 58.3%）
2. 不在 forward path 添加新模块 → 完全避免梯度干扰问题
3. 基于确认有效的 PSG 架构（58.3%），不依赖尚未验证的 PDS
4. SPD 结果可指导后续方向选择：若有效 → 说明 backbone 过度依赖 pose → 继续 PVR；若无效 → pose 始终有用 → 尝试 PCRA
5. 论文 story 清晰："PSG 教 backbone 在哪里关注，SPD 防止过度依赖"

**执行结果**: mAP 57.9% (-0.4% vs PSG)。SPD 略微负面，证明 pose 信号在 Occluded-Duke 上一致有用

---

### [2026-03-11 20:20] 决策 #27

**上下文**: exp026 (SPD) 完成，结果轻微负面 (-0.4%)。需要选择下一个实验方向。

**选项**:
  A. PCRA (Pose-Contrastive Representation Alignment) — 修改 triplet 距离度量
  B. PVR (Pose-Guided Variance Regularization) — 辅助 loss 约束特征分布
  C. 不同 SPD dropout rate (p=0.1/0.5)

**红蓝队辩论**:
- 🔴 红队（方案 A: PCRA）核心论点: PCRA 是唯一操作在未被探索维度（loss 距离度量）上的方案。26 个实验证明所有 forward path 改动都干扰 PSG，但 PCRA 不修改 forward path 也不添加 aux loss——它只改变 hard mining 中的距离计算。这是论文方法的"第三层"贡献（特征层 PSG + 架构层 PDS + 度量层 PCRA）。实现仅~20 行代码，0 新参数。风险极低：最差退化为 PSG baseline。Occluded-Duke 中 pose 相似的 negative 才是真正的 hard case（两个只露上半身的不同人），标准 triplet loss 对此毫无感知。PVR 的弱点在于 exp020 (PRA) 已证明辅助 loss 方向中性/负面(-0.5%)，且需要 part feature 提取（与 exp008 的失败模式重叠）。SPD 调参无论文价值。信心: 8/10
- 🔵 蓝队（方案 B: PVR）核心论点: PVR 零 forward path 改动，在 26 个实验证明"所有 forward 改动都干扰 PSG"的历史下最安全。与 PSG 理论互补（PSG 约束幅度分布，PVR 约束语义结构）。复用现有 heatmaps_to_parts() 基础设施。exp020 (PRA) 失败是因为重建任务梯度方向与 ID loss 不一致，PVR 的结构正则化方向与 ID loss 一致（同部位同 ID → 应相似）。PCRA 的弱点在于修改了核心度量学习过程（exp003 修改 loss 权重导致 -6.4%），且引入 O(B²) 批内 pose 比较和新的超参数设计空间。信心: 7/10
- 综合判断: 选择 A (PCRA)。红队论点更有力——PCRA 确实操作在全新维度（距离度量）而非 aux loss，与 PRA (exp020) 的失败模式不同。PCRA 的实现更简洁（~20 行 vs ~50 行），超参数更少（1 个 vs 2 个）。两方都正确指出了对方的弱点，但 PCRA 的核心优势——"不修改 loss 项数量，只修改距离函数"——使其与所有历史失败模式都不同。

**选择**: A (PCRA)
**理由**:
1. 26 个实验穷尽了 forward path 方向，PCRA 操作在全新维度
2. 不添加 aux loss，不修改 forward path，风险最低
3. 论文 story 价值最高（三层 pose 利用的完整体系）
4. 红队信心 8/10 > 蓝队 7/10
**执行结果**: ❌ PCRA (alpha=0.2) 得到 mAP 57.8%, R1 66.8%，低于 PSG -0.5%/-1.1%。pose similarity 调制在 triplet 距离中引入了训练不稳定性（锯齿形 mAP 波动）。17 维 pose signature 不够精确区分姿态差异。

---

### [2026-03-11 22:35] 决策 #28

**上下文**: exp027 PCRA 结果中性偏负 (-0.5% mAP)。至此，所有在 PSG 基础上的单点改进（forward path 添加、aux loss、距离度量调制、dropout 正则化）均未能超越 PSG。唯一成功的方向是 PDS+StopGrad (exp023, +2.9% mAP)，但 exp024 证明其中 PSG 的贡献很小（仅 0.3%）。需要决定下一步方向。

**选项**:
  A. 改进 PDS+StopGrad（改善 Part 分支收敛，如 Part LR boost、Part warmup、feature distillation）
  B. 全新范式（PGFS 硬 token 选择等）

**红蓝队辩论**: 代理超时未完成，基于主 agent 分析做决策。

**选择**: A — 改进 PDS+StopGrad，具体为 Part LR Boost (3x)
**理由**:
1. PDS+StopGrad 是唯一成功超越 PSG 的方向，直接在其上改进是最高 ROI 的选择
2. Part 分支收敛不充分是已知问题（ID loss 2.0 vs Global 0.2），Part LR boost 直接解决此瓶颈
3. Part-only 56.7% → concat_scaled 59.1% < global-only 59.5%，说明 Part 是弱项
4. 实现极简（~5 行代码改 optimizer），风险低
5. 全新范式（20 个 PSG 改进全失败的历史）成功概率不高
**执行结果**: 🟡 中性。exp028 得到 mAP 59.3%, R1 68.9%（equal_concat 测试）。Part ID loss 从 exp023 的 ~2.0 显著降至 0.4，Part tri loss 与 Global 完全对齐。但测试性能仅 59.3%（vs exp023 global-only 59.5%）。**关键发现：Part 收敛不是 PDS 性能瓶颈。** 即使 Part 完全收敛，concat 特征也未能超越 global-only，可能是 Part 特征过拟合或与 Global 冗余。

---

### [2026-03-12 01:06] 决策 #29

**上下文**: exp028 表明 Part LR boost 不是 PDS 的改进方向。至此尝试的所有 PDS 改进均未超越 exp023：
- exp025 Delayed StopGrad: mAP 58.9% (-0.6%)
- exp028 Part LR 3x: mAP 59.3% (-0.2%)
- exp023 的 global-only 59.5% 仍然是最佳

关键洞察：
1. PDS 中 Part 分支的信息与 Global 高度冗余（都从共享 Stage 0-2 提取）
2. stop_grad 的主要价值不是让 Part 学好，而是保护 Global 不被 Part 梯度污染
3. 融合方式（concat）无法利用 Part 的独特信息

**选项**:
  A. 继续改进 PDS 融合方式（如 attention-weighted fusion、confidence-based weighting）
  B. 回归 PSG 路线但走全新范式（如 Pose-Guided Feature Selection — 基于热图做 token pruning/routing）
  C. 探索 PSG + PDS 的更深层结合（如 Part 分支用不同的 pose modulation）

**选择**: B — 基于 pose 热图的 token 选择/路由（Pose-Guided Token Selection, PGTS）
**理由**:
1. PSG 和 PDS 都是在 backbone 输出后处理，但 token-level 操作是 Transformer 的原生语言
2. Pose 热图天然可以判断哪些 token 对应可见的身体部位（高响应 = 可见，低响应 = 遮挡/背景）
3. 遮挡 ReID 的核心问题是噪声 token 稀释有效特征，热图引导的 token 选择直接解决此问题
4. 这个方向有更强的论文 story：从"全局调制"到"局部选择"的范式升级
5. 实现简单：用 pose 热图的空间最大值作为 token 重要性分数，加权 GAP 或 TopK 选择

**执行结果**: exp029 (PWP) 完成。mAP 57.9%, R1 67.5% — 低于 exp007 (58.3%) -0.4%。**PWP 是 PGTS 的 soft 版本，结果中性偏负。** Post-backbone 的加权 pooling 在 PSG 已完成空间选择的情况下是冗余操作。关键启示：如果要做 token-level 操作，必须在 Stage 3 **内部** 做（如 hard token pruning），而不是在 pooling 阶段。

---

### [2026-03-12 03:30] 决策 #30

**上下文**: exp029 (PWP) 完成后，29 个实验全部结束。PDS+StopGrad global-only 59.5% 仍是最佳。所有 PSG 改进（21 个）和 Part 改进（exp028, exp029）均失败。需要决定下一步方向。

**选项**:
  A. Learnable Part Query (LPQ) — 用 cross-attention 替代 Part 分支的 heatmap mask pooling
  B. 论文整合阶段 — 多 seed 验证、可视化、效率分析、停止新模块实验

**红蓝队辩论**:
- 🔴 红队（方案 A: LPQ）核心论点: Part 分支失败不是方向错误而是方法不对。29 个实验中从未在梯度隔离的 Part 分支测试过 cross-attention 查询机制。LPQ 从"被动聚合"到"主动查询"的范式转换，与热图 mask 池化有本质区别。成本仅 2-3 天，StopGrad 提供安全网（global-only 59.5% 下限不变）。论文最大软肋是 Part 特征无效，LPQ 可能补全这个 gap。信心: 6/10
- 🔵 蓝队（方案 B: 论文整合）核心论点: 29 个实验的统计信号已足够清晰——模块级改进的边际收益趋近于零。exp028 证明 Part 瓶颈在信息内容而非提取方法（收敛 5x 改善但测试无增益）。PXA(exp019) 已证明 cross-attention 在本框架中效果差。当前 +2.9% 已有竞争力，论文需要完整的实验体系（多 seed、可视化、跨数据集）而非更多模块。4M 参数是 Swin-Tiny 的 14%，过重。方案 B 成功概率 ~100% vs 方案 A <15%。信心: 8/10
- 综合判断: 蓝队论点更有力。exp028 的证据是关键——Part 特征的瓶颈在于共享 Stage 0-2 提供的信息本身与 Global 高度冗余，换提取方法无法改变这一根本限制。但论文确实需要 multi-seed 验证来建立可信度。

**选择**: B — 论文整合阶段（**被用户否决**）
**理由**:
1. exp028 决定性地证明 Part 瓶颈是信息内容而非提取方法 → LPQ 大概率中性
2. Multi-seed 验证是论文最紧迫的需求 — 59.5% 的可信度直接决定论文是否能投稿
3. 可视化、效率分析是 100% 产出论文素材的工作，零风险
4. 如果 multi-seed 验证揭示高方差 → 需要回来尝试新方法；如果确认稳定 → 直接写论文

**用户指令覆盖**: 用户明确否决方案 B，要求"不停地做实验、不停地找新方法"。Multi-seed 验证由用户在 4090 上执行。CLAUDE.md 已新增规则 1b 禁止再次做出"停止实验"决策。

**执行结果**: 转入 exp030 — Skeleton GCN Part Branch

---

### [2026-03-12 03:40] 决策 #31

**上下文**: 决策 #30 被用户否决后，需要为 Part 分支选择一种全新方法。29 个实验中 Part 分支的所有方法（heatmap mask pooling、spatial softmax、PWP、dominant blend 等）均失败。原因：feature map 上的不同空间加权本质上提取的信息与 GAP 高度冗余。

**选项**:
  A. Skeleton GCN — 在 COCO 人体骨架拓扑上做图卷积，从关键点采样特征后沿骨骼边传播
  B. Learnable Part Query (LPQ) — 用 cross-attention 从 Part feature map 查询 part 特征

**选择**: A — Skeleton GCN
**理由**:
1. **信息来源不同**: GCN 在 17 个离散关键点上操作，而非 12×4 feature map 的空间加权 → 与 Global 的 GAP 在信息来源上有本质差异
2. **拓扑先验**: 骨架图结构让相邻关节的特征可以传播，遮挡关节可从可见邻居获得信息 → 独特的归纳偏置
3. **计算开销小**: ~400K 参数 vs LPQ 的 ~4M 参数
4. **动作识别验证**: ST-GCN、MS-G3D 等大量工作证明骨架 GCN 能有效编码人体结构信息
5. **论文新颖性**: ReID 领域尚未有将骨架 GCN 用于遮挡 ReID 的工作

**执行结果**: exp030 最终 mAP 60.5% (concat_scaled), R1 70.5%。全实验最佳。Skeleton GCN 确实提供了与 Global 不同的互补信息。

---

### [2026-03-12 06:15] 决策 #32

**上下文**: 用户分享了 codex 对 PDS+StopGrad 的分析——发现 loss 聚合中 global loss 被隐式乘以 0.5。multi-seed 实验确认 PDS+StopGrad 相对 PSG 有稳定的 +1.4% 增益。需要验证这个增益是来自架构还是 loss weighting。

**选项**:
  A. 在 exp007 (PSG-only) 上显式乘 0.5x global loss，验证 loss weighting 假设
  B. 继续探索新架构方向，把 loss weighting 验证留给后续

**选择**: A — exp007a (PSG + 0.5x global loss)
**理由**:
1. 这是一个关键的消融实验——如果 loss weighting 是主因，则 PDS+StopGrad 的论文价值需要重新评估
2. 实现成本极低（仅改 1 个 config 参数）
3. 结果明确：如果 exp007a ≈ 59%，假设成立；如果 < 58%，假设不成立

**执行结果**:
- exp007a 最终 mAP **59.5%**, R1 **69.8%** — 与 exp023 (PDS+StopGrad) 的 59.5%/69.5% **完全一致**！
- **假设完全确认**: PDS+StopGrad 的增益 100% 来自 loss weighting 正则化效果
- **重大发现**: 仅 +102K params (PSG) + 0.5x loss = 6.3M params (PDS) + StopGrad 的相同效果
- 这个发现从根本上改变了对 PDS 架构的理解

---

### [2026-03-12 08:30] 决策 #33

**上下文**: exp007a 确认 loss weighting 是 PDS 增益的主因。exp030 (PDS+GCN) 达到 60.5%（GCN 额外 +1.0%）。需要决定下一步：验证 GCN 能否在无 PDS 的架构中工作？还是先做 loss scale 网格搜索？

**选项**:
  A. Loss Scale Grid Search (exp007b-d): 测试 0.3/0.7/0.1 三个 scale 值
  B. PSG + 0.5x Loss + Skeleton GCN 无 PDS (exp030a): 验证 GCN 能否不需要独立 Stage 3

**红蓝队辩论**:
- 🔴 红队（方案 A）核心论点: 0.5x 是唯一数据点，需要理解曲线形状。干净的消融表 {0.1,0.3,0.5,0.7,1.0} 是论文核心证据。近零风险。但 3 个实验 ~15h GPU 时间。信心: 8/10
- 🔵 蓝队（方案 B）核心论点: 架构简化问题的信息价值远高于超参调优。如果成功，整个方法简化为 PSG+loss+GCN (~0.6M params vs PDS 6M+)，论文 story 极其简洁。仅 1 个实验 ~5h。即使失败也有价值（说明 PDS 必要）。信心: 8/10
- 综合判断: 选择 B。架构问题的优先级高于超参优化。grid search 可以后续作为论文 appendix 补充。

**选择**: B — exp030a (PSG + 0.5x loss + GCN, 无 PDS)
**理由**: 信息价值/GPU 时间比远高于 grid search。如果成功，论文 story 变为"3 个正交轻量组件"，极具吸引力。

### [2026-03-12 10:55] 决策 #34

**上下文**: exp030a 训练完成。结果：PSG+GCN (无 PDS) equal_concat 达到 mAP 61.1%, R1 73.7%，为全实验最佳。完全验证 GCN 不需要独立 Stage 3。

**实验结论**:
- exp030a 在所有 4 种测试模式上都 ≥ exp030 (PDS+SG+GCN)
- equal_concat (61.1%) > concat_scaled (60.5%) → exp030 中 concat_scaled 更好可能是偶然
- GCN-only (58.2%) 远超 PDS part-only (56.7%)，说明共享 Stage 3 的特征对 GCN 更好
- 参数从 6.3M → ~500K，减少 92%

**下一步选项**:
  A. 消融实验：分离 loss scaling 和 GCN 的独立贡献（exp030b: PSG+GCN+1.0x loss，即不使用 0.5x scaling）
  B. 论文完善：在 4090 上用 Swin-Small 复现最佳结果
  C. 更多 test-time 增强测试（NFC + re-ranking）

**选择**: 先 A 再 C。exp030b 消融是论文必需的证据（证明 GCN 的增益独立于 loss scaling），且只需跑一个实验。C 可以在训练期间并行用 test.py 跑。

### [2026-03-12 13:30] 决策 #35

**上下文**: exp030b 完成。核心发现：w_p=0.01 时 GCN 几乎未训练（ID_part loss 5.1 vs exp030a 的 0.17），但 global mAP = 60.6%，远超预期的 58.3%（exp007），甚至超过 exp030a global (59.8%)。

**关键数据对比**:
| 实验 | 模型差异 | Global mAP |
|------|---------|-----------|
| exp007 | PSG only | 58.3% |
| exp007a | PSG + 0.5x loss | 59.5% |
| exp030a | PSG + GCN (w_p=1.0, 隐式 0.5x) | 59.8% |
| exp030b | PSG + GCN (w_p=0.01, ≈1.0x) | 60.6% |

这四个实验的 global mAP 从 58.3% 到 60.6%，差异达 2.3%，但没有一致的规律（exp030b 应该最低却最高）。

**分析**:
1. **训练方差假说**：不同模型类初始化消耗不同随机状态 → DropPath mask、数据增强序列不同 → 2% 范围内的 mAP 差异不可信
2. **Loss scaling 效果可能被高估**：exp007a 的 +1.2% 可能部分/全部来自方差
3. **GCN 特征贡献可量化**：exp030b equal_concat (60.5%) ≈ global (60.6%)，证明未训练的 GCN 特征确实无价值
4. **exp030a GCN 贡献**：exp030a equal_concat (61.1%) > global (59.8%) = +1.3%，但这 1.3% 可能也受方差影响

**选项**:
  A. 立即设计多种子实验 (3-5 seeds)，在 3090 上跑 baseline/PSG/PSG+GCN 各 3 次
  B. 继续新方向实验（如新的 GCN 改进、更好的 part 特征融合）
  C. Loss scale grid search 作为论文消融分析

**选择**: A — 多种子实验。这是最关键的待办，没有多种子数据就无法区分真实增益和方差。但考虑到 3090 每个实验 120 epoch 需约 2 小时，跑 15 个实验 (5 configs × 3 seeds) 需要 30 小时，这在时间上太长。折中方案：选 3 个最关键配置 × 3 seeds = 9 个实验 (≈18 小时)，先确认核心结论。

**关键配置**:
1. exp007 (PSG only, 1.0x loss) — 基线
2. exp007a (PSG + 0.5x loss) — loss scaling 效果
3. exp030a (PSG + GCN, equal_concat) — GCN + loss scaling 组合

**执行**: 编写多种子脚本，后台跑。期间可设计下一个实验方向。

### [2026-03-12 18:45] 决策 #36

> 注：本决策中的“loss scaling 无效”等判断已被后续的 **决策 #37** 更正；保留这里只是为了记录当时的判断路径。

**上下文**: 用户在 4090 上已完成多种子验证（exp000 × 3, exp007 × 3, exp023 × 3），结果存放在 `4090_log/multiseed/`。用户选择了 exp023 (PDS+StopGrad) 而非原计划的 exp030a (GCN) 和 exp007a (0.5x loss)。

**多种子核心结论**:

| 方法 | Mean mAP | Std | vs Baseline | p-value |
|------|---------|-----|-------------|---------|
| Baseline | 56.50% | 0.53% | — | — |
| PSG | 57.83% | 0.50% | +1.33% | ~0.054 |
| PDS+StopGrad (global) | 59.20% | 0.50% | +2.70% | ~0.014 ✅ |

**关键发现**:
1. PSG +1.33% 所有 seeds 正向，但边缘显著
2. PDS+StopGrad (global) +2.70% 统计显著
3. PDS+SG vs PSG = +1.37%，极一致 (1.4/1.3/1.4)
4. Loss Scale 实验(0.25/0.75/1.0 均 58.3-58.6%) 证明 loss scaling 无效
5. 因此 PDS+StopGrad 的额外 +1.37% 不是 loss scaling 效果，机制待解释

**对论文的影响**:
- PSG 确认为有效贡献 (+1.33% mean)
- PDS+StopGrad 提供更大增益 (+2.70%) 且统计显著
- 但 PDS+StopGrad 增加 ~8.8M 参数（完整 Stage 3 复制），效率比 PSG 差很多
- GCN 仍需多种子确认
- 需要理解 PDS+StopGrad 为何在 loss scaling 无效的情况下仍优于 PSG

**选择**: 继续实验探索，重点方向：
1. 理解 PDS+StopGrad 的真实机制（不是 loss scaling，那是什么？）
2. GCN 多种子验证（安排 4090）
3. 基于确认的结论设计更强的方法

### [2026-03-13 01:10] 决策 #37

**上下文**: 4090 上又补齐了 `exp007a` 的 3 个 seed，以及 `exp030a` 四种测试模式的 3 个 seed。现在可以正式修正 #36 中关于 loss scaling 和 GCN 的错误推断。

**新增多种子结果**:

| 方法 | 模式 | Mean mAP | Std | Mean R1 |
|------|------|----------|-----|---------|
| exp007a | global | **59.37%** | 0.32% | **69.43%** |
| exp023 | global | **59.20%** | 0.50% | **68.63%** |
| exp030a | global | **59.33%** | 0.40% | **68.87%** |
| exp030a | concat_scaled | **60.20%** | 0.44% | **73.13%** |
| exp030a | equal_concat | **60.73%** | 0.47% | **72.57%** |

**修正后的关键结论**:
1. **#36 中“loss scaling 无效”的判断是错的**
   `exp007a vs exp007` 的 paired diffs = `(1.3, 1.6, 1.7)`，two-sided paired t-test `p=0.0061`。
   这不是方差，而是稳定增益。

2. **PDS+StopGrad 的 global-only 收益基本被 exp007a 复现**
   `exp007a = 59.37%`，`exp023-g = 59.20%`，差异 `+0.17%`，`p=0.3377`。
   所以 PDS+StopGrad 更像一个揭示机制的中间实验，而不是必须保留的主方法。

3. **GCN/KPP branch 的贡献主要发生在 fusion，不发生在 global**
   `exp030a-global = 59.33%` 与 `exp007a = 59.37%` 几乎相同。
   但 `exp030a-eq = 60.73%` 对自身 global 的 paired diffs 为 `(1.3, 1.1, 1.8)`，`p=0.0214`。

4. **`equal_concat` 应替代 `concat_scaled` 成为主模式**
   `exp030a-eq vs exp030a-cs` 的 paired diffs = `(0.6, 0.5, 0.5)`，`p=0.0039`。
   之前围绕 `concat_scaled` 的主叙事需要全部下调。

5. **exp030b 与 exp032 的最终定位**
   - `exp030b`: 证明 `w_p=0.01` 时 branch 基本没学好。
   - `exp032`: 证明 keypoint pooling 本身就是强 branch baseline。
   - 两者共同支持的正确 story 是：**KPP 贡献 branch 主体信息，GCN 负责 refinement。**

**对论文的影响**:
- PSG 仍是主创新点之一（稳定正向、参数极轻）
- `0.5x global loss` 可以作为一个重要机制发现 / training recipe
- PDS+StopGrad 不再适合作为主创新
- 如果保留 skeleton branch，应写成 **KPP + GCN refinement + equal_concat fusion**

**选择**: 全面重写实验总结文档与论文 story，统一改为：
1. PSG
2. `0.5x global loss`
3. skeleton branch 的 fusion 增益（以 `equal_concat` 为主）

### [2026-03-13 01:40] 决策 #38

**上下文**: 下一阶段准备把 `visibility` 引入当前的 skeleton branch / keypoint pooling 路线。但当前多人图中仍存在一个更基础的问题：`pose_data` 和 branch 侧并没有稳定地区分“crop 中的目标人”和“旁边靠得很近的其他人”。如果先做 visibility，再去解决 target assignment，visibility 语义会被污染，后续所有结论都不可靠。

**当前问题定义**:
1. `visibility` 是 **target-specific** 语义，不是 scene-level 语义
2. 只要 target person 还会和邻近人物混淆，`visibility`、`KPP`、`GCN` 都会学脏
3. 因此必须先解决 **target assignment**，再做 visibility 融合

**决策**: visibility 路线按 5 个阶段推进，严格顺序执行，不跳步。

#### Stage A — 先解决主要人物识别（必须先做）

**exp033: target assignment**
- 目标：为每张多人图稳定找出 target person
- 方法：给每个检测到的人计算 `targetness`
- 第一版启发式允许简单：
  - bbox 中心距离 crop 中心
  - bbox 面积
  - bbox 落在 crop 内的占比
  - 平均 keypoint score
- 产出：
  - `target_person_idx`
  - `target_score`
  - `target_margin`
- 验证：
  - 随机抽样 `100~200` 张多人图可视化
  - 明确标出 target person
  - 人工检查摘要写入实验文档

**停止条件**:
- 如果 target assignment 明显不稳定，就继续修 exp033
- 在 exp033 没过之前，不进入 visibility 训练

#### Stage B — 让数据流和模型路径 target-aware

**exp034: target-aware dataloader / branch**
- dataset 中保证 target person 被明确索引，最好固定为 `person 0`
- `KPP / GCN` 分支只消费 target person
- 保持向后兼容，不能破坏旧实验读取

**验证目标**:
- 先不引入 visibility
- 单独跑一个 “只修正 target assignment” 的对照实验
- 判断仅靠 target-aware branch 是否已经改善稳定性

#### Stage C — 先用现成 visibility 做最小闭环

**exp035: visibility 最小消融**
- 暂时**不重训 ViTPose**
- 直接使用已经提取好的 `visibility / visibility_binary`
- 只先改：
  - keypoint pooling
  - branch readout
  - fusion
- 不先改 GCN 消息传播

**最小对比组**:
1. `score only`
2. `visibility only`
3. `score * visibility`
4. `binary visibility`

**判断标准**:
- 必须回答：`visibility` 相对现有 `score/confidence` 是否有独立价值
- 如果连最小闭环都没有稳定收益，就不要急着做更复杂的 visibility-aware GCN

#### Stage D — 只有最小闭环有效，才做动态组装

**exp036: visibility-guided composition**
- 把 visibility 用于 keypoint / part 向量如何组成 branch feature
- 重点写法应是：
  - visible evidence 的动态组装
  - unreliable evidence 的降权
- 不要只做“0/1 mask 一乘了之”的弱版实现

#### Stage E — 最后才做 visibility-aware graph

**exp037: visibility-aware GCN refinement**
- 只有在 exp035 / exp036 已经证明 visibility 确实带来额外价值时才继续
- 此时 visibility 才进入：
  - 节点更新
  - 边权
  - 消息传播

**原因**:
- 否则会把两个变量混在一起：
  - 是 visibility 有用？
  - 还是 graph 本身有用？

**当前不做的事情**:
1. 不先重训 ViTPose
2. 不先做 visibility-aware GCN
3. 不在 target assignment 未解决前讨论 visibility 创新性

**文档要求**:
- 每推进一步，都要同步更新：
  - `design.md`
  - `monitor.md`
  - `results.md`
- 如果新结果推翻旧判断，必须直接修正文档措辞，不能只在聊天里说明

**实验基线约束**:
1. visibility 系列实验**全部以 `exp030a` 为代码与实验基线**
2. 主汇报模式固定为 **`equal_concat`**
3. 机制对照模式固定为 **`global`**
4. 必要时才额外查看 **`gcn_only`**
5. 不再基于 `exp023 / exp030 / exp032` 单独开 visibility 主线；这些实验只保留为历史对照

**选择**: 先执行 `exp033 -> exp034 -> exp035`。只有这三步打通，才继续 `exp036 / exp037`。

### [2026-03-13 06:25] 决策 #40

**上下文**: exp035 visibility 消融实验的前两个子实验已完成。

**实验结果**:
- exp035a (score, baseline): mAP 61.1%, R1 73.8% — 与 exp030a seed=1234 一致
- exp035b (score*visibility): mAP 60.4%, R1 71.6% — 比 score-only 差 0.7% mAP / 2.2% R1

**选项**:
  A. 继续跑 035c (visibility_only) 和 035d (binary_visibility) 完成完整消融
  B. 跳过剩余变体，转入新方向

**选择**: B — 跳过剩余变体

**理由**:
- score_visibility 是预期最强的 visibility 模式（兼顾 score 和 visibility 信号），但结果为负
- visibility_only 和 binary_visibility 更激进（完全不用 score），预期不会更好
- 花 4h 再跑两个预期为负的实验，不如转入更有价值的方向
- 但当前证据只覆盖 `score*visibility` 这一条实现路径，不能把单个负结果直接上升成整条 visibility 路线的最终结论

**关键教训（收紧版）**: 目前只能说 `score*visibility` 在当前 keypoint-level 加权池化中未带来正向证据。Visibility 是否能在其它位置（如 retrieval-time reasoning、pairwise masking、target-aware setting）发挥作用，仍需后续更精确的问题定义来判断。

### [2026-03-23 03:30] 决策: MaxSim (ColBERT Late Interaction) 方向确立

**上下文**: exp148 PCVT 和 exp151 PVAT 全部失败。训练集 95.8% 可见率使得所有 visibility-dependent 训练方法无效。

**关键数据发现**:
1. 训练集 95.8% keypoint 可见 → 训练时 visibility 几乎没有变化信号
2. 但 MaxSim test-time 在所有 checkpoint 上稳定 +1.0~1.5% mAP
3. 文献搜索确认：**没有任何 ReID 论文用过 MaxSim training**

**选择**: 从 test-time trick 升级为 training paradigm：
1. **MaxSim test-time**: 已验证（exp030a: +1.1%, PAA: +1.0%, PAA+ROA: +1.5%）
2. **MaxSim training**: 用 Soft-MaxSim 距离替换 GCN branch 的 pooled triplet loss

**论文范式定义**: Occluded ReID = partial-set-to-partial-set matching, 不是 vector-to-vector matching

**执行**:
- exp152 (soft MaxSim, tau=0.05) → 远程
- exp152b (hard MaxSim, tau=0.005) → 本地
- 两者形成 soft vs hard 的强消融

### [2026-03-23 03:50] 决策: 代码大规模减肥

**上下文**: 151 个实验积累了 5370+ 行死代码。31 个模块文件、processor 1598 行。

**执行**:
- 删除 23 个 model/modules/ 文件
- pose_backbone_model: 1110 → 352 行
- processor: 1598 → 765 行
- make_loss: 713 → 186 行
- **验证**: exp066 复现测试 loss/acc 与原始完全精确一致（每个 iteration 数字相同）

### [2026-03-23 02:30] 决策 #N: exp148/149/151 结论与训练集 visibility 关键发现

**上下文**: exp148 PCVT、exp149 SCFA、exp151 PVAT 三条线同时或先后推进，试图从不同角度解决 "single-image support incomplete"。

**关键数据发现**: Occluded-Duke 训练集 person-0 的可见关键点比例均值 **95.8%**，中位数 **100%**。95.6% 的训练图可见率 > 80%。训练数据几乎没有遮挡。

**实验结论**:
1. **exp148 PCVT**: 早期加速（ep30 +2.4 mAP），后期被基线追平并反超（ep100 -0.9 mAP）。3-view 训练的 1/3 主损失稀释 + 训练数据缺乏 visibility 多样性。
2. **exp149 SCFA**: ep30 即止损（-1.5 mAP / -4.7 R1）。bilateral gap case 太少。
3. **exp151 PVAT**: 中性。pvat_acc 从未下降（0.83 constant），gradient reversal 无法影响 backbone，因为训练数据几乎全可见。

**选择**: 彻底放弃所有训练时 visibility-dependent 的创新方向。

**理由**:
1. 训练集 95.8% 可见 → 任何依赖训练时 visibility 变化的机制都缺乏信号
2. 真正的 occlusion gap 在 test-time（gallery/query 有严重遮挡）
3. 连续 3 条不同范式（data augmentation / structural / adversarial）全部失败，排除了实验噪声

**后续约束**: 不再在训练侧做任何 visibility-conditioned 机制。如果要改善遮挡鲁棒性，只有两条路：(a) 数据增强缩小 train-test visibility gap (b) 改善 test-time matching。

### [2026-03-23 全天] 主要成果

**有效发现**：
1. **PLBOA** (Pose-guided Lower-Body Occlusion): +1.6 mAP (2-seed mean +1.57)。数据端创新。
2. **MaxSim** (ColBERT Late Interaction): +1.0~1.5 test-time。NLP 迁移。
3. **STD-PR** (Structural Token Decomposition): 单独 -2.4，但 +PLBOA 后 63.4 (+2.3 vs baseline)。
4. **训练集 95.8% 可见**：解释了所有 visibility-dependent 方法的失败。

**无效方向（今天确认）**：
- MaxSim training (replace: -3.3, additive: neutral)
- Evidential DL (neutral)
- SPLADE (neutral)
- PCVT (neutral)
- PVAT (neutral)
- 200 epochs (不是创新，浪费时间)

**待解决**：
- 还没有一个足够 "eye-catching" 的单一主贡献
- PLBOA 和 MaxSim 是 trick 级别，STD-PR 离 paradigm shift 还有距离
- 需要继续思考真正的范式级创新

### [2026-03-13 06:30] 决策 #41

**上下文**: exp035 完成后，需要选择下一个实验方向。

**选项**:
  A. Learnable Keypoint Attention — 用可学习的 MLP 替换固定置信度加权
  B. Part-level Triplet Loss — 对 GCN 分支的 17 个关键点特征独立施加 triplet loss

**红蓝队辩论**:
- 🔴 红队（方案 A）核心论点: exp035 证明固定权重无效，可学习注意力是自然下一步。新颖性高、可视化价值大、实现简单。攻击 B: 缺少新颖性，梯度干扰风险。信心: 8/10
- 🔵 蓝队（方案 B）核心论点: Phase 1 已验证 GiLt +0.5%，零风险，验证 GCN 关键点级判别性是论文核心证据。攻击 A: 小数据集过拟合风险，alpha suppression 前车之鉴。信心: 8/10
- 综合判断: 先 B 后 A。"先确保特征质量，再优化融合方式"是更稳健的策略。

**选择**: B — Part-level Triplet Loss (exp036)
**理由**: 已有正面信号(+0.5%)，零架构风险，实现快，为 A 提供更好的基础
**执行结果**: ❌ exp036 最终 mAP 60.6%，vs exp035a 61.1%，-0.5%。Per-keypoint triplet loss 未能提升。GCN 消息传递已使特征充分判别。

### [2026-03-13 08:50] 决策 #42

**上下文**: exp036 (per-keypoint triplet) 失败。Phase 1 GiLt 的 +0.5% 来自更弱的 part features（PCFC pooling）。GCN 增强后的 keypoint features 已经足够判别，额外 triplet 约束反而干扰（-0.5%）。

需要选择下一个实验方向。当前 GCN 分支的核心瓶颈不在"特征质量"，而在"融合方式"——equal_concat 是一个固定的、非自适应的融合策略。

**选项**:
  A. Learnable Keypoint Attention (LKA) — 用可学习 MLP 替换固定置信度加权，允许模型自动发现哪些关键点对检索最重要
  B. Adaptive Feature Fusion — 用可学习门控替换固定的 equal_concat 融合，让模型自适应地混合 global 和 GCN 特征
  C. Multi-Scale Keypoint Features — 从多个 backbone stage 采样关键点特征，丰富 GCN 输入的语义层次

**红蓝队辩论**:
- 🔴 红队（方案 A - LKA）核心论点: 最低风险最低成本（~600 params, ~20 行），exp035b 证明权重方案敏感（score→score*vis -0.7%），confidence ≠ 判别重要性。可解释性高（训练后可可视化哪些关键点最重要）。攻击 B: CAPSG 前车之鉴（内容自适应门控失败 -1.1%），fusion 已接近最优。攻击 C: exp005 灾难。信心: 7/10
- 🔵 蓝队（方案 B - AFF）核心论点: exp036 证明 GCN 特征已饱和，继续优化 GCN 内部（LKA/C）是优化已饱和子系统。equal_concat vs concat_scaled p=0.0039 证明融合权重极敏感。AFF 完成 PSG→GCN→AFF 的"感知→补全→融合"完整链条，论文故事最佳。攻击 A: alpha suppression 前车之鉴，LKA 在已饱和系统上增量有限。信心: 8/10
- 综合判断: 两个方案都低风险、高价值。LKA 更快（2h vs 4h），AFF 更直接攻击瓶颈。选择先 A 后 B 的策略：如果 LKA 证明聚合权重已最优，则确认瓶颈在 fusion → 做 AFF；如果 LKA 有效，可与 AFF 叠加。

**选择**: A → B 顺序执行。先 exp037 (LKA)，再 exp038 (AFF)
**理由**: LKA 实现更快，完成关键点加权调查线（score→visibility→learnable），无论结果如何都为 AFF 提供信息

### [2026-03-13 10:55] 决策 #43

**上下文**: `exp035b / exp036 / exp037` 连续三步都指向同一个信号：继续在 GCN branch 内部做权重/损失微调，越来越像局部调参。根据 `AGENTS.md`，此时必须先进入论文/代码学习模式，再决定下一步。

本轮完成的学习：
- KPR（ECCV 2024）论文 + 官方代码
- BPBreID（WACV 2023）论文 + 官方代码
- FRT（TIP）摘要 + 官方仓库状态
- QPM 摘要

**核心发现**:
1. 近年的强路线把问题定义在 **target ambiguity / common visible support / retrieval-time reasoning**，而不是“再学一个融合权重”。
2. BPBreID / KPR 的 visibility 主要落在 **query-gallery pairwise distance**，不是只改 pooling。
3. 我们当前代码线的真实缺口是：
   - `exp030a` 已证明 branch 的价值主要体现在 fusion
   - 但测试时仍用 `equal_concat`
   - 即：结构化 keypoint branch 被训练出来后，在检索阶段被过早压成单向量
4. QPM / 类似质量感知工作已经覆盖了“adaptive weighting / quality-aware fusion”叙事，`AFF` 难以作为主线创新。

**选择**: 暂不把 `exp038 = AFF` 作为默认主线；下一步优先改为 **共同可见关键点检索诊断**。

**理由**:
1. 这个方向的问题定义更强：它直接针对 partial observation 下“哪些局部证据可比较”。
2. 它和 `exp030a` 的既有证据链更一致：branch 的价值既然发生在 fusion，就应继续追问“fusion 的信息到底来自哪些共同可见关键点”。
3. 机制上也更接近文献 gap：我们可以利用已有的 keypoint/skeleton branch，而不去复刻 parsing-based part methods。
4. 证据路径清晰，可先做低风险诊断：
   - `global`
   - `equal_concat`
   - keypoint-only pairwise distance
   - global + common-visible keypoint distance

**执行约束**:
1. 由于用户新增规则“不要主动停下来”，`exp037` 继续自然跑完，不主动中止。
2. GPU 只有 1 张卡；因此在 `exp037` 结束前，优先完成文档校正、文献沉淀和下一实验实现准备。
3. 若后续仍保留 `AFF`，它只作为备选/消融，不再作为默认主线。

### [2026-03-13 11:10] 决策 #44

**上下文**: `exp037` 已自然结束。

**实验结果**:
- `exp037 (LKA)` = `60.7% mAP / 71.7% R1`
- `exp035a (score baseline)` = `61.1% mAP / 73.8% R1`
- 差距 = `-0.4% mAP / -2.1% R1`

**判断**:
1. `LKA` 没有显示出稳定正向收益。
2. 这与 `exp035b`、`exp036` 形成一致信号：**继续在 branch 内部调关键点权重/损失，不是当前最值得推进的主线。**
3. 该结果进一步支持 **决策 #43** 的方向修正。

**选择**: 立即启动 `exp039` 的两个评测子实验：
- `039a`: `cvk_only`
- `039b`: `cvk_hybrid`

**理由**:
1. 先用低风险 retrieval-time diagnostic 回答“共同可见关键点支撑”是否存在。
2. 若诊断为正，再决定是否值得进入更重的训练端或 pair-specific mechanism。
3. 若诊断为负，也能更清楚地界定 branch 增益到底来自哪里。

### [2026-03-13 11:15] 决策 #45

**上下文**: `exp039` 两个 retrieval-time diagnostic 子实验已完成。

**实验结果**:
- `039a cvk_only` = `59.3% mAP / 72.9% R1`
- `039b cvk_hybrid` = `61.9% mAP / 73.2% R1`
- 对照 `exp035a equal_concat` = `61.1% mAP / 73.8% R1`

**结论**:
1. 纯共同可见关键点距离不能替代当前主距离：
   `cvk_only` 的 mAP 明显低于 `equal_concat`。
2. 但共同可见关键点支撑不是噪声：
   `cvk_only` 的 R1 仍有 `72.9%`，说明 keypoint-level pairwise signal 是真实存在的。
3. `cvk_hybrid` 给出正信号：
   - mAP `+0.8%`
   - R1 `-0.6%`
   这符合“keypoint common-support 更适合作为补充项”的判断。

**选择**: 继续推进这条线，但先做更干净的基线复核，而不是立刻调权重。

**下一步**:
1. 直接在 `exp030a` 原始 checkpoint 上复核 `cvk_hybrid`
2. 若仍为正，再进入：
   - 权重敏感性 (`global : cvk`)
   - 多 seed / 多 checkpoint 验证
3. 在那之前，不把 `exp039` 的单 checkpoint 结果上升为最终论文结论

### [2026-03-13 11:31] 决策 #46

**上下文**: `exp040` 已在 `exp030a` 原始 checkpoint 上完成直接复核。

**实验结果**:
- `040a equal_concat` = `61.1% mAP / 73.7% R1`
- `040b cvk_hybrid` = `61.9% mAP / 73.2% R1`
- 差距 = `+0.8% mAP / -0.5% R1`

**附加观察**:
- `040b` 与 `039b`（`exp035a` checkpoint）几乎一致：
  - mAP 相同 `61.9%`
  - R1 相同 `73.2%`
- 因此 `cvk_hybrid` 的正信号不是 bundled checkpoint 偶然现象。

**判断**:
1. retrieval-time 的共同可见关键点 reasoning 已具备 **可复核的单 checkpoint 正向证据**。
2. 当前收益模式也更稳定了：两次都表现为 **mAP 提升、R1 小幅回落**。
3. 这进一步说明它更像整体排序修正项，而不是单纯替代 `equal_concat` 的 top-1 强化器。

**选择**: 继续推进该主线，下一步先做权重敏感性，而不是立刻上升到论文最终结论。

**下一步**:
1. 固定 `exp030a` checkpoint
2. 调整 `TEST.CVK_GLOBAL_WEIGHT / TEST.CVK_KP_WEIGHT`
3. 判断 `1:1` 是否接近最优，或是否存在更稳的工作区间

### [2026-03-13 11:37] 决策 #47

**上下文**: `exp041` 已完成 `cvk_hybrid` 的最小权重敏感性验证。

**实验结果**:
- `1:1`（来自 `exp040b`）= `61.9% mAP / 73.2% R1`
- `2:1`（`041a`）= `61.6% mAP / 72.6% R1`
- `1:2`（`041b`）= `61.6% mAP / 73.6% R1`

**判断**:
1. `1:1` 是当前测试点中的 **mAP 最优点**。
2. 两侧偏移都会使 mAP 从 `61.9%` 回落到 `61.6%`，说明这条线依赖于 global 与 CVK 的平衡，而不是单侧主导。
3. `1:2` 的 R1 接近 `equal_concat`，但 mAP 仍下降，说明“更强 CVK”更像在做 top-1 偏置，而不是整体排序最优。

**选择**: 暂不继续做更细的权重网格调参，下一步转向更高价值的证据加固。

**理由**:
1. 当前已经得到足够清晰的机制信号：`1:1` 不是偶然点，方向解释也比较稳定。
2. 再做更细权重搜索会逐渐滑向 test-time 参数调优，不符合当前主线要求。
3. 更值得投入的是：
   - 多 checkpoint / 多 seed 复核
   - 失败样例 / pair 类型分析

### [2026-03-13 11:49] 决策 #48

**上下文**: `exp042` 已完成 `equal_concat` vs `cvk_hybrid` 的 query-level 差分分析。

**关键结果**:
- `positive_delta_ap = 1129`
- `negative_delta_ap = 822`
- `zero_delta_ap = 259`
- `top1_fixed = 47`
- `top1_degraded = 58`

**判断**:
1. `cvk_hybrid` 的 mAP 增益不是靠少数样例偶然暴涨，而是来自 **更多 query 的整体 AP 改善**。
2. `top1_fixed < top1_degraded` 解释了为什么它会稳定表现成：
   - `mAP` 上升
   - `R1` 小幅下降
3. 这意味着当前最准确的机制描述应改成：
   **common-visible keypoint reasoning 主要作用于 deeper-rank correction，而不是 top-1 boosting。**

**选择**: 继续沿这条 story 推进，但下一步优先做可视化与更多资产复核，而不是再开新调参实验。

**下一步**:
1. 从 `query_deltas.csv` 中抽取最典型的改进 / 退化样例做可视化
2. 并行继续查找是否存在遗失的多 seed checkpoint 资产

### [2026-03-13 11:53] 决策 #49

**上下文**: `exp043` 已完成 qualitative case study 生成。

**结果**:
- `top_improved.png`
- `top_degraded.png`
- 并已同步到 `paper_materials/figures/qualitative/`

**判断**:
1. 现在这条 CVK 主线已经同时具备：
   - aggregate metric
   - query-level 差分统计
   - qualitative 样例图
2. 这比继续刷新的小调参更接近可投稿 story 所需的证据链闭环。

**选择**: 下一步继续追资产层面的多 seed / 多 checkpoint 复核；在此之前，不再开新的 test-time 参数实验。

### [2026-03-13 15:15] 决策 #50

**上下文**: `exp045` 已在 `exp044` 重建出的 `seed42` checkpoint 上完成第二个 checkpoint 复核。

**实验结果**:
- `045a equal_concat` = `60.2% mAP / 72.7% R1`
- `045b cvk_hybrid` = `61.1% mAP / 73.2% R1`
- 差距 = `+0.9% mAP / +0.5% R1`

**附加观察**:
1. `045a` 的 mAP 与既有 seed42 记录 `60.2%` 完全一致，说明 `exp044` 的重建 checkpoint 可用于后续复核。
2. `045b` 的 mAP 增幅与 `exp040` 的 `+0.8%` 非常接近，因此当前最稳定的信号已经从“单 checkpoint 正例”推进到“至少两个 checkpoint 上都能转正的 mAP 信号”。
3. 但这次 R1 没有像 `exp040` 那样回落，反而变成 `+0.5%`，说明先前的 “R1 小降” 不能再写成固定规律。

**判断**:
1. 当前最稳妥、且已经被多 checkpoint 支撑的结论是：
   **`cvk_hybrid` 能稳定改善 mAP。**
2. `exp042` 对 `040a/040b` 的 deeper-rank correction 解释仍然成立，但应明确它首先是在那个 checkpoint 上得到的机制证据，而不是所有 checkpoint 都必须伴随 R1 下降。
3. 因此论文叙事应聚焦：
   - mAP 的跨 checkpoint 正复核
   - common-support reasoning 对整体排序的修正作用
   而不是把 “R1 小降” 当作这条线的必要代价。

**选择**: 继续推进资产恢复，优先补第三个 seed，而不是回到 test-time 权重细调。

**下一步**:
1. 重建 `exp030a seed2024` checkpoint
2. 在第三个 checkpoint 上补 `equal_concat / cvk_hybrid`
3. 再决定是否可以整理成更正式的多 checkpoint 汇总表

### [2026-03-13 15:30] 决策 #51

**上下文**: 在 `exp046` 继续重建 `seed2024` 的同时，基于 `exp039-045` 现有证据和新一轮文献/代码学习，评估是否应提前启动训练端方向。

**已知事实**:
1. `cvk_hybrid` 已在两个 checkpoint 上复核出正 mAP：
   - `exp040`: `+0.8% mAP`
   - `exp045`: `+0.9% mAP`
2. 当前 test-time 权重敏感性已经基本收敛，继续细调性价比很低。
3. KPR / BPBreID / QPM / FRT 等工作都说明：
   - common-visible / pair-specific matching 不是新概念
   - 但主流落点仍多在 retrieval-time matching，而不是把这份 pairwise signal 迁入训练期 mining

**判断**:
1. 现在已经有足够证据支持我们 **不等第三个 seed 完整结束，就先启动训练端候选设计**。
2. 但这不等于可以把当前 `cvk_hybrid` 直接包装成训练端创新；中间还缺一个清晰机制。
3. 当前最值得优先尝试的训练端候选不是 AFF 或新的局部权重模块，而是：
   **CSGT（Common-Support-Guided Triplet）**
   - 用 `kp_weights` 构造 batch 内 pairwise common-support overlap
   - 在 global triplet 上增加 support-aware hard mining 约束

**选择**:
- 保持 `exp046` 继续跑
- 并行准备 `exp047 CSGT`

**理由**:
1. 这符合 `AGENTS.md` 的方向切换要求：先写清楚为什么切，再明确相对 `exp030a` 的单变量改动。
2. `CSGT` 触及的是 **partial observation 下 pair comparability mismatch**，问题定义比“再学一个融合权重”更强。
3. 代码侧已有 `kp_weights` 与 GCN branch 输出，落地成本可控，值得先做最小原型。

### [2026-03-13 17:30] 决策 #52

**上下文**: `exp046` 已完成 `exp030a seed2024` checkpoint 重建，需要决定下一步是否继续停留在资产恢复。

**已知事实**:
1. `exp046` 最终 `Epoch 120` 结果为：
   - `60.1% mAP / 72.9% R1 / 84.0% R5 / 87.6% R10`
2. 这意味着本地现在已经补齐：
   - 原始 `seed1234` checkpoint
   - 重建 `seed42` checkpoint
   - 重建 `seed2024` checkpoint
3. `cvk_hybrid` 的正 mAP 证据已经在前两个 checkpoint 上成立：
   - `exp040`: `+0.8% mAP`
   - `exp045`: `+0.9% mAP`

**判断**:
1. `exp046` 的角色应定义为 **资产恢复实验**，而不是新的方法证据。
2. 第三个 checkpoint 资产既然已经补齐，就不应再把“缺 checkpoint”当作拖延训练端方向的理由。
3. 当前最高优先级应从资产恢复切换到 `exp047 CSGT` 的实际训练验证。

**选择**:
- 立即结束 `exp046` 文档收尾
- 启动 `exp047` 训练

**理由**:
1. `exp046` 已经完成它唯一的任务：把第三个 checkpoint 补回本地。
2. 后续如果 `exp047` 或 `cvk_hybrid` 需要第三 checkpoint 复核，当前资产已经足够支撑。
3. 继续停留在 checkpoint 恢复不会新增论文机制证据，而 `CSGT` 才是当前真正待验证的训练端创新候选。

### [2026-03-13 21:00] 决策 #53

**上下文**: `exp047 CSGT` 失败（pos/neg overlap 几乎相同，机制无法区分正负 pair）。需要决定下一步方向。

**选项**:
  A. **SGMKC (Skeleton-Guided Masked Keypoint Completion)**: 在 GCN 训练时加入 masked keypoint prediction 辅助任务。训练时随机 mask 30% 关键点特征，GCN 通过骨架图传播恢复，辅助 MSE 重建 loss。
  B. **放弃训练端创新，转入 1-2 天文献精读和新问题定义**: 47 个实验已充分说明当前框架训练端改进空间极小，应寻找全新方向。

**红蓝队辩论**:
- 🔴 红队（方案 A）核心论点: SGMKC 实现成本极低（~15 行代码，无新参数），属于不同类别的改进（训练方法论 vs 架构/loss 添加），skeleton graph + masked prediction 组合是真正新颖的（FCFormer 用 transformer decoder，MAE 用 random patches，没有人在 skeleton graph 上做过 masked completion for ReID）。47 个失败实验都是架构添加或 loss 变体，SGMKC 是 self-supervised 训练策略——不同搜索空间。即使失败，负面结论也有论文价值。信心: 6/10
- 🔵 蓝队（方案 B）核心论点: 47 个实验中 21/21 训练端改进全部失败，贝叶斯后验 P(失败) ≈ 95.7%。CLAUDE.md 明确说"不要做 GCN 小变体"。SkeletonMAE (2023) 已在动作识别上做过 skeleton graph masked autoencoding，novelty 受限。即使涨 1%，"在 GCN 上加 MAE 辅助任务"的 story 不够支撑 B 会。应该花时间精读 2024-2025 新工作（uncertainty-aware ReID、diffusion-based augmentation 等），找到 paradigm shift 级别的创新。信心: 8/10

**综合判断**: 选择折中方案——**快速实现 SGMKC 并启动训练（~2h GPU），同时在训练期间做文献精读**。理由：
1. GPU 当前空闲，不用是浪费
2. SGMKC 实现确实极快，机会成本极低
3. 文献精读不需要 GPU，可以完全并行
4. 如果 SGMKC 失败，文献精读已经准备好了下一步方向
5. 但我认同蓝队的核心判断：SGMKC 更可能是 supporting experiment 而非 main contribution

**关键约束**: SGMKC 只跑一次，不做变体。成功则记录，失败则止损并全面转入新方向。

**执行结果**: SGMKC 完整训练 120 epoch，最终 mAP 58.9%（vs exp030a 60.5%，-1.6%）。蓝队判断正确——GCN branch 上的训练端改进已穷尽。

### [2026-03-13 22:45] 决策 #54

**上下文**: exp047（CSGT）和 exp048（SGMKC）连续失败。已 48 个实验，其中 GCN/KPP branch 训练端改进全部为负或中性。根据 CLAUDE.md 止损规则，必须转入文献/代码学习和新方向探索。

**选项**:
  A. **直接实现 PAMC（Pose-Aware Masking Consistency）**: 之前在 innovation_brainstorm.md 已设计的方向。SimSiam 风格自监督 + body-aware masking。
  B. **先做更深入的文献调研，再设计实验**: 研究近期（2024-2025）occluded ReID、masked image modeling for ReID、uncertainty-aware ReID 等方向的新进展，确保 PAMC 不与已有工作重叠，并寻找可能更好的方向。

**选择**: B — 先做文献调研
**理由**:
1. 48 个实验后仍未找到 paradigm shift 级创新，说明需要从外部获取新灵感
2. PAMC 虽然满足创新门槛，但尚未确认是否与 2024-2025 年新工作重叠
3. CLAUDE.md 明确说"连续负结果后应优先切到读论文、下载并阅读代码、写 gap analysis"
4. GPU 空闲时间可用于文献调研（不浪费 GPU），调研完成后再高效利用 GPU

### [2026-03-14 02:32] 决策 #55

**上下文**: exp050 PAMC 训练完成。最终结果：mAP 60.7% / R1 72.2%，与 exp030a 3-seed mean (60.73% / 72.57%) 几乎完全一致。PAMC 的 pose-aware masking + consistency loss 未提供任何有意义的改善。这是连续第 3 个训练端辅助 loss 方向失败（exp047 CSGT, exp048 SGMKC, exp050 PAMC）。

**选项**:
  A. 继续在训练端尝试不同类型的辅助 loss（如对比学习、蒸馏等）
  B. 彻底放弃"在 exp030a 上加辅助 loss"路线，转向全新方向

**选择**: B — 彻底放弃训练端辅助 loss 路线

**理由**:
1. 连续 3 个辅助 loss 实验（CSGT/SGMKC/PAMC）分别代表了 mining filter、自监督重建、一致性正则化这三种完全不同的辅助 loss 范式，全部失败/中性
2. 失败的共同根因：PSG+GCN 已将 ID+Triplet 的训练优化到一个局部最优，任何辅助 loss 要么干扰主目标（SGMKC），要么信号太弱被主目标淹没（PAMC），要么机制不成立（CSGT）
3. 继续在这条路线上"换一种 loss 再试"的预期收益极低，应把时间投入到更有可能产生突破的方向

**下一步**: 进入文献学习，寻找不是"加辅助 loss"而是"改变特征提取或匹配范式"的新方向

### [2026-03-14 05:00] 决策 #56

**上下文**: 文献/代码深入调研完成。研究了 PADE (ICASSP 2024), ProFD (ACM MM 2024), PersonViT, Pose2ID (CVPR 2025), P3E, CION (NeurIPS 2024), SEAS (CVPR 2024), Camera Bias (ICLR 2025 Spotlight) 等近期工作。核心发现：领域正从"更好的特征提取"转向"更智能的匹配/检索"。

**选项**:
  A. **Pose-Aware Metric Learning (PAML)** — 修改 GCN 分支 part triplet loss 的距离计算方式，从聚合 skeleton feature 距离改为逐关键点 confidence 加权 pairwise 距离。对齐训练和 CVK 测试目标。不添加新模块/新参数/新辅助 loss。
  B. **Probabilistic Keypoint Embedding (PKE)** — 修改 GCN 输出为概率分布（均值+方差），用 reparameterization trick 训练，KL 散度正则化。

**红蓝队辩论**:
- 🔴 红队（方案 A: PAML）核心论点: 50 个实验证明任何新增机制都失败。PAML 不添加任何东西，只修改已有距离函数。CVK 已证明逐关键点距离有效（+0.8-0.9% mAP），对齐训练目标是自然延伸。0 新参数，30 行代码，1 次实验验证。与 exp036（per-keypoint triplet）本质不同——PAML 用单一 triplet 但距离通过逐关键点聚合。攻击 B: KL 是另一个辅助 loss，3 连败教训；概率嵌入训练不稳定（方差 collapse、权重敏感）；与 CVK 不兼容。信心: 8/10
- 🔵 蓝队（方案 B: PKE）核心论点: PKE 是表征级创新而非 loss trick。论文 story 更强（概率建模 vs 距离对齐）。可视化价值高（方差热图）。与 CVK 可协同（方差替代 confidence）。攻击 A: 创新性不足（KPR 已有类似 matching），train-test alignment 不一定成立。信心: 6/10
- 综合判断: 红队论点更有力（8 vs 6），核心优势是"不添加任何新东西，只对齐已有逻辑"，与 50 个失败的"添加新机制"实验本质不同。

**选择**: A — PAML
**理由**:
1. CVK 正信号直接支持逐关键点距离的有效性
2. 不添加新模块/参数/loss，避免重蹈 3 连败覆辙
3. 如果成功，自然延伸为"训练端-测试端距离对齐"的完整 story
4. 如果失败，也是有价值的消融（证明距离方式不是瓶颈）
5. PKE 作为后续候选保留

### [2026-03-14 05:31] 决策 #57

**上下文**: exp051 PAML 完成，结果中性。equal_concat: mAP 60.7% / R1 72.7%（≈3-seed mean），CVK hybrid: mAP 62.0% / R1 73.6%（vs exp030a CVK 61.9%/73.2%）。训练-测试 metric alignment 假设未验证。这是训练端辅助 loss/距离修改方向连续第 5 次未能超越 exp030a 基线。

**决策**: **训练端辅助 loss 方向彻底关闭**。下一步需要进入深度文献学习 + 新机制探索。

**执行结果**: exp051 PAML 中性，方向关闭。连续失败列表：exp047 CSGT、exp048 SGMKC、exp050 PAMC、exp051 PAML、exp036 Per-KP Triplet。

### [2026-03-14 05:45] 决策 #58

**上下文**: exp051 PAML 中性后，训练端辅助 loss 方向彻底关闭（5 次连续失败）。需要选择全新方向。

**选项**:
  A. **KP-RPE (Keypoint Relative Position Encoding)** — 在 Swin Stage 3 WindowMSA 中注入关键点相对距离编码，修改 backbone attention pattern
  B. **PPE (Probabilistic Pose-Aware Embeddings)** — 将 GCN 输出改为高斯分布，概率匹配

**红蓝队辩论**:
- 🔴 红队（方案 A: KP-RPE）核心论点: 基础设施已就绪（extra_attn_bias），~2.8K 参数，零初始化安全退化。PAB 失败是因为 additive decomposition 不够表达（bias=val[i]+val[j] 丢失 pairwise 信息），KP-RPE 用 r_ij=d_i-d_j 计算真正的 pairwise 结构关系。论文 story 清晰："将 RPE 从空间推广到身体结构空间"。不引入新 loss，避开 5 连败雷区。攻击 B: 实现复杂度高（方差监督+概率距离+pipeline 兼容性），多个未验证假设，审稿风险高。信心: 7/10
- 🔵 蓝队（方案 B: PPE）核心论点: 重新定义遮挡问题（从 visibility mask 到 uncertainty quantification），pose confidence → feature uncertainty 的因果链是全新机制。论文理论深度更强（概率嵌入+信息几何）。与 PSG 正交互补（PSG 管特征提取，PPE 管匹配）。P3E/DUL 已验证概率嵌入在 ReID 中可训练。攻击 A: PAB 已在同范式失败（attention bias 变体），CLAUDE.md 警告"attention trick"，2.8K 参数太小难以撑论文主贡献，12x4 分辨率太粗糙。信心: 7/10
- 综合判断: KP-RPE 是 backbone 机制改变（非 branch 内 trick），实现简单（~1 天），失败成本低（零初始化），诊断价值高。PPE 作为 backup 保留。

**选择**: A — KP-RPE
**理由**:
1. KP-RPE 是 backbone 级改动（与 PSG 同类），不在 CLAUDE.md 警告的"branch 内 trick"范围内
2. 实现快、风险低，可快速验证后再决定是否转向 PPE
3. PAB 的失败有明确的技术原因（additive decomposition），KP-RPE 直接修复该问题
4. 即使失败，诊断信息清晰（确认 attention bias 在 PSG 存在时是否冗余）
5. PPE 作为后续方向保留——如果 KP-RPE 中性，再投入 PPE 的开发时间

### [2026-03-14 08:15] 决策 #59

**上下文**: exp052 KP-RPE 完成，最终结果：61.0% mAP / 72.7% R1（equal_concat），对比 exp030a 3-seed mean 60.73% / 72.57%，差距 +0.27% / +0.13%，在方差范围内。Global 模式下 KP-RPE 反而微负（-0.3% mAP / -1.1% R1）。训练过程中 mAP 10/12 checkpoint 为正（均值 +0.76%）但最终收敛至基线水平。

这标志着：
- 注意力偏置方向（PAB unary + KP-RPE pairwise）正式关闭
- 训练端辅助 loss 方向（5次失败）已关闭
- Visibility 方向已关闭
- 在 PSG+GCN 框架上的增量修改空间已耗尽

**选项**:
  A. 继续扩展 KP-RPE（更大容量、更多层、与 PSG 更紧密集成）
  B. 彻底放弃增量修改，转向全新架构方向

**红蓝队辩论**:
- 🔴 红队（方案 A）核心论点: KP-RPE 中期信号强（+1.6% mAP），2736 参数可能太小导致收敛后被 overwhelm，扩展到更大容量/更多层可能释放潜力。实现成本低（半天），诊断价值高。风险：继续做变体可能又是中性。信心: 3/10
- 🔵 蓝队（方案 B）核心论点: 7 次增量修改全部中性/失败（5 loss + 2 attention bias），信号极其清楚——当前框架已饱和。继续做变体是沉没成本谬误。新方向有更高的创新潜力和论文价值。可能的新方向：pose-guided contrastive learning、cross-attention decoder、MoE routing。信心: 8/10
- 综合判断: 蓝队论点压倒性优势。数据清楚地表明增量修改空间已耗尽。

**选择**: B — 放弃增量修改，转向全新方向
**理由**:
1. 7 次增量修改全部在方差内或负面，继续做变体的期望值极低
2. 训练中期"正信号消失于收敛"是注意力偏置的系统性问题，不是容量问题
3. 需要全新的问题定义或机制类别才能突破当前上限
4. 具体方向将通过文献学习和 gap analysis 确定

**执行结果**: 待填

### [2026-03-14 08:20] 决策 #60

**上下文**: 需要选择全新方向。分析已有实验模式：
- 有效：PSG（乘性门控）、GCN（结构特征 fusion）、CVK（test-time pairwise matching）
- 无效：所有加性/辅助修改
- 核心规律：直接改变特征的加工方式有效，添加额外信号无效

从论文笔记中发现的未探索方向：
1. DPEFormer 的 pose-guided token selection（从 feature map 中动态选择人体 token）
2. SSSC-TransReID 的 pose-aware contrastive learning（遮挡增强 + 一致性训练）
3. Cross-attention decoder（keypoints 作为 queries 对 feature map 做 cross-attention）
4. Pose-Aware Mixture of Experts（根据 occlusion 模式路由到不同 expert heads）

**选项**:
  A. **Pose-Guided Token Selection + Cross-Attention (PGTCA)**: 用 PSG 热图做 token 重要性评分，选出可靠 token，再用 keypoint-guided cross-attention 提取 part 特征。本质上替换当前 GCN branch 为更强大的 cross-attention 解码器。
  B. **Pose-Conditioned Occlusion Augmentation (PCOA)**: 用关键点位置生成语义化遮挡增强，结合 consistency loss 训练。是数据增强层面的创新，不改变模型架构。

**选择**: A — PGTCA
**理由**:
1. 这是一种架构级改变（替换 GCN branch 为 cross-attention decoder），与我们的发现一致（直接改变特征加工方式更有效）
2. Cross-attention decoder 用 keypoint 作为 queries 是在 pose-guided ReID 中未被清晰实现的机制
3. 可以利用已有的 PSG 基础设施，在其上构建解码器
4. 论文 story：PSG (backbone 注入) + Cross-Attention Decoder (结构化解码) 形成完整的 encode-decode 范式
5. 数据增强方向（方案 B）不改变模型，创新深度不够支撑论文主贡献

**执行结果**: 待填

### [2026-03-16 10:20] 决策 #61

**上下文**: `exp066-074` 已完成 `PAA` 系列探索，`exp075` 正在补多 seed。联网复盘了 KPR、ProFD、Pose2ID、DPEFormer、SSSC、FCFormer、TTPM 等 2024-2025 工作，并重新评估当前成果是否足以支撑 B 类会议/期刊主线。

**已知事实**:
1. 当前最强训练端结果是：
   - `exp066 PAA` = `61.6% mAP / 74.2% R1`
   - `exp067 PAA+ROA` = `62.0% mAP / 73.7% R1`
2. `ROA` 已在 DPEFormer / FCFormer 这一类工作中出现，不能再作为主创新。
3. `PAA` 虽然有效，但本质仍是 pose-conditioned additive adapter；若没有更强的问题定义，容易被审稿人视为“再加一个 adapter”。
4. KPR / TTPM 已把问题推进到：
   - `target ambiguity`
   - `non-target pedestrian occlusion`
   而我们当前主线还没有显式处理 target 与 distractor 的冲突。
5. `exp070` 的负结果只否定了 naive `target-only PAA`，不能上升为“target-aware 路线无价值”。

**判断**:
1. **当前成果还不够支撑 B 类主线**。
   它更像是一个很强的研究基线 + 一个有效新模块 (`PAA`)，但问题定义、机制新意、证据闭环仍不够强。
2. 下一步不应继续刷新的 `PAA` 小变体，也不应回到新的 branch 内 learnable weighting / extra loss。
3. 当前最合理的新方向应切到：
   **Target-Distractor Pose Conditioning (TDPC)**。

**选择**: 在 `exp075` 完成后，默认把下一周主线切换为 `TDPC`。

**相对基线**:
- 以 `exp066` 作为训练端主对照（不带 ROA，避免把已有增强混进主创新）
- 单变量改动：
  - 保留 `PSG + GCN + PAA + 0.5x loss`
  - 只新增 `target / distractor` pose conditioning 机制

**TDPC 机制草案**:
1. `PSG` 继续使用 `scene_heatmap`
2. `PAA` 路径额外拿到：
   - `target_heatmap`
   - `distractor_heatmap = max(non-target persons)`
3. 计算 ambiguity score
4. 只在高歧义样本上增加 target-distractor differential conditioning

**理由**:
1. 它把问题从“更好的 pose injection”升级到“如何在多人遮挡里区分 target 与 distractor”。
2. 它能直接复用 `exp033 / exp034` 已有的 target-aware 基础设施。
3. 它没有被 `exp070` 直接证伪，因为 `exp070` 试的是 hard switch，不是 `scene + target-distractor` 的联合机制。
4. 一周内可实现、可训练、可分析，并且可以自然补上 subset / case study 证据。

**执行约束**:
1. `exp075` 完成前，不启动新的长期主线训练，避免和多 seed 验证抢资源。
2. `TDPC` 第一轮只跑单 seed，不做变体扩散。
3. 若 `TDPC` 在 2-3 天内无明显正信号，则 fallback 到 retrieval-time `common-support recovery`，不继续做 `TDPC` 小修小补。

### [2026-03-16 22:00] 决策 #62

**上下文**: exp076-078 TDPC 方向全面失败。exp079 发现 ROA≈PAA+ROA。exp081 PQTD decoder 不够收敛。需要决定下一步。

**已确认的事实**:
1. PAA 是 multi-person specialist (+1.69% multi / -1.61% R1 single)
2. ROA(无PAA) = 62.0% ≈ PAA+ROA = 62.0%
3. 所有 target-aware PAA 变体失败
4. Transformer Decoder 在 120ep 不可行

**选择**: 启动 exp083 PGFI (Pose-Guided Feature Inpainting) — 在 feature map 空间恢复遮挡区域特征。

**理由**:
1. "recover" 是与 "suppress/inject/select" 完全不同的范式
2. FCFormer 在这个方向上有 SOTA 结果
3. 但我们的 PGFI 用 Conv inpainter (不是 Transformer decoder)，适合 15K 数据量
4. 如果有效：可以讲 "PSG suppress + PGFI recover" 的互补 story
5. 如果无效：说明 12×4 分辨率上 inpainting 太粗糙

**后续计划**:
- 如果 PGFI 中性或负面：转向整体方法叙事（"pose-guided multi-granularity representation"）
- 不再追求单一新模块的大突破


### [2026-03-19 12:45] 决策 #63

**上下文**: `exp107 DACHM` 完成了第一轮 retrieval-time 原型验证。该实验用 `exp030a equal_concat` 做主基线，显式构建多 person hypotheses，并比较：
- `raw_counterfactual_signed`
- `dachm_signed`
- `raw_counterfactual_penalty`
- `dachm_penalty`

**已确认结果**:
1. `base_equal_concat = 61.14% / 73.71%`
2. 四个 DACHM 变体全部低于基线；最佳 `dachm_penalty` 仍为 `60.72% / 73.17%`
3. 子集上同样没有局部正信号：
   - `clean multi`: `63.99 / 77.27` → `63.24 / 75.83`
   - `duplicate-suspect multi`: `61.36 / 76.71` → `60.64 / 76.46`
4. duplicate-aware pruning 没有把 coarse pooled hypothesis rerank 救回来。

**判断**:
1. `target/distractor ambiguity` 这个问题定义本身还没有被证伪。
2. 但当前这条具体机制：
   **pooled person embedding + counterfactual margin rerank**
   已可视为负面。
3. 这与 `cvk_hybrid / SGCFR` 的正结果一起说明：
   真正有效的 pair-specific reasoning 很可能必须发生在 `per-keypoint / common-support` 粒度，而不是 pooled person feature 粒度。

**选择**: 停止 `exp107` 的当前实现，不继续在该公式上做小调参。

**理由**:
1. signed 与 penalty-only 都负面，只是 penalty-only 负得更少，说明不是单纯的评分函数符号问题。
2. 去重版和不去重版一起负面，说明不是 detector duplicate artifact 单独导致失败。
3. 若继续在这个 pooled hypothesis 空间上调 `alpha/topk/threshold`，高概率只是局部调参，不足以形成论文主贡献。

**后续方向**:
- 若继续沿 `ambiguity` 主线推进，下一步应优先尝试：
  **duplicate-aware / confuser-aware 的 per-keypoint common-support reasoning**
- 不再继续 `exp107` 当前公式的参数扫点。


### [2026-03-19 14:55] 决策 #64

**上下文**: `exp108 DACCM` 完成了第二轮 retrieval-time 原型验证。该实验把 `exp107` 的思路从 pooled person embedding 下沉到 `per-keypoint / common-support` 粒度，并以 `exp030a cvk_hybrid` 为主基线，比较：
- `raw_daccm_penalty`
- `daccm_penalty`

**已确认结果**:
1. `base_cvk_hybrid = 61.88% / 73.26%`
2. `raw_daccm_penalty = 61.35% / 72.85%`
3. `daccm_penalty = 61.39% / 72.94%`
4. 关键子集同样负面：
   - `multi`: `64.07 / 76.51` → `63.16 / 75.87`
   - `clean multi`: `65.06 / 76.26` → `64.12 / 75.40`
   - `duplicate-suspect multi`: `62.31 / 76.96` → `61.47 / 76.71`

**判断**:
1. `exp107` 和 `exp108` 组成了一个更强的负证据链：
   - coarse pooled hypothesis rerank 负面
   - per-keypoint common-support penalty 仍负面
2. 这说明问题不只是粒度太粗，而是：
   **当前 retrieval-time 反事实 penalty 机制本身不构成稳定可用的排名信号。**
3. 因而不能再把 `ambiguity/confuser rerank` 当作主创新方向继续做 test-time 小修小补。

**选择**: 停止 retrieval-time ambiguity 线，不再继续 `exp107/108` 公式调参。

**理由**:
1. 两轮原型都给出整体和子集层面的负结果，已经超过“单次试错”。
2. dedup 版始终只带来很小的退化缓解，说明 duplicate artifact 不是该方向失败的主因。
3. 继续扫 `alpha/topk/threshold` 很难形成论文级结论，只会落入局部调参。

**后续方向**:
1. 若还要继续 `ambiguity` 问题定义，必须进入训练端机制，而不是 test-time rerank。
2. 在进入新实验前，先补一轮文献与代码学习，确认近两年关于 occluded ReID / multi-person ambiguity / retrieval-time reasoning 的真实 gap。


### [2026-03-19 16:12] 决策 #65

**上下文**: `exp109` 完成了 `Oracle Support Bank` 上界诊断。该实验使用 `exp030a cvk_hybrid` 的 target keypoint features，在 query+gallery 上用 GT same-ID 样本构造 leave-one-out 的 per-keypoint prototype。

**已确认结果**:
1. `base_cvk_hybrid = 61.88% / 73.26%`
2. `oracle_feat_only_cvk = 66.15% / 77.87%`
3. `oracle_feat_weight_cvk = 70.40% / 81.36%`
4. 极低可见 query 的 headroom 极大：
   - `target_vis<=8`: `29.42 / 26.92` → `78.26 / 84.62`
   - `target_vis<=5`: `16.85 / 14.29` → `78.43 / 85.71`

**判断**:
1. 这不是可汇报结果，因为它使用了评测集 GT same-ID oracle。
2. 但它足以证明：
   **当前性能缺口里有一大块确实来自“support 不完整”，而不是 confuser suppression 失败。**
3. 因而 `support-complete distillation` 已从“想法”升级为“有强 headroom 支撑的训练主线候选”。

**选择**: 启动训练版最小原型，优先做 `per-keypoint prototype distillation`。

**理由**:
1. `oracle_feat_only` 已经大幅转正，说明关键不只是 weight 修正，而是 feature completion 本身。
2. 这条线能自然解释：
   - 为什么 `SGCFR` 有效
   - 为什么 `DACHM/DACCM` 无效
   - 为什么 batch-local recovery 方法此前难以成功
3. 相比继续做 retrieval-time trick，这条线更有机会形成训练端方法贡献。

**执行约束**:
1. 第一版必须保持最小改动，只新增 prototype bank distillation，不叠加 decoder / pairwise rerank / 新 backbone 模块。
2. 仍以 `exp030a` 为主基线。
3. 若第一版单 seed 明显负面，不允许连续做多版小修小补，需先复盘 prototype 更新与蒸馏目标是否合理。


### [2026-03-19 18:20] 决策 #66

**上下文**: `exp110 SCKD` 已完成第一版训练端最小原型。该实验在 `exp030a-eq` 基础上新增 `per-identity / per-keypoint prototype bank`，对 low-visibility keypoint 做蒸馏，除此之外不引入 decoder、rerank 或 backbone 改动。

**已确认结果**:
1. `exp110 ep120 = 61.2% / 73.7%`
2. 匹配单 seed 对照 `exp030a-eq seed1234 = 61.1% / 72.9%`
3. 因而当前单 seed 相对提升为 `+0.1 mAP / +0.8 R1`
4. `sckd` 分项在 `epoch > 20` 后稳定在约 `0.19~0.21`

**判断**:
1. 第一版最小原型已经通过了最关键的门槛：
   **这条训练端主线不是负面的，而且能在不破坏 `0.5` global/part 平衡的前提下转成弱正向。**
2. 但当前增益不大，且仍是单 seed；还不能把它写成“已确认主方法”。
3. 从实现细节看，当前更可能的瓶颈是 teacher 可靠性，而不是蒸馏是否应该存在：
   - `MIN_COUNT=1` 允许只凭单个 high-visibility 样本就构造 teacher
   - 这和“support-complete”要表达的 multi-view support 概念并不完全一致

**选择**: 继续 `support-complete` 主线，但下一步只做 teacher reliability 的单变量改动。

**理由**:
1. `exp110` 已经给出正信号，没有理由因为增益小就回到 retrieval-time 小修小补。
2. 用户特别提醒过 `0.5x global loss` 很关键；当前实验已经保留了这一点，因此不应随意动 global/part 主损失平衡。
3. 下一步最合理的改动是提高 bank teacher 的可信度，而不是调大蒸馏权重。

**后续方向**:
1. 启动 `exp111`，仅改 `POSE_SCKD_MIN_COUNT`，要求 prototype 至少由多个支持样本支撑后再参与蒸馏。
2. 若 `exp111` 转强，说明当前 gap 主要来自 noisy teacher。
3. 若 `exp111` 反而变差，再回头看覆盖率与蒸馏触发范围，而不是直接堆新模块。


### [2026-03-19 18:46] 决策 #67

**上下文**: `exp111` 已完成。该实验在 `exp110_sckd` 基础上只改一个变量：`POSE_SCKD_MIN_COUNT: 1 -> 4`。

**已确认结果**:
1. `exp111 ep120 = 61.1% / 73.8% / 85.2% / 88.6%`
2. `exp110 ep120 = 61.2% / 73.7% / 84.7% / 88.2%`
3. 因而相对 `exp110` 的变化为 `-0.1 mAP / +0.1 R1`

**判断**:
1. `count gating` 没有把 `SCKD` 的弱正向放大，整体更接近等价结果。
2. 这说明当前 teacher reliability 的主要瓶颈，不像是“prototype 需要更多支持样本数”。
3. 更值得继续追的方向是：
   **提高 prototype 写入纯度，而不是单纯提高 prototype 生效所需的 count。**

**选择**: 结束 `exp111`，不再沿 `MIN_COUNT` 继续扫点。

**理由**:
1. `1 -> 4` 已经是足够强的单变量测试；若还继续试 `2/3/5`，高概率只会落入局部调参。
2. 当前并行的远程 `exp112`（`UPDATE_THR=0.7`）在 `ep50` 已给出更强的正信号，因此更值得优先跟进。
3. 这与当前论文主线也更一致：关键不只是“有多少 support”，而是“teacher support 是否足够干净可信”。


### [2026-03-19 19:05] 决策 #68

**上下文**: `exp112` 与 `exp113` 已完成当前阶段使命并被提前停表。
- `exp112`：`UPDATE_THR=0.7`
- `exp113`：在 `exp112` 设置上增加 `SCKD` 诊断统计

**已确认结果**:
1. `exp112` 到 `ep80` 的最佳观测为 `59.7% / 71.6%`
2. 相比 `exp110 ep80 = 59.8% / 71.4%`，其表现更接近 **弱正向/近乎等价**
3. `exp113` 显示：
   - `sckd_pairs / active_ratio / elig_ratio / proto_conf` 基本稳定
   - `proto_count` 持续快速上升
   - `sckd_cos` 从 `~0.83` 下降到 `~0.79`
   - raw `sckd` 从 `~0.17` 升到 `~0.21`

**判断**:
1. 当前 `support-complete` 主线没有被否定；相反，它的瓶颈已比之前更清楚。
2. `count gating` 不是关键，`purity gating` 有价值但还不够强。
3. 当前更核心的问题是：
   **online prototype bank 在 student 蒸馏阶段继续增长，导致 teacher 持续变硬，形成 non-stationary target。**
4. 因此下一步最值得验证的，不再是继续扫 `MIN_COUNT / UPDATE_THR`，而是直接控制 teacher 是否继续变化。

**选择**: 启动 `exp114`，测试 **freeze-after-warmup teacher bank**。

**理由**:
1. 这是对 `exp113` 诊断结论最直接、最干净的单变量响应。
2. 若冻结 teacher 后表现变好，就能把当前主创新从“prototype distillation”进一步推进到：
   **reliable support-complete learning**
3. 若冻结 teacher 后仍不涨，则说明问题不在 non-stationary teacher，而在 distillation target hardness / weighting 设计。

### [2026-03-20 06:00] 决策 #69

**上下文**: SCKD 系列 7 个变体（exp110-116）全部完成或接近完成，给出了一个极其清晰的结论：

| 变体 | 核心改动 | 最终 mAP | 最终 R1 |
|------|----------|----------|---------|
| exp110 | 基础 SCKD (online, thr=0.5) | 61.2% | 73.7% |
| exp111 | MIN_COUNT=4 | 61.1% | 73.8% |
| exp112 | UPDATE_THR=0.7 | 59.7%* | 71.6%* |
| exp113 | 诊断实验 | — | — |
| exp114 | freeze epoch 20 | 61.3% | 73.6% |
| exp115 | freeze epoch 30 | 61.3% | 73.6% |
| exp116 | SCFR (直接替换) | ~61.2* | ~73.5* |

(*exp112 在 ep84 提前停表；exp116 仍在运行，预测值)

**核心发现**:
1. **所有变体收敛到 61.2-61.3 mAP**，无论调整 count、purity、freeze 时机、还是替换策略
2. teacher non-stationarity 不是主要瓶颈（freeze 不帮忙）
3. 直接替换 ≈ 间接蒸馏（SCFR ≈ SCKD）
4. **SCKD 框架的天花板已确认：~+0.1 mAP / +0.7 R1 vs exp030a-eq**

**选择**: 在 exp116 完成后，记录 SCKD 系列的最终负结论，转入新方向。

**理由**:
1. 7 个变体已经穷尽了 prototype bank 方向的合理设计空间
2. 继续在该框架下做微调变体违反 CLAUDE.md 的止损规则
3. oracle headroom (+8.5%) 说明问题真实存在，但 EMA prototype bank 不是正确的解决方案
4. 下一步应转向：
   - 文献/代码学习（寻找不同的 feature completion / cross-view learning 机制）
   - 围绕已确认的 `support incomplete` 问题重新设计新机制
   - 不应因为 prototype bank 线遇到天花板，就直接回退到 generic GCN 小改动或多模块组合扫点


### [2026-03-20 11:15] 决策 #70

**上下文**: 接手复核后发现，`exp117` 与 `exp118` 已经偏离当前主线。
- `exp117`：`VCGA`，属于 generic graph routing 小改动
- `exp118`：在 `exp085 (PAA+ROA)` 上继续叠加 `VCGA` 的组合实验

**已确认事实**:
1. `exp117` 相对 `exp030a-eq seed1234` 为 **61.1% / 73.5%**，本质上是中性结果。
2. `exp118` 改用了 `exp085` 作为对照，不再锚定当前主基线 `exp030a`。
3. `exp118` 还是一个组合实验，不符合当前阶段“围绕论文核心机制做单变量推进”的要求。
4. 这条线会把 story 从 `support incomplete / support-complete learning` 拉回到“GCN 小模块 + 组合扫点”。

**判断**:
1. `exp117` 可以保留为一次旁路线探索，但不能升级为新的论文主线。
2. `exp118` 明显偏离当前方向，继续跑完只会增加噪声，不会提升主线清晰度。
3. 当前更需要的是回到最近一轮有效问题定义之上，再决定下一个真正值得做的新机制。

**选择**: 停止 `exp118`，并把 `exp117/118` 明确标记为旁路线探索，不作为当前主线继续推进。

**理由**:
1. `exp117` 没有形成新的问题定义，只是把 visibility 又塞回 GCN 路由，创新门槛不够。
2. `exp118` 同时违反了主基线锚定和单变量原则。
3. 当前最宝贵的是论文叙事的一致性，而不是继续积累一个组合结果。

**后续要求**:
1. 后续实验重新以 `exp030a` 及其直接问题链为锚点。
2. 若要切到新方向，必须先说明它相对 `support incomplete` 主线的关系，而不是直接跳到模块叠加。


### [2026-03-20 11:40] 决策 #71

**上下文**: 在停止 `exp118` 后，需要为下一实验系列重新选择主线。复核表明：
- `exp047 CSGT` 失败的是 overlap mining，而不是 `pair comparability` 问题本身
- `exp051 PAML` 中性，是因为它只改了 part triplet 的距离定义，没有把 pairwise teacher 几何迁到 global embedding
- `exp109-116` 则说明 `support-complete` 若被压成 `per-ID prototype`，会丢失太多 pair-specific 细节

**判断**:
1. 不能因为“generic auxiliary loss 大多失败”就把所有训练端 pairwise 机制一并否掉。
2. 当前最值得继续赌的，不是 feature prototype，也不是 generic local matching，而是：
   **用已经被 `cvk_hybrid` 验证过的 common-support pairwise 几何，直接蒸馏 global embedding 的关系结构。**
3. 这条线同时承接了两条已确认事实：
   - `cvk_hybrid` 的正信号是真实的
   - prototype bank 的问题在于压缩过强，而不是 pairwise teacher 本身无用

**选择**: 启动 `exp119`，验证 `CSRD (Common-Support Relational Distillation)`。

**理由**:
1. `CSRD` 不是再做 overlap filter，也不是再做 prototype averaging。
2. 它把 skeleton branch 作为 **batch-wise pairwise teacher**，比 `exp047/051` 更直接地作用于 `global` 几何。
3. 若它转正，论文主线可自然收束为：
   - partial observation 下存在 pair comparability mismatch
   - pose/keypoint branch 可作为 privileged pairwise teacher
   - global embedding 需要被蒸馏成更符合 common-support geometry 的空间

**执行约束**:
1. 第一版保持最小改动，只新增 `CSRD` 一个 loss。
2. 仍以 `exp030a` 为唯一训练基线。
3. 若 `ep20/30` 即明显落后，不连续扫多个权重版本，先回到机制层面复盘。


### [2026-03-20 14:20] 决策 #72

**上下文**: `exp119 CSRD` 已完成训练与正式评估：
- `exp119-eq = 61.1% / 73.2%`
- `exp119-g = 60.4% / 70.3%`
- `exp119-cvk = 62.0% / 73.2%`

直接对照为：
- `exp030a-eq seed1234 = 61.1% / 72.9%`
- `exp030a-g seed1234 = 59.8% / 69.9%`
- `exp040b cvk_hybrid = 61.9% / 73.2%`

**判断**:
1. `CSRD` 已经证明：**训练端 pairwise relational teacher 不是伪命题**。
2. 当前最清楚的增益落在 `global`（`+0.6 / +0.4`），说明它确实把 common-support 几何迁进了 backbone/global 空间。
3. `equal_concat` 仍接近持平，说明第一版 teacher 还不够强；瓶颈更像 teacher 的 `support incompleteness`，而不是 relational distillation 这件事本身无效。
4. 因而 `exp109` 的高价值结论仍应保留：真正缺的不是再换一个 loss 形式，而是 **更 support-complete 的 teacher**。

**选择**: 不扫 `CSRD` 权重/温度，直接进入下一步：
**把 `exp109` 的 support-complete bank 降级为 teacher enhancer，而不是 pointwise distillation target，构造 support-complete relational teacher。**

**理由**:
1. 这条线同时保留了 `exp109` 的核心 headroom 和 `exp119` 已验证的 pairwise 机制。
2. 它避免回到 `exp110-116` 已经穷尽的 prototype-pointwise 蒸馏。
3. 相比扫 `tau / weight`，它更像问题层面的推进，而不是调参。

**执行约束**:
1. 下一实验相对 `exp119` 只改 teacher 构造，不改 backbone、batch size、主 loss 配比。
2. bank 只用于增强 `CSRD teacher`，不直接给 student 施加 pointwise cosine loss。
3. 默认沿用已经验证更可靠的 `update_thr=0.7`，避免把旧的 bank 噪声重新带回来。


### [2026-03-20 16:20] 决策 #73

**上下文**: `exp120 SCRD` 已在 `ep90` 人工停表。关键事实是：
- 指标轨迹：
  - `ep40 = 55.5 / 67.8`
  - `ep50 = 56.2 / 69.3`
  - `ep60 = 57.5 / 69.7`
  - `ep70 = 58.0 / 71.2`
  - `ep80 = 59.2 / 71.7`
  - `ep90 = 59.9 / 73.2`
- 对照 `exp119`：
  - `ep50 = 56.8 / 69.3`
  - `ep90 = 60.1 / 73.7`
- 机制统计：
  - `csrd_sr ≈ 0.145`
  - `csrd_sn ≈ 157~159`
  - `csrd_tgap ≈ 0.55`
  - `csrd_sgap ≈ 0.53`

**判断**:
1. `support-complete teacher` 并没有“没生效”，相反，它已经稳定地增强了 teacher 几何。
2. 但这种增强没有自动转成更好的检索指标，说明当前瓶颈已经不再是 “teacher 够不够完整” 本身。
3. 结合 `exp109` 的 oracle 结论，更合理的新解释是：
   **support-complete 监督的价值集中在 support-incomplete 样本；如果对所有 anchor 等权蒸馏，clean 样本会稀释掉这份增益。**
4. 因此，`exp120` 否定的不是 `exp109 -> exp119` 这条主线，而只是：
   **“只增强 teacher 内容、但不改变 supervision 分配” 还不够。**

**选择**: 启动 `exp122`，验证 **Support-Gap Weighted SCRD**。

**理由**:
1. 相对 `exp120`，这是最干净的下一跳：teacher、bank、主 loss 全不变，只改 `CSRD` 的 anchor 加权方式。
2. 它直接把 `exp109` 的低可见 headroom 转译成训练机制，而不是再做 generic loss 调参。
3. 若这一步转正，论文主叙事可以自然升级为：
   - 单图遮挡带来 support incomplete
   - pose branch 提供 support-complete relational teacher
   - 但 distillation 必须 **selective**，聚焦真正存在 support gap 的 anchor

**执行约束**:
1. `exp122` 必须以 `exp120` 为唯一直接对照。
2. 唯一改动是：`CSRD` 按 sample-level `replace_ratio` 做 anchor weighting。
3. 不同时改 `tau / weight / bank freeze`，避免再次混入多个变量。


### [2026-03-20 17:15] 决策 #74

**上下文**: `exp122 SGW-SCRD` 已在 `ep43` 提前终止。关键事实是：
- `ep40 = 55.4 / 68.2`
- 对照：
  - `exp119 ep40 = 55.9 / 68.7`
  - `exp120 ep40 = 55.5 / 67.8`
- 机制统计：
  - `csrd_ar ≈ 0.56`
  - `csrd_aw ≈ 0.145`
  - `csrd_tgap ≈ 0.49`
  - `csrd_sgap ≈ 0.44`

**判断**:
1. `exp122` 不是实现失败；sample-level selective weighting 确实被正确接入并稳定工作。
2. 但它没有把 `support-complete teacher` 的增强转成更好的指标，反而更像削弱了有效监督总量。
3. 这说明当前问题不该再写成“监督该打给哪些样本”，而应进一步收紧成：
   **监督该聚焦哪些 pair/relations。**
4. `support-complete` 主线本身仍然成立；被否定的只是 sample-level `replace_ratio` 作为路由信号太粗。

**选择**: 启动 `exp123`，验证 **Pair-Delta Focused SCRD**。

**理由**:
1. 相对 `exp120`，这仍是单变量：teacher bank、teacher 替换、主 loss、batch size 全不变，只改 `CSRD` 对 pair 关系的聚焦方式。
2. 它直接回应 `exp122` 的失败：真正该被强调的不是“这个样本补了多少 keypoint”，而是 **support-complete teacher 实际改变了哪些 pair 几何**。
3. 若这一步转正，story 会比 sample-level weighting 更扎实：
   - 单图遮挡带来 support incomplete
   - support-complete teacher 改变一部分 pairwise comparability
   - distillation 应聚焦这些 **teacher-change pairs**，而不是对整个样本等权放大/缩小

**执行约束**:
1. `exp123` 必须以 `exp120` 为唯一直接对照。
2. 唯一改动是：`CSRD` 由 sample-level anchor weighting 改为 pair-level delta focusing。
3. 不同时引入 freeze、改 tau、改 bank 写入阈值，避免再次混入多个变量。


### [2026-03-20 10:36] 决策 #75

**上下文**:
- `exp121 freeze30` 已完成收敛：
  - `ep120 = 60.6 / 74.0`
  - 对照 `exp119 ep120 = 60.4 / 73.4`
- `exp123 pair-delta` 已运行到中后期：
  - `ep40 = 55.5 / 68.9`
  - `ep50 = 56.2 / 69.9`
  - `ep60 = 57.8 / 70.9`
  - 对照 `exp119 ep60 = 57.7 / 70.5`
  - 对照 `exp120 ep60 = 57.5 / 69.7`
- 同时 `exp123` 的机制统计一直显示：
  - `csrd_pd = 0.002~0.003`
  - `csrd_pf = 1.06~1.08`

**判断**:
1. `stable teacher` 已经被 `exp121` 明确坐实为有效 supporting mechanism，但它不是当前主突破口。
2. `pair-level teacher-change focusing` 到 `ep60` 首次同时超过 `exp119/120` 同阶段，说明这条线本身是成立的。
3. 当前最明显的瓶颈不是“pair focus 没用”，而是 **focus 放大力度偏弱**：
   - `pair_delta` 很小
   - 导致 `pair_focus` 长期只有 `1.06~1.08`
   - 正向收益兑现得慢且浅

**选择**: 保持本地 `exp123` 继续跑，同时利用空出的远程资源启动 `exp124`，只测试 **更强的 pair focus 放大**。

**理由**:
1. 这一步仍然严格遵守单变量原则：相对 `exp123` 只改 `POSE_CSRD_PAIR_WEIGHT_ALPHA`
2. `exp123` 已给出“方向对”的证据，因此下一跳最有信息量的不是换题，而是放大当前有效信号
3. 远程 `exp121` 已收尾，当前最合理的资源利用方式是并行验证 `alpha` 是否就是当前瓶颈

**执行约束**:
1. `exp124` 必须以 `exp123` 为唯一直接对照。
2. 唯一改动是把 `POSE_CSRD_PAIR_WEIGHT_ALPHA` 从 `1.0` 提高到一个更强但仍保守的值。
3. 不同时引入 `freeze30`、改 `tau`、改 bank 写入阈值，避免再次混入多个变量。


### [2026-03-20 12:05] 决策 #76

**上下文**:
- `exp123` 已完成正式评估：
  - `equal_concat = 61.1 / 73.4`
  - `global = 60.2 / 70.3`
  - `cvk_hybrid = 61.9 / 73.2`
- 对照 `exp119`：
  - `61.1 / 73.2`
  - `60.4 / 70.3`
  - `62.0 / 73.2`
- 同时远程 `exp124 alpha=4.0` 到 `ep40` 的关键信号是：
  - `ep20 = 47.7 / 62.0`
  - `ep30 = 53.2 / 66.4`
  - `ep40 = 55.6 / 68.6`
  - `csrd_pf = 1.24~1.29`

**判断**:
1. pair-level `teacher-change focusing` 主线没有被否定，但 `alpha=1.0` 的第一版正式结果只做到与 `exp119` 近乎等价。
2. 远程 `exp124` 又说明：单纯把 `alpha` 放大确实能显著增强 `pair_focus`，但到 `ep40` 仍然只是近乎持平，尚未形成明确更强的中期优势。
3. 因而当前更合理的瓶颈不再是“有没有 pair focus”或“alpha 够不够大”，而是：
   **teacher-change pairs 本来就稀疏，若仍对所有 pair 做连续平滑加权，真正有信息量的 changed pairs 仍会被大量近零变化 pair 稀释。**

**选择**: 保持远程 `exp124` 继续跑，同时在本地启动 `exp125`，验证 **Sparse Pair-Delta SCRD**。

**理由**:
1. 这一步仍然严格锚定 `exp109 -> exp119` 主线，不回到 sample-level，也不回到 generic 模块叠加。
2. 相对 `exp123`，`exp125` 只改 pair 路由机制：从连续加权改为稀疏 top-delta 选择。
3. 它直接回应当前最具体的证据：
   - `exp123` 说明“平滑 pair focus 太弱”
   - `exp124` 到 `ep40` 说明“单纯增大 alpha 也未必足够”
4. 若 `exp125` 转正，主创新会从“加一个权重”升级成更清楚的机制：
   **只把被 support completion 真正改变过的 comparability relations 蒸进 global embedding。**

**执行约束**:
1. `exp125` 必须以 `exp123` 为唯一直接本地对照。
2. 唯一改动是 `CSRD` 的 pair 路由：`delta -> delta_top`，并固定一个保守的 top ratio。
3. 不同时改 `alpha`、不改 teacher bank、不断开 `support-complete` teacher，避免再次混入多个变量。


### [2026-03-20 14:32] 决策 #77

**上下文**:
- 本地 `exp125 delta_top` 已运行到后期：
  - `ep70 = 58.3 / 72.1`
  - `ep80 = 59.4 / 72.0`
  - `ep90 = 60.1 / 73.9`
- 远程 `exp124 alpha=4.0` 也进入后期：
  - `ep90 = 59.8 / 72.6`
  - `ep110 = 60.1 / 73.1`
- 同时 `exp125` 的机制统计持续显示：
  - `csrd_psr = 0.90+`
  - `csrd_pf = 1.10~1.14`

**判断**:
1. `exp125` 的 late-stage 表现已经把 `pair routing` 从“弱正向候选”提升成了当前最强的训练主线候选之一。
2. 相对 `exp124`，更结构化的 `delta_top` 当前已经证明比“仅增大 alpha”更强，至少在 late-stage 的 `R1` 上明显占优。
3. 但 `exp125` 也同时明确暴露出：当前实现依然没有形成真正的 sparse routing，`top-25%` 在阈值式实现下被严重 tie 扩散。

**选择**:
1. 本地 `exp125` 继续跑到正式收敛与后续正式评估，不中途停表。
2. 远程 `exp124` 继续自然收尾，但不再追加任何同类 `alpha` 扫点。
3. 下一条真正的因果验证主线切到 `exp126 exact top-k sparse routing`。

**理由**:
1. 现在已经不需要再证明“pair routing 有没有用”，而需要回答“真稀疏 routing 会更强还是更差”。
2. `exp125` 说明当前主线是有效的，因此不能中途停掉。
3. `exp126` 是相对 `exp125` 的最小单变量下一跳，直接检验当前最关键的机制缺口。

**执行约束**:
1. `exp126` 只允许改 pair 选择实现，不能同时改 `alpha/top_ratio/teacher bank`。
2. `exp124/125` 只作为 late-stage 证据保留，不再围绕其参数做横向扫点。
3. 后续文档与 story 必须把“当前收益来自结构化 pair focus，但不等于已证明真稀疏 routing”写清楚。


### [2026-03-20 23:28] 决策 #78

**上下文**:
- 本地 `exp125` 已完成训练：
  - `ep110 = 60.4 / 73.8`
  - `ep120 = 60.5 / 73.5`
- 远程 `exp124` 已完成训练：
  - `ep120 = 60.2 / 72.9`
- 远程 `exp126 exact top-k` 已自动接续启动：
  - `ep20 = 47.7 / 62.0`
  - `csrd_psr = 0.292`

**判断**:
1. `exp125` 已把 “结构化 pair routing 有效” 这件事坐实，当前它是已完成训练中最强的 pair-routing 版本。
2. `exp124` 证明了单纯增大 focus 强度也有效，但最终不如 `exp125`，因此它应退居 supporting branch。
3. `exp126` 的首个关键信号已经不是指标，而是机制：
   - exact top-k 终于把 `pair_select_ratio` 从 `0.90+` 压到了 `0.292`
   - 说明我们现在第一次真正开始测试“真稀疏 routing”

**选择**:
1. 本地优先转入 `exp125` 正式评估与结果落盘。
2. 远程继续监控 `exp126` 到 `ep30/40`，这是当前最有信息量的主线实验。
3. 不再启动任何新的 `alpha` 或 `delta_top` 变体，直到 `exp126` 给出明确方向。

**理由**:
1. `exp125` 的训练问题已经回答完了，继续做的是评估而不是再训练。
2. `exp126` 现在是唯一能回答关键机制问题的实验：
   - 真稀疏 routing 是更强，还是其实 `exp125` 的“伪稀疏”更优？
3. 若 `exp126` 转正，论文主机制会显著变硬；若转负，我们也能据此更准确地解释为什么 `exp125` 有效。

**执行约束**:
1. 在 `exp126` 得到 `ep30/40` 前，不再开启新的并行主实验。
2. `exp125` 评估必须沿当前正式配置完成，不额外加 test-time trick。
3. 文档中要明确区分：
   - `exp125`: 有效的结构化 pair focus
   - `exp126`: 真稀疏 pair selection 的因果验证


### [2026-03-20 16:18] 决策 #79

**上下文**:
- 远程 `exp126` 已在跑，它负责回答：
  - “真稀疏 pair routing” 是否优于 `exp125` 的伪稀疏版本
- 但本地 3090 此时空闲
- 同时当前 `exp109` 主线已经暴露出另一个未被打透的缺口：
  - `SCKD` 太间接
  - `SCFR` 太硬
  - 二者都没有把 oracle support-complete 上界真正兑现出来

**判断**:
1. 现在继续在本地扫 `alpha / top_ratio / freeze epoch` 属于低价值调参，不符合当前阶段目标。
2. 但这不意味着要离开 `exp109`；相反，最合理的下一步仍然是沿 `support incomplete -> support-complete learning` 这条主线，直接测试更强的 feature-level 兑现机制。
3. `SCFR≈SCKD` 只能说明 “hard replace 不优于 loss-only”，不能说明 “feature-level support completion 整体无效”。

**选择**:
1. 本地启动 `exp127: SCRC (Support-Conditioned Residual Completion)`。
2. 该实验保持 `bank`、`warmup`、`threshold` 与 `exp116` 同量级，只改 low-vis keypoint 如何利用 support-complete prototype：
   - 从 `hard replace`
   - 改为 `learnable residual fusion`

**理由**:
1. 这是沿 `exp109` 主线的下一阶段机制，而不是换题。
2. 它直接回应了 `SCFR` 的两个已知问题：
   - 分布错位
   - 原始 instance-specific 线索被硬覆盖
3. 相对 `SCKD/SCFR`，`SCRC` 既更直接，又不那么硬，属于比纯 loss / 纯 routing 更接近方法核心的一跳。

**执行约束**:
1. `exp127` 只测试 `residual completion` 本身，不同时叠加 `CSRD` 或新的 pair routing。
2. 主对照优先用 `exp116 SCFR`，其次才是 `exp110 SCKD` 与 `exp030a-eq seed1234`。
3. 若 `exp127` 仍只得到 `SCFR≈SCKD` 的结果，再考虑结束这条 feature-level bank 兑现线。


### [2026-03-20 18:35] 决策 #80

**上下文**:
- `exp127 SCRC` 到 `ep100 = 60.5 / 73.1`
- 对照:
  - `exp116 ep100 = 60.7 / 73.4`
  - `exp110 ep100 = 60.8 / 73.4`
  - `exp114 ep100 = 60.9 / 73.4`
- 同时 `SCRC` 日志显示:
  - `scrc_g ≈ 0.999`
  - `scrc_gm = 1.000`
- 本地 `exp128 freeze30` 已按用户要求手动终止，不再继续 `freeze` 线

**判断**:
1. `SCRC` 没有把 feature-level support completion 推成更强结果，反而 late-stage 基本塌成了“近似 hard replace”。
2. 因而 `exp109` 被否定的不是 `support incomplete` 问题定义，而是：
   - per-ID prototype 的 direct feature completion 兑现方式
3. `freeze20/30` 的既有证据已经足够说明它只是弱 supporting mechanism，不值得继续占用本地算力。
4. 当前最有价值的缺口不再是 “teacher 还该不该更稳定”，而是：
   **support-complete teacher 的新增 correction 仍被完整 teacher target 稀释。**

**选择**:
1. 关闭 `SCRC` 这条本地主线，不再追加 direct completion 变体。
2. 关闭本地 `freeze` 线，不再继续 `exp128` 类实验。
3. 本地启动 `exp129: Residual-Correction SCRD`。

**理由**:
1. `exp120/123/125` 的共同现象说明：
   - support-complete teacher 的增量信息是真实存在的
   - 但当前 full-teacher distillation 没把这部分新增修正单独抽出来学
2. `exp129` 相对 `exp125` 仍然是单变量：
   - 保留在线 teacher
   - 保留 `delta_top` pair routing
   - 只把 distillation target 从 `full teacher` 改成 `residual correction`
3. 这一步比继续扫 `alpha/top_ratio/freeze` 更接近方法机制，也更贴合 `exp109` 的问题定义。

**执行约束**:
1. `exp129` 只改 `POSE_CSRD_TARGET_MODE`，不同时改 sparse 规则、teacher 更新或 backbone。
2. 文档中要明确区分：
   - `exp125`: 有效的结构化 pair focus
   - `exp126`: 真稀疏 pair selection 的因果验证
   - `exp129`: target dilution 是否是当前主瓶颈的因果验证


### [2026-03-20 21:20] 决策 #81

**上下文**:
- 本地 `exp130 residual_kl` 已跑满:
  - `ep110 = 60.1 / 73.4`
  - `ep120 = 60.1 / 73.1`
- 直接对照 `exp125`:
  - `ep110 = 60.4 / 73.8`
  - `ep120 = 60.5 / 73.5`
- 同时 `exp130` 的后期 `csrd` 统计始终稳定:
  - `csrd = 0.011~0.013`
  - `csrd_pf = 1.12~1.14`
  - `csrd_psr = 0.90~0.91`

**判断**:
1. `residual_kl` 不是实现失败，也不是因为训练信号太弱；它在后期一直稳定工作。
2. 但它到收敛都没有压过 `exp125`，因此当前不能再把“target dilution”当作主瓶颈。
3. 这意味着当前最值得继续推进的，不是 `target form`，而是：
   **changed pairs 的覆盖与选择机制本身。**
4. 因而 `exp130` 的价值主要是负向因果证据：
   - 完整 teacher target > residual target
   - 主线应回到 `pair coverage / pair selection`

**选择**:
1. 正式结束 `exp130` 这条 `residual target` 支线，不再追加 `target_mode` 变体。
2. 保留 `exp125` 作为当前本地最强的在线 `SCRD` 版本。
3. 本地下一轮不再改 target，而改 **relation coverage**：
   - 从 batch 内 sparse pair routing
   - 推进到 cross-batch changed-pair coverage

**理由**:
1. `exp125` 已证明结构化 pair focus 有效。
2. `exp126` 正在远程回答“真稀疏 routing 本身是否更优”。
3. 在此基础上，本地最合理的下一问已经不是 “teacher 应该长什么样”，而是：
   **单个 batch 里的 changed pairs 是否太少，导致 `SCRD` 的有效监督覆盖不够。**

**执行约束**:
1. 新实验必须继续以 `exp125` 为唯一直接本地对照。
2. 新实验只允许新增一种 relation-coverage 机制，不同时改 `target_mode / pair_weight_mode / teacher bank`。
3. 按用户最新规则，启动前必须先完成 Claude 审查并生成 `claude_review.md`。


### [2026-03-21 02:25] 决策 #82

**上下文**:
- 本地 `exp131 cross-batch queue` 已跑满:
  - `ep110 = 60.4 / 73.7`
  - `ep120 = 60.5 / 73.7`
- 直接对照 `exp125`:
  - `ep110 = 60.4 / 73.8`
  - `ep120 = 60.5 / 73.5`
- 同时 `exp131` 的 queue 统计始终真实工作:
  - `csrd_qn = 256`
  - `csrd_qr = 0.427~0.441`
- 另一个重要背景是：
  - `exp089 PAMN` 只有 design/review 草案，从未真正接入 checkpoint 与测试检索流程，因此**不能**算作“learned pair module 已被证伪”

**判断**:
1. `cross-batch changed-pair coverage` 不是当前主瓶颈；queue 明显参与了监督，但没有带来实质性的 mAP 提升。
2. 至此，`target form` 与 `relation coverage` 两个方向都已经被较干净地排除了主矛盾地位。
3. 当前更合理的主假设应收紧为：
   **pair-specific support-complete correction 不能被当前单向量 student 充分吸收。**
4. 这意味着下一步不该再继续做：
   - `queue size`
   - `target_mode`
   - `freeze / alpha / top_ratio`
   这些局部改动，而应切到一个真正进入检索路径的 learned pair module。

**选择**:
1. 正式结束 `exp131` 这条 `relation coverage` 支线，不再追加 queue 类变体。
2. 本地启动新方向 `exp132`：
   - **LTCS / 自适应共同支撑融合**
   - 用一个真正挂在模型里的 pair-adaptive fusion head
   - 学习每个 pair 该在多大程度上信任 `global distance` 与 `CVK distance`
3. 该 head 必须：
   - 被保存进 checkpoint
   - 被测试期 evaluator 真正调用
   - 不能再犯 `exp089` 那种“训练了但不进入检索”的错误

**理由**:
1. `exp125` 已说明 pair-level correction 是真实有价值的，但把它蒸进 embedding 的上限已经开始显现。
2. `exp131` 说明即使给 student 更多 changed pairs，它也没有自动学得更好，问题更像在 **correction 的表示形式**，而不是 **监督覆盖**。
3. `exp040/045` 的固定 `cvk_hybrid` 已经证明 pair-specific common-support correction 在检索时能转成稳定正信号。
4. 因而当前最值得赌的新机制不是“再蒸一次”，而是：
   **学习一个真正的 pair-adaptive correction rule，并把它直接接入检索。**

**执行约束**:
1. `exp132` 必须先完成 Claude 审查，再允许启动。
2. `exp132` 必须把 learned pair module 接到模型和 evaluator，两端都要打通。
3. `exp132` 的直接对照应同时保留：
   - `exp030a-eq seed1234`
   - 固定 `cvk_hybrid`
   - `exp125`（作为“蒸进 embedding”的当前最强对照）


### [2026-03-21 07:20] 决策 #83

**上下文**:
- `exp132 LTCS` 已跑满并完成同 checkpoint 正式评估:
  - `exp132a cvk_adaptive = 62.1 / 72.8 / 84.8 / 88.1`
  - `exp132b cvk_hybrid  = 62.1 / 72.8 / 84.8 / 88.1`
- 同时训练监控终点也是：
  - `ep120 = 62.1 / 72.8`
- 当前日志里没有落出 `ltcs_*` 统计，因此这轮实验只能靠正式结果做主要判定

**判断**:
1. `exp132` 已较干净地否定了“第一版 learned alpha-fusion 就足以超过固定 `cvk_hybrid`”这一命题。
2. 这不等于 learned pair module 大方向被证伪；真正被证伪的是更具体的实现：
   - 单个 `alpha`
   - 在 `global distance` 与 `CVK distance` 之间做凸组合
   - 用 teacher distance 做回归监督
3. 当前最合理的解释是：
   - pair-adaptive correction 的表示能力不够
   - 且当前监督不够 ranking-aligned
4. 因而主矛盾已从“要不要进入检索期 learned pair module”进一步收紧为：
   **需要更强的 pair scorer，而不是更聪明的标量融合权重。**

**选择**:
1. 正式结束 `exp132` 这一版 `LTCS alpha-fusion` 主线，不再追加：
   - hidden dim
   - warmup
   - teacher bank 小调参
   - `alpha` 初始化/范围
2. 保留 `exp132` 作为一个重要负证据：
   - retrieval-time learned pair module 值得做
   - 但 scalar fusion 不够
3. 下一步切向新的主线：
   - **ranking-aligned learned pair scorer / pair residual correction**

**理由**:
1. 如果 learned head 连同一 checkpoint 下的固定 `cvk_hybrid` 都无法超过，继续扫 `alpha` 头局部参数没有意义。
2. `exp125` 与 `exp132` 合起来说明：
   - pair-specific correction 真实存在
   - 但它既不适合继续只蒸进 embedding
   - 也不适合只压成一个融合权重
3. 因而更有价值的升级，不是继续做 distance mixing，而是直接学习：
   **这个 pair 应该被额外加/减多少 correction score**

**执行约束**:
1. 新实验必须继续通过 Claude 审查后才能启动。
2. 新实验相对 `exp132` 只能改一个核心变量：
   - 从 `alpha-fusion` 改为更强的 `pair scorer / residual score`
3. 新实验必须保留同 checkpoint 正式对照：
   - learned scorer
   - 固定 `cvk_hybrid`
   不能再只看训练监控曲线下结论。


### [2026-03-21 08:55] 决策 #84

**上下文**:
- 本地 `exp133 LPCS` 已跑到:
  - `ep40 = 56.5 / 67.8`
  - `ep50 = 58.3 / 69.6`
- 远程 `exp134 Sparse LPCS` 已跑到:
  - `ep10 = 35.7 / 49.9`
  - `ep20 = 46.4 / 58.1`
- 但两个日志都在 `epoch 21+` 后始终没有任何 `lpcs_*` 统计
- 代码排查确认：
  - `processor.py` 中 `kp_aux_data` 的构建条件漏掉了 `ltcs_enabled / lpcs_enabled`
  - 结果是 `lpcs_teacher_feats` 永远不会生成
  - `LPCS` loss 实际从未加入训练

**判断**:
1. `exp133/134` 的当前数值全部 **不能** 用于支持或反驳 `LPCS`
2. 这不是方法结论，而是共享接线 bug 导致的失效 run
3. 当前最重要的决策不是“继续观察曲线”，而是立即止损并重跑干净实验
4. 这也解释了为什么：
   - 曲线像 baseline
   - 日志里完全没有 `lpcs_*`
   - 之前所有关于 `exp133/134` 方法有效性的判断都不可靠

**选择**:
1. 立即停止本地 `exp133` 与远程 `exp134`
2. 在文档中显式把两轮实验标记为 **失效 run**
3. 修复共享接线 bug：
   - `kp_aux_data` 构建条件补上 `ltcs_enabled / lpcs_enabled`
4. 以新实验编号重启：
   - `exp135`: corrected `LPCS`
   - `exp136`: corrected `Changed-Pair Sparse LPCS`

**理由**:
1. 继续跑当前进程只是在浪费 3090 和 5060 Ti 算力
2. 共享接线 bug 一旦存在，任何后续阶段数值都没有研究意义
3. 既然方法还没真正被测试到，就不能把当前线判负，更不能切题
4. 正确做法是：
   - 修 bug
   - 重新做 clean run
   - 再看 `lpcs_*` 和正式结果

**执行约束**:
1. `exp135/136` 必须重新走 Claude 审查流程
2. 文档必须明确写清：
   - `exp133/134` 为失效 run
   - 不得再引用其数值作为方法证据
3. `exp136` 仍需保持相对 `exp135` 的单变量原则：
   - 只改 sparse pair routing


### [2026-03-21 19:40] 决策 #85

**上下文**:
- 本地 `exp135 corrected LPCS` 已跑满:
  - `ep110 = 61.2 / 72.3`
  - `ep120 = 61.1 / 72.3`
- 远程 `exp136 corrected sparse LPCS` 已跑满:
  - `ep110 = 61.0 / 72.0`
  - `ep120 = 60.9 / 72.1`
- 同时机制统计已经足够清楚:
  - `exp135`: `lpcs_psr / lpcs_pf = 1.000 / 1.000`
  - `exp136`: `lpcs_psr = 0.254`、`lpcs_pf ≈ 3.0`

**判断**:
1. 修复后的 `LPCS` 方向已经第一次被真正测到，不能再把它视为失效实现
2. `exp135` 是有效线，但它的最终形态是：
   - `mAP` 更强
   - `R1` 偏弱
   更像排序修正型收益，而不是最优主突破
3. `exp136` 已经把“真稀疏 routing”机制完整坐实，但到收敛也只得到与 full-pair `LPCS` 近似等价的结果
4. 因而当前最值得收紧的判断不是“routing 是否有效”，而是：
   **当前 `LPCS` 的主瓶颈更像 ranking objective 本身，而不是 pair coverage / sparse routing 语义**

**选择**:
1. `exp136` 到此结案，保留为 supporting 证据
2. 不再围绕 sparse routing 本身继续开变体
3. 立刻在本地切到下一轮：
   - **更 ranking 对齐的 `LPCS` 聚合方式**
   - 例如从全 pair 平均转为 harder / top-k hard 组合

**理由**:
1. `exp135` 已证明 full-pair `LPCS` 能工作，但它没有把 `R1` 也一起拉起来
2. `exp136` 又证明“改 routing 语义”本身不是万能答案；真稀疏最终也不自动变强
3. 这两条证据合起来最自然地指向：
   - 监督该怎么聚合
   - 比监督该选哪些 pair 更像当前瓶颈

## [2026-03-21 13:48] 决策：`exp137 Hard-Rank LPCS` 到 `ep80` 终止，判为过强 hard selection 的负边界

**现状**:
- 本地 `exp137` 已跑到:
  - `ep70 = 58.3 / 69.1`
  - `ep80 = 60.1 / 70.4`
- 对照:
  - `exp135 ep80 = 60.8 / 71.9`
  - `exp125 ep80 = 59.4 / 72.0`
- 同时机制统计持续稳定:
  - `lpcs_rsr = 0.254`
  - `lpcs_psr / lpcs_pf = 1.000 / 1.000`

**判断**:
1. `exp137` 不是失效实现；`hard-rank` 确实被真正测到了
2. 但它到 `ep80` 为止稳定伤害 `R1`，没有出现预期的 late recovery
3. 这说明当前负边界已经很清楚：
   - **“只保留 hardest 25% ranked pairs” 太激进**
   - 它不是当前 `LPCS` 的正确升级方向

**选择**:
1. 立即终止本地 `exp137`
2. 不再继续 `hard_top ratio` 一类的更激进 hard-selection 变体
3. 下一步若继续沿 `LPCS` 主线推进，应转向：
   - 更平滑的 `top-sensitive` 或 `rank-decayed` correction

**理由**:
1. `exp136` 说明 routing 不是主突破口
2. `exp137` 说明极端 hard ranking 也不是主突破口
3. 因而当前最合理的升级方向是：
   - 不丢弃大部分 pairs
   - 但对 top-ranked mistakes 施加更平滑、更定向的强调

## [2026-03-21 13:56] 决策：并行准备 `exp138/139`，分别验证“平滑 rank 强调”与“query 上下文 correction”

**上下文**:
- `exp136` 已给出结论：
  - 真稀疏 routing 成立
  - 但最终只与 full-pair `LPCS` 近似等价
- `exp137` 已给出结论：
  - hard-top 25% ranking 机制接线正确
  - 但到 `ep80` 稳定伤害 `R1`

**判断**:
1. 当前 `LPCS` 主线还值得继续，但不能再沿着“更稀疏”或“更硬选择”推进
2. 最合理的下一步应同时覆盖两个不同创新点：
   - `exp138`: 不删 pairs，只做更平滑的 top-sensitive rank-decay
   - `exp139`: 不改聚合，而是让 pair scorer 感知 query-level context
3. 这两条线分别回答两个不同问题：
   - `exp138`：问题是不是出在 hard selection 太离散
   - `exp139`：问题是不是出在 scorer 缺少 query 级语境

**选择**:
1. 本地候选线定为 `exp138 Rank-Decayed LPCS`
2. 远程候选线定为 `exp139 Query-Context LPCS`
3. 两条线都先完成：
   - 代码接线
   - 设计文档
   - 全面 Claude 审查
4. 在用户告知审查完成前，不启动训练

**理由**:
1. 用户明确要求两个服务器跑不同创新点，不能只做同一条线的小变体
2. `exp138/139` 相对当前主线都满足单变量原则
3. 两条线共享 `LPCS` 框架，但机制问题不同，适合并行筛选

**执行约束**:
1. 新实验必须继续以 `exp135` 为唯一直接本地对照
2. 只改一个核心变量：
   - `LPCS` 的 ranking 聚合方式
3. 启动前仍必须先通过 Claude 审查

## [2026-03-21 14:24] 决策：放行 `exp138`，驳回当前版 `exp139` 并重构为无标签 context

**上下文**:
- `exp138` 的 Claude 全面审查已完成，结论为“允许启动”
- `exp139` 的 Claude 全面审查已完成，结论为“不允许启动”
- `exp139` 当前暴露的不是小实现问题，而是两个 blocking:
  1. test-time descriptor 维度与 `PairResidualScorer` 输入维度不匹配
  2. query context 依赖 `label` 构造，训练与测试不一致

**判断**:
1. `exp138` 可以直接作为本地主线启动
2. 当前版 `exp139` 不能进入远程训练，否则即使勉强补零也无法解释结果
3. `exp139` 若要保留“query 上下文 correction”这条创新点，必须改成：
   - train/test 都可构造
   - 完全不依赖 label
   - evaluator 与训练共用同一 descriptor 语义

**选择**:
1. 本地立刻启动 `exp138 Rank-Decayed LPCS`
2. 远程暂不启动当前版 `exp139`
3. 将 `exp139` 重构为无标签 query-context 版本后，重新做一轮完整 Claude 审查

**理由**:
1. `exp138` 仍在 `LPCS` 主线内，并且当前最接近“平滑 top-sensitive”这一合理升级
2. `exp139` 的当前失败不是点子无效，而是设计没有闭环到测试路径
3. 若把 `exp139` 改成无标签 context，它仍然代表与 `exp138` 不同的第二创新点，适合远程并行

## [2026-03-22 00:14] 决策：终止 `exp138`，将 `exp139` 升为当前唯一主候选

**上下文**:
- `exp138 Rank-Decayed LPCS` 已跑到停表窗口，最新有效验证为：
  - `ep80 = 60.7 / 71.7`
- 对照：
  - `exp135 ep80 = 60.8 / 71.9`
- 同时这条线的机制统计已经非常稳定：
  - `lpcs_rwm = 0.177`
  - `lpcs_dm / lpcs_ds ≈ 0.43 / 0.21`
  - `lpcs_fg > lpcs_bg`
- 远程 `exp139 Query-Context LPCS` 则已给出：
  - `ep20 = 47.6 / 60.0`
  - `ep40 = 57.0 / 68.8`
- 对照：
  - `exp135 ep40 = 56.7 / 68.3`
  - `exp138 ep40 = 56.8 / 68.6`

**判断**:
1. `exp138` 可以被明确收口为：
   - 机制有效
   - 相比 `hard-rank` 更稳
   - 但不足以把 `LPCS` 主线推成更强版本
2. `exp139` 则首次同时满足：
   - train/test 对称、无标签 context 已真实接入
   - `lpcs_ctxm` 显著大于 `0`
   - 指标在中期已稳定超过 `exp135/138`
3. 因而当前最值得押注的不是“更平滑的 rank 强调”，而是：
   - **query-level context-aware pair correction**

**选择**:
1. 立刻终止本地 `exp138`
2. 保留远程 `exp139` 持续跑到下一个关键验证点
3. 本地后续新实验若继续沿 `LPCS` 主线推进，应优先围绕：
   - `query_ctx` 的更强版本
   - 而不是继续做 `rank_decay` / `hard-rank` / `sparse-routing` 小变体

**理由**:
1. `exp138` 已经提供了足够的负边界：平滑 top-sensitive 只能算 supporting 机制
2. `exp139` 是当前唯一同时拥有机制证据与中期正信号的升级线
3. 这条线更贴近论文级叙事：
   - pose 定义 common support
   - query context 决定 pair correction 应如何解释该 support

## [2026-03-22 00:23] 决策：本地转向 `exp140`，验证“correction confidence calibration”而不是继续 rank 变体

**上下文**:
- `exp138` 已停表，结论为 supporting 线
- `exp139` 正在远程继续跑，并已成为当前唯一主候选
- 本地主卡已释放，不能空等远程结果

**判断**:
1. 本地下一条线不应再继续：
   - `rank_decay`
   - `hard-rank`
   - `sparse routing`
2. 当前更值得测试的不同创新点是：
   - **pair correction 是否需要显式的 confidence calibration**
3. 这条线和 `exp139 query-context` 不是一个问题：
   - `exp139` 问的是 scorer 是否缺少 query 语境
   - `exp140` 问的是 scorer 会不会过修正，因为它不知道该不该信自己

**选择**:
1. 新本地候选定为 `exp140 Confidence-Calibrated LPCS`
2. 先完成：
   - 代码接线
   - 设计文档
   - 本地自检
   - 全面 Claude 审查
3. 在用户告知审查完成前，不启动训练

**理由**:
1. 这是当前最合理、且与 `exp139` 机制不同的并行探索点
2. 它直接对应 `exp135` 一直存在的现象：
   - `mAP` 能涨
   - `R1` 不够稳
3. 如果成立，它能把 story 从“pair correction 会修”推进到：
   - **pair correction 知道什么时候该修、什么时候该收手**

## [2026-03-22 00:39] 决策：`exp139` 在 `ep50` 前后已强化为当前唯一主候选，`exp140` 作为本地并行线正式接上

**上下文**:
- `exp139 Query-Context LPCS` 最新已到：
  - `ep40 = 57.0 / 68.8`
  - `ep50 = 58.7 / 70.4`
- 对照：
  - `exp135 ep50 = 57.8 / 69.5`
  - `exp138 ep50 = 57.9 / 69.5`
- 同时机制统计继续增强：
  - `lpcs_ctxm ≈ 0.465 ~ 0.473`
  - `lpcs_fg > lpcs_bg`
  - `lpcs_dm / lpcs_ds ≈ 0.41 / 0.22`
- `exp140 Confidence-Calibrated LPCS` 的 Claude 审查已通过，且本地已正式启动

**判断**:
1. `exp139` 已经从“最值得继续盯”升级成：
   - 当前唯一真正接近论文主创新点的 active run
2. 到 `ep50` 为止，它不再只是早期波动，而是：
   - 中期持续领先 `exp135/138`
   - 机制增强与指标增强同步
3. `exp140` 的角色则是：
   - 不与 `exp139` 抢同一个问题定义
   - 而是并行测试“confidence calibration”是否才是 `R1` 瓶颈

**选择**:
1. 继续优先监控远程 `exp139`
2. 本地转入 `exp140` warmup 观察
3. 当前不再新开第三条线，先看这两条并行主候选的中期走势

**理由**:
1. `exp139` 已经给出当前最硬的正信号
2. `exp140` 是最合理、且与 `exp139` 机制互补的本地并行实验
3. 现在更重要的是把主候选跑清楚，而不是继续横向发散

## [2026-03-22 01:22] 决策：`exp139` 继续作为主候选推进，`exp140` 首轮 run 作废并准备按原假设 clean rerun

**上下文**:
- `exp139 Query-Context LPCS` 最新已到：
  - `ep60 = 57.9 / 69.0`
  - `ep70 = 59.5 / 71.0`
- `exp140 Confidence-Calibrated LPCS` 在 `ep20 = 46.8 / 59.0` 后进入机制阶段时崩溃

**判断**:
1. `exp139` 到 `ep70` 为止仍是当前唯一主候选
2. `exp140` 这轮不能被解读成机制失败，因为：
   - 崩溃发生在 `epoch 21+`
   - 根因是 `sigmoid + BCELoss` 在 AMP 下实现不安全
3. 这次 `exp140` 只能算：
   - 有效 warmup
   - 无效机制阶段

**选择**:
1. 继续保持 `exp139` 为当前主候选并盯后续 `ep80/90`
2. 不把 `exp140` 当前 run 计入结论
3. 修复实现为 logits 版 confidence head 后，对 `exp140` 做 clean rerun
4. rerun 前重新做一轮全面 Claude 审查

**理由**:
1. 这是实现问题，不是研究假设被推翻
2. `exp140` 与 `exp139` 仍然是当前最合理的双线并行：
   - `query context`
   - `confidence calibration`
3. 现在最重要的是避免把无效 run 错写成负结果

## [2026-03-22 02:00] 决策：继续双线推进，但当前优先级仍是 `exp139`

**上下文**:
- `exp139 Query-Context LPCS` 最新已到：
  - `ep80 = 60.8 / 71.6`
- `exp140 Confidence-Calibrated LPCS` clean rerun 最新已到：
  - `ep20 = 46.8 / 59.5`
  - `epoch 21+` 后 `lpcs_cf / lpcs_ctm / lpcs_cl` 已首次真实出现

**判断**:
1. `exp139` 到 `ep80` 为止，已经基本追平当前最强 supporting 线 `exp135`
2. `exp140` clean rerun 已从“启动健康”进入“机制真实生效”阶段
3. 但当前 `exp140` 的 `conf_mean` 明显高于 `conf_target_mean`，说明这版 confidence gate 仍偏激进，暂时不能提前下正结论

**选择**:
1. 远程继续主盯 `exp139`，至少看到 `ep90/100`
2. 本地继续盯 `exp140`，当前关键节点是 `ep30`
3. 现在不再新开第三条线

**理由**:
1. `exp139` 仍是当前最有机会收敛成论文主机制的方向
2. `exp140` 刚刚第一次拿到有效机制证据，现在停掉太早
3. 当前最重要的是看：
   - `query context` 是否能在后期稳住优势
   - `confidence calibration` 是否会把早期高 gate 形态兑现成真实收益

## [2026-03-22 02:38] 决策：`exp140` 当前版本止损，远程 `exp139` 继续跑完

**上下文**:
- `exp139 Query-Context LPCS` 最新已到：
  - `ep80 = 60.8 / 71.6`
  - `ep90 = 60.6 / 72.1`
  - `ep100 = 60.8 / 71.9`
- `exp140 Confidence-Calibrated LPCS` clean rerun 最新已到：
  - `ep30 = 52.8 / 63.1`
  - `ep40 = 56.4 / 67.6`
  - `ep50 = 57.4 / 68.5`

**判断**:
1. `exp139` 到 `ep100` 为止仍是当前唯一可持续推进的主候选
2. `exp140` 当前版本虽然真实接上了，但 `confidence gate` 已明显退化：
   - `lpcs_cf -> 0.99`
   - `lpcs_ctm ≈ 0.10`
   - `lpcs_dm ≈ lpcs_rdm`
3. 因此 `exp140` 已不再是在做“confidence calibration”，而越来越像：
   - `residual scorer + 几乎恒开 gate + 额外 auxiliary loss`

**选择**:
1. 手动终止本地 `exp140`
2. 远程 `exp139` 继续跑到收敛
3. 本地下一步不再沿当前 `confidence target` 形式修小补小

**理由**:
1. `exp140` 的问题已经不是“还没收敛”，而是机制本体退化
2. `exp139` 目前虽未显著超过 `exp135`，但它仍是当前最接近论文主机制的 active line
3. 现在更合理的是把本地算力从已退化的 `exp140` 释放出来，转给下一条真正不同的创新点

## [2026-03-22 02:49] 决策：本地转向 `exp141 Competition-Context LPCS`

**上下文**:
- `exp139` 证明 query-level context 有价值，但到 `ep100` 为止仍只是稳住主候选
- `exp140` 当前版本已止损，原因是 confidence gate 退化为接近常数 1
- 本地主卡已释放，必须进入真正不同的创新点

**判断**:
1. 本地下一条线不应再沿：
   - confidence target
   - rank weighting
   - query mean/std 摘要
2. 当前更合理的不同创新点是：
   - **candidate competition-aware pair correction**
3. 它与 `exp139` 的区别是：
   - `exp139` 问“这个 query 整体是什么语境”
   - `exp141` 问“这个 pair 在当前 query 的候选竞争里处于什么位置”

**选择**:
1. 新本地候选定为 `exp141 Competition-Context LPCS`
2. 相对 `exp135` 保持单变量：
   - 只把 `POSE_LPCS_CONTEXT_MODE` 改成 `comp_ctx`
3. 先完成：
   - 代码接线
   - 设计文档
   - 本地自检
   - 全面 Claude 审查
4. 在用户告知审查完成前，不启动训练

**理由**:
1. 这是当前与 `query_ctx / confidence-gate` 都不同的真正新点
2. 它仍然紧扣 retrieval 本质，而不是退回 backbone 模块堆叠
3. 如果成立，它能把 story 从“query 语境”进一步推进到：
   - **candidate competition 语境**

## [2026-03-21 19:02] 决策：不启动 `exp141`，本地转向更大的 feature completion 主线 `exp142`

**上下文**:
- `exp141` 的二次 Claude 审查已经通过
- 但 `exp141` 本质上仍属于 `LPCS` 家族内部的 context 变体
- 用户已明确要求：
  - 不要继续围绕同一个小点浪费时间
  - 本地应转向真正不同、真正可能带来收益的大改动
- `exp109` 的 oracle 结论始终没有被推翻：
  - 真正 headroom 来自 `single-image support incomplete`

**判断**:
1. `LPCS` 家族已经给出足够多证据：
   - pair correction 不是完全无效
   - 但它当前更像 supporting 机制，而不是确定的论文主方法
2. 如果本地继续开 `exp141`，即使成功，也大概率仍是：
   - `LPCS` 家族内部的小幅体感优化
3. 当前更合理的本地大转向应回到 `exp109` 根问题本身：
   - 不在距离层修正
   - 而在特征层直接补全 keypoint-level support

**选择**:
1. 暂不启动 `exp141`
2. 本地主线切到 `exp142 SKC (Support-Conditioned Keypoint Completion)`
3. 按用户要求，先写正式设计，再改代码
4. 设计阶段只做文档，不启动训练

**理由**:
1. 这是比 `query_ctx / comp_ctx / confidence` 都更大的方法级改动
2. 它直接回应 `exp109` 的核心发现，而不是继续在 scoring 层修修补补
3. 如果成立，它比 `LPCS` 家族更有机会支撑 B 类论文主创新

---

### [2026-03-22 05:58] 决策 #N+1

**上下文**: exp142 SKC 训练完成。最终结果 mAP 60.3% / R1 71.8%（equal_concat），相对 exp030a -0.8% mAP / -1.9% R1。feature-level support-supervised completion 方向确认失败。

**核心发现**:
1. SKC completion 模块虽然活跃（gate≈0.26, delta_norm≈1.5），但 skc_pre≈skc_post 说明修改方向不是向 support prototype 靠近
2. gate 无限制增长（0.12→0.26）导致后期过度修改特征
3. 这与 SGMKC, SCRC, SCKD 的结论一致：feature-level completion 在 15K 数据集上不可行

**选项**:
  A. SKC gate clamp ablation（限制 gate 增长上界）
  B. SASA（Skeleton-Aware Self-Attention，零参数注意力偏置）
  C. PGCO（Pose-Guided Curriculum Occlusion，课程式遮挡增强）

**选择**: B — SASA
**理由**:
1. feature-level completion 方向已被彻底证伪（5+ 次尝试），不值得继续做 ablation
2. SASA 代表全新方向：通过骨架拓扑修改注意力路由（而非特征值）
3. 零参数（纯归纳偏置），符合"只有强归纳偏置才能在 15K 数据上生效"的核心教训
4. KP-RPE 虽然中性，但 SASA 使用拓扑距离（图论）而非欧氏距离（几何），是本质不同的信息
5. 已实现、已审查通过，可立即启动

**执行结果**: exp143 SASA 已在本地启动，训练正常进行。

---

### [2026-03-22 09:35] 决策：停止沿 `exp141-147` 的小修补漂移，重新回到 `exp109` 的大问题定义

**上下文**: 夜间 `exp141-147` 收尾后，已经有足够证据说明：
- `competition-context LPCS` 失败
- `SKC` 再次确认 feature-level completion 失败
- `SASA` 系列中性
- `0.25x global loss / PAA recipe` 这类老路线不该继续占机器

**核心发现**:
1. `single-image support incomplete` 这个问题定义没有被推翻
2. 被反复否掉的是：
   - cross-image prototype completion 的兑现方式
   - retrieval scorer 的局部修补
   - attention bias 这类轻量归纳偏置
3. 如果继续沿这些线扫点，得到的大概率只是 recipe，不是主创新

**选项**:
  A. 继续在已有 retrieval scorer / completion / bias 路线上小修小补  
  B. 回到 `exp109` 的问题定义，重新设计两个真正不同的大方向  

**选择**: B

**理由**:
1. 现阶段更缺的是新的训练对象 / 新的结构对象，而不是新的小模块
2. 两个仍然开放、且足够大的 gap 是：
   - 单图能否被改写成“伪多 support 学习”对象
   - 单图内部的双侧同源冗余是否被浪费
3. 这两条线都比继续改 scorer/gate 更有可能支撑 B 类论文主 story

**执行结果**:
- 新增方向复盘文档 `paper_21_20260322_direction_reset.md`
- 新增两条新主线设计：
  - `exp148 PCVT`
  - `exp149 SCFA`
- 当前进入“先设计、后实现、再广范围 Claude 审查”的新阶段

---

### [2026-03-22 15:48] 决策：停止 `SCFA`，将本轮方向收缩为“继续追 PCVT，远程空出来给下一条真正不同的新机制”

**上下文**:
- `exp148 PCVT` 已给出连续 `ep10/20/30` 的稳定 `mAP` 正向
- `exp149 SCFA` 已在预设的 `ep30` 快速止损窗口内明确判负

**核心发现**:
1. 两条“大方向”已经出现分化：
   - `PCVT` 至少跑出了连续可见的验证正信号
   - `SCFA` 则在当前数据集上缺少足够强的 bilateral gap case
2. `SCFA` 的失败不是代码没接上，而是问题前提太弱：
   - `scfa_an` 并未塌成 `0`
   - 但 `scfa_pg` 长期只有 `0.086~0.093`
3. `PCVT` 当前最主要风险不是方法失效，而是：
   - 最终是否会收敛成 `mAP` 增益但 `R1` 不明显的 trade-off

**选项**:
  A. 继续让 `SCFA` 占远程卡，赌后期反转  
  B. 立即终止 `SCFA`，保留其负结论，并把远程算力留给下一条真正不同的新方向  

**选择**: B

**理由**:
1. `SCFA` 已达到预设止损条件，继续跑的边际价值很低
2. 当前更重要的是：
   - 继续把 `PCVT` 跑清楚
   - 并尽快设计一个与 `PCVT` 真正不同的新大方向占用远程卡
3. 这样才能避免再次陷入“小修补并行跑满两台机器”的低价值循环

**执行结果**:
- 已终止远程 `exp149`
- 已确认远程 GPU 完整释放
- 本地主线继续保留 `exp148 PCVT`
- 远程进入“等待下一条真正不同的新机制”状态

### [2026-03-26 15:40] 决策 #91

**上下文**: exp190-195 系列实验完成，揭示了 OA-SD 和 3-view parallel aug 的组合关系，以及 OA-SD global-only 解决 SupCon 梯度冲突的新机制。

**关键实验结论**:
- exp190 (3-view+CE): 64.2/75.6 — 3-view 是最强单一技术
- exp191 (OA-SD+CE): 63.2/75.4 — OA-SD 独立有效
- exp192 (decay=0.99): 62.6/74.9 — decay 不敏感
- exp193 (3-view+OA-SD+CE): 64.4/76.5 — additive! R1 追平 SupCon
- exp194 (weight=2.0): 63.4/74.8 — weight 不敏感
- exp195 (SupCon+OA-SD global-only): ep70=60.2/73.4 — 梯度冲突解决!

**选项**:
  A. exp196: 3-view + SupCon + OA-SD global-only（终极组合，验证所有创新 additive）
  B. 探索全新方向（如 SCL, Part Token Codebook 等）

**选择**: A

**理由**:
1. OA-SD global-only 是一个新机制（满足创新门槛 #2 和 #3），不是简单组合
2. 如果 exp196 > exp187 (64.9/76.6)，则创论文主表新高
3. exp195 已验证 SupCon+OA-SD global-only 兼容，exp193 已验证 3-view+OA-SD additive
4. 这个实验的论文价值极高：证明"global distillation + per-token contrastive" 职责分离可行

**执行结果**: (后续补填)

### [2026-03-26 19:40] 决策 #92

**上下文**: exp196 (3-view + SupCon + OA-SD global-only) 在 ep70 持续落后 exp187 (3-view + SupCon) -1.8/-0.9。OA-SD global-only 的 distillation 信号 (oa_sd=0.01) 过弱。

**发现**:
- OA-SD + CE 有效 (+2.9/+2.6)
- OA-SD + SupCon (all-token) 有梯度冲突 (exp188 负向)
- OA-SD + SupCon (global-only) 无梯度冲突但信号太弱 (exp195/196)
- 结论: **OA-SD 和 SupCon 本质上互斥**——选一个用

**选项**:
  A. 最终配置用 SupCon (exp187: 64.9/76.6) — 测试时最强
  B. 最终配置用 OA-SD+CE (exp193: 64.4/76.5) — R1 几乎一样
  C. 论文中两条路线都展示，作为 ablation

**选择**: C

**理由**:
1. 两条路线都是有效创新，mAP/R1 各有优劣
2. SupCon 路线: mAP 更高 (+0.5)
3. OA-SD 路线: R1 几乎一样，且 OA-SD 概念更新颖 (self-distillation 范式)
4. 论文可以展示："两条互补但互斥的训练范式"
5. 用户做 3-seed 验证后决定主结果用哪个

### [2026-03-27 03:30] 决策 #93

**上下文**: 5 个研究 agent 并行调研后，发现 Relational Knowledge Distillation (RKD, CVPR 2019) 可能解决 OA-SD vs SupCon 互斥问题。OA-SD 在 token 级别 match 个体特征 → 与 SupCon 冲突。RKD match pairwise similarity structure → 不碰个体特征 → 理论上与 SupCon 兼容。

**选项**:
  A. OA-RD (Relational Distillation): Teacher 和 student 的 batch-level pairwise similarity 一致
  B. BMKCA (Batch-Mate Keypoint Cross-Attention): 跨图 cross-attention 补全
  C. Multi-Granularity Contrastive: 多粒度层次化 SupCon

**选择**: A (OA-RD)

**理由**:
1. 直接解决已知问题 (OA-SD vs SupCon 冲突)
2. 满足创新门槛: 问题层面(关系级 vs 特征级) + 机制层面(RKD in occluded ReID) + 证据链(OA-SD→OA-RD)
3. 实现简洁 (~100 行): 计算 pairwise similarity matrix + KL divergence
4. 如果成功: SupCon + OA-RD + 3-view 可能是真正的终极配置

### [2026-03-30 09:15] 决策 #94

**上下文**: exp199 (OA-RD+SupCon) ep60=-1.5/-3.4 vs exp187，exp200 (OA-RD+CE) ep60=-1.1/-3.4 vs exp191。OA-RD (relational distillation) 也是负结果。

**核心发现**: 任何形式的 EMA self-distillation (OA-SD/OA-RD) 都与 SupCon 不兼容。
- OA-SD (feature-level): exp188/196 失败
- OA-RD (relation-level): exp199 失败
- 原因可能是 EMA teacher 本身引入的学习干扰，不是 distillation 的形式问题

**选项**:
  A. 继续尝试其他 distillation 变体（如 CRD contrastive distillation）
  B. 放弃 distillation+SupCon 组合，接受两条互斥路线
  C. 转向完全不同的方向（从 5 个研究 agent 的其他建议中选）

**选择**: C

**理由**:
1. distillation+SupCon 已尝试 4 种方式 (all-token, global-only, relational, RD) 全部失败
2. 继续尝试 = 浪费 GPU 时间
3. 应探索研究 agent 提出的其他有前途的方向
4. 最有前途的候选：Batch-Mate Keypoint Cross-Attention (BMKCA) 或 Multi-Granularity Contrastive

**执行结果**: (后续补填)

### [2026-03-30 15:10] 决策 #95

**上下文**: exp197-201 连续 5 个负结果。所有在 exp187 (64.9/76.6) 基础上的改进尝试都失败。

**失败实验链**:
- exp197 (STM + SupCon): -0.8/-0.6 — token mixup 只加速不改善
- exp198 (STM + OA-SD): ±0 — 同上
- exp199 (OA-RD + SupCon): -1.5/-2.1 — relational distillation 也与 SupCon 冲突
- exp200 (OA-RD + CE): -0.3/-1.5 — OA-RD 不如 OA-SD
- exp201 (global SupCon): ~-1.5/-3.6 — global SupCon 压缩特征空间

**模式**: 所有额外 loss/constraint 在早期加速训练，但后期限制模型的 fine discrimination 能力。
**结论**: exp187 的配置已接近 Swin-Tiny 的 performance ceiling (~65% mAP)。

**选项**:
  A. 继续在 Swin-Tiny 上尝试新方向
  B. 接受 64.9/76.6 为 Swin-Tiny 最佳，转向论文撰写和用户做 Swin-Small/Base scaling
  C. 尝试一个完全不同维度的改进（如 test-time 优化、数据增强改进）

**选择**: A (再尝试 1-2 个方向，如果仍负则转 B)

**理由**:
1. CLAUDE.md 禁止"论文收尾模式"
2. 但也不能无限尝试——已有 5 个连续负结果
3. 最后再试一个真正不同的方向：不加 loss、不加 distillation、不加 module
4. 候选：改变 SupCon 的 temperature (T=0.03 在 1-view 曾有好结果)，或 PLBOA prob 调优

### [2026-04-01 06:00] 决策 #96

**上下文**: 3个研究 agent 发现：
1. GLOBAL_LOSS_SCALE=1.0 在所有 Small/Base 实验中 → 0.5x 从未测试 (+1.53% on Tiny)
2. KPR test-time prompting 不是 reranking，可以用 (+1.8%)
3. Swin-Base 是到 76% 的关键 lever

**路径到 76/85 (目标)**:

| Step | 方法 | 预期 mAP |
|------|------|---------|
| 当前 | Small GCN+PAA+CE+OA-SD | 70.5% |
| +1 | **0.5x global loss** (未测试!) | 72% |
| +2 | **Swin-Base** (exp207 进行中) | 74-75% |
| +3 | **Base + 0.5x** | 75-76% |
| +4 | **KPR-style test-time prompting** | 76%+ |

**选择**: 按优先级执行
1. exp207 Base 跑完后确认 Base 增益
2. 立即在下一个实验中加 0.5x global loss
3. 如果 Base + 0.5x ≈ 75%，实现 KPR prompting 冲 76%

**执行结果**: 
- exp208 (0.5x global loss) = NO-OP（GCN list-loss 已隐含 0.5x），取消
- exp209 (STD-PR+CE+OA-SD) ep30=56.0/69.3，落后 5%，终止
- MaxSim hybrid 发现: +1.8% mAP 无需重训！
- OA-SD teacher Critical bug 修复并部署

### [2026-04-01 10:30] 决策 #X — MaxSim + PKC + Fixed OA-SD 路线

**上下文**: MaxSim hybrid 在 exp206 checkpoint 上无需重训给 +1.8% mAP (70.3→72.1)。OA-SD teacher bug 已修复。PKC (Per-Keypoint Contrastive) 开始测试。

**新路径**:

| Step | 方法 | 预期 mAP |
|------|------|---------|
| 已确认 | Small GCN+PAA+CE+OA-SD + maxsim_hybrid | **72.1%** |
| exp210 | + PKC (进行中) | 73-74% |
| exp207 | Swin-Base 3-view (进行中) | 74-76% |
| 最终 | Base + PKC + maxsim_hybrid | **76%+** |

**选择**: 双线并行，Base + PKC 为主攻方向

### [2026-04-02 02:10] 决策 — Per-Keypoint Loss 路线全面失败

**上下文**: 尝试了 5 种 per-keypoint loss 方案，全部失败或无效：

| 实验 | 方法 | 结果 |
|------|------|------|
| exp210 | PKC weight=0.5 (detached GCN) | 灾难 (3.6%) |
| exp210b | PKC weight=0.05 (detached GCN) | 无效 (= baseline) |
| exp211 | MST weight=0.5 (detached GCN) | 无效 (= baseline, 所有 loss 完全一致) |
| exp213 | PKC+MST 组合 (detached) | 灾难 (40.6%) |
| exp215 | BA-PKC weight=0.1 (NON-detached backbone) | 灾难 (0.5%) |

**根本原因**: 
- 有 detach: 梯度只更新 GCN 200K params, 对 50M backbone 无影响 → 无效
- 无 detach: SupCon 梯度直接打到 backbone, 与 CE 冲突 → 灾难

**重要结论**: 
1. **per-keypoint loss 路线已证伪** — 架构约束使其不可能有效
2. MaxSim hybrid (+1.7-1.8%) 是纯 test-time 方法，不受此限制
3. **当前最佳: 72.4/83.1 (exp210b + maxsim)**
4. 需要完全不同的训练端创新来突破

**新方向候选**:
1. 更长训练 (200/240 epochs)
2. 更强数据增强 (多种 occlusion 组合)  
3. 改变 GCN 架构让 Part 分支更强
4. 回到 STD-PR+SupCon 路线（已知 67.9+maxsim ≈ 69.7，不如 GCN+OA-SD）
5. **多分辨率/多尺度特征融合** (未试过)

### [2026-04-02 05:48] 决策 — OERL / per-keypoint loss 全面失败总结

**测试过的所有 per-keypoint 训练方案:**

| 实验 | 方法 | detach? | 结果 |
|------|------|---------|------|
| exp210 | PKC w=0.5 on detached GCN | Yes | 灾难 3.6% |
| exp210b | PKC w=0.05 on detached GCN | Yes | 无效 (=baseline) |
| exp211 | MST w=0.5 on detached GCN | Yes | 无效 (所有 loss 完全一致) |
| exp213 | PKC+MST combo | Yes | 灾难 40.6% |
| exp215 | BA-PKC w=0.1 non-detached | No | 灾难 0.5% |
| exp212 | LR=0.0008 | — | 灾难 0.8% |
| exp217 | OERL w=1.0 non-detached cosine | No | `62.2/75.2`，相对 `exp191 63.2/75.4` 为 `-1.0/-0.2` |

**核心教训:**
1. detached 路径: 梯度不到 backbone → 无效
2. non-detached SupCon: 与 CE 冲突 → 灾难
3. non-detached cosine alignment: 与 OA-SD 竞争 → 负面
4. **per-keypoint training loss 路线已全面证伪**

**剩余可行路径:**
- MaxSim hybrid test-time matching (+1.7-1.8%，已确认)
- 完全新的架构创新 (not loss tuning)
- 回到 IDEA 1 (PACI: per-identity part prototype bank) 或 IDEA 3 (CIPCM: cross-instance correspondence)

### [2026-04-02 09:45] 决策 — PACI 证伪 + MaxSim Ceiling 发现

**PACI (exp218/219) 结果:**
- PACI + OA-SD (exp218): `61.9 / 74.2` (vs `exp191 63.2 / 75.4` = **-1.3 / -1.2**)
- PACI-only (exp219): 已从远程补回 `train_log`，当前可直接复核到 `ep10=37.7/50.5`、`ep20=47.5/60.4`、`ep30=51.9/64.9`；但尚无 final，因此它仍只能作为 early stop-loss 证据，不能写成正式最终结果
- **PACI 证伪。Consistency loss on detached GCN = 无效。**

**当时已完成三条 Tiny 线的 MaxSim 对照：**

| 方法 | equal_concat | maxsim_hybrid |
|------|------|------|
| OA-SD-only | **63.2** | 64.2 |
| OERL+OA-SD | 62.2 | 64.3 |
| PACI+OA-SD | 61.9 | 64.1 |

这一步更准确的结论不是 “OA-SD 已达 64.4 ceiling”，而是：
1. 在 `OA-SD / OERL / PACI` 这三条已完成 Tiny 线内部，`maxsim_hybrid` 都落在 `64.1~64.3`
2. `MaxSim` 对 `OA-SD` 本身仍是正向的（`63.2 -> 64.2`），只是 `OERL/PACI` 并没有把这个 test-time 上限继续抬高
3. 因而当时更合理的判断应是：
   - `extra loss` 没有改善 Tiny 上的 `MaxSim` 上限
   - 但 “Tiny 的硬天花板就是 64.2” 这个表述还不能下

**根本问题诊断:**
1. detached GCN 是架构瓶颈 → 任何 per-keypoint loss 只更新 200K GCN params → 无效
2. non-detached losses 与 CE/OA-SD 冲突 → 灾难
3. MaxSim 是 test-time matching 修正；它是否继续涨，强依赖 per-keypoint consistency，而不只是 global 强弱
4. 后续 `exp220` 已把 Tiny `maxsim_hybrid` 推到 `64.6`，因此这里原先的 `~64.4` / `~64.2` ceiling 表述应视为阶段性误判

**接下来的方向必须是:**
1. **改变 Part 架构** — 不是 loss tuning，而是改 GCN 本身或替换为更强的 part 机制
2. **或者找到不依赖 detach/non-detach 的全新训练范式**
3. **Small/Base scaling** 通过 MaxSim 已达 72.4%，用户在 4090 上跑

### [2026-04-02 17:12] 决策 — PADPQ + GSPB + 全面 Tiny 探索总结

**新增实验结果:**

| Method | equal_concat mAP/R1 | maxsim mAP/R1 |
|--------|------|------|
| OA-SD-only | 63.2/75.4 | 64.2/77.1 |
| GSPB+OA-SD (scale=0.05) | 62.9/74.3 | **64.6/76.0** |
| PADPQ K=4+OA-SD | **63.7/74.5** | 63.9/74.8 |
| PADPQ K=8+OA-SD | 进行中 | 进行中 |

**关键发现:**
1. GSPB: 早期加速 +5.8% at ep10，按当前测试记录 `maxsim_hybrid` 相对 OA-SD 为 `+0.4`，是目前 Tiny 线上最高的 `maxsim` mAP
2. PADPQ: mAP +0.5 但 R1 -0.9, MaxSim 仅 +0.2 (deformable 破坏了 cross-image consistency)
3. GSPB 只在 Tiny 有效，Small 上灾难 (ep10=2.3%)

**Tiny 上的硬天花板: ~64.6% maxsim, ~63.7% equal_concat**

**接下来的方向:**
- 研究 agent 提出的 KAMP (多尺度 keypoint 融合) 还未尝试
- 或者接受 Tiny ceiling, 聚焦整理论文材料
- 用户在 4090 上跑 Small: 72.4% (maxsim) 是 Small 最佳

### [2026-04-03 20:40] 决策 — BT-PKD 系列证伪，Non-Detached Gradient 方向关闭

**上下文**: exp229-232 全面测试了 BT-PKD (Backbone-Through Per-Keypoint Distillation):
- constant weight (Tiny/Small): ~-1.0 mAP at final
- cosine decay (Tiny/Small): ~-1.5 mAP at final
- 所有变体展示相同模式: 早期加速 +3-5%, 后期干扰 -1~-2%

**选择**: 关闭所有 non-detached gradient 方向

**理由**:
1. BT-PKD 是最温和的 non-detached 梯度 (cosine distillation from EMA teacher)
2. 即使最温和的梯度也无法避免后期干扰
3. Cosine decay schedule 也不解决 (干扰在 active 阶段已发生)
4. 这是一个根本性限制，不是梯度类型或 schedule 的问题

**已证伪的 non-detached 变体汇总**:
- BA-PKC (SupCon): catastrophic (0.5%) 
- GSPB scale≥0.01 on Small: catastrophic (2.3-15.1%)
- GSPB scale=0.05 on Tiny: -0.3 at final
- GSPB scale=0.005 on Small: ~0 (with PADPQ: +1.0 mAP, -1.8 R1)
- BT-PKD constant: -1.0/-0.4
- BT-PKD decay: -1.5/-1.1

**接下来**: 需要完全不同的方向。等论文搜索结果后决定。

### [2026-04-04 15:40] 决策: exp242 PPA+GCN Small 灾难性失败

**上下文**: PPA+GCN 在 Tiny 上 +0.5/-0.1, 需要在 Small 上验证泛化性。
**结果**: 60.9/73.4 vs baseline 70.6/82.6 = **-9.7/-9.2**

**关键发现**:
1. PPA 的 non-detached 梯度在 Small backbone 上造成灾难性干扰
2. 对比: PPA on Small (exp240) 也是中性 (70.7/81.1 vs 70.6/82.6 = +0.1/-1.5)
3. Non-detached part gradients 与 backbone 强度负相关: Tiny OK, Small catastrophic
4. 这进一步验证了 detach barrier 的根本性——不仅 detached features 不行, non-detached gradients 也不行 (在大模型上)

**选择**: 放弃 PPA 作为主线方向。转向 LGPA (CLIP-based part assignment)

**理由**: LGPA 使用 CLIP frozen text prototypes 作为语义锚, cross-attention 机制与 PPA 不同, 可能在梯度控制上更好。

### [2026-04-04 15:40] 决策: 启动 exp243 LGPA

**上下文**: 寻找范式级创新, 结合 VLM + pose 做 part assignment。
**选择**: LGPA = CLIP text embeddings + cross-attention + pose masks
**理由**:
1. 首次将 VLM 语义知识 + 几何 pose 信息结合用于 occluded ReID
2. CLIP 提供语义锚定 (PPA 缺乏), pose 提供空间约束
3. Cross-attention 比 softmax assignment 更灵活
4. 如果成功, 创新性足以支撑论文核心贡献

### [2026-04-04 21:10] exp243 LGPA 结果分析 (GPU crash at ep88)

**结果**: ep80 60.9/72.5, delta -1.1/-1.9 vs baseline。

**关键发现**:
1. CLIP 语义锚定有效: ep20-40 +3.5~+4.1 mAP, 超过所有 PPA 变体
2. Cross-attention 梯度干扰 > PPA: 后期 delta (-1.1 at ep80) 比 PPA+GCN (+1.2 at ep80) 差 2.3
3. **Detach barrier 是根本性问题**: 无论用什么机制 (linear assignment, cross-attention, SupCon, L2 distillation), non-detached part branch 都在后期干扰 backbone

**洞察**: CLIP 的价值在语义初始化而非梯度训练。未来方向:
- LGPA with detached features (仅用 CLIP 做更好的 part pooling, 不传梯度)
- CLIP-guided GSPB (用 CLIP 语义控制 gradient scale)
- 完全放弃 non-detached part branch, 另寻创新方向

### [2026-04-05 04:10] exp244 LGPA-Detach — 突破性结果! ⭐⭐⭐

**结果**: 65.3/75.7 (+2.1/+0.3 vs exp191 GCN+OA-SD)

**这是首个在 120 epoch final 仍保持正向 delta 的 Part branch 方法。**

**关键发现**:
1. LGPA-D 全程 mAP delta 均为正 (ep10~ep120), 前所未有
2. detach 完全消除了 non-detach 的后期干扰 (exp243 -1.1 → exp244 +2.1)
3. CLIP 语义做 part assignment 比 GCN skeleton graph 更有效 (+2.1 mAP)
4. LGPA-D 无 OA-SD (63.6) ≈ GCN + OA-SD (63.2): CLIP 价值 ≈ OA-SD

**论文价值**: 
- 核心贡献: "Language-Grounded Part Assignment" — 首次用 VLM 语义做 occluded ReID part 表征
- 消融故事清晰: non-detach (exp243) vs detach (exp244) 证明 detach 必要性
- CLIP 语义 vs GCN skeleton: 语义优于结构
- 与 OA-SD 正交: 可叠加

**下一步**: 在 Small 上验证泛化性

### [2026-04-05 10:15] 用户判定 VCSR 不够新 + 深度调研结果

**用户反馈**: VCSR = 5/10 novelty (不是 7/10)。VPM/PVPM/QPM/BPBreID/KPR/PAFormer/ProFD/MoS 都覆盖了 VCSR 的组件。训练集 95.8% visible → 训练端 visibility gating 无效。

**深度调研结果** (Claude + Codex 双路调研):
推荐方向: **Pose-Conditioned Feature Differencing (PCFD)**
- Novelty 7.5/10, Feasibility 8/10, Combined 7.5
- 核心: "不问两人有多像, 问在共同可见部位上有什么不同"
- 训练差异分类器, 用 per-part feature 差异做 same/diff-ID 判断
- 与 LGPA-D 完美配合: LGPA-D 提供 per-part features, PCFD 做 pair-level 精细比较
- 工作在 retrieval-time, 不依赖训练集 visibility (解决 95.8% 问题)

**选择**: 暂时搁置 VCSR 作为消融实验, 主线转向 PCFD

**理由**: 
1. PCFD 重新定义问题 (相似度排序 → 差异分类)
2. PCFD 是 retrieval-time 创新, 不受训练集 visibility 限制
3. PCFD 与现有 LGPA-D pipeline 正交叠加

### [2026-04-08 16:45] 决策 — exp249 完成后下一步

**上下文**: exp249 (Small LGPA-D+GCN) 完成: 71.9/81.8 equal_concat, 73.3/83.2 MaxSim。
POT test-time: m=0.6 在 mAP 上超 MaxSim +0.3 (两个 Small checkpoint 都确认)。
两台 GPU 都空闲。

**选择**:
A. 本地 Tiny: 启动消融实验 (论文需要 Tiny 数据)
B. 远程 Small: 新创新实验

**本地选择: 不再启动新训练实验。** 原因:
1. Tiny 消融数据 (exp244, exp246b) 已经足够完整
2. 用户在 4090 上做多 seed 验证，不需要 Claude 做
3. CLAUDE.md 说 "不为刷 0.1% 做无意义调参"
4. **本地 GPU 给用户同学使用**

**远程选择: 也暂不启动。** 原因:
1. 所有 "安全" 创新方向已试完或被证伪
2. 需要先与用户确认论文策略再决定下一步
3. 当前结果 (73.3/83.2) 已可投稿

**当前论文素材已具备**:
- LGPA-D: +2.1 mAP (Tiny), +0.3 mAP (Small) vs GCN baseline
- GCN dual branch: +0.3/+0.2 (Small) — R1 中期增益更大
- MaxSim: +1.4/+1.4 test-time
- POT: mAP 73.3 > MaxSim 73.0 — 理论补充
- 完整消融: detach vs non-detach, CLIP vs GCN, 250 experiments

### [2026-04-08 00:50] 决策 — CCF-B 创新方向评估

**上下文**: LGPA-D novelty 4.5/10, 需要更深层创新达到 CCF-B 级别。
已完成 VCSR (exp247, 失败) 和 PCFD (exp248, 失败) 两个创新尝试。

**深度分析** (3 轮 Opus 子代理评估):

| 方向 | Novelty | 可行性 | 结论 |
|------|---------|--------|------|
| POT (Partial OT) | 5/10 | 高 | 5×5 太小, 可能无法超 MaxSim |
| CPRE (Cross-Part Relations) | 7/10 | 中 | 关系编码 pose 而非 identity, HOReID 已有 |
| AQGP (非对称 Q-G) | 8/10 | 中-高 | 退化为 vis-weighted pooling |
| Pose-Conditioned Masking | 6/10 | 低 | 类似 PLBOA feature-level |

**根本发现**: 250 实验反复证明：
1. 只有 backbone 修改有效 (PSG, OA-SD)
2. 训练集 95.8% visible → 训练端创新空间极有限
3. 当前系统 (73.0% MaxSim) 距 SOTA (75.1%) 仅差 2.1% (Swin vs ViT 差异)

**选择**: 
A. 短期: 完成 exp249, 快速测试 POT (test-time, 无训练需求)
B. 论文策略: 以 LGPA-D (CLIP 语义 part assignment) 为核心贡献, 配合完整 pipeline 消融
C. 如需更强创新: 需要跳出 Swin + detach 框架 (换 ViT 或全新问题定义)

**理由**: 
1. POT 低成本快速验证, 作为 secondary contribution
2. LGPA-D 虽然 single novelty 4.5/10, 但与 PSG+OA-SD+MaxSim 组成完整 framework novelty 更高
3. exp249 (LGPA-D+GCN on Small) 有潜力达到 73-74% → 与 SOTA 竞争力足够

### [2026-04-15 18:30] 决策 — PRCV 方向重审，停止把 LGPA/MaxSim 当主故事

**上下文**: 用户要求重新阅读 `CLAUDE.md`、rules、全量实验文档，并进行新一轮线上文献调研；明确要求“不要被 Claude 的旧 story 误导，只看实验和结果”。  
本轮重审后确认：
- `exp109` oracle support bank 仍是仓库内最强问题证据
- `exp257-259` 已基本说明当前 `exp255` recipe 空间耗尽
- `LGPA-D + GCN + OA-SD + MaxSim + flip` 虽然结果强，但主问题定义仍偏弱
- 文献上 visible-part / prompt / recovery / test-time matching 这些线都已有大量前人

**选项**:
  A. 继续沿 `LGPA-D + MaxSim/POT + test-time` 故事收论文
  B. 回到 `exp109`，把主线改成“single-image support incomplete”的训练对象重写

**选择**: B

**理由**:
1. `exp109` 已给出巨大 headroom：`61.88 -> 66.15 -> 70.40 mAP`，而后续任何实验都没有正面回应这个 gap
2. `LGPA-D` 更像 detached semantic part asset，不足以单独撑起新的问题定义
3. `MaxSim / POT / flip` 主要仍是 test-time supporting evidence，不能作为训练端主贡献
4. 文献上最接近的前人（如 VPM/PGFA/PVPM/PAT/BPBreID、FRT/MVI²P、CLIP-ReID/Instruct-ReID/KPR）分别占据了 visible matching、recovery、language/prompt 等位置，但**“把单图不完整支持改写成可训练对象”** 仍有空位

**新的主线定义**:

**PSCD: Pose-defined Support-Complete Distillation**

即：
1. 用 pose 定义互补 support 伪视图，而不是随机多视图分类
2. 用互补视图组装 support-complete teacher token set
3. 只对低支持 token 做 confidence-gated distillation
4. 再用轻量 set-level alignment 与已验证有效的 `MaxSim` 行为对齐

**执行顺序**:
1. 先做“单图互补视图 oracle”诊断；如果这一步 headroom 不明显，立即止损
2. 若 oracle 为正，先在 Tiny 做 30-40 epoch 趋势验证
3. 若 Tiny 为正，再上 `exp255` Small scaffold
4. 必做消融：
   - pose-defined vs random complementary masking
   - assembly teacher vs 3-view classification
   - low-confidence-only distill vs all-token distill
   - token-only vs token+set alignment

**补充说明**:
- `LGPA-D / GCN / OA-SD / PLBOA / MaxSim` 现在统一降级为资产与 supporting evidence
- 详细文献压缩与路线说明见 `experiments/paper_notes/2026-04-15_prcv_reset.md`

### [2026-04-15 19:20] 决策 — 用户确认：PRCV 先收敛到 PSG 主线，旧实验允许重跑

**上下文**: 用户明确表示当前目标是“把现在的探索结果包装成一个故事和创新点，先发 C 类”；并进一步确认：  
1. 不必强行切到全新路线  
2. 所有实验都可以重跑  
3. 可以重新设计消融实验，不必被旧实验组织方式束缚

**选项**:
  A. 继续沿刚提出的 `PSCD/support-complete` 新路线展开
  B. 回到 `PSG` 主线，把 `multi-stage PSG` 重新做成干净可辩护的扩展版本

**选择**: B

**理由**:
1. `PSG` 本体已有最稳的证据链：`exp007` 单次正向，且 3-seed mean 明确成立
2. 当前最强系统 `exp255` 使用的就是 `2-stage PSG`
3. `exp255 vs exp255b` 给出最强信息：在 `GCN512` 高容量结构分支下，`2-stage PSG` 带来 `+1.7 / +1.4`
4. 虽然 `exp009 / exp251 / exp253` 不支持“multi-stage 普遍更强”，但这恰好说明需要**重跑干净消融**，而不是放弃 PSG 主线
5. 对 PRCV 来说，“PSG 为主创新，2-stage PSG 为 scalable extension” 比临时强切新问题定义更稳

**新的文档口径**:
1. `PSG` = 主创新
2. `2-stage PSG` = 当前最终版本 / scalable extension
3. `LGPA-D / GCN / OA-SD / MaxSim / POT / flip` = supporting assets

**下一步实验任务**:
1. 设计并重跑基础 PSG stage 消融：
   - no PSG
   - 1-stage PSG
   - 2-stage PSG
   - 3-stage PSG
2. 固定 branch 容量，重跑结构分支依赖性消融：
   - GCN256 + 1-stage
   - GCN256 + 2-stage
   - GCN512 + 1-stage
   - GCN512 + 2-stage
3. 视时间再补 semantic 分支依赖性消融

**执行结果**: 待后续新一轮 PSG 消融实验补充。

### [2026-04-19 03:00] 决策 — 本地 3090 挂了，Phase 1 Base 推迟（不丢）

**上下文**: 本地 3090 已挂；`phase1_design.md` 原把 Base 3 个 run（exp263/266/269）全部排在 3090 上。剩余资源仅 srvA/B/C 三台 5060 Ti 16G，已在跑 Phase 1 前 3 个 Tiny/Small run（exp261/262/264）。

**更正（2026-04-19 同日）**: 用户指出 `MODEL.WITH_CP: True`（gradient checkpointing，已在 `configs/occluded_duke/prcv_best_base.yml:14` 打开）下，**Base 显存只 6–8 GB，5060 Ti 16G 完全够**。显存不是瓶颈，真正瓶颈是时间分配优先级。

**选项**:
  A. Base 立刻并排 5060 Ti（挤占 Tiny/Small 进度）
  B. PRCV 只上 Tiny + Small 两行，Base 留 rebuttal
  C. 先把 Tiny + Small × 3 数据集 6 个 run 打完，再把 Base 3 个 run 排进同三台

**选择**: C

**理由**:
1. deadline 2026-04-30，11 天。Tiny/Small 6 run 三机并行约 22–28h，Base 3 run @ with_cp 三机并行约 18h，Phase 3 消融约 30h。仍能在 deadline 前出全稿
2. `exp260b Base = 73.9/83.2`（旧协议，本地 3090）可作 Base 行 reference
3. 现在三台都在跑 Tiny/Small，并排 Base 反而拖慢现有进度

**执行结果**:
- `experiments/prcv_2026_psg/todo.md` Phase 1 表 Base 3 行标 DEFERRED，机器列改为"srvA/B/C 任一"（不再绑定 local）
- Phase 4 multi-seed 短期 Small 优先；Phase 1 Base 跑完再补 Base multi-seed
- 同步条目落在 `experiments/prcv_2026_psg/decisions.md`
- Phase 1 当前运行: srvA=exp262(Small OD) e70, srvB=exp261(Tiny OD) e106, srvC=exp264(Tiny OP) e83；接下来按 srvB→exp267, srvC→exp265, srvA→exp268 顺序排队；Tiny/Small 6 run 完成后立即评估是否把 Base 3 run 并入 Phase 1
