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
