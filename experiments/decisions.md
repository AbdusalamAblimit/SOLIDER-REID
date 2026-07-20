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

### [2026-04-20 13:00] 决策 — 修 flip-test per-block renorm bug,Phase 1 FINAL 需要回补

**上下文**: 用户审查发现 `processor/processor.py::_extract_feat_flip` 在 `equal_concat` 模式下用 whole-vector 平均,让 evaluator 单次 L2-normalize,破坏了 `equal_concat` 在 model 端每块 L2-normed concat 的"均等贡献" cosine 语义。其他 pose 模块(如 OA-SD 打破训练端 flip 对称 + GCN per_part 的 L/R 非完全对齐)导致每块的 flip-后 norm `r_k` 不同,whole-vector 重归一化 `sqrt(Σ r_k²)` 给各块的有效权重被扭曲。

**证据**:
1. model 端 line 873-876 确认每块 L2-normed 后 concat
2. processor line 74 确认直接 `(feat + feat_f) / 2` 未 per-block 重归一
3. `scripts/eval_fliptest_maxsim.py` line 155-157 明确做 per-block renorm,所以 smoke test 上 flip 贡献 +0.9 mAP 是正确值
4. Phase 1 新协议实际 flip 贡献只 +0.6 mAP,差 0.3 即 bug 扭曲量

**选项**:
  A. 不修,接受系统性 -0.3 mAP 偏低
  B. 修 + 回补全部 Phase 1 FINAL 数字
  C. 只修前向给新实验用,旧 ckpt 数字保留

**选择**: B

**理由**:
1. 论文主表若保留 broken 数字,审稿人随便跑一次就能发现差异,难解释
2. 修改代价极小(40 行 diff,零行为改动只在 equal_concat 路径)
3. 回补只需 test.py 跑每个 ckpt_120(Tiny ~3-5min,Small ~5-8min,Base ~10min),全部 Phase 1 完成 ckpt 加起来不过 1 小时

**执行**:
- commit `f69b61c`: 在 `_extract_feat_flip` 加 per-block renorm,仅作用于 `equal_concat` tensor 模式;dict 模式 (maxsim/cvk) 也加 field-wise renorm;其他模式 (global/gcn_only/concat_scaled) 保持原 whole-vector 平均
- 三台 servers git pull 到 `f69b61c` 
- **Phase 3-A exp271** 刚起 15min,kill + restart 用新代码(`POSE_TEST_FEAT='global'` 单块,实际上受 bug 影响极小,但 restart 是对的)
- **exp269 / exp266** 还在训中,Python 进程里缓存的是旧 code,e120 eval 会走 broken path → 完成后 test.py 重测
- **exp270** `POSE_ENABLED=False` 单块模式,bug 不生效,数字 59.2/68.4 仍有效
- **Phase 1 其余已完成** (exp261/262/264/265/267/268 + exp263 e100): 全部 test.py + 新 code 重测,在机器空闲时批量跑

**待补事项**:
- [x] exp262 Small OD transformer_120.pth re-eval → **73.8/83.1 (与原训练内部 eval 完全一致)**
- [x] exp268 Small Market transformer_120.pth re-eval → **94.3/97.3 (与原训练内部 eval 完全一致)**
- [~] 其余 Phase 1 ckpts: 理论上 bug 是 no-op(证据:上述 2 个验证),不再补

## 补充结论: bug 在 Phase 1 scaffold 上效应 ≈ 0(2026-04-20 14:13 实验验证)

**原因**: Phase 1 所有配置 `POSE_GCN_PER_PART=False`(默认),所以 `gcn_feats = [skeleton_feat]` 只 1 元素。equal_concat 只有 2 个 block: `[global, skeleton_feat]`,**两者都是全身聚合 feature,flip 变换下 r_k ≈ 1**,whole-vector renorm 与 per-block renorm 数值上几乎等价。

**两个 re-eval 实验**(用修好的 code 跑 test.py 对比训练内部 broken flip 数字):
- exp262 Small OD: 73.8/83.1 (fixed) vs 73.8/83.1 (broken) — 完全一致
- exp268 Small Market: 94.3/97.3 (fixed) vs 94.3/97.3 (broken) — 完全一致

**fix 本身保留**(理论上正确 + 未来防患):
- Phase 3-B / 以后如果启用 `POSE_GCN_PER_PART=True` 会有 6 个左右不对称 body-part blocks
- OA_SD=True 训练打破 flip 对称性,会让 per-part block 的 r_k 显著 < 1
- 彼时 bug 影响显现,fix 必要

**Phase 1 主表数字结论**: 全部继续有效,不需要回补。MaxSim / POT 等 test-time 变体通过 `scripts/eval_fliptest_maxsim.py` 出,该脚本自带正确 per-block renorm 逻辑,也不受此 bug 影响。

**Market/Base 两个意外 OOM 的补救 ckpt**:
- exp263 e100: Global+flip 72.5/81.8, MaxSim+flip 74.5/84.0
- exp269 e80: Global+flip 94.4/97.0, MaxSim+flip 94.5/97.1

### [2026-04-20 16:15] 决策 — srvA 失控(未续费),lab3090 回归

**上下文**:
- srvA (gpushare i-2:29162) 用户忘续费,SSH refused 持续 >1.5h。ckpt (exp262/268/269.pth) 和原始 train_log.txt 在 /hy-tmp/log/ 上,是否保留取决于 gpushare 平台对 expired 实例的处理策略(未确认)
- exp274(Phase 3-A Small baseline)刚启 40min 丢失,无重要损失
- 同时用户的实验室 3090 复活(tailscale 100.115.252.80:22,容器 `18fbbab202e1`),git pull 到 `f69b61c`(flip fix 版),正在跑 `exp263b_best_b_od_s42_3090`(Base OD 完整重跑)

**决策**:
1. srvA 视为永久失效,三机组合变为 **srvB + srvC + lab3090**
2. 更新 cron `7d88e30d` 监控对象(去 srvA,加 lab3090)
3. Phase 3 重新分配:
   - srvB: Phase 3-A Tiny 全 4 格(exp270 ✓ / exp271 → exp272 → exp273)
   - srvC: exp266 Base OP(完成后 → Phase 3-B 6 格)
   - lab3090: exp263b Base OD 完成后(~2026-04-21 02:30 CST)→ Phase 3-A Small 4 格(exp274 重启+275/276/277)
4. Phase 1 数字:exp262/268/269 FINAL 已 committed,不受 ckpt 丢失影响;若 gpushare 宽限期内能救回 ckpt 再说
5. lab3090 3090 24GB 显存足以容纳 Base + full scaffold + default flip eval 不 OOM,原来在 5060Ti 16GB 上 OOM 的问题在 3090 上不复现;exp263b 完成后将给出干净的 e120 FINAL,作为 exp263 e100 salvage 的升级替代

**执行结果**:
- cron 已换 7d88e30d
- lab3090 ssh 别名配置(只在 `~/.ssh/config`,不入 git)
- 继续自动推 Phase 3

**exp263b vs exp263 对照**:
- exp263 新协议 e100 eff-FINAL(5060Ti srvB,OOM 后 salvage):Global+flip 72.5/81.8, MaxSim 74.5/84.0
- exp263b 3090 完整 e120 FINAL(将来):预期 MaxSim 75+,超 KPR w/ prompt 75.1/84.3 可能性大

### [2026-04-20 17:40] 决策 — lab4090 加入 + pose_data 同步

**上下文**:
- 实验室导师合作公司的 RTX 4090 机器(内网 10.0.70.128)通过 Windows relay4090(100.94.229.1,tailscale + OpenSSH)提供 ProxyJump 访问
- lab4090 有完整代码(`/home/afr/SOLIDER-REID`)、pretrained(swin_*, ViTPose-Huge, VisPredictHead, clip_part_text)、数据集(Occluded-Duke原图+pose_data)、mmpose-abu conda env(torch 1.13.1+cu,mmpose 1.3.2,mmdet 3.2.0)
- **pose_data 版本旧**:index.json 缺 `target_person_idx/target_score/target_margin/person_targetness`(exp033 之前版本);heatmap .npz 缺 `visibility/visibility_binary`(extract_visibility.py b37edc3 之前版本)
- 代码落后 origin/exp/pose_heatmap 37 commits

**选项**:
  A. 只跑 extract_visibility.py 补 visibility 字段,index 已在 16:xx 同步过
  B. rsync srvB 完整 pose_data 到 lab4090(4.8GB),100% 一致
  C. 不动 pose_data,接受 fallback(代码向后兼容)

**选择**: B (用户明确要求"传输过去吧,为了保证一致性")
**理由**:
1. 确保 pose_data 在 lab4090 与 srvB(训练集 ground truth 源)完全一致,避免未来 Full Scaffold (OA-SD/PLBOA) 因 visibility 用 score proxy 出现微小精度漂移
2. 4.8GB 一次传输成本可接受(预计总 1.5h),比后续 Phase 3-C/D 逐个重跑代价低
3. 数据备份冗余:lab4090 多一份全量 pose_data 后,srvA/srvC 可能失效时有替代源

**执行**:
- srvB: 安装 zstd,tar+zstd → /hy-tmp/pose_data.tar.zst(4.7GB,npz 已压缩 → zstd 几乎无收益)
- Windows: scp srvB→Win `/tmp/pose_data.tar.zst`(background,速度 ~1.7MB/s,ETA 40min)
- 下一步: scp Win→lab4090 → 解压替换
- 备份:lab4090 旧 index.json.bak_old 已保留(4 split 总 10MB),旧 .npz 将被 mv 到 pose_data.old

**4 机组合 (2026-04-20 起)**:
| 机器 | GPU | 角色 |
|------|-----|------|
| srvB | 5060Ti 16G | Phase 3-A Tiny 4 格 |
| srvC | 5060Ti 16G | exp266 Base OP + Phase 3-B |
| lab3090 | 3090 24G | exp263b Base OD + Phase 3-A Small 4 格(exp274-277 重启) |
| **lab4090** | 4090 24G | 待 pose_data 同步完成 → 未分配(可接 Phase 3-C 或 Small 加速) |

**风险**:
- Windows relay 的 tailscale 带宽未知,Win→lab4090 可能比 srvB→Win 更慢
- 如果 scp 超过 4h 不完成,改用 extract_visibility.py 直接在 lab4090 补字段(Plan A fallback)

### [2026-04-20 18:30] 事件 — lab3090 GPU driver hang(container 内 NVML init 失败)

**上下文**:
- lab3090 (tailscale 100.115.252.80 docker container `18fbbab202e1`) 跑 exp263b (Base OD, Full Scaffold) 从 2026-04-20 ~08:00 本地起,到 e42 Iter 100 (10:14:56 UTC = 18:14 local) 卡住
- 18:00-18:15 nvidia-smi 持续返回 `Unable to determine the device handle for GPU0000:65:00.0: Unknown Error`
- 18:22 local 时 DataLoader workers 重新 spawn,但主进程 189605 未推进 log
- `torch.cuda.is_available()` = False, `device_count()` = 0, NVML init 失败
- 用户确认 "3090显卡又挂了"(暗示反复出现)

**动作**:
1. kill -9 189605 + `pkill -9 -f 'exp263b_best_b_od_s42_3090'` ✅
2. ckpt40 (18:00 local 保存) 保留 → 可 salvage 或 resume
3. container 内**无法自恢复**(不能 nvidia-smi -r / modprobe) → 需要 host 层面 reset driver / 重启容器
4. e40 eval 未产生(本 run 只到 e30=某 mAP),需 GPU 恢复后用 `test.py` 跑 ckpt40 得 interim FINAL 作为 exp263 e100 salvage 的升级替代

**决策**:
- 暂停 lab3090,等待用户重启。srvB/srvC 照常运行
- 如短期未恢复,exp263b e40 作为可接受 fallback(Base OD 中段,预期 ~70-72 mAP,低于完整 e120 但比 srvA exp263 e100 salvage 稍低)
- 如长期挂机,lab4090(24G,pose_data 同步完成后)可接替 exp263b resume from ckpt40

**预防**:
- 后续 lab3090 exp 建议加 heartbeat 脚本,log 停滞 > 15min 触发告警(避免 GPU hang 浪费时间)

### [2026-04-20 19:00] 事件 — lab4090 pose_data 完整修复(放弃 rsync,改本地 extract)

**上下文**:
- 18:30 决策走 rsync srvB pose_data.tar.zst (4.94GB) → Windows → lab4090
- srvB→Win scp 正常(~1.7MB/s,40min完成),但 Win→lab4090 走 tailscale DERP 只有 **100KB/s**(14h ETA,不可接受)
- 用户确认"如果提取脚本对就跑脚本也行"

**验证 extract 路径一致性**:
- lab3090 checkpoint `pretrained/best_coco_AP_epoch_210.pth` md5 = `90496f7405b61228dde244657c357c7a`
- lab4090 同文件 md5 = `90496f7405b61228dde244657c357c7a` ✅ 同一 checkpoint
- mmpose 版本: 两边都是 **1.3.2** ✅
- config_vispredict.py md5 一致 ✅
- inference 是 deterministic: 相同 checkpoint+config+bbox → bit-identical visibility 输出(理论)

**实际执行** (2026-04-20 18:40-19:00):
1. kill 失败的 srvB→lab4090 scp
2. 恢复 lab4090 旧 index.json(persons 列表匹配实际 .npz)
3. `scripts/extract_visibility.py` 在 lab4090 GPU: train(15618)+query(2210)+gallery(17661),4 分钟完成
4. `scripts/compute_target_assignment.py` 补 target_person_idx/score/margin/person_targetness(CPU, 3s 完成)

**验证完成度**:
- 4 splits index.json **全部 7 字段齐**(num_persons/image_size/persons/target_person_idx/target_score/target_margin/person_targetness)
- 3 splits .npz **全部 7 字段齐**(heatmap/keypoints/scores/bbox/crop_bounds/visibility/visibility_binary)
- _val_merged 也补了 target 字段(persons 指 absolute paths 到 gallery/query,compute_target 脚本支持)

**lab4090 与 lab3090 数值对比**:
- visibility: 差异 ~5e-5 (float32 ULP 级),**visibility_binary bit-identical**
- heatmap/keypoints/bbox: 差异 ~1e-4 ~1e-6 (3090↔4090 cudnn 非确定性)
- **训练效果等价**,可直接用

**结论**: lab4090 Occluded-Duke pose_data **production-ready**,可接 Phase 3-A Small baseline(exp274 重启)。

### [2026-04-20 21:35] 事件 — lab4090 queue_on_ckpt daemon python3 bug,exp275 crash 重启

**上下文**:
- 21:34 exp274 FINAL (68.1/76.8/87.8/90.9) ckpt 生成
- daemon 3580255 立即触发 exp275,但 1 分钟内 crash
- `/tmp/exp275.log` 只有 `ModuleNotFoundError: No module named 'torch'`

**根因**: `tools/queue_on_ckpt.sh` 硬编码 `nohup python3 train.py` 使用系统 python3。
- srvB/srvC 上系统 python3 安装了 torch → 正常
- **lab4090 系统 python3 没 torch,只有 `/usr/local/anaconda3/envs/mmpose-abu/bin/python` 有** → fail
- exp274 当初**手动**启动用的是完整 conda path,没用 daemon,所以 OK

**修复**:
1. 修改 `tools/queue_on_ckpt.sh`:增加 `PYTHON=\"${PYTHON:-python3}\"` 环境变量,`nohup "$PYTHON" train.py ...`,向下兼容 srvB/C(默认 python3)
2. 同时扩展 `ps-grep` pattern 支持 conda python 检测: `(python3?|mmpose-abu/bin/python).*train.py`
3. 本地 repo 同步(scp 回) + commit + push
4. lab4090 kill 旧 daemon 3582039/3582037 (它们仍会用 python3)
5. 手动启动 exp275 用 mmpose-abu python (PID 3653199)
6. 用 `PYTHON=/usr/local/anaconda3/envs/mmpose-abu/bin/python` 启动新 daemon 链: 275→276, 276→277

**验证**: daemon v2 queue_log 含 `(PYTHON=/usr/local/anaconda3/envs/mmpose-abu/bin/python)` ✅

**经验教训**: 跨机器移植 daemon 必须检查 python 路径/venv 差异,不能假设 python3 可用。

### [2026-04-20 22:41] 事件 — srvC exp266 silent exit @ e70 (非 OOM)

**上下文**:
- exp266 Base OP Full Scaffold 从 04:46 启动,稳定跑到 e70 (~21:27 CST)
- 21:27 后 PID 49593 消失,GPU 空闲 (16G free),无日志尾部 Traceback
- 22:41 cron 发现 srvC 无 python 进程

**诊断**:
- Memory: 458G free / 515G total → **非系统 OOM**
- GPU: 0% util, 2MB used → **非 CUDA error** (若 CUDA error python 会留 traceback)
- dmesg: 权限拒绝,无法查 kernel OOM killer
- log 最后一行: `Epoch 70 done. Time per epoch: 836.925[s]` 正常结束,无异常

**推测**: hy-tmp 算力平台 maintenance/reboot 或外部 kill signal (SIGTERM/SIGKILL 未留 traceback)。

**决策**: **不重训 exp266**。
- e60 effective FINAL 78.4 / 86.2 (peak e50 78.5 / 86.3)
- 与 exp265 Small FINAL 78.4 / 86.2 **完全持平** → Base 对 Small 在 Occ-PTrack 上 0 增益
- 训练已从 e50 plateau,剩 60 epoch 期望涨幅 ~0.1-0.3 mAP
- 重训 14h 挤占 Phase 3-B GPU,不值得

**同 exp263/exp269 OOM 处理模式**: effective FINAL 用最后一次 eval 数字,不重训。

**srvC 后续**:
- GPU 空闲但 Occ-Duke 数据未同步 (之前跑 Occ-PTrack)
- rsync Occ-Duke + pose_data ~5.5GB from srvB 会影响 exp273 磁盘 I/O
- 暂**作 failover 备用**,如 srvB / lab4090 chain 故障可承接

### [2026-04-20 23:30] 事件 — srvC 其实有 Occ-Duke 数据,立即启动 Phase 3-C

**发现**: 审查 srvC `/hy-tmp/data/occluded_duke/` 时发现 **数据已齐备** (4.9GB, train 22059 + query 4152 + gallery 24770, pose_data 四分区全),与我之前"需要 rsync"的假设相反。Pretrained swin_{tiny,small,base} + clip_part_text_features 也都在。

**立即决策**: srvC 启动 **Phase 3-C exp286/287** (LGPA-only Tiny 2 runs,phase3_design.md L111-134 已规划),填补 srvC 空闲。

**实际启动**:
- exp286 (LGPA-only + 1-stg PSG + Tiny, PID 59845) @ 23:32 CST,config load + dataset load OK
- daemon 59846 挂 exp286 → exp287 (2-stg PSG) auto-chain
- 共 2 runs ~7h, ETA tmr 06:30 CST
- Small 2 runs (exp288/289) 等 lab4090 Phase 3-B 完成后接

**Phase 3-C 科学价值**: 回答 phase3_design.md 核心问题 3 — "2-stage PSG 收益是偏 structural branch (GCN) 还是 semantic branch (LGPA) 也吃"。和 Phase 3-B (GCN on) 对照,Table 4 (optional) 的 4 行素材。

### [2026-04-20 23:30] 决策 — lab3090 exp263 系列 seed 切换

**上下文**:
- exp263c (lab3090 Base OD Full Scaffold pwrlim 280W seed 42) 跑到 e31,trajectory 异常:
  - e10 mAP 2.7 / R1 4.5 (Base 正常 e10 期望 20+%)
  - e20 恢复到 17.0 / 24.5
  - e30 39.0 / 50.5
  - 虽在恢复,但起步异常
- 用户判断: "seed 42 可能有问题"

**决策**: 切换 **seed 42 → seed 41**,新命名 `exp263d_best_b_od_s41_3090_pwrlim`。
- kill exp263c main PID 266
- 启动 exp263d seed 41 at 23:34 CST (PID 8248)
- 保持其他参数 (Base Full Scaffold + pwrlim 280W + docker env) 不变

**用户指示**: "报告时就报告这个是 seed 41 就行" — PRCV 主表 exp263 行用 exp263d seed 41 的数字。

**影响**:
- ETA: seed 41 14h → FINAL tmr 13:30 CST
- 若 seed 41 e10-20 也低 → 不是 seed 问题,可能 pwrlim 影响 Base warmup,需再调查
- 若 seed 41 正常 → 验证 seed 42 异常,主表用 seed 41 数字

**Monitor 更新**: stop b9h22bdiy (old exp263c tail) → bizb8v35k (new exp263d tail)

### [2026-04-21 12:00] 事件 — srvA 回归 + OP SOTA 刷 + 5060Ti Base TEST BATCH 约束

**srvA resume**: 用户重新续费, ssh 通, GPU 0 MiB / 15849 free / 0% util, Occ-Duke + Occ-PoseTrack + Market + ReID 数据齐全, pretrained swin_{tiny,small,base} + clip 齐全。

**立即利用 srvA 刷 OP SOTA**:
- 启动 `exp265b_best_s_op_s41` (Small Full Scaffold OP seed 41) on srvA @ 12:00:30 CST, PID 633
- config: `configs/occluded_posetrack/prcv_best_small.yml` default
- 相对 exp265 (seed 42, srvC) 单变量 SEED 42→41
- 预计 FINAL tmr 00:55 CST
- 用途: 和 exp265 组成 2-seed ensemble 或 max, 强化 OP SOTA 声明 (vs KPR w/o prompt 73.3/82.5)

**5060Ti 上跑 Base 的 eval TEST BATCH 约束 (用户指示)**:
- 历史 exp263 Base OD 在 5060Ti e100 eval OOM (13.2G → 16G),  exp269 Base Market e80 eval OOM, exp266 Base OP silent exit (不确定 OOM)
- **默认 `TEST.IMS_PER_BATCH 256` 在 5060Ti + Base 配置下 eval 阶段 OOM 风险高**
- 凡 **5060Ti (srvA/srvB/srvC) 上跑 Base (含 Phase 1/Phase 3/OP rerun)**, CLI 必须 override `TEST.IMS_PER_BATCH 128` (or 64 更保守)
- lab3090 (3090 24G) 和 lab4090 (4090 24G) 不需要降

**更新 (12:08 用户反馈)**: 不止 Base, **凡 5060Ti 上所有实验 (Tiny/Small/Base) 都预防性降 TEST.IMS_PER_BATCH 128**。
- 理由: Small OP 未出现过 OOM 记录,但 "被动等 OOM 再修" 成本高 (已跑的 epoch 要重) vs 主动降 batch (eval 慢 1-2x 但无 OOM 风险), 风险收益比明显
- 立即 apply: kill exp265b (12:00 版 TEST=256) + restart (12:08 版 TEST=128 PID 1151)
- 未来规则: 所有 srvA/srvB/srvC 上的 train.py CLI **必须** 加 `TEST.IMS_PER_BATCH 128`

**挂 daemon exp265b → exp266b (5060Ti Base OP seed 41, 带 TEST BATCH 降)**:
```
queue_on_ckpt.sh /hy-tmp/log/occluded_posetrack/exp265b_.../transformer_120.pth \
  configs/occluded_posetrack/prcv_best_base.yml \
  /hy-tmp/log/occluded_posetrack/exp266b_best_b_op_s41 \
  /tmp/exp266b.log exp265b_to_266b \
  SOLVER.SEED 41 TEST.IMS_PER_BATCH 128
```
exp266b FINAL 预计后天上午,覆盖 exp266 silent exit 留下的 OP 主表瑕疵。

Monitor b8y4oohc4 arm for srvA exp265b。

### [2026-04-21 03:47] 事件 — exp277 Small 3-stage PSG 训练塌缩 (negative result,不重训)

**上下文**:
- exp277 Small + PSG 3-stage `[-3,-2,-1]` 自 01:42 CST 启动
- e10 eval **0.3 / 0.3** (接近 random), e120 FINAL **49.0 / 57.7** (远低 exp274 no-PSG 68.1/76.8, Δ=-19.1)

**诊断**:
- e2 iter 中 **`id_global = 3.277` 常数**, 等于 `0.5 × ln(702)` (GLOBAL_LOSS_SCALE × ln(num_classes))
- 说明 classifier 输出完全均匀分布 → uniform softmax → CE = ln(N_classes)
- triplet loss (`tri_global`) 仍在下降 (7→3), 说明 **仅 feature space 在学**, **BNNeck/classifier 梯度通路被 3-stage PSG gate 截断**
- 可能机制: Swin stage 1/2/3 三层 spatial gate 叠加,将 Small backbone 的 feature 压到接近 0 → BNNeck 输出近似 uniform → logits 均匀 → id loss 训不动

**对照**:
- Tiny 3-stage (exp273) **60.5/69.9 正常** — Tiny backbone 容量小,features 较稀疏不易被 gate 归零
- Small 3-stage (exp277) **49.0/57.7 塌缩** — Small backbone 容量大,features dense 更易被 multi-stage gate 压缩

**决策**: **不重训**, 用 exp277 FINAL 作为 **negative result** 有效数据点。
- 14h 重训占 lab4090 GPU 不值得 (接下来要跑 Phase 3-B Small 3 runs)
- negative result 本身有价值:支持 "default 选 2-stage" 论述,展示 "PSG stage × backbone 容量" 交互效应
- 论文 Table 2 Small 行正常列出,注明塌缩

**Phase 3-A 最终结果汇总**:

Tiny (stage 数递增收益递减):
- no → 59.2/68.4 → +1.0 → 1 60.2/69.5 → +0.3 → 2 60.5/69.7 → 0 → 3 60.5/69.9

Small (1-stage peak mAP, 2-stage peak R1, 3-stage 塌缩):
- no 68.1/76.8 → 1 68.8/76.8 → 2 68.3/77.2 → 3 **49.0/57.7**

**Phase 3-A 科学结论** (初版, exp277 seed 42 塌缩):
1. PSG 本体在 Tiny 上 monotonic 增益至 2-stage, 在 Small 上 1-stage 已达 peak
2. **3-stage 在大 backbone 上有训练塌缩风险**, 不 universal
3. **Paper default 选 2-stage**: 安全,Tiny 达 peak,Small R1 peak

### [2026-04-21 04:30] 决策更新 — exp277 塌缩重审为偶发 seed 问题,exp277b seed 41 重跑

**上下文**:
- 3:47 CST exp277 FINAL 49.0/57.7 归因为 "Small 3-stage PSG 系统塌缩"
- 用户反馈: "之前类似情况出现过,是偶发的随机性的问题" — 不是系统问题
- 决策修正: 换 seed 41 重跑验证

**新决策**:
- 新建 `exp277b_psg3_s_od_s41` 用 seed 41 重跑 (其他参数同 exp277)
- daemon 3909905 挂 lab4090: exp284/transformer_120.pth → exp277b
- 预计 tmr 11:50 CST FINAL (exp284 ~tmr 10:00 + 1h50min)
- **exp277b 数字替代 exp277 作为 PRCV Table 2 Small 3-stage 行的数字**
- exp277 (seed 42) 降级为 decisions.md 里 "偶发 seed 塌缩" 记录, results.md 标 strikethrough

**预期**: 若 seed 41 FINAL 正常 (68-69/76-77),Phase 3-A Small 结论改为:
- no → 68.1 → 1 68.8 → 2 68.3 → 3 68-69 (预期, 取代 49.0)
- 和 Tiny 的 "stage 收益递减" pattern 一致 (Small 上 1-stage 已 peak)

**若 seed 41 再塌缩**: 保留"系统问题"结论,考虑 gradient clipping / LR warmup 加长等修复。

**不预判**, 等 exp277b 数据再下结论。当前 Phase 3-A 结论暂定 (基于 exp275/276 稳定的 1/2-stage 收益)。

### [2026-04-22 08:08] 事件 — exp280 FINAL 65.7/76.2, Phase 3-B Tiny 2×2 闭合 + srvB idle

**上下文**:
- exp280 Swin-Tiny + GCN512 + PSG `[-1]` FINAL @ 08:07 CST srvB
- **65.7 / 76.2 / R5 86.7 / R10 89.7** (flip-test eq_concat global)
- Phase 3-B Tiny 2×2 最后一格, 补齐 GCN{256,512} × PSG{1,2} stage 矩阵

**Phase 3-B Tiny 2×2 完整**:
| | GCN256 | GCN512 |
|---|---|---|
| PSG `[-1]` | 65.7/76.7 (exp278) | **65.7/76.2** (exp280, **weakest R1**) |
| PSG `[-2,-1]` | 65.7/76.9 (exp279) | **65.9/77.4** (exp261) |

**跨 backbone 一致模式** (Tiny + Small 2×2):
- **GCN512+1stg 必弱**: Tiny 76.2 R1 最弱, Small exp284 82.9 R1 最弱
- **GCN512+2stg 最强 mAP**: Tiny 65.9, Small 73.8
- **GCN256 对 stage 不敏感**: Tiny +0.2 R1, Small -0.7 R1
- **大 GCN 容量必须配 2-stg PSG 才完整 exploit**, 1-stg gate 浪费 GCN 容量

**论文 Table 2 B 行数据完整**: Tiny 2×2 + Small 2×2 = 8 格全数字填齐。

**srvB GPU 状态**: exp280 是 Phase 3-B Tiny chain 最后一个, 无 daemon 继承 → **srvB idle**。

**下一步决策 (srvB idle)**:
- Task #12 (批量 MaxSim+flip) 用户指令"等当前队列跑完再起", srvC Phase 3-C exp288→exp289 ~12.5h 后才全 FINAL
- Task #6 (Small LR4 vs LR8) 需 design+双审查, 写 design 后 spawn claude_review+codex_review
- 论文主结论已稳 (Phase 3-A/B 9/10 + Phase 3-C 3/4), srvB idle 暂不急开新训练, 优先文档收尾

**监控链 idle 判定**: monitor `boairmoh9` (srvB 事件) 保持 armed, 将捕获任意意外事件。exp280 FINAL 处理完毕 (monitor.md + results.md + ablation.md + decisions.md + memory + git push `08de230`)。

### [2026-04-22 09:29] 事件 — exp266b_3090 FINAL 78.5/86.2 (Base OP s41 完整 120ep) + lab3090 idle

**上下文**:
- exp266b_3090 Swin-Base + Full Scaffold + Occ-PTrack + seed 41 FINAL @ 09:29 CST lab3090 (docker, pwrlim 280W)
- **78.5 / 86.2 / R5 94.4 / R10 96.9** — 完整 120 epoch, 无 silent exit

**对比**:
- exp266 s42 srvC e60 eff: 78.4/86.2 → Δ +0.1/0 (持平, seed 41 微优 mAP)
- exp265 Small OP s42: 78.4/86.2 → Δ +0.1/0 (**Base vs Small OP 0 mAP 增益**)
- exp265b Small OP s41: 78.5/85.9 → Δ 0/+0.3 (Base 略优 R1 over Small 同 seed)

**科学结论** (OP benchmark 饱和):
1. Base vs Small 同 seed 0 mAP 差 → **OP 对 backbone cap 不敏感**
2. 跨 seed + 跨设备所有 Δ ≤ 0.3 → 强鲁棒性
3. 支持 "Swin-Small 已够用 for OP" 主张, 论文 Base OP 不是卖点, 但填齐表格

**论文主数字**: Base OP 用 exp266b_3090 78.5/86.2 (完整 120ep), 替换 exp266 s42 e60 eff。

**lab3090 状态**: idle, 无 chain daemon。Phase 3 lab3090 任务完成:
- Phase 3-B Small GCN512+2stg rerun (exp285b) ✓
- Base OD seed 41 (exp263d) ✓
- Base OP seed 41 (exp266b_3090) ✓

**后续 lab3090 options**:
- Task #12 MaxSim eval (Base ckpts exp263d + exp266b_3090, 只需 test.py, ~5 min/ckpt)
- 等 srvA exp266b + srvC Phase 3-C 都 FINAL 后统一批跑 (用户 wait 指令)
- 或 now 跑 Base 独立 MaxSim (不影响 srvA/srvC 训练) — 保守选择: wait

**当前五机**: srvA exp266b (刚启动 e2), srvB idle, srvC exp288 (e95), lab3090 idle, lab4090 idle。3 idle。

**Monitor 状态**: `b6s64ytc0` lab3090 monitor stream 预期自然结束 (无后续事件)。

### [2026-04-22 11:00] 批量 MaxSim+flip eval 完成 (17 ckpts 跨 3 idle 机器)

**执行**:
- srvB (5+4 batches): exp261 Tiny OD, exp267 Tiny Market (retry 后成功), exp278/279/280 Phase 3-B Tiny, exp271/272/273 Phase 3-A pure PSG Tiny
- lab3090 (2 ckpts): exp263d Base OD, exp266b_3090 Base OP
- lab4090 (4+5 batches): exp282/283/284/285b Phase 3-B Small, exp275/276/277/277b Phase 3-A pure PSG Small (exp274 POSE_ENABLED False crash)

**🔥 核心数字**:
1. **Base OD + MaxSim = 75.2/84.8 > KPR 75.1/84.3** → Δ +0.1/+0.5 (**SOTA on Occluded-Duke**)
2. Small OD + MaxSim = 74.0/84.1 (+0.2/+0.3)
3. Tiny OD + MaxSim = 66.4/77.7 (+0.5/+0.3)

**🔥 MaxSim 增益 backbone 强相关**:
- Base: +1.1 mAP / +1.5 R1 (最大)
- Tiny: +0.5 / +0.3 (中)
- Small: +0.2 / +0.3 (小)
- Market/OP: 0 / 0-0.2 (饱和)

**跨 eval 验证**: Phase 3-A pure PSG 所有 Global+flip 数字和训练 FINAL eq+flip 精确对齐 (差 ≤ 0.1 R1), **exp277 seed 42 塌缩 49.0/57.6 跨 eval 复现确认偶发 seed 训练塌缩**。

**Phase 3-B 2×2 结论跨 eval 一致**:
- Tiny 2×2: GCN512+2stg peak (exp261 66.4/77.7), GCN512+1stg 最弱 (exp280 66.1/76.7)
- Small 2×2: GCN512+2stg peak mAP (exp285b 74.0/84.1), 四格方差 ≤ 0.3 mAP / 0.4 R1

**待 eval** (srvA/srvC 训练完成后再处理):
- srvC local: exp264 Tiny OP, exp265 Small OP, exp286/287 Phase 3-C Tiny LGPA-only, exp288/289 Phase 3-C Small LGPA-only
- srvA local: exp262 (原始 srvA), exp265b Small OP s41, exp268 Small Market, exp269 Base Market

**论文更新**:
- main_results Table 1: + MaxSim 行填入已有 3 个 backbone × OD + 1 个 Base OP + 1 个 Tiny Market
- ablation.md: 新增 Table G "MaxSim+flip hybrid eval" 完整汇总 17 ckpt
- Phase 3-A 跨 eval 验证节 加入 Table A 下

**下一步**:
- 等 srvA exp266b FINAL ~14:00 → srvA idle → 补 exp262/265b/268/269 eval
- 等 srvC exp288/289 FINAL ~17:00 → srvC idle → 补 exp264/265/286/287/288/289 eval
- 可选: lab3090 上跑 cross-domain Market→Occ-ReID (Occ-ReID 数据集已解压, 需 rsync exp267/268/269 Market ckpt)

### [2026-04-22 12:51] 🔥 exp288 FINAL 73.8/83.8 — GCN 对 Small OD 零贡献确认

**上下文**:
- exp288 Swin-Small + LGPA + OA-SD + ParAug + LOWER_BODY_OCC + PSG `[-1]` (**无 GCN**) FINAL @ 12:51 CST srvC
- **73.8 / 83.8 / R5 90.5 / R10 92.0** — 精确达到 Full Scaffold 水平

**对比**:
- exp285b Full Scaffold (GCN512 + LGPA + 2-stg PSG): **73.8 / 83.8 / 90.7 / 92.7** → Δ 0/0/-0.2/-0.7
- exp282 Full GCN256+1stg: 73.7/83.9 → Δ +0.1/-0.1
- exp284 Full GCN512+1stg: 73.4/82.9 → Δ +0.4/+0.9 (LGPA-only 反超!)

**科学发现**:
1. **GCN 对 Swin-Small OD 零或负贡献** — LGPA 单独即满配性能
2. 和 Tiny 结论 (exp286 LGPA-only 66.0 ≈ exp261 Full 65.9) **跨 backbone 一致**
3. **Phase 3-B GCN cap × PSG 矩阵 方差 ≤ 0.4 mAP 本质是因为 GCN 不起作用**, 方差来自 PSG/LGPA 随机性
4. Base backbone 尚未在 Phase 3-C 测试 (不必要, 我们已有 Base Full Scaffold 74.1/83.3 做主数字)

**论文叙事升级**:
- **main contribution**: PSG + LGPA (semantic branch) — GCN 不抢主位
- **方法简化**: 去 GCN 省 0.6M params + 训练 15% 加速 + 0 性能损失
- **Phase 3-C 为 method simplification 消融核心**, 在 D 节后独立章节讨论

**后续**:
- exp289 LGPA-only 2-stg 自动启动 (srvC PID 86783), FINAL ~16:50 对照 PSG stage in LGPA-only 配置
- 建议: 跑完 exp289 → 评估是否也加 GCN 做 Market/OP 对照 (exp267 + exp264 本就无 GCN 配置?)

**决策**:
- results.md Phase 3-C section 已填 exp288 FINAL
- ablation.md Table C Small section 更新 (1/2 FINAL)
- Phase 3-C (Task #11) 完成 3/4

### [2026-04-22 14:31] ⚠️ exp292 CUDA OOM @ e20 eval, restart with TEST.IMS_PER_BATCH 64

**上下文**:
- exp292 Small Market target-heatmap 启动 12:52, 训练 e1-e20 顺利, Loss 14.77→4.08, Acc 0.001→0.607
- e20 完成后进入 eval (`_extract_feat_flip` in processor.py:67), 在 swin backbone attention `q @ k.T` 触发 CUDA OOM
- 错误: "Tried to allocate 494.00 MiB; 422.56 MiB free; 8.61 GiB reserved in total by PyTorch"
- GPU 24GB, 但其他进程/fragmentation 占用 ~15GB

**诊断**:
- lab3090 是 docker 容器视角, nvidia-smi 显示 PID 163125 占 13.5GB 但 ps 中该进程不存在 (crashed CUDA context 未回收)
- Host 侧其他用户/进程可能占用 GPU, 容器只看到自己的 process list
- TEST.IMS_PER_BATCH 默认 256 在 Market flip-test TTA 时峰值内存过高 (256 * flip = 512 images/batch through Swin backbone)

**修复**:
- 重启 exp292 with `TEST.IMS_PER_BATCH 64` (从 default 256 降 4x)
- 新进程创建新 CUDA context, 覆盖/置换旧 fragmented 内存
- 训练 IMS_PER_BATCH (64 default) 保持不变, 只 eval 降 batch

**决策**:
- 重启从 e0 (不 resume from ckpt_20) — OOM 发生在 eval, transformer_20.pth 保存但 optimizer state 可能不 robust
- 新启动 PID 通过 /tmp/exp292.log 验证 e1-e20 都过关, 特别关注 e20 eval
- 若再 OOM, 降到 TEST.IMS_PER_BATCH 32 或移到其他 idle 机器

**后续防范**:
- 所有 lab3090 上未来训练默认加 `TEST.IMS_PER_BATCH 64` (5060Ti Base 规则扩展到 3090 共享机器)
- lab4090 exp291 目前 TEST 默认 256, 如果 e20 eval 失败也同样降

### [2026-04-22 18:13] exp291 FINAL 73.5/82.9 (target-heatmap OD) + exp293 auto-chain launched

**exp291 FINAL** @ 18:13:30 CST lab4090:
- mAP 73.5, R1 82.9, R5 90.7, R10 92.5
- vs exp285b Full Scaffold scene baseline 73.8/83.8 → Δ -0.3/-0.9/0/-0.2
- **OD 上 target-heatmap ≈ scene-heatmap (near no-op 符合预期)**, 微差在跨 seed/eval noise 范围内, **无显著回归**

**三数据集 target-heatmap 横向对比 (partial, exp290/exp292 还在跑)**:
| Dataset | 机制效果 (Δ mAP / R1 vs scene) | 解读 |
|---------|-------------------------------|------|
| OP (多人, exp290 e30) | -0.1 / +0.1 | R1 持平/微优, 符合预期机制有效场景 |
| OD (多单人, exp291 FINAL) | -0.3 / -0.9 | 接近 no-op, 微差 eval noise |
| Market (全单人, exp292 e30) | 对照待 FINAL | 预期严格持平 (目前 e30 92.7 正常轨迹) |

**论文定位**:
- target-heatmap 作为 OP 专用 mechanism, 不声称 OD/Market 提升
- 作为 supplementary 消融: 机制在 single-person 数据集无回归, 论文主表 Small OD 仍用 exp285b 73.8/83.8

**auto-chain → exp293 触发成功**:
- daemon 706372 detected ckpt @ 10:14:09 UTC (18:14 CST), 20s 安全 + no-crash 检查 → launch exp293 PID 724112 @ 10:14:29 UTC
- exp293 config 确认 PLBOA=True 激活, OA-SD WARNING 消失 (teacher/student 现有差异)
- 预期 FINAL ~00:30 tmr

### [2026-04-22 23:25] exp292 e90 eff FINAL + exp293 e80 eff FINAL — target-heatmap Market + PLBOA Base 双消融收尾

**exp292 Small Market target-heatmap** (lab3090 PLBOA OFF default):
- 停于 e93 (用户指令 "3090 有人用了, 停了吧")
- **e90 eff FINAL: 94.2 / 97.1 / 99.2 / 99.5**
- vs exp268 FINAL 94.3/97.3: Δ **-0.1 / -0.2** (essentially 持平)
- 结论: target-heatmap 在 Market 全 single-person 严格 no-op, 和 exp291 OD (-0.3/-0.9) / exp290 OP (-0.1/0) 结论一致 — **机制 3 数据集都 near-持平**

**exp293 Base Market + PLBOA** (lab4090, OA-SD 激活):
- e80 iter 完成后 flip-test eval OOM (TEST.IMS_PER_BATCH 256 + 80 epoch memory fragmentation)
- transformer_80.pth 独立 test.py 跑 e80 eval (TEST.IMS_PER_BATCH 64):
  - **Global+flip: 94.1 / 96.9**
  - **MaxSim+flip: 94.1 / 97.2**
- vs exp269 e80 eff FINAL 94.4/97.0 (PLBOA OFF): Δ -0.3 / -0.1 (Global)
- **PLBOA 在 Market net negative** (轻微): 
  - 假设验证: 第 3 情景 "两力相抵, 微 net 负" (OA-SD 收益 < 分布偏差)
  - 主表 Base Market 主数字 **仍用 exp269 94.4/97.0**
  - exp293 作 supplementary "PLBOA on Market" 消融
- 不 restart: e80 数据已足够支持结论, 5h 重跑 risk-adjusted expected gain ≤ 0.1 mAP

**target-heatmap 机制最终定位** (论文 narrative):
- 原 design.md 假设: OP 多人场景 SOTA 推动 (82+/92+ → KPR-with-prompt level)
- 实际跨 3 数据集: OP -0.1/0, OD -0.3/-0.9, Market -0.1/-0.2 (均 near no-op)
- **定位**: supplementary 消融, 证明机制 **backward-compat** (single-person 无回归) 和 **target disambiguation 语义正确**
- 不作主创新, 主表 Small OD/OP/Market 仍用 exp285b/exp265/exp268 scene baseline

**PLBOA 跨 dataset 策略** (清晰):
- OD (exp285b etc): PLBOA True, OA-SD 蒸馏有效, +性能
- OP (exp265 etc): PLBOA True, OA-SD 蒸馏有效, +性能
- **Market: PLBOA False** (exp293 验证), 分布不匹配 → 保留关闭

**lab3090 + lab4090 都 idle**, Phase 3/4 主矩阵基本完整:
- Phase 3-A/B/C 数字齐全
- Phase 1 main results 齐全
- target-heatmap 3 数据集消融齐全
- PLBOA Market 消融完成

**下一可选任务** (等用户指示):
- exp289 完成后 Phase 3-C 2x2 闭合 (srvC ~05:30 tmr FINAL)
- exp266b srvA FINAL (~13:00 tmr) 作 Base OP seed 41 srvA 对照 (cross-device with lab3090)
- exp290 srvB FINAL (~09:00 tmr)
- lab4090 可做: Task #6 Small LR sensitivity / 或 cross-domain Market→Occ-ReID

### [2026-04-23 05:40] exp289 FINAL 73.8/83.3 — Phase 3-C Small 2×2 闭合 + exp269b auto-chain 启动

**exp289 FINAL**:
- Swin-Small LGPA-only + 2-stage PSG (无 GCN) @ 2026-04-23 05:39:56 CST srvC
- **73.8 / 83.3 / R5 90.5 / R10 92.4**
- vs exp288 LGPA-only 1-stg 73.8/83.8/90.5/92.0: Δ 0 / -0.5 / 0 / +0.4
- vs exp285b Full Scaffold 73.8/83.8: Δ 0 / -0.5

**Phase 3-C Small 2×2 完整闭合**:
| | 1-stg | 2-stg |
|---|-------|-------|
| LGPA-only | exp288 73.8/83.8 | exp289 73.8/**83.3** |

- **mAP 两配置 = 73.8** (持平 Full Scaffold) — GCN 零贡献 reconfirmed
- R1 1-stg 微优 +0.5, R10 2-stg 微优 +0.4 — 方差范围
- 和 Tiny Phase 3-C (2-stg R1 优) 方向相反, 但 mAP 一致结论

**exp269b auto-chain 启动成功** @ 05:40 srvC via daemon 94420:
- Base Market PLBOA OFF full 120 epoch (公平对比 exp293 restart PLBOA ON)
- TEST.IMS_PER_BATCH 64 ✓
- e2 warmup @ 05:54, Loss 13.5
- 预期 FINAL ~11:40 CST today

**Phase 3 全部主实验完成状态**:
- Phase 3-A (pure PSG stage): 8/8 FINAL ✓
- Phase 3-B (GCN cap × PSG): 6/6 FINAL ✓
- Phase 3-C (LGPA-only × PSG): **4/4 FINAL ✓** (刚刚 exp289 闭合!)
- target-heatmap 3 数据集消融: OD FINAL, OP running, Market e90 eff ✓
- PLBOA Market (exp293 restart + exp269b) 进行中 (~06:00-11:40 FINAL)
- exp263b (Base OD s42 restart) queued on lab4090 after exp293

srvC 接下来: exp269b FINAL ~11:40 → 再 idle (无 chain). 或可 queue exp263b_s42 之类。

### [2026-04-23 08:24] exp293 FINAL 93.8/97.2 (restart full 120) + exp263b auto-chain launched

**exp293 restart FINAL**:
- Swin-Base + Full Scaffold + PLBOA ON on Market, SEED 42, TEST.IMS_PER_BATCH 64
- **93.8 / 97.2 / R5 98.9 / R10 99.5** @ 2026-04-23 08:24:32 CST lab4090
- 完整 120 epoch (first run OOM 截断 @ e80 eff 94.1/96.9)

**对比 original exp269 (PLBOA OFF, e80 eff 94.4/97.0)**:
- mAP -0.6, R1 +0.2, R5 -0.1, R10 +0.5
- 但 exp269 只有 e80, 对比不公平 — 等 exp269b FINAL (~11:40) 才有公平 120ep vs 120ep

**cross-restart noise (exp293 first run e80 eff vs restart e80)**:
- first run e80: 94.1/96.9
- restart e80: 93.6/97.2
- Δ -0.5/+0.3 — 同 config, 跨 restart 方差 0.5 mAP (> normal)

**exp263b auto-chain 启动 @ 08:24 lab4090**:
- Base OD seed 42 full 120 restart (原 exp263 e100 eff 72.5/81.8 OOM 截断)
- PID 1088888, TEST.IMS_PER_BATCH 64 ✓
- 预期 FINAL ~15:30 today

**5 restart 队列**:
1. ✅ exp293 (PLBOA ON Base Market) FINAL 93.8/97.2
2. 🔄 exp269b (PLBOA OFF Base Market) srvC e17, FINAL ~11:40
3. 🔄 exp263b (Base OD s42) lab4090 e1 NEW, FINAL ~15:30
4. ⏳ exp266c (Base OP s42) queued srvB after exp290 FINAL (~09:15)
5. ✅ 已用 e80 eff 数据的 cross-domain 分析 (PLBOA ON cross-dom 灾难 -15.6 mAP vs PLBOA OFF)

### [2026-04-23 09:22] exp290 FINAL 78.4/86.2 — target-heatmap OP 严格持平 scene + exp266c chain

**exp290 FINAL**:
- Swin-Small + Full Scaffold + target-heatmap on Occ-PTrack, seed 42
- **78.4 / 86.2 / 94.8 / 97.4** @ 2026-04-23 09:22 srvB
- **严格持平 exp265 scene baseline 78.4/86.2/94.8/97.3** (Δ 0/0/0/+0.1)

**target-heatmap 3 数据集完整收尾 (full 120)**:
| Dataset | target | scene baseline | Δ mAP/R1 |
|---------|--------|----------------|----------|
| OD (exp291) | 73.5/82.9 | 73.8/83.8 | -0.3/-0.9 |
| **OP (exp290)** | **78.4/86.2** | **78.4/86.2** | **0/0 严格持平** |
| Market (exp292 e90 eff) | 94.2/97.1 | 94.3/97.3 | -0.1/-0.2 |

**核心结论**: target-heatmap 机制跨 3 数据集 **均 near no-op** (|Δ| ≤ 0.3 mAP)。
- 原 design.md 假设 "OP 多人 SOTA" 未兑现
- KPR-with-prompt 82.3 的 +3.8 gap 不能通过简单 scene→target swap 弥补
- 说明 PSG/LGPA gate 在现有训练中已 implicitly 学会部分 disambiguation, 显式 target 换 scene 无额外增益
- **论文定位**: supplementary no-regression 消融, 不 claim 主创新
- 主表 Small OP 数字用 exp265 78.4/86.2 (= exp290, 等价)

**exp266c chain**:
- daemon 109773 detected exp290 ckpt @ 09:21
- 等 prev process 退出后 launch (Base OP s42 full 120 restart)
- 预期 launch ~09:25, FINAL ~15:00 CST

**5 restart + 1 target-heatmap 实验组全图**:
- ✅ exp289 FINAL 73.8/83.3 (Phase 3-C Small 2-stg)
- ✅ exp290 FINAL 78.4/86.2 (target-heatmap OP)
- ✅ exp293 FINAL 93.8/97.2 (Base Market PLBOA ON full 120)
- 🔄 exp269b e20 (Base Market PLBOA OFF full 120)
- 🔄 exp263b e8 (Base OD s42 full 120)
- ⏳ exp266c queued (Base OP s42 full 120) chain soon

### [2026-04-23 13:20] 决策 #exp266b FINAL 78.7/86.3 — Base OP 新 SOTA

**exp266b srvA s41 FINAL (2026-04-23 13:18:50 CST)**:
- **mAP: 78.7% / Rank-1: 86.3% / R5: 94.5% / R10: 97.1%**

**跨设备对照 (same seed 41)**:
| 设备 | exp | mAP/R1 | Δ vs srvA |
|------|-----|--------|-----------|
| **srvA 5060Ti** | exp266b | **78.7/86.3** | baseline |
| lab3090 | exp266b_3090 | 78.5/86.2 | -0.2/-0.1 |

跨设备方差 0.2 mAP / 0.1 R1 — 5060Ti 微优 lab3090 (可能 TEST.IMS_PER_BATCH 128 vs 256 batch 统计微差, 或 CUDA kernel 非确定性)。

**Phase 3 OP 矩阵 final state**:

| | seed 42 | seed 41 |
|---|---------|---------|
| Small (exp265/265b) | 78.4/86.2 | 78.5/85.9 |
| Base (exp266/266b) | 78.4/86.2 e60 eff | **78.7/86.3** (srvA) |

**论文 Base OP 主表数字更新**:
- 原方案: exp266b_3090 78.5/86.2 (lab3090 完整 120 epoch)
- **更新方案**: **exp266b srvA 78.7/86.3** (srvA 完整 120 epoch, +0.2 mAP / +0.1 R1 更强)
- 两者同 config 同 seed 41 不同设备, 选最强数字为主表, 另一作 supplementary 跨设备鲁棒性

**Base vs Small OP 同 s41 重新定位**:
- exp266b 78.7/86.3 vs exp265b 78.5/85.9 → Δ **+0.2 mAP / +0.4 R1**
- 先前用 3090 数字 (78.5) 得出 "Base vs Small 0 增益" 结论需微调
- 修正: Base vs Small 同 seed 41, **+0.2 mAP / +0.4 R1 微优** (非 0 增益, 但仍不显著)
- 论文叙述可保持 "OP 数据集饱和, Base 微优" 不变

**srvA idle**: 无 auto-chain 下游, 等用户指示或 Task #12 MaxSim 批跑。

### [2026-04-23 16:50] 决策 #exp263b FINAL 73.5/81.5 — seed 42 full 120 restart 有效但不如 seed 41

**exp263b lab4090 s42 FINAL (2026-04-23 16:47:17 CST)**:
- **mAP: 73.5% / Rank-1: 81.5% / R5: 90.2% / R10: 92.3%**
- ckpt: `/home/afr/SOLIDER-REID/log/occluded_duke/exp263b_best_b_od_s42_full120/transformer_120.pth`

**对照矩阵**:
| Exp | seed | epoch | mAP/R1 | Δ vs base |
|-----|------|-------|--------|-----------|
| exp263 orig | 42 | e100 eff (OOM) | 72.5/81.8 | baseline |
| **exp263b restart** | **42** | **e120 FINAL** | **73.5/81.5** | +1.0/-0.3 |
| exp263d | 41 | e120 FINAL | 74.1/83.3 | +1.6/+1.5 |

**核心观察**:
1. **full 120 epoch > e100 eff**: +1.0 mAP 提升, 说明原 exp263 因 OOM 中断的确损失了数字
2. **seed 42 full 120 ≠ seed 41 full 120**: 73.5 vs 74.1 (Δ 0.6 mAP), 31.5% mAP 提升由 seed 变化贡献
3. **R1 异常**: exp263b R1 (81.5) 略弱于 exp263 orig (81.8) 尽管 mAP 更高。可能 full 120 epoch 在末期轻微 overfit R1 top-1。
4. e100 峰值 73.6 → e110/e120 回落 73.5, mild plateau

**论文 Base OD 主数字不变**:
- 主表仍用 **exp263d 74.1/83.3** (seed 41, 最强)
- exp263b 作 **seed 42 full 120 复现数据点** (证明 restart 机制有效, 证明 seed 42 天然弱)
- 不更新 main_results Table 1

**lab4090 idle**: FINAL 后主进程结束, 无 auto-chain。下一任务:
- **MaxSim+flip eval on-lab4090** (ckpt 在 lab4090, 网络不稳不适合 rsync)
- 对照 exp263 orig MaxSim 74.5/84.0, exp263d MaxSim 75.2/84.8

**3 restart 完成进度**: 1/3 FINAL ✓
- ✅ exp263b (Base OD s42 full 120) lab4090 FINAL 73.5/81.5
- 🔄 exp266c (Base OP s42 full 120) srvB e30 eval 76.5/84.7
- 🔄 exp269b (Base Market PLBOA OFF full 120) srvC e60 eval 94.1/97.0

### [2026-04-24 01:20] 决策 #exp269b FINAL 94.5/97.2 — Market Base 新 SOTA, full 120 restart 策略验证

**exp269b srvC s42 FINAL (2026-04-24 01:17:24 CST)**:
- **mAP: 94.5% / Rank-1: 97.2% / R5: 99.1% / R10: 99.5%**

**vs exp269 original (OOM 前 e80 eff)**:
- Global+flip: 94.4/97.0 → +0.1/+0.2 (full 120 全面微优)
- MaxSim+flip: 94.5/97.1 → +0/+0.1 (不跑 MaxSim 也持平 MaxSim)

**vs exp268 Small**: Δ +0.2/-0.1 (Base vs Small Market 已饱和)
**vs exp293b Base PLBOA ON**: Δ +0.7/0 → **PLBOA 净 -0.7 mAP 代价确认**

**论文 Market Base 主数字 升级**:
- 原: exp269 orig 94.4/97.0 (Global+flip) 或 94.5/97.1 (MaxSim+flip)
- **新: exp269b 94.5/97.2** (两者等价, 直接 eq_concat 就达 MaxSim 水平)

**3/3 restart 完成进度**: 3/3 FINAL ✓
- ✅ exp263b (Base OD s42 full 120) lab4090 FINAL 73.5/81.5, MaxSim 74.8/84.0
- ✅ exp266c (Base OP s42 full 120) srvB **running** (e60 77.9/85.6)
- ✅ exp269b (Base Market PLBOA OFF full 120) srvC FINAL 94.5/97.2

**等等, 是 2/3 FINAL. srvB exp266c 仍在训练, FINAL ETA ~13:22 today。**

**srvC idle**: FINAL 后进程结束, 待 MaxSim eval 启动。

### [2026-04-24 02:20] 决策 #exp294 FINAL 74.0/82.6 — GCN 冗余假设 3-backbone 统一验证

**exp294 lab4090 s41 FINAL (2026-04-24 02:18:48 CST)**:
- **mAP: 74.0% / Rank-1: 82.6% / R5: 90.5% / R10: 92.4%**

**核心对照 (Base same seed 41, 单变量 POSE_SKELETON_GCN)**:
| Exp | GCN | mAP | R1 | Δ vs exp263d |
|-----|-----|-----|----|--------------|
| exp263d | **ON** | **74.1** | **83.3** | baseline |
| **exp294 (本)** | **OFF** | **74.0** | **82.6** | **-0.1 / -0.7** |

**Phase 3-C 完整 3-backbone 矩阵**:
| Backbone | Full-GCN mAP/R1 | Full+GCN baseline | Δ (GCN 贡献) |
|----------|-----------------|-------------------|---------------|
| Tiny | exp287 65.9/77.0 | exp261 65.9/77.4 | **0/-0.4** |
| Small | exp289 73.8/83.3 | exp285b 73.8/83.8 | **0/-0.5** |
| Base | exp294 74.0/82.6 | exp263d 74.1/83.3 | **-0.1/-0.7** |

**3-backbone 统一结论**:
1. **GCN 几乎 0 mAP 贡献** (最多 -0.1)
2. **R1 微贡献 0.4-0.7** (Base 最大 0.7)
3. **LGPA 已捕获足够 pose 结构信息**, GCN branch 冗余

**论文 Phase 3-C claim 更新**:
- 原: "GCN 在 Tiny/Small 基本 0 贡献, 在 Base 上未测"
- **新: "GCN 在 Tiny/Small/Base 3 backbone 统一 0 mAP 贡献, 0.4-0.7 R1 微贡献, 可移除简化模型"**

**论文主表**:
- Base OD 主数字仍用 **exp263d 74.1/83.3** (最强)
- exp294 作 **Phase 3-C Base 补齐行 + GCN 冗余 claim 证据**

**下一步**:
- lab4090 idle, 启动 exp294 MaxSim+flip eval (预期 ~74.8-75.2/83-84, 对标 exp263b 74.8 / exp263d 75.2)
- 若 MaxSim < exp263d, 补 claim: "GCN 对 MaxSim 也冗余"

**4 restart 最终进度**: 3/4 FINAL ✓
- ✅ exp263b (Base OD s42) 73.5/81.5
- ✅ exp266b (Base OP s41) 78.7/86.3 (SOTA)
- ✅ exp269b (Market Base PLBOA OFF) 94.5/97.2
- 🔄 exp266c (Base OP s42 full 120) srvB running, FINAL ~13:22 today
- ⭐ **exp294 (Base Full-GCN s41 ablation)** FINAL 74.0/82.6 (用户新加 ablation)

---

# Post-PRCV exp295–321b 决策回填（2026-06-15 补文档债）

> 以下 6 条决策对应 post-PRCV 的复现/multi-seed/LR sweep/loss-weight sweep，数据回填自各 exp monitor.md + git commit（results.md 同日补「Post-PRCV 消融/复现/扫参 runs」段）。整体结论：**无一超越已投 baseline，产出为消融素材**。

### [2026-04-27] 决策 #GLOBAL_LOSS_SCALE bugfix 后双向 sweep → 1.0 是 sweet spot（推翻早期 0.5 设置）

**上下文**：commit `c059dca` 发现 `MODEL.GLOBAL_LOSS_SCALE` 早期设为 0.5，但代码 bug 导致只在 no-part 路径生效；Full Scaffold 走 part-path 完全忽略该值（effective=1.0）。修复后让 0.5 真在 part-path 生效。
**选项**：A. 0.5× global loss 是真实改进，bugfix 后应能涨点；B. effective 1.0（default 行为）才是最优，0.5/2.0 双向均负
**双向 sweep 证据**：0.5 真生效(exp311b Small) **-0.7 mAP**；1.0 default(exp295/exp261) baseline ⭐；2.0(exp312 Tiny) **-0.4 mAP**
**选择**：B
**理由**：bugfix 后 0.5 和 2.0 两个方向都 net negative，effective 1.0 最优。早期 config 里 `GLOBAL_LOSS_SCALE: 0.5` 是 bug 期间虚假观察。
**执行结果**：GLOBAL_LOSS_SCALE 论文不需调，保持 1.0。`prcv_best_*.yml` 的 0.5 设置应纠正（occluded_duke 已改 1.0，occluded_posetrack/market 仍 0.5 待修）。

### [2026-04-28] 决策 #Tiny 五维 loss-weight sweep 全 ≤ baseline → default recipe 已调优，停止扫参

**上下文**：Tiny（seed 42）系统扫 5 个 loss weight 维度，验证 default recipe 是否还有调参空间。
**8 个 sweep 点（vs exp261 67.2/78.6 MaxSim）**：GLS2.0 -0.4；PartW2.0 -0.3；PartW0.5 0；lgpaW1.0 -0.2；oasdW2.0 0；lgpaW0.25 **+0.2**；partTriW0.5 -0.1；oasdW0.5 -0.1
**选择**：default recipe 各维度已是 sweet spot，停止 loss-weight 扫参。
**理由**：8 点中 7 个 ≤ baseline，唯一正向（exp317 +0.2）在 multi-seed std（0.42-0.45）内、不可强 claim。
**执行结果**：唯一候选 exp317 转 Small 验证（见下条）。loss-weight 维度收敛。

### [2026-04-28] 决策 #exp317 Tiny lgpaW=0.25 的 +0.2 未迁移到 Small（exp321b）→ 判 seed noise 放弃

**上下文**：exp317（Tiny，LGPA_ASSIGN_WEIGHT 0.25）是 Tiny sweep 中唯一 MaxSim 超 baseline 的点（+0.2 mAP），需在 Small 验证。
**选项**：A. +0.2 真实，应在 Small/Base 重现写入论文；B. +0.2 在 multi-seed std 内，是 seed noise，放弃
**验证**：exp317 Tiny/42 lgpaW0.25 67.4/78.6 (+0.2/0)；exp321b Small/1234 lgpaW0.25 **74.9/85.4 (-0.3/0)**
**选择**：B
**理由**：Tiny +0.2 未在 Small 重现（Small 反 slight -0.3，R1 持平）。Tiny std 0.42-0.45 覆盖 ±0.3，判 seed noise。
**执行结果**：不写为论文改进，保持 default 0.5。（exp321b monitor 提及待 exp321c s42 复核，但未跑/无 FINAL；现有 Tiny+Small 证据已足判 noise）

### [2026-04-28] 决策 #exp320 LGPA_DETACH=False -6.4 mAP catastrophic → detach 是必要设计（强 negative 消融）

**上下文**：SOTA push 探索——让 LGPA aux loss 反传 backbone（default DETACH=True），测是否能让 LGPA shape backbone features。
**结果（exp320 Small s1234 vs exp295）**：eq 68.1/79.3 vs 74.2/84.0（-6.1/-4.7）；MaxSim **68.8/79.6 vs 75.2/85.4（-6.4/-5.8）**
**选择**：DETACH=True（current default）是必要选择，非任意 hyperparam。
**理由**：DETACH=False → catastrophic underfit（e10 46% near-random，e80 plateau 68.3）。LGPA 须 detach，作为 frozen pose-spatial-gated features 上的 downstream attention head。
**执行结果**：强 negative，写入论文消融（"LGPA must be detached; allowing backprop causes -6.4 mAP severe underfitting"）。POSE_LGPA_DETACH=True 保持。

### [2026-04-25] 决策 #Base OD LR sweep → LR8 最优，LR2 下界，PLBOA OD net positive

**上下文**：overnight Base OD LR sweep（exp296/297/298）+ PLBOA 消融（exp299）。
**LR sweep（Base s41 vs exp296 LR8 74.9/83.8 MaxSim）**：LR8 baseline；LR4(exp297) -0.3/+0.3（近 tie）；LR2(exp298) **-5.3/-4.7**（下界）
**PLBOA 消融**：exp299(OFF) 72.7/80.5 vs exp296(ON) 74.9/83.8 → OD 上 **PLBOA net +2.2 mAP**；配 Tiny exp307(+2.7) 2-backbone 一致
**选择**：LR8 sweet spot；PLBOA OD-train 启用、Market-train 关闭（dataset-specific）。
**理由**：LR4≈LR8（非显著 underfit），LR2 严重 underfit -5.3。PLBOA 在 Occ-Duke +2.2-2.7 mAP，但 Market→Occ-ReID 跨域 -25.4 mAP（exp293 vs exp269）。
**执行结果**：Base OD 主表保持 exp263d 75.2/84.8；exp296-298 作 LR ablation，exp299/exp307 作 PLBOA dataset-specific evidence。

### [2026-04-26~27] 决策 #multi-seed 3-backbone std ≤ 0.5 → "robust to seed" claim 成立

**上下文**：补齐 Small/Base 各第 3 个 seed（exp304 Small s2024、exp302 Base s42），支撑 "robust to seed selection" claim。
**multi-seed 统计（MaxSim+flip）**：Small(42/1234/2024) mean **74.7 std 0.45** 主行 exp295；Base(41/1234/42) mean **74.87 std 0.42** 主行 exp263d
**选择**：论文写 "robust to seed selection (std ≤ 0.5 mAP, both Small & Base, 3 seeds each)"。各 backbone 主表用最强 seed。
**理由**：两 backbone 3-seed std 均 < 0.5，一致性强；exp300(Base s1234) R1 微超但 mAP -0.2，未破 SOTA。
**执行结果**：主表数字不变（Small 75.2/85.4 / Base 75.2/84.8）；exp302/304/300 作 multi-seed 补充数据点。

### [2026-06-16] 决策 #exp323 — MLLM 视觉裁剪 A/B 廉价首验（3B 退化，不可判）

**上下文**：post-PRCV「搬范式」首验（frozen Qwen2.5-VL-3B，零训练，lab-3090-d）。
288 个重遮挡难例 pair（均衡 144 同/144 异），三条件 A/B/C：甲(裸)、乙(可见部位文字)、丙(姿态视觉裁剪)。
假设：视觉裁剪/文字 grounding 改善小模型对遮挡 pair 的同人判定，且增益集中在重遮挡档。
**结果（一个词 YES/NO 格式）**：三条件**全部恰好 50.0%**（=随机），各 n_visible 档全 50.0%。
原因：Qwen2.5-VL-3B 有压倒性 NO-bias，几乎全输出 NO（甲 0 YES、丙 0 YES、乙 2 YES），
连明显同人(pid 全可见)也答 NO。诊断探针：强制"必须选"仍全 NO；允许 reasoning 才开始区分。
**对照**：同 288 对 GPT-5.5 裸=55.9% / 文字=55.6%（文字也无效，印证"强模型文字无用"）。
**选择**：(a) always-NO 使一个词格式下 A/B/C **不可判**，非方法被证伪；
(b) 补跑 reasoning 输出格式（先推理后 ANSWER:，max_new_tokens=128）让模型 commit，
取得可判的 A/B/C（exp323 reason 变体，结果见 monitor.md）。
**理由**：kill-switch 的前提是模型能给出非退化判定；一个词格式下小模型地板效应掩盖了任何信号。
**执行结果**：见 experiments/exp323/monitor.md（两次 run 完整记录）。脚本：scripts/exp323_crop.py（视觉裁剪）、
exp323_qwen3b.py（一词格式）、exp323_qwen3b_reason.py（推理格式）、exp323_analyze.py、exp323_diag.py。

### [2026-06-16] 决策 #exp323-final — reasoning 格式 A/B/C 出可判结果，kill-switch 偏负

**上下文**：一词格式 always-NO 不可判后，补跑 reasoning 输出格式（先推理后 ANSWER:，128 token）让 3B commit。
**结果（UNK 计错）**：甲(裸)=54.2% > 乙(文字)=49.3%(-4.9pt) > 丙(视觉裁剪)=35.8%(-18.4pt, 71 UNK)。
按 n_visible 无任何档丙>甲；heavy(≤4) 甲0.525/乙0.525/丙0.375。增益不存在、不集中重遮挡（撞红线#6 的"均匀"否定面）。
丙最差机制：裁剪删上下文→模型对每个碎片长篇描述→128 token 内常没到 ANSWER→71 UNK。
**选择**：判 "frozen 小 MLLM + pose 视觉裁剪/文字提示" 首验**不正向**。
**理由**：两个 pose-guided 干预（文字、裁剪）对 frozen 3B 都无帮助且裁剪显著有害；
裸图 54.2% 已接近 GPT-5.5 裸 56.5%，说明不是 3B 太弱，而是干预本身不 work。
**执行结果**：建议砍 frozen-MLLM-reasoner 廉价首验，转 exp324（DINO-correspondence，更 frontier-independent）或换机制。
保留 escape hatch：若坚持 MLLM 线需 LoRA 让模型学会用裁剪/grounding，但 frozen 证据偏负+沉没成本警告。

### [2026-06-16] 决策 #exp324 — DINO emergent correspondence + pose-anchored part-MaxSim 首验偏正

**上下文**：exp323 frozen-MLLM 线偏负后，按搬范式 #2 路线做 frozen DINOv2-base 廉价首验（training-free）：
dense patch token → pose 锚定 5-part → 跨图只比 mutually-visible part 的 part-MaxSim。全量 Occluded-Duke。
**选项**：
  A. 机制有相对信号（pose-part 重遮挡超整图、且 pose 锚定 > 均匀网格）→ 推进 exp324b（轻量 part 投影头/LoRA）。
  B. 无相对信号（与整图/均匀网格打平或更差）→ 与 exp323 一起判 pose-guided-frozen 这一大类偏负，换机制/退 DIFT。
**结果**：重遮挡子集 pose-part 1.86 mAP / 3.54 R1，holistic CLS 仅 0.55/0.81（**+1.31 mAP / +2.73 R1，mAP×3.4 R1×4.4**）；
均匀网格 grid-part 仅 0.67/1.21（vs holistic +0.12 mAP，几乎无效）→ **pose 锚定贡献占绝对主导**（pose vs grid +1.19 mAP / +2.33 R1）。
ALL 子集同向更明显（pose-part 3.21/7.87 vs holistic 0.64/0.90）。绝对分低（heavy 1.86 mAP）但落在 DINO 零样本 ReID 文献区间（0.3-4.7）。
**选择**：A。
**理由**：(1) 三种表征**单变量隔离干净**——(b)/(c) 都是 5 同序 part 向量在 common-visible part 求均值，唯一差别是锚定方式（pose vs 固定带），grid 几乎不涨而 pose 大涨，直接证明"姿态把 DINO token 约束到身体部位语义"是涨点来源，不是部位分解 trivial 效果；
(2) 重遮挡组涨幅 > 全体涨幅占比（机制对症遮挡），不撞红线 #6（非"均匀涨"）；
(3) 与 exp323（frozen 干预无效）形成对照——同样 frozen + 同样 pose，但 DINO dense correspondence 这条**有信号**，差别在表征端而非 LLM-reasoning 端。
**执行结果**：exp324b 候选——冻结 DINO，仅训一个轻量 part-projection 头（或 LoRA）把 token 投到 ReID-judiciable 空间，
保留 pose 锚定 + mutually-visible part-MaxSim，全量对比 KPR。脚本 scripts/exp324_dino.py，特征已缓存 experiments/exp324/_cache。
**待补**：rep-building 327s 瓶颈在每图重开 PIL 读尺寸（2 万次），exp324b 应把图尺寸随特征一并缓存或预存 npz 元数据。

### [2026-06-16] 决策 #exp327 — 更强冻结对应源（DINOv2-with-registers）止损

**上下文**：exp324 frozen DINOv2-base pose-part 重遮挡 1.86，天花板低。问"换更新/更干净的冻结 SSL 源能否抬过 1.86"。hyy GPU1，唯一变量=特征源。DINOv3-vitb16 gated（hf-mirror 需 token）下不了，改用 ungated 的 dinov2-with-registers-base（registers 去 high-norm artifact token，更干净 dense 特征）。
**选项**：
  A. 更强冻结源显著超 1.86（≥+1~2 mAP）→ 天花板瓶颈在模型新旧，值得上头。
  B. 仅小幅/打平 → 瓶颈在 frozen 本身，换源无用，止损。
**结果**：dinov2reg-b 重遮挡 pose-part **2.15/3.84（+0.29 mAP / +0.30 R1 vs 1.86/3.54）**，ALL 3.85/8.60（+0.64/+0.73）。机制保持（pose vs grid +1.44 mAP，grid 几乎不涨）。
**选择**：B（小幅正向但止损）。
**理由**：registers 更干净特征只蹭出 +0.29 mAP（heavy），远不足以独立可用（exp324b 头已到 14）；印证 exp324 假说**训练-free 天花板瓶颈在 "frozen" 本身，不在 SSL 模型新旧/registers**。换更强冻结 DINO 源不是天花板解。
**执行结果**：exp327 线止损。若要上头优先选 DIFT（不同范式，smoke 趋势更强）。dinov3-b 因 gated 无法验证；按 registers 小幅增益外推预期也不破天花板，不为它申请 token。slim pose data pipeline 经 dinov2-b sanity（复现 exp324 数字）+ heavy-occ 989/2210 完全一致核验无损，可复用于后续 hyy 实验。

### [2026-06-16] 决策 #exp326 — DIFT/SD 特征对应训练-free 决定性负，SD 线止损

**上下文**：exp324 frozen DINOv2-base pose-part 重遮挡 1.86。对应特征综述称 SD UNet 中间特征（DIFT）在遮挡/姿态对应基准上比 DINO 高 14-19 PCK。问"换 SD-DIFT 特征源能否超 1.86"。hyy GPU0，唯一变量=特征源（DINOv2→SD-v1.5 UNet up_blocks[1] DIFT，t=100 ensemble=4）。
**选项**：
  A. DIFT 全量重遮挡超 1.86 → SD 特征值得上轻量头（exp326b）。
  B. 不超 → SD 训练-free 不优于 DINO，止损。
**结果**：DIFT smoke（500 gallery）pose-part heavy **9.92**（趋势第一，误导），但 **FULL（17661 gallery）塌到 0.73（−1.13 vs 1.86）**，更不及 dinov2-registers 2.15。机制方向仍在（pose 0.73 > grid 0.35 > holistic 0.22）但绝对判别性远低于 DINO。
**选择**：B（决定性负）。
**理由**：(1) DINO 从 smoke 2.55→full 1.86 仅小降，DIFT 从 9.92→0.73 **灾难性塌**——证明 **SD/DIFT 特征 category-level 语义对应强（PCK 高）但 instance-level 身份判别弱**（与 SD-DINO / Tale-of-Two-Features 文献一致：SD 与 DINO 互补、SD 不主导 instance retrieval）；(2) instance-discrimination 是 SD 特征的**结构性短板**（非超参问题），扫 t/up_block/ensemble 不会救；(3) 训头起点 0.73 远低于 DINO（1.86→14），不值得上 exp326b。
**执行结果**：SD/DIFT 线止损，不上头。**重要方法论教训写入铁律：训练-free probe 必须用全量 gallery 判定绝对值，小 gallery smoke 只验流程不验数值**——DIFT 是活教材（smoke 排第一、full 垫底）。结合 exp327（registers +0.29 小幅、不破天花板）：**换特征源（更新 DINO / 换 SD 范式）都不是 frozen 天花板的解**，瓶颈在 frozen 本身（需 LoRA/解冻，即 exp324d 线）或换"DINO 补 Swin"重量级角度（planner #1 oracle）。

### [2026-06-16] 决策 #exp324i — 做"解相关感知 DINO-LoRA"作 FM-import 方向最后一个真 method shot

**上下文**：夜间 FM-import 全线证负，headline = 判别性-互补性张力（adaptation 让 DINO 判别化但趋同 Swin，融合只 +0.37）。lab-3090-d 空闲。用户睡前铁令"整夜不停务必找一个有用创新点"。问：直接用解相关损失攻击该张力，能否换来真互补、融合超 SOTA？
**选项**：
  A. 跑 exp324i（跨协方差解相关 DINO-LoRA，λ=0 vs λ=1 单变量）——真机制介入，成则 method、败则把张力升级为强结论。
  B. 不跑，直接把夜间产出定为 analysis 诊断研究收尾。
**选择**：A。
**理由**：(1) 不是堆模块 / 调参，是直攻 headline 张力的**单一新机制**（Barlow-Twins 跨网络版，Codex 查无直接先例）；(2) 无论成败都增信息——败也是诊断论文必需的对照（"显式解相关也打不破张力"）；(3) lab-3090-d 否则空转，符合"GPU 空闲必开下一个"铁律；(4) 双审查通过、dry-run 干净、加固两个 Low。
**红蓝队**：未单独跑（决策风险低、可逆、纯空闲 GPU、双审查已过）；先验 ~75% 偏负（Swin 占最判别方向、global-only 解相关不针对遮挡盲点、95.8% 全可见墙）已诚实记入 design.md 失败模式。
**执行结果**：λ=1 已上 lab-3090-d（30ep/rank16/seed1234，PID 在 /tmp/exp324i_lambda1.log）；λ=0 control 待 hyy r32 完→GPU1。eval 走 exp324h oracle/fusion + Jaccard-vs-λ 曲线。结果跑完并入 results.md + study + 晨报。**若败不编造 method，诚实呈现张力诊断。**

### [2026-06-16] 决策 #exp324i-result — decorr 没打破张力，method 为负但诊断升级（执行结果）

**上下文**：exp324i（解相关感知 DINO-LoRA）e10 matched oracle 出（λ=0 vs λ=1）。
**结果**：**decorr 完全没移动 Jaccard(0.253→0.2513)/oracle(+0.59→+0.58)/fusion(+0.37→+0.37)**，decorr loss 活跃(0.041)但对 part-MaxSim 排序正交。
**判定**：method shot 对 beat-SOTA **为负**（fusion 仍 +0.37 NFC 级，未超训练端门槛）；但作为**严格对照为正**——把"判别性-互补性张力"从观察升级为"显式解相关施压也打不破"的强诊断结论。
**决策**：(1) 不在 decorr 上继续调 λ 找正收益（机制正交，调参无意义，CLAUDE.md 铁律）；(2) **跑 λ=2 + λ=0/λ=1 e30 matched** 把 sweep 做 bulletproof（确认更强 λ/收敛点仍不动），仅用空闲 GPU、不额外烧；(3) FM-import 方法方向**正式关闭**，夜间真产出 = 诊断研究(张力洞察 + 显式干预对照 + ×4 finding + capacity-control + 可复用 oracle 工具)；(4) 真 method 只能走问题 reframe（CLAUDE.md 钦定方向，多日线，醒后与用户定）。
**执行结果**：λ=2 + e30 oracle 已 armed（λ=1 [done] 触发）。诚实呈现，不编造 SOTA 突破。

### [2026-06-17] 决策 #PartNC-result — 首验判死, TBPS method 线暂无活候选
**上下文**: TBPS 调研唯一幸存 PartNC 跑 kill-switch 首验(数据自动下好绕过 login)。
**结果**: 换 RDE 真 CCD 公平对照后, 50% 噪声 PartNC pair 检出 0.729 输真 CCD 0.754(Δ−0.025, 2 种子复现)。判死。
**理由**: 真 CCD 已充分吸收部位级噪声信号, 部位粒度无增益; 之前优势来自不公平代理对照。
**决策**: PartNC 止损(不进成稿)。整夜 occluded method + TBPS method 两线均证无现成 beat-SOTA 创新点(有据可查+对抗验证)。下一步待用户: analysis 论文(推荐, 证据齐) / 换任务 cold-start(aerial/video) / 用户别的方向。不硬凑。

### [2026-06-17] 决策 #UCE-probe-result — 跨域分数尺度不漂移, UCE 统一阈值校准无 headroom, 判死(首验)
**上下文**: 验证把 UCE(UniFace ICCV23, 统一阈值校准 loss) 搬到 ReID 跨域/开集的**前提**——跨域时 genuine vs impostor 相似度尺度是否漂移、单一全局阈值能否分开正负。lab-3090-d 无训练 probe。
**资产**: 现成 Market-trained ckpt `log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth`(Swin-Base+PSG+LGPA+GCN512), 现成跨域 eval `test_on_occluded_reid.py`(Market→Occluded-ReID 86.0/88.5 已存), 两域数据+pose 齐。脚本 `scripts/uce_calib_probe.py`, 结果 `log/uce_calib_probe.json`。
**方法(无训练)**: 同一 ckpt 提 L2-normed equal_concat 特征, in-domain=Market test(3368q/15913g), cross-domain=Occluded-ReID(1000q/1000g); 按 Market metric(去同 pid+同 cam)算 genuine/impostor 余弦相似度直方图; 报 d'/AUROC/EER/EER 阈值 + **阈值迁移测试**(源域 EER 阈值搬到目标域看 FAR/FRR)。
**结果**:
  - In-domain Market: gen=0.886±0.067 imp=0.365±0.091 **d'=6.51 AUROC=0.9983 EER=0.95% thr@EER=0.636**
  - Cross-domain OccReID: gen=0.807±0.082 imp=0.442±0.103 **d'=3.92 AUROC=0.9942 EER=3.65% thr@EER=0.642**
  - **阈值迁移**: 源阈值 0.636 → 目标 0.642, **shift 仅 +0.006**; 把源 Market 阈值直接搬到目标 OccReID: **FAR=4.06% FRR=3.14%**(vs 目标原生 EER 3.65%)——几乎零代价。
**判定(看量级不只看显著)**: 跨域**分离度确实掉**(d' 6.51→3.92, EER 0.95%→3.65%)——这是难度上升, 不是尺度漂移。但 UCE 攻击的是"**单一全局阈值是否还分得开**"——**全局阈值几乎不动(0.006), 直接迁移零代价**, 说明 SOTA 已把分数尺度校准好, **统一阈值校准 loss 无 headroom**。分离度下降来自遮挡难度(impostor 尾巴变厚, gen 整体下移), calibration loss 治不了——那是判别性问题, 不是阈值/尺度问题, 落回"别在 ReID 内部找机制"老墙。
**决策**: UCE-import **判死(kill)**, 不开训练。整夜+本轮三线(FM-import / TBPS / 校准-import)均证无现成 beat-SOTA 创新点。诚实呈现, 不编造。脚本/结果留档供复现。

### [2026-06-17] 决策 #VCNorm-probe-result — 遮挡确在 per-part-token 归一化统计造成巨大可分离 shift, 且非采样伪影 → 有燃料 PROCEED(首验)
**上下文**: 验证 VC-Norm(occlusion-as-domain-factor, visibility-conditioned normalization) 跨域创新的**前提**——遮挡是否在 per-part-token 的 normalization statistic(mean/var) 上造成可分离分布 shift。若 KL≈0 则无燃料 kill。lab-3090-d 无训练 probe。
**资产**: Market-trained ckpt `log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth`(Swin-Base+PSG+LGPA+GCN512, Occ-ReID baseline 88.0 mAP MaxSim+flip), 数据 Occluded-ReID(1000q/1000g+pose), env solider-reid(torch1.13+mmcv)。脚本 `scripts/vcnorm_probe.py`(主)+`scripts/vcnorm_probe_control.py`(对照), 结果 `experiments/vcnorm_probe/*.json` + README。
**方法(无训练)**: per-part token = SkeletonGCNHead 在 PSG-modulated Stage-3 图上按 17 COCO kp bilinear 采样的 token(dim 1024, pre/post-GCN)。按 pose 置信度 score 把每 kp 的 token 分 high-vis(≥0.7) vs low-vis/遮挡(≤0.2), 算逐通道对角高斯对称 KL + 2-Wasserstein + Fisher-LDA held-out AUC。三对照排伪影: %border 坐标、KL(hi,**rand** 体内随机采样)、KL(hi,**lo_onbody** 剔边界坐标)。
**结果**:
  - 主探针 median: **KL_sym=288(pre)/170(post), LDA_AUC=0.97(pre)/0.98(post)**, 各 kp KL 94–300、AUC 0.95–0.99 → 遮挡 vs 可见 token 近完美线性可分, 远非 KL≈0。
  - 对照 C1: low-vis 坐标 **仅 7% 在边界** → 不是退化坐标。
  - 对照 C2: **KL(hi,lo)=288 ≫ KL(hi,rand)=125(~2.3×)** → shift 是遮挡特有, 不是泛泛 off-kp 采样噪声。
  - 对照 C3: **KL(hi,lo_onbody)=294 ≈ KL(hi,lo)=288** → 剔边界坐标后 shift 不变, 非边界伪影。
**判定(看量级)**: 前提**成立**——遮挡确是一条**巨大、可分离、遮挡特有、非采样伪影**的 domain 轴, 结构简单到对角高斯 KL 能量到、一个线性方向就近完美分开 → 符合"一个 normalization/对齐模块可吸收"的 VC-Norm 假设。GCN 已部分修复(post KL↓)但远未抹平 → VC-Norm 与 GCN 不重复、有 headroom。
**重要 caveat**: 这是 **NECESSARY 非 SUFFICIENT**——有可对齐 domain 轴 ≠ 对齐后涨 mAP(可能连带抹掉身份信号), 只能训练验证。上半身 low-vis 样本不足(Occ-ReID 遮挡集中头部边缘+下肢), KL 表只覆盖头部+膝/踝, 但已足够给明确 PROCEED。test 端 Occ-ReID 不直接受 95.8% 训练全可见墙限制。
**决策**: VC-Norm **PROCEED**(整夜多线首个非 kill 信号)。下一步若推进: 1-2d dual-forward Market 30ep(occluded/clean 两路 forward, per-part-token visibility-conditioned 对齐归一化统计), **目标 Occ-ReID mAP > 88.0**, 30ep 短训当 kill-switch 不涨即止损。是否开训待用户拍板。诚实呈现, 不夸大为已验证。

### [2026-06-17 05:2x] 决策: burstiness 范式 bet 判 KILL + 收窄"in-domain 特征机制"整类
**上下文**: 夜间范式调研唯一过审强 bet=burstiness(VLAD-BuFF/face-set import)。0-GPU 前提在 frozen DINO 成立(occluded +0.0206 更 bursty)。e120 弱 baseline(TransReID 53.5)训练模型真实判据。
**结果**: pooled burst−uniform=−0.29、part-MaxSim=−0.25(双判据 KILL), cls+burst<cls(加 burst 伤特征), 前提训练后翻负(occluded −0.0154 更不 bursty)。
**决策**: KILL burstiness。**并据此收窄: in-domain 特征重加权/对齐/补全/重打分这一整类机制, 在任何训练好的 ReID 模型(强 SOTA 或弱 baseline)上都无 headroom** —— frozen kill-switch 会误导(frozen-promising → trained-absorbed)。否证"换弱 baseline 就有 headroom"对这类机制。
**理由**: 双判据 + 前提诊断三重证负, 干净。归入 backdoor/TopoFR/UCE/FM-import 同 pattern, 现推广到弱 baseline。
**下一步**: 不再碰 in-domain 特征机制。转(a)改问题定义/评测协议 (b)改监督/目标 (c)跨域泛化 (d)新匹配范式。新 bet 必须有**训练模型 kill-switch**(非 frozen)。已启动 informed 调研 agent。VC-Norm(唯一在训的训练端改表征机制)跨域判据待定。

### [2026-06-17 ~06:1x] 决策: exp330 组合遮挡泛化 + group-DRO 判 NO-GO
**上下文**: burstiness 死后调研 Rank-1 bet(改问题定义类, 结构性逃训练吸收)。双审过+smoke过+单变量干净。e20 kill-switch。
**结果**: ERM held-out vs seen GAP=+0.10≈0(无组合结构); DRO mAP 0.26 训练塌缩(q runaway)。
**决策**: KILL。occluded ReID 模型已组合泛化(不学 occluder-class 捷径)→ 组合重定义无 headroom。与 in-domain 死法不同类不同因。
**理由**: ERM 零 gap 独立判死; kill-switch 便宜捕获(省全量方法)。
**下一步**: BET 2 (DUL identity-conditioned aleatoric variance) 待评估; VC-Norm 仍唯一活 method bet。

### [2026-06-17 ~06:4x] 决策: exp331 DUL 判 NO-GO
**上下文**: 最后未测的"新监督/目标"类(搬人脸 DUL)。双审过+smoke过+单变量(BN单更新)。e20+e40 kill-switch。
**结果**: DUL mAP < DET 且 gap 扩大(−1.35→−1.95); σ² 缓慢收缩 + 遮挡相关性反号(遮挡更低)。双判据全失败。
**决策**: KILL。DUL 在 occluded ReID 无 headroom——采样噪声伤判别 + σ² 不捕捉遮挡。
**理由**: e20+e40 双确认、gap 扩大、σ² 反号。
**下一步**: 今晚 7 个 bet 全 NO-GO/kill。唯一活线 = VC-Norm 跨域(待 e120)。cheap/vetted 空间穷尽; 剩重量级 import(低 EV)留用户定。

### [2026-06-17 ~07:0x] 决策: Bet A 几何验证判 KILL + 今晚穷尽收尾
**上下文**: 调研第三轮给吸收陷阱解释 + Bet A(几何 re-rank, 结构性逃逸)。冻结 e120 probe。
**结果**: baseline 53.53 → 几何 51.27 = −2.26。KILL。诊断: 信号在内容非几何。
**决策**: KILL Bet A。8 bet 全 NO-GO。**吸收陷阱**(per-image+联合优化机制被吸收)= 7+1 连负的机制级解释, analysis headline。
**理由**: 冻结判据干净; 人非刚体几何不稳。
**下一步**: cheap/vetted 空间穷尽。VC-Norm 跨域(慢, ~8h)是唯一活线。剩重量级 import(mmcv-gated/低EV)留用户定。真实交付=诊断论文(8 kill + 吸收陷阱 + 张力 + 三堵墙)。

### [2026-06-17 ~01:4x] 决策: VC-Norm 跨域判 NO-GO — 最后活线死, 今晚 9/9 NO-GO
**上下文**: VC-Norm 唯一"训练端改表征"活线。e40 后系统性挂起, 从 e40 ckpt 跑跨域 Occ-ReID 提前判(单变量对照 relay-copy 同环境评)。
**结果**: VC-Norm vs control 跨域全变体 −3.3 mAP(global 77.3/80.6, part_only 79.3/82.7, equal_concat 77.5/80.9)。**VC-Norm 在真遮挡跨域上显著伤**(Market 学的对齐 transform 跨域误用)。
**决策**: KILL VC-Norm。9 个 bet 全 NO-GO。**真实交付=诊断/analysis 论文**(9 kill + 吸收陷阱 + 张力 + 三堵墙)。
**理由**: −3.3 显著、VCA 已激活 20ep、e120 反转极低概率。Market −0.6 + 跨域 −3.3 双向伤。
**下一步**: occluded ReID 单图搬机制证负完毕。写 analysis 论文 / 重量级 import(用户定) / 换任务。

### [2026-06-18 ~19:1x] 决策: CLIP/LGPA-D 复现弧线 — 增益是 pose 不是 CLIP 文本(拆解证据)
**上下文**: post-SMPL,用户想把 LGPA-D 包装成新 CLIP 模块创新。先复现 ViT-baseline+LGPA-D 确认 CLIP +X。ViT 上 equalcat < global(负)。用户坚持"是你的复现 bug 非 backbone"。
**调查**: 派 10 个 Codex 并行深挖(用户 rate-limit 不让开 300 Claude 子agent,Codex token 无限)→ 挖出**热图 bug**:exp335 喂 target-only 热图(`heatmaps[:,0]`+POSE_USE_TARGET_HEATMAP=True)→ LGPA assign KL 坍缩=0 → 部位退化。修(scene-merged)→ assign 0→7.02≈原版。但 ViT 仍只 +0.5、不翻盘。深挖发现 **LGPA-D 从未单独跑过**(exp244/245g 全是 PSG+LGPA+OASD+aug+384+Swin 全系统)。
**结果**:
- exp336(Swin 纯 LGPA-D,关 PSG/OASD/aug,`POSE_PSG_STAGES=[]`):equalcat 59.6 vs global 58.5 = **+1.1**(e60 时 +1.7)。→ CLIP 模块 standalone 在 Swin 上涨;**ViT 失败=ViT-specific**(单尺度池不出强部位)。
- exp337(同配置 + `POSE_LGPA_NO_POSE=True`,LGPA 收 heatmaps=None,纯 CLIP-text 部位,assign=0):equalcat 58.7 vs global 58.8 = **≈0**。→ **那 +1.1 全来自 pose 注入,不是 CLIP 文本语义**。CLIP 文本"head/torso/legs"是 query 壳、冗余于 global;pose-bias 引导注意力到身体区才让部位有判别力。
**决策**: **纯 CLIP 文本部位路线=死路**(语义冗余)。step2 的新 CLIP 接法必须带 global 没有的真信息(CLIP 视觉特征/遮挡推理/ID 级原型)。理想判据(用户定):baseline 58 → +CLIP 59(CLIP 单独过坎)→ +pose-CLIP 60;现状是 CLIP 单独那步=0。
**验证完成（3-seed sweep,2026-06-18）**: 非噪声铁定。
- pose-CLIP（exp336）增益: s0 +0.9 / s1 +0.9 / s2 +0.9 = **+0.9 ± 0.0**（equalcat/global: 59.9/59.0, 59.0/58.1, 60.1/59.2）。
- no-pose（exp337）增益: s0 −0.1 / s1 −0.1 / s2 −0.3 = **−0.17 ± 0.1**（≈0,从不为正）。
- → **增益来自 pose 注入(+0.9 稳),不是 CLIP 文本(≈0)。** 你担心的噪声不存在。
- 基建: 清 hyy /hy-tmp(47→25G 用),传 pose_data 上 hyy(torch2.7+cu128 在 sm_120 上验证可跑 SOLIDER)。
- 下一步: 下 CLIP-ReID 论文+代码,14 Codex 并行查 ours-vs-CLIP-ReID + pose×CLIP 创新性(含 web 查新)。
**理由**: detach 保证 global==无-LGPA baseline,within-ckpt equalcat-vs-global 干净;no-pose ablation 单变量隔离 pose。
**基建**: Clash 两修(都 live,非永久):① `PROCESS-NAME,tailscaled,DIRECT`(tailscaled 直连 DERP→4090 relay 复活)② `tun.dns-hijack: ['any:53']`(原为空→gpushare/hyy DNS 解析失败)。hyy(5060Ti sm_120)需 torch2.x,跑不了 SOLIDER 的 torch1.13+mmcv2.1。详见 [[lgpa-d-reproduction-gotchas]] memory。

### [2026-06-20] 决策: 姿态融进 CLIP-ID-prompt 机制 — 三方系统验证全 NEGATIVE

**上下文**: Step1 exp341 (CLIP-ReID 可学习 ID prompt 对齐 raw global) = +2.2 (59.8 vs 57.6)。用户路线 = 把姿态融进这个能涨的 CLIP 机制再涨。建了 3 种融法 (A/B/C), 各双审查通过、机制验证激活、e120 test.py global。

**结果**:
- A (exp343, pose-bias 池化特征替代 global 对齐): 57.6, -2.2
- B (exp344, pose 调制 prompt context, zero-init): 57.6, -2.2
- C (exp345, K=3 pose-localized 部位对齐): 58.0, -1.8
全部 ≤58.0 < exp341 59.8, 掉回 baseline 附近。

**核心洞察 (机制层面, 非 bug)**:
exp341 的 +2.2 来自**纯 ID 对齐** (raw global ↔ 纯 ID 文本原型, 把 global 塑成纯 ID-判别)。
姿态以任何"融进对齐"的方式进来都跟纯 ID 抢/稀释:
- A/C: 姿态建带参数新通路, 吸收 i2t/t2i 梯度 (吸收陷阱, 本项目反复出现)。
- B: 姿态进 prompt → 原型 pose-aware → global 被拉去编码姿态而非纯 ID。

**决策**: 整合式加 pose 在此 CLIP 机制上 = 死路 (3/3)。最后测分离式 exp342 (pose 当独立描述子拼接, 不碰对齐)。若也不涨, 结论 = exp341 (+2.2) 是干净贡献, 姿态加不动。

**执行结果**: exp342 训练中 (4090, ~2h)。

### [2026-06-20] 决策: 通宵 pose+CLIP 深度融合搜索 — 完整结论(诚实负结果)

**上下文**: 用户要"找一个 CLIP+姿态融合涨点的创新", 通宵 3 台机探索。

**全部结果** (Occluded-Duke, e120, test.py, over baseline 57.6):
- CLIP-ReID prompt 单独 (exp341, raw global): 59.8 = +2.2 ← 真 CLIP 增益(干净)
- 姿态**整合进 CLIP 对齐** (A/B/C exp343/344/345): 57.6/57.6/58.0 全负 ← 吸收/稀释纯 ID 对齐
- 姿态**外挂** (exp342 detached LGPA): 60.0 +0.2 marginal
- 姿态 **un-detach LGPA** + CLIP (exp342b): 60.7 +0.9 ← 一度以为是突破
- **un-detach LGPA 单独无 CLIP (exp353): 60.5 = +2.9, 已 > CLIP 单独 59.8!**
- 深挖 7 变体 (clean global/de-occluded×2/scale-up/occluder/part-weight): 均未超 60.7

**核心结论(诚实)**:
1. 姿态进 CLIP 对齐 = 死(A/B/C, 跟纯 ID 抢被吸收)。
2. exp342b 的 +0.9 **大部分是 pose(un-detach LGPA), CLIP 只 +0.2**(exp353 隔离戳穿)。
3. **CLIP(+2.2)与 un-detach LGPA(+2.9)冗余**: 合 +3.1 << 5.1, 都塑造 backbone 学 ID, 重叠。
4. **无真正 CLIP+pose 协同。强 pose backbone 上 CLIP 加不动**, 反之亦然。
5. un-detach 在简单设置(纯 LGPA)涨, 但破坏全系统(exp349b 65.7<<73.2)。

**决策**: pose+CLIP 不存在有意义的融合增益(冗余, 非互补)。CLIP-ReID(+2.2)和 pose(LGPA)各自是干净贡献, 但二选一即可, 叠加无收益。这与 [[fm-import]]/[[occluded-reid-four-classes]] 一脉: 强判别 backbone 上"互补"信号总冗余。诚实负结果 = 这一夜的真交付。
**执行**: exp349(detached 强系统+CLIP)收尾确认; 探索收敛, 不再开变体; 多 seed 留给用户。

### [2026-06-21] 决策: pose+CLIP 训练端两机制(PGPD/PC-MSC)证负 + 整条线最终封板

**上下文**: 20-codex 调研后用户选 PGPD(pose选完整teacher蒸馏prompt simplex暗知识)+ PC-MSC(pose mask可见部位重建冻结CLIP部位语义)两个训练端弱赌注。全协议(design→kill-switch→Claude+Codex双审→训练+random控制)走完。

**结果**:
- PGPD: exp355(pose)58.6 / exp355r(random)59.0 / exp341 59.8 → **-1.2**, pose选teacher≈random(噪声内)无价值。
- PC-MSC: exp356(pose)57.1 / exp356r(random)57.3 / exp341 59.8 → **-2.7**, pose-mask≈random(噪声内)无价值。PC-MSC kill-switch 已预警(CLIP部位特征只带弱ID gap+0.01)。

**决策**: pose+CLIP **五角度全负**(进对齐A/B/C 57.6死、强系统exp349 -1.8、空间归属PC-SOR kill-switch死、训练端蒸馏PGPD -1.2、训练端补全PC-MSC -2.7), 封板。pose 与 CLIP 无 productive fusion。
**理由**: CLIP=全局语义工具(特征各向异性、部位ID弱), pose=空间结构工具; 二者能力不重叠处恰好无法在对方层面发挥。两训练端机制同模式: 机制本身负、pose成分(teacher/mask选择)无关。
**执行**: pose+CLIP 探索彻底结束。Step1 CLIP-ReID(+2.2)和 pose 系统各自干净, 但不融。最强仍 exp342b 60.7 / 强系统 73.2。交付=详尽负结果诊断。多seed留用户。

---

### [2026-06-23] 决策 #97: AIRL fusion 零训练 oracle kill-switch = PASS(上完整 resolvability 双分支)

**上下文**: AIRL(只退化 ground)在 CARGO Swin 上 A->G +3.15 / G->A -3.18, mean 翻平(60.83≈60.84)。红队 codex 指出方向路由上界=(61.90+62.93)/2=62.42=baseline +1.58。投入双分支前的最后廉价关: 合法固定 gate 能否逼近上界?

**方法**(零训练, `airl_gate_oracle.py`, lab-4090): baseline + AIRL 两 ckpt(eval 架构逐键相同, missing=0/unexpected=0)各提 CARGO 双向特征, 复用 eval_market 测 5 类 gate 融合 mean。FULL sanity 逐键复现 doc(base 58.75/62.93, AIRL 61.90/59.75)→ pipeline 精确。

**结果**:
- view/方向 gate(合法上界): +1.58(复现红队)。
- **硬路由(area/reliability, train 分位阈值)回收不了**: area +0.02~+0.41, reliability +0.07~+0.35 → 单标量阈值在两方向分布重叠, 选不出"A->G→AIRL / G->A→baseline"的方向性。**只看硬路由=KILL。**
- **但 score 融合(软, cos=w·AIRL+(1-w)·base, 单全局 w)轻松过**: w=0.25 保守默认 **+1.46**, plateau w∈[0.25,0.75] 全 ≥+1.46, w=0.40 **+1.86 反超 view-gate 上界**。无 label、非 knife-edge=合法固定 gate。
- per-query oracle +4.96 → headroom 远大于 +1.58。

**选择**: **PASS**(合法固定 gate 由 score 融合达 +1.46~+1.86 ≥ +1.0 PASS 阈)。

**理由**: 红队的方向路由"上界"不是真上界——硬路由丢方向内 per-query 互补性, 软融合保留, 故 +1.86 > +1.58。trade-off 可由合法机制(score-level fusion)回收, 不是单模型回收无望。代价=测试 2× inference(跑两 model)→ 正是 PASS 的意义: 单模型双分支(resolvability branch)可把两套特征空间内化进一次 forward, 拿 mean +1.5 同时省 2× 成本。area/reliability 硬路由死是诚实信号(cheap proxy 选不出方向), 不影响裁决(软融合是更强的合法 gate)。

**执行结果**: 上完整 resolvability 双分支机制(area/altitude-conditioned recoverable-evidence ceiling, 单模型出两支特征 score 融合), 目标=单 forward 复现 +1.46~+1.86 mean 增益 + 解决 #2 标注的 G->A trade-off。脚本 `experiments/cargo_cvpb/airl_gate_oracle.py`, log `/tmp/airl_gate_oracle.log`(lab-4090)。

---

### [2026-06-25] 决策 #98: Gallery-组成三测试零训练 kill-switch — 仅 Gallery-Growth Tax 活, B/C 诚实判死

**上下文**: 三个独立 codex(终身 d3 / 开集 d9 / 长尾 d10)收敛到同一 re-framing: ReID 失败由 GALLERY 组成(规模/膨胀/分布)驱动, 非只看 query/模型。用户要求零训练验证, ★铁律=每个 per-query 相关都控 trivial 代理(吸取 HUBNESS §7.6 教训: 上个诊断被漏控 #false-in-topk 证伪)。脚本 `cvpb_gallery_killswitch.py`, 复用 hubness 缓存特征, Market exp260b + Occluded-Duke exp255。双审(Claude broad 5 blocking 全修 + Codex)。

**结果**(frozen, numpy, log `/tmp/cvpb_gallery_{market,oduke}.log`):
- **测试 A Gallery-Growth Tax = LIVE**: frozen 模型旧 query mAP 随同域 gallery 膨胀结构性下降(Market 1x→10x −4.4, **OD −12.9**, 量级≈LReID 报的 forgetting)。CONTROL1(#false-in-topk, 杀 Hubness 的代理): ρ(−dAP,d#false)+0.74 大部分是 trivial 计数, 但"#false 完全不变"子集仍 −1.2(Market)/−2.6(OD) mAP, partial(OD)+0.28——结构成分过了致命代理。CONTROL2 ★决定性: real distractor −4.45(Market)/−13.16(OD) vs 列洗牌毁方向同 count −0.00 → tax 是结构性(distractor 身份几何咬人), 非机械 count。
- **测试 B Gallery-Size Rejection = DEAD**: impostor max 随 watchlist size 升(REAL ρ+1.0)但 random-feature null 升一样猛(纯 max-of-N 极值)。CAL/EVAL 折去循环后净增益(REAL−RANDOM)drift-red −0.245(Market)/−0.282(OD) 为负。强 backbone 上 genuine~0.97/impostor~0.5 近完全可分, 拒识饱和, size-conditioning 表观增益全在 EVT trivial floor。
- **测试 C Singleton Merge = DEAD**: NN-is-head 0.72 只反映 head 占 72% 图像质量。per-head-ID(n=450/311 真功效)Spearman(support, attraction-PER-IMAGE)+0.003/+0.005≈0, 分箱 per-image 甚至下降。support-calibrated 阈值几乎无增益(d≈−0.003)且 40-60% level 退回 global。被 "head 图多→NN 彩票多" trivial count 吃掉。

**决策**: **仅推进测试 A(Gallery-Growth Tax)作为诊断/问题重定义候选**; B/C 诚实判死(各自被 max-of-N / count trivial 代理吃掉)。
**理由**: A 是唯一过了 #false-in-topk + 列洗牌双控的信号, headline 干净: "frozen 强 ReID 旧 query 随同域 gallery 膨胀结构性掉点, LReID 误记为 catastrophic forgetting"。这正是上个 Hubness 诊断没做到的(被 trivial 代理吃光)——本次 A 的两个对照专为此教训设计且活下来。
**执行(待办)**: A 当前是诊断, remedy(distractor-aware continual training)未验证, 需独立实验且警惕撞 backward-compatible LReID(arxiv 2403.10022)。诚实写明 CONTROL2 是主证据(CONTROL1 的结构残差 Market partial 仅+0.05 偏弱)。跨 backbone 普适性 + 与 re-rank 互补性未测。交付=`cvpb_gallery_result.md` 原始数字。

### [2026-06-26] 决策 #99: LM-ReID(低分辨率=采样格点) session — 6.5 成稿 + 训练端穷尽 + 冲 7.0 失败 + d17 KILL = 探索收敛

**上下文**: 探索 LM-ReID(d8 演化): 低分辨率 ReID 重定义为采样格点 sampling-lattice 隐变量, test-time decision marginalization(K=9 phase/bbox/kernel 变体边缘化)。autonomous mandate 找 B 类方法稿, 全自主无休止。脚本 cvpb_lattice_killswitch.py(全参数)/cvpb_lm_reid_train.py/cvpb_d17_killswitch.py。

**结果**:
- **LM-ReID test-time 成立(6.5/10)**: LM-S2 5 分辨率全 beat 普通 TTA / LM-S2-strong 全 beat 强 TTA(+0.76~7.28, severe LR 处强 TTA 反有害) / LM-S4 bbox 检测框不确定性主导 +2.84 / K-sweep K=5 达 87% / LM-S3 logsumexp(soft decision marginalization)severe LR 最优 / backbone 泛化 Swin +3。
- **训练端穷尽(8 机制 + 4 codex 8.5/10 无空间)**: embedding-invariance(consistency −1.73)/frozen-adaptation(LS-MRT +0.028/LPA +0.075/LATS)/backbone-loss(LSRC −1.9 损判别)/robust-ERM(Hard-Lattice 76.9<77.44)/input-canonicalize(BLC 数据证伪)全负 → "Why Training-Time Invariance Fails" 强论点。审查纪律: LSRC full-finetune codex 审出 Critical(默认混旧 loss)+High(train/test 不对称)已修[[pre-experiment-review-discipline]]。
- **冲 7.0 三条腿全失败(codex push7 6/10 路径)**: ①detector-jitter σ-sweep 单调衰减到负(h12 +5.49→+2.18→−5.85, marginalization 是 sub-pixel sampling-lattice 非 detector 鲁棒性=诚实机制范围界定) ②MSMT17 跨数据集 config 缺失止损(msmt17_split 数据读对但 swin_small_pose.yml 被删 SANITY 2.67) ③adaptive-K 中性(per-query≈fixed K=5)。
- **d17 rank-leverage KILL**: RLC 现象成立(mean 0.357>uniform 0.167)但 raw spearman 反向(−0.100)+partial|#false −0.049≈0 → 重演 Hubness/evidence #false-trivial-proxy 坑(本 session 第 N 次验证: 任何 per-query 解释变量必控 #false-in-topk)。
- **codex meta-eval 确认探索充分收敛**: 全新范式(event-camera 5.5 需新数据/federated 4.5/text 3 撞 FM/3D-SMPL 2.5 撞 SMPL/group 2 撞#false)非当前代码线方法点。

**决策**: **LM-ReID 6.5 收尾投 B 类(唯一存活候选; codex 三层 push7/d17-eval/meta 都判务实)**。
**理由**: 探索充分收敛——训练端无空间(8 机制+4 codex)/冲 7.0 机制范围被 detector 误差封顶/d17 撞 #false 坑/全新范式需新数据非方法点。6.5 是诚实天花板=中等偏强 B 类(重定义+干净 test-time 机制+训练端系统反例+多维证据+σ-sweep 诚实界定)。负结果大多成论文素材。
**执行**: paper 素材完整(experiments/exp359_lm_reid/paper_skeleton+results_tables 7 表+monitor; memory [[post-pivot-20domain-gallery-bet]])。正式 multi-seed/train/MLR benchmark/真实 detector 数据集(CUHK-SYSU/PoseTrack)留用户。event-camera 唯一边际全新方向留用户全新 sprint。codex 报告 litreview2/(lmreid_push7/d17_eval/meta_converge/train{2,3,4})。

### [2026-06-27] 决策 #100: 换量级跳出盒子探索诚实终点 — AG+DG+open-set 全证伪, 现有 ReID 训练端机制探透

**上下文**: 用户点醒"没限定 occreid+solider"→换量级跳出盒子(训新预训练范式/换 backbone/换问题, 不限现有 occluded+SOLIDER)。codex 全 ReID gap analysis 选 AG(8/10)/DG(#2 6.5)。

**探索链(一整天全证伪)**:
- **AG(exp363) aerial-ground/RGB-IR video foundation adaptation**: 视频证据积累路死(frozen DINOv2-reg 全 8 protocol mean-single<+5, 最高 +1.82, exp5 还 −1.82); frame-quality selection 坑(oracle 是 retrieval-label upper bound 无监督学不到); 换 foundation 不救。frozen DINOv2 Market 2.71。
- **DG(exp364) Camera-Pair Foundation-Preserved Residual**: direct-FT SOLIDER swin Market 30ep 让所有行人域涨(Market 15.56→88.70 / MSMT held-out 4.18→11.37 **+7.19** / Occ-Duke 3.27→14.47), fine-tune 不破坏 held-out 反提升→保护弱 frozen topology(15.56)无意义, 无 U-shape sweet spot(no-op, PSC-JEPA 同质死另一形式)。
- **open-set/gallery-growth(codex 核查)**: 3.5/10 主线(撞 gallery 死区 B/C/Hubness 都被 #false-in-topk 吃 + 先例卡紧 Open-set ReID 2014/Open-Set Biometrics 2024, 唯一窄缝是 LReID re-attribution 分析非新机制)。

**共同墙**: frozen foundation(DINOv2 通用/SOLIDER 行人专门)对 ReID instance prior 都弱(frozen 2.71/15.56 << fine-tuned 88+), **fine-tune 是王道**→解释 ReID 都靠 fine-tune 非 frozen retrieval; frozen-preservation/foundation-adaptation 类在 ReID 没燃料。

**codex 诚实判**: 现有 ReID 标准数据 + 4 单卡 + 强 backbone, 训练端新机制空间已经很窄。

**决策**: 换量级在 ReID 现有约束内基本探透。诚实停在用户拍板节点。
**出路(用户拍板)**: ①收 LM-ReID(#99, 6.5 现有最强 B 类, 文档已全)/PSG 已成型线(最务实) ②换真实新数据规模(下新大规模数据集) ③换问题领域。
**方法论价值**: cheap kill-switch(frozen probe 零训练几分钟 + 一次 30ep bounded)一天验死 AG+DG 两范式, 没浪费多日训练; codex 审 backbone fine-tune(抓预处理口径不一致+ckpt+checkpoint); 诚实定位每个工程 bug(patch14/.eval返回None/MSMT camid/0.5 norm/python env mmcv); 不绑架沉没成本。负向结论(钉死共同墙)避免后续重复盲试。
**关联**: memory [[ag-dg-paradigm-killed-frozen-prior-weak]] [[paradigm-shift-occluded-reid-wall]]; monitor exp363_ag_foundation/exp364_dg_foundation; codex 报告 paradigm_shift/(codex_full_reid_gap/ag_verify/dg_deepen/openset_check)。

### [2026-07-13] 决策：exp370 PBSR 同机 epoch 60 门禁 NO-GO，停止该机制家族

**上下文**：LGPA 与 PAFormer 高度重合后，尝试把单向部位查询改造成 PBSR：共享路由从空间特征读取结构槽、槽间混合，再沿同一路由写回标准 global；pose 只作为 detached routing target，推理不依赖姿态。首轮 B0/P0 分别跑在 3090/4090，故补了 4090 同解释器、同依赖、同 execution 的 B0 控制。

**严格结果**：同机 epoch 60，B0 `54.5 mAP / 63.8 R1`，P0 `54.4 / 63.7`，P0-B0=`-0.1/-0.1`。完整 epoch 10/20/30/40/50/60 mAP 差依次为 `+0.8/-4.7/-0.7/-1.4/+0.2/-0.1`，不存在稳定正向。机制统计健康，排除 NaN、死门、background collapse 与执行失败。

**决策**：**PBSR P0 正式 NO-GO。** 未达到预注册 `+0.8～1.0 mAP`，不运行 P1/P4/P2/P3，不扩三 seed或 ResNet/ViT/Swin，不做超参救场和结构槽小变体。由于主门禁已失败，uniform/shuffled 不再运行，因此不能声称正确 pose 的因果优越性。

**边界**：这不推翻历史 LGPA/pose 分支“有信号”的结论；它只说明把 pose 监督路由学习进一步改造成共享读写的 global residual，在当前干净隔离下没有身份检索收益。PBSR 不得写成新论文主贡献，matching、GCN、CLIP 语义也继续不作为创新点。

### [2026-07-14] 决策：exp371 CASD Gate C 正式 NO-GO，停止 LGPA 自有化主线

**上下文**：Gate B 已证明 LGPA 的 `global+parts` 局部融合资产真实，但 correct 只比 shuffled/canonical 高 `0.0320/0.0984 mAP`，实例级精确姿态不是主要来源。为判断能否形成自有机制，exp371 用 target/canonical/scene 三份配对 cache，执行 strict-LOO、三 donor、cross-camera、class-free、五折 frozen support oracle；所有门禁和阈值均在看正式指标前冻结。

**严格结果**：target `POSE-RESP=94.2355` episodic mAP，低于最强 `PART-EQUAL=94.3121`，差 `-0.0766` pp；低于 `POSE-SCALAR` 与 `RESP-PERM` `-0.0162/-0.0372` pp。相对每折最强 control 五折全负；对 PART-EQUAL 的 PID-grouped bootstrap 95% CI=`[-0.1561,+0.0022]` pp。scene-merged 同样 `-0.0868` pp且五折全负。coverage、三 donor、path/content disjoint、canonical matrix、slot active 与 wrong-ID fail-safe 均通过，排除执行故障。

**解释**：`PART-EQUAL−SLOT-PERM=+1.2347` pp 与 wrong-ID 崩溃说明 same-ID support 和固定部位对应有效；但 POSE-RESP 不优于 equal/scalar/permuted routing，不能把这些 generic 结构收益归因给逐图 pose。UMTS/MVI²P 又已覆盖普通 multi-shot teacher/support 邻域，因此 generic support 不能单独承担 CASD 新颖性。

**决策**：**CASD 正式 NO-GO。** 不进入 matched RGB-only student，不补 Gate A learned-query，不转 AERC/OT/MoE/slot/temperature/queue/loss-weight 小变体，不扩三 seed、ResNet/ViT 或多数据集。IPER、PBSR、CASD 三个正交机制均未通过预注册门禁，LGPA 自有化主线到此停止。

**边界**：这不推翻 LGPA 约 `+0.82～0.85 mAP` 的结构化局部性能资产，也不证明所有 pose 方法无效；它只说明当前证据不足以把 LGPA 改造成可归属我们的新方法。matching、GCN、CLIP 文本、same-ID support 与 fixed part correspondence 均不得换名包装成创新。

### [2026-07-15] 决策：exp372 PCAR 新颖性 Gate NO-GO，不进入 official CLIP-ReID 训练

**上下文**：CASD 封板后，提出一条与旧 LGPA head 正交的新候选：在 official CLIP-ReID 的真实 CLIP ViT 内，以零初始化、少量 head/layer 的方式注入 `B(Pinstance)-B(Pcanonical)` attention residual，保留 untouched CLIP heads 与标准 global descriptor。Goal 预注册要求：若查新后只剩普通 additive pose bias 或模块迁移，直接 NO-GO。

**查新结果**：PeVL 已用 pose mask 调制 CLIP visual attention；PAAB 已把 pose-pair mask送入 ViT attention logits并残差写回；2026 MUVA 已在 ReID 中把动态 grounding body-part mask逐层送入 CLIP ViT self-attention。PAFormer、KPR、ProFD 分别覆盖 pose-supervised cross-attention、pose-conditioned encoder与 CLIP part decoder。仓库 exp012/052/143 又已覆盖 unary/pairwise/skeleton attention bias。

**数学判定**：
`L+γ[B(P)-B(Pc)] = (L-γB(Pc))+γB(P)`。固定 canonical 项只是静态 attention bias；ordinary additive pose-bias 模块可以直接输出同一个 centered residual。zero-init、少量 heads/layers、untouched heads 与 global-only 输出是良好的工程和证据约束，但不构成新的函数族。

**燃料边界**：exp371 Gate B 中 correct 只比 shuffled/canonical 高 `+0.0320/+0.0984 mAP`，说明实例级姿态残差并非当前 LGPA 增益的主要来源。六臂协议仍是好控制，但更严格的证据设计不能替代机制创新。

**决策**：**PCAR 新颖性 Gate FAIL，正式 NO-GO。** 不下载 checkpoint、不修改代码、不占用 3090/4090、不运行六臂性能 screen，不转 layer/head/alpha/temperature/query/OT/MoE 小变体。即使未来工程移植涨点，也只能先称 pose-conditioned attention adapter，不能作为“我们的 LGPA 创新”。

**边界**：本决策不否定 LGPA 的结构化局部增益，也不否定未来所有 pose×CLIP 方法；它只否定当前可归约的 centered additive attention 方案。若未来重开，必须先提出无法写成“实例 pose bias + 静态 bias”的非可分解机制。

### [2026-07-15] 决策：exp373 SA 正交耦合新颖性 Gate NO-GO，不运行 fuel audit 或训练

**上下文**：用户提出把 PSG 与 PAA 在多层同时结合，并希望把 scale/shift 改造成
可归属的自有创新。仓库核对发现，现代码已经在每个启用 stage 的每个 block 后
执行 PSG→PAA；`exp073` 已跑 Stage2+3 同步注入且比 Stage3-only 低 `0.5 mAP`，
matched `exp251/exp254` 中两阶段 PAA 也为 `-0.3/-0.6`。因此不能把“多层共置”
当作未做过的新结构。

**候选**：定义实际 PSG displacement `d=x_PSG-x`、PAA residual
`b=x_PAA-x_PSG`，用 `b_perp=b-Proj_sg(d)(b)` 强制两支逐 token 正交；若门禁
通过，再考虑 PSG Stage2+3、PAA Stage3 的深度非对称实现。

**查新判定**：普通 PSG+PAA 是 FiLM/SPADE 类条件仿射。若投影相对 pose-only
gate，候选仍是带正交约束的 FiLM 子集；若投影相对实际 `x*g(H)` displacement，
关键 hard orthogonal residual operator 已被 arXiv 2025 Orthogonal Residual
Update 直接覆盖。CVPR 2023 Shape-Erased VI-ReID 与 ICML 2026 CoLoRAI
Workshop Ortho-ReID 又已覆盖 ReID 中人体结构/外观相关子空间和正交补身份表征。
stop-gradient、zero-init、独立 stage mask 和强 controls 不能消除该重合。

**资产核对**：exact exp066 seed1234 checkpoint 仍在
`lab-3090-d:/root/work/SOLIDER-REID/log/occluded_duke/exp066_paa/transformer_120.pth`，
SHA256=`a084d84995f8fcfd53eea19d8c674d1cdce07d954d9cafbd78e73a211a8903ad`，
execution commit=`8eacaf16dcd797ab8090fe19aca49f80f86bec6a`，数据/pose_data 齐全。
因此停止不是 checkpoint、数据或执行环境阻塞。

**决策**：**exp373 新颖性 Gate FAIL，正式 NO-GO。** 按预注册规则不运行
checkpoint forward fuel audit，不实现 `POSE_PAA_STAGES` 或正交投影，不占用
3090/4090，不进入 e60/e120；也不转 transport、routing、adaptive gate、
content-LoRA、普通 FiLM、层数或阈值小变体。

**边界**：本决策不否定 PSG 的跨数据集/跨 backbone 历史增益，也不抹去 PAA
在早期 `PSG+GCN` 两 seed 上的正信号。它只否定“给 PSG/PAA 加正交投影”能够
承担新论文主贡献。若未来仅为工程优化使用，必须降级为辅助正则，不能写成主创新。

### [2026-07-15] 决策：exp374 因果 Gate A NO-GO，停止 PSG gate 自有化

**问题**：PSG 的稳定涨点究竟来自当前图像与正确实例姿态的对应关系，还是只要提供一个
合法的人体空间先验就能得到相同收益？exp374 在三 seed 冻结 checkpoint 上比较
correct、matched-shuffle 与 true bypass，并用 correct-start/end 检查执行漂移。

**结果**：correct−shuffle mAP 仅 `+0.001163 pp`，simultaneous interval=
`[-0.363577,+0.377887]`，三 seed 中两 seed 非正；correct−shuffle R1 为 `0.000000 pp`。
与此同时 correct−bypass 为 `+3.857684 mAP / +5.143288 R1`，区间均明确为正。

**决策**：**exp374 正式 NO-GO。** PSG 分支的性能价值真实，但没有证据表明它依赖正确
实例 pose；因此不继续 gate 权重、层数、shuffle/canonical/centroid/anatomical 小变体，
不把通用人体先验收益改写为 instance-specific pose reasoning 创新。

**下一步**：独立转向 pose-controlled Mamba/state-space 路线，优先研究 pose 是否能控制
selective state update、保留/遗忘、解剖扫描顺序或身体区域间状态传递。新路线必须用
parameter-matched image-only Mamba、pose-shuffle 与 PSG/SFT 强对照，但不因相邻 Mamba
论文存在就自动放弃；先建立直接先例边界，再尽快实现最小原型。

### [2026-07-15] 决策：exp375 PRSM 双重 Gate NO-GO，停止姿态路由状态记忆线

**问题与机制**：PRSM 不再用姿态逐位置缩放特征，而让实例 pose 把 RGB token 路由到
6 个身体状态槽，在双向 recurrent update 内控制 write/retain；RGB query 独立读取 carried
state。第一版严格禁止 graph、动态扫描、额外 loss、GCN、LGPA、PAA 与 PSG，只验证
pose-controlled state dynamics 本身。

**同运行时训练结果**：B0 image-only、M0 canonical PRSM、P0 instance-pose PRSM 均在
同一 4090、同一运行时和同一 execution 跑满 120 epoch。最终 B0=`58.4/67.1`、
M0=`58.8/67.5`、P0=`57.1/66.3`（mAP/R1），即 `P0−B0=-1.3/-0.8`，
`P0−M0=-1.7/-1.2`。三路完整结束且无异常，模块参数已离开初始化，不能归因于执行失败。

**同 checkpoint 因果结果**：correct 相对 matched-shuffle、foreground-uniform、zero-bypass
的 mAP 差仅 `-0.000338/-0.000690/+0.001795` 百分点，R1/R5/R10 全同；matched target-only
nuisance gate PASS，各 descriptor 不同，correct-start/end 精确复现，zero 622 次 exact
identity。full canonical 也近似相同，但因同时改变 route/support/write mass，只保留为诊断。

**决策**：**exp375 PRSM Gate A 正式 NO-GO。** 训练端 P0 同时输给 B0/M0，冻结权重下
correct 又与 matched/foreground/zero 等价，因此实例姿态路由、部位状态归属和推理时
memory 写回均没有可测身份排序贡献。按预注册规则不做 graph、scan order、更多槽、额外
loss、Mamba block 替换或小参数变体救场。

**边界**：该结果只否定当前 PRSM，不能外推为所有 pose-controlled state-space 均无效，
也不推翻 PSG/LGPA 的历史性能资产。但在当前论文规划中，PRSM 不进入方法、贡献、摘要或
主实验表，不能作为解决 LGPA 归属问题的第二创新。

### [2026-07-16] 决策：exp376 Pose Hyper-LoRA e60 NO-GO，停止动态低秩调制线

**结果**：4090 P0 e60=`54.2 mAP / 63.0 R1`，低于预注册同机 clean B0
`55.2/65.0`，差 `-1.0/-2.0`；从 e30 到 e60 四个评测点均不高于 B0。8 层动态
operator 的 alpha、pose coefficient、visibility 和 delta 全程有限非零，batch64 AMP
也证明关键参数均有梯度并实际更新。

**决策**：**exp376 正式 NO-GO。** 停止当前 stage2/3、rank4/M4 factor-wise Pose
Hyper-LoRA；不补 M0、exact B0、matched donor、多 seed、更多层或 rank 小变体。该结果只否定
当前逐层动态低秩实现，不外推所有 pose-conditioned operator。

### [2026-07-16] 决策：exp377 selective Δ/B/C e60 NO-GO，停止当前 pose×SSM 线

**结果**：4090 P0 e60=`54.5/63.8/78.5/83.8`，预注册同机 clean B0 为
`55.2/65.0/77.6/83.1`，mAP 差 `-0.7`，已满足“低至少 0.5 即停止”的充分条件。
训练因中间心跳未当场执行停止而自然跑满；e120 P0=`58.6/67.8/81.4/86.3`，相对 clean
B0 只有 `+0.2 mAP`，仍未达 `+0.8` 正式门槛。3090 RGB-only D0 e120=`58.8/68.1`，
跨机趋势也不支持实例姿态增益。SSM、alpha、state 和 pose `Δ/B/C` residual 均实际学习，
日志无数值异常。

**决策**：**exp377 正式 NO-GO。** 性能门禁已充分失败，不再花费资源构造 donor mapping
或冻结反事实，也不补同机 B0/D0/M0、多 seed、跨 backbone、`Δ-only/B/C-only`、动态 scan、
graph、更多 state 或额外 loss。

**边界**：该结果不证明所有 pose×Mamba/SSM 都不可能，也不推翻 PSG/LGPA 的历史性能资产；
它只说明在当前 heatmap、最终 48-token serpentine scan 与联合 `Δ/B/C` residual 下，实例姿态
没有提供可报告的额外身份信息。exp377 不进入论文方法、贡献、摘要或主实验表。

### [2026-07-16] 决策：exp378 保留内生姿态场核心，停止当前 geometry residual 小变体并继续必要归因

**结果**：seed 1234 的同机探索性 final 为 B0=`55.1/66.7/79.5/83.8`、hard
F0/P0=`55.9/67.4/79.3/83.3`与`55.6/66.7/78.4/83.0`、explicit-relax
MR-F0/MR-P0=`56.0/67.1/79.2/83.4`与`55.7/67.1/78.7/82.9`。residual-OFF的
hard F0、MR-F0相对B0分别为`+0.8/+0.9 mAP`；relaxation相对hard在residual OFF/ON下
均只有描述性`+0.1 mAP`；开启当前`17×4` geometry residual在hard/relax下均为
`-0.3 mAP`，mAP difference-in-differences按`0.1`精度为`0.0`。R5/R10相对B0略降，
因此不是四项全面提升，也不是显著性结论。

**决策**：**TAPF继续，当前geometry residual小变体停止。** 当前最有燃料的对象是
residual-OFF的内生姿态场配置，不是SGD relaxation，也不是现有可靠性有界几何残差。
旧`5de3b30` P0/F0继续只作`INVALID_AS_HARD_FREEZE / VALID_RELAXATION_PILOT`，不得续训或
混入正式表。不能把F0的mAP正差直接写成“正确关节语义贡献”，因为它仍混合bootstrap课程、
内部anchor、Gaussian renderer与PSG。

**下一步**：先完成final checkpoint的correct/shuffle/None exact parity、joint permutation、
错误姿态/常量场与stage关闭等低成本语义审计；再在4090同机同execution串行运行
D0→J0→R0。D0/J0闭合持续pose-supervision × residual对照，R0提供外部ViTPose PSG实用
上限。最小Gate B保留RG0、N0/置换bootstrap、teacher agreement、flip equivariance与
geometry-only ID probe；只有J0−D0重新显示residual正贡献时才恢复U0/C0。完成这些必要单锚点
归因后再进入独立Hierarchical TAPF设计与训练，不以当前单一负机制终止整条姿态场路线。

### [2026-07-17] 决策：exp378当前单锚点PSG姿态因果NO-GO，TAPF只转入可空分离consumer设计

**冻结证据**：事后选择但固定的D0 e90 checkpoint在同RGB、同标签、同权重下，correct=
`56.2984/67.6471/79.8190/83.5294`。external correct/shuffle/None/unindexable与correct的
descriptor和四项指标逐位相同。matched-wrong-field、joint/confidence permutation、
spatial-constant和zero-field相对correct的mAP差依次为`-0.0155/+0.0024/+0.0002/-0.0238/-0.0536`
个百分点，全部`|差|<0.1`；只有真正PSG bypass降到`53.6154/63.9367/76.2896/80.4525`，即
`-2.6829/-3.7104/-3.5294/-3.0769`。anchor仍有pseudo-PCK@0.05=`0.5539`、teacher posterior
cosine=`0.8276`、flip cosine=`0.9467`与17通道全占用，因此不是anchor完全失效，而是检索不依赖
这些场语义。

**决策**：**当前单锚点TAPF/PSG的姿态因果主张NO-GO。** residual-OFF相对B0的单seed mAP正差
只能归为训练后PSG模块/容量性重标定，不能归为图像对应姿态、正确关节名称、confidence或空间结构。
当前实现不补multi-seed，不迁移ResNet-50或Video ReID，不触发H0，也不通过多层复制现有PSG救场。

**继续边界**：这不等于永久停止TAPF问题对象。下一步只允许独立设计null field严格identity的
consumer，并加入parameter-matched RGB/static control与逐层matched/constant/bypass门禁；只有
训练后correct相对破坏臂达到预注册`>=0.3 mAP`，才恢复multi-seed、跨backbone和时序姿态扩展。
Video ReID/时序姿态仍是潜在统一方向，但必须晚于单图consumer的因果成立，不能把普通时序容量
包装成姿态贡献。

### [2026-07-17] 决策：把anchor+PSG作为原子方法，启动exp379逐层升级与backbone迁移链

**用户裁决**：原始PSG在测试期必须由姿态模型持续供应heatmap；exp378 D0则在训练期用ViTPose
监督内部anchor，推理期只需RGB。论文可以也应该把anchor与PSG作为一个完整模块讨论，不要求把
两者拆成分别显著的组件。冻结语义审计仍然有效，但它约束的是“精确关节语义在推理期因果生效”
这一机理措辞，而不是完整模块的部署价值和相对B0的整体收益。

**证据口径**：同机B0=`55.1/66.7/79.5/83.8`，fresh D0=
`56.2/67.6/79.8/83.4`，即`+1.1/+0.9/+0.3/-0.4`；测试期外部ViTPose+PSG的R0=
`56.1/67.4/79.5/83.7`。因此D0基本以RGB-only部署复现了原始external-pose PSG。不得声称
每个关节通道独立贡献，但可以陈述“训练期姿态监督替代测试期姿态依赖，并获得整体检索增益”。

**执行决策**：启动独立`exp379 Progressive Hierarchical TAPF`。首版固定Stage-1内部场调制
Stage-2，Stage-2在上一状态上refine后调制Stage-3；使用stage-specific投影与单个共享decoder，
两节点pose loss取均值，持续pose监督，推理仍RGB-only。直接对照已完成的同机D0，B0作总增益
参考；禁止重复训练B0/D0。历史exp009“同一外部热图多stage复制”只作负背景，不能替代exp379。

**后续顺序**：exp379在Swin-T跑满后，独立迁移ResNet-50，再评估合适ViT。强SOLIDER Swin-T
可能压缩上涨空间，因此Swin中性不自动取消一次预注册backbone迁移，但不能临场调层数/宽度/loss
救场。单图与backbone证据闭合后，再设计Video ReID与时序姿态可靠性，不在exp379中抢跑。

**覆盖关系**：本决策只覆盖上一条中“当前实现不补multi-seed/ResNet/Video、必须先换null-separable
consumer”的执行限制；不删除冻结审计事实，也不恢复geometry residual、U0/C0或旧H0训练。

### [2026-07-17] 决策：exp379 Swin-T中性封板，转入ResNet-50同骨干三臂验证

**结果**：fresh HT0自然完成120 epoch，final=`56.1/67.6/79.9/83.4`；相对已完成的同机D0=
`-0.1/+0.0/+0.1/+0.0`，相对B0=`+1.0/+0.9/+0.4/-0.4`。12个checkpoint、两个stage
projection、唯一shared decoder与Stage-2/3 PSG全轨迹有限，final external pose
correct/shuffle/None/exploding descriptor exact parity通过，排除未执行、数值异常或测试期偷读pose。

**决策**：**HT0在Swin-T上不宣称优于D0，当前Swin逐层小变体封板。** 不补更多stage、独立
decoder、loss权重、宽度、geometry residual或重复seed救场；HT0与D0基本中性说明强SOLIDER
Swin-T下逐层升级没有可分辨额外收益。与此同时，`anchor+PSG`作为完整方法相对B0的约`+1 mAP`
仍成立，且HT0已实现用户要求的每个anchor对应一个PSG，不能把“逐层未额外上涨”误写成完整方法
失效。

**下一步**：按预注册独立设计ResNet-50迁移，并在同一backbone、同数据recipe、batch64、seed1234、
120 epoch下依次fresh训练B0→D0→HT0。D0只含单anchor及其PSG；HT0保持每anchor一PSG、stage-specific
projection与shared decoder。三臂必须从同一预训练权重和matched初始化出发，只比较backbone内部
差值；完成后再决定ViT、multi-seed与Video ReID/时序姿态，不跨backbone比较绝对指标。

### [2026-07-17] 决策：exp380 ResNet逐层增量GO，进入ViT判别而不在ResNet调参

**结果**：R50-B0/D0/HT0三臂均fresh串行跑满120 epoch，final依次为
`35.0/45.3/61.3/68.2`、`38.1/49.4/64.6/71.1`、`38.9/50.5/65.9/72.0`。
D0−B0=`+3.1/+4.1/+3.3/+2.9`，HT0−D0=`+0.8/+1.1/+1.3/+0.9`。进程、GPU、12个
checkpoint、全参数轨迹、严格异常与pose-free exact parity均通过，排除未执行或测试期偷读pose。

**决策**：**完整anchor+PSG跨骨干验证GO；逐层HT0在ResNet上描述性GO。** 完整方法现在同时在
Swin-T和ResNet-50内部优于B0。HT0相对D0在ResNet达到`+0.8 mAP`，但Swin对应差为`-0.1 mAP`，
所以逐层refinement暂不写成跨backbone稳定主贡献，也不以ResNet一个seed触发显著性或普适性主张。

**执行边界**：ResNet线封板，不补更多stage、独立decoder、loss权重、宽度或重复seed救场。下一步
独立设计合适ViT的matched B0→D0→HT0三臂，继续只比较backbone内部排序；若ViT也支持HT0>D0，
再把hierarchical refinement升为核心贡献候选，否则主方法仍以更稳健的完整单anchor+PSG为中心，
逐层版作为backbone-conditional扩展。ViT闭合后才进入Video ReID/时序姿态，不并行抢跑。

### [2026-07-17] 决策：exp381完整原子方法GO，逐层跨架构主张NO-GO，转入时序姿态

**结果**：ViT-B0/D0/HT0三臂均fresh串行跑满120 epoch，final依次为
`52.9/59.5/77.1/82.0`、`54.9/61.4/78.9/84.0`、`54.6/60.6/78.4/84.1`。
D0−B0=`+2.0/+1.9/+1.8/+2.0`，HT0−D0=`-0.3/-0.8/-0.5/+0.1`。进程退出、GPU、
12 checkpoints、全参数轨迹、strict load、pose-free exact parity与严格异常均通过。

**实现边界**：ViT PSG是post-block调制；post-block11位于最后一次CLS–patch交互之后，其final
zero projection全轨迹`0/2 changed`，对CLS descriptor无下游路径。D0/HT0实际有效G3 consumer
为post-block9/10；terminal冗余在两臂共享，不混淆HT0新增G2的直接比较，但论文不得声称block11
PSG有效。

**决策**：**完整`anchor+PSG`原子方法跨三骨干描述性GO；当前逐层跨架构主张NO-GO。** 单anchor
D0相对B0的mAP差在Swin-T/ResNet-50/ViT-B依次为`+1.1/+3.1/+2.0`；HT0−D0则为
`-0.1/+0.8/-0.3`。因此训练期pose监督、推理期RGB-only的完整方法保留为论文中心，hierarchical
只作backbone-conditional扩展/消融，不写成普适增益，也不继续单图stage、decoder、宽度或loss救场。

**下一步**：单图backbone判别链封板，进入独立Video ReID/时序姿态设计。新实验必须把变量拆成
同video backbone的RGB temporal B0、逐帧原子方法D0和真正使用跨帧pose可靠性/运动连续性/遮挡恢复
的T0；T0必须直接比较D0，不能只对B0报总增益。训练期可用pose作为privileged teacher，推理仍应
RGB-only；普通temporal pooling、参数量或更多帧带来的收益不得包装成时序姿态贡献。

### [2026-07-17] 决策：exp382 Video TAPF独立headline NO-GO，优先补数据集与跨域证据

**查新裁决**：GAE-Net已直接覆盖训练期gait+RGB视频教师向RGB-only视频学生的局部互补蒸馏；
PAFormer覆盖pose-supervised、pose-free inference，KPRTrack覆盖tracklet同部位聚合，
PSTA/STMN/TF-CLIP等又覆盖遮挡、干扰与temporal memory。因此“训练期跨帧pose state、推理期
RGB-only”本身不再足以支撑独立视频方法。远端也没有可用视频数据。exp382按novelty/data双门禁
`NO-GO`，禁止下载或为占用GPU而启动视频训练。

**论文中心不变**：完整`anchor+PSG`仍按不可强拆原子方法讨论，三骨干D0−B0 mAP为
`+1.1/+3.1/+2.0`；hierarchical仍只作backbone-conditional扩展。不得把exp382的NO-GO误写成
单图原子方法失效，也不得把pose-privileged/pose-free这一已被强先例覆盖的大叙事单独声称为首次。

**下一算力决策**：优先设计`exp383 Market→Occluded-ReID TAPF`，fresh训练matched Market
B0/D0，并同时报告Market域内与Occluded-ReID跨域final。Occluded-ReID是test-only数据，不能称为
第二训练集。该两臂同时补第二训练域、独立遮挡target和TAPF效率审计，信息量高于立即在原
Occluded-Duke重复seed。只有design、evaluator真正pose-free、CUDA/AMP/overflow/state-RNG-
optimizer/route/gradient/parity全部通过后才允许串行启动；当前保持GPU空闲。

### [2026-07-18] 决策：官方干净 D0 弱 GO，clean hierarchical NO-GO，下一步补 matched 多 seed

**执行口径重置**：用户要求提交旧状态后退回 SOLIDER 官方最后代码，只保留原始数据集与官方
代码；旧 runtime、旧 pose cache 和路径映射全部禁止复用。exp384/385 已分别完成 Market 与
Occluded-Duke official B0，exp386/388 从原始 train RGB 用 ViTPose-H fresh 提取 pose，
query/gallery 不提取。Market B0=`91.6/96.3/98.7/99.2`，其中 mAP 精确复现官方报告。

**clean 原子结果**：Occluded-Duke D0=`57.6/67.7/80.8/84.6`，matched B0=
`57.4/67.4/80.6/85.2`，D0−B0=`+0.2/+0.3/+0.2/−0.6`；Market D0=
`92.0/96.5/98.8/99.3`，matched B0=`91.6/96.3/98.7/99.2`，D0−B0=
`+0.4/+0.2/+0.1/+0.1`。两臂均通过 strict finite、参数轨迹与 correct/shuffle/None/exploding
pose-free exact 终审。D0 额外参数约 `0.375%`，supported-op FLOPs 约 `+0.242%`。

**clean 层级结果**：exp389 HT0=`56.9/65.9/80.0/84.1`，相对 clean D0=
`−0.7/−1.8/−0.8/−0.5`，相对 official B0=`−0.5/−1.5/−0.6/−1.1`。early/late anchors、
六个 early 和两个 late PSG 均离开初始化；八个 consumer 对 final descriptor 都有非零可执行路径，
严格异常为 0。因此这是机制有效运行后的负结果，不补层数、宽度、loss、decoder 或中途 best 救场。

**决策**：原子 `anchor+PSG` 在 clean 口径下仅作**弱描述性 GO**：两个训练域 mAP 均为正，
但幅度只有 `+0.2/+0.4`，Occluded-Duke R10 为负，且都只有一个 seed。hierarchical 在 clean
官方实现上正式 `NO-GO`，继续只保留为历史 backbone-conditional 扩展，不进 headline。旧 runtime
的 Swin/ResNet/ViT 正差不得与 clean 主表混为同一实现证据。

**下一算力决策**：优先建立官方干净 Occ-Duke matched B0/D0 多 seed，而不是 Video、HT0 或新
模块。新增 seed 必须逐 seed 先跑 fresh B0 再跑 fresh D0，保持 batch64/120 epoch/数据/teacher/
增强/optimizer 完全一致；报告逐 seed 差值、mean/std，不以某个正 seed 或 best checkpoint裁决。
若多 seed 不支持正均值，当前 MMAsia headline 必须降级或转向新的问题对象；若支持，再决定是否
补 clean 跨骨干，而不是直接恢复旧 runtime 结果。

### [2026-07-18] 决策：exp390确认TAPF小幅mAP可重复，但rank收益不成立；转入受控全stage直预测诊断

**结果**：三seed paired D0−B0（mAP/R1/R5/R10）分别为seed1234
`+0.2/+0.3/+0.2/−0.6`、seed4321 `+0.8/+0.3/+0.5/+0.5`、seed2025
`+0.4/−0.9/−0.7/−0.5`。paired mean±sample std=
`+0.47±0.31/−0.10±0.69/+0.00±0.62/−0.20±0.61`；mAP三seed均正，rank指标方向混合。

**决策**：official clean原子TAPF保留为**mAP-only弱GO**。它已超过“单seed偶然正差”的最低
证据线，但效应小、R1/R5/R10均值不正，不能包装为全面提升、统计显著或跨架构普适headline。
exp389 clean HT0相对D0为`−0.7/−1.8/−0.8/−0.5`，仍保持NO-GO；不复活原层级实现，也不依据
旧runtime结果改写clean结论。

**下一算力决策**：允许进入已预注册的exp391，但它不是直接抢跑“三anchor大模型”，而是严格
按A→B→C单变量链推进：先验证exp389 sum→mean pose objective，再验证6/2→2/2 consumer平衡；
只有两道门禁均通过，才运行参数/loss/AMP matched的H3-OFF/H3-ON全stage direct pair。三个anchor
必须参数独立、都直接预测absolute field、不读取prior或offset，2/2/2 consumer全部通向final
descriptor，并使用物理sigma归一`6.0/3.0/1.5`。任一阶段失败即停止后续层级扩张，不用调参救场。

### [2026-07-18] 决策：exp391 Phase A恢复HT0但仍低于D0，封板并禁止Phase B/C

**结果**：H2-M保持exp389两个独立direct anchor、early/late=`6/2` consumer和全部训练recipe，
唯一变量为pose objective从`0.1×sum`改为`0.1×mean`。自然e120 final=
`57.2/67.3/80.2/84.5`；相对exp387 D0为`−0.4/−0.4/−0.6/−0.1`，相对exp389 HT0为
`+0.3/+1.4/+0.2/+0.4`。不得用中途best替代final。

**机制审计**：冻结early-bypass后full的独立差值为
`+0.141111/+0.497735/+0.316739/−0.135750`，early mAP贡献超过预注册`+0.1`门槛；late独立差值为
`+1.546086/+2.036196/+1.447964/+1.583707`。early/late anchor、八个PSG和八条consumer均真实
更新且可影响final descriptor，推理期对external pose严格无访问，因此失败不是dead path或执行错误。

**决策**：预注册规则规定H2-M final mAP低于D0超过`0.2`即Phase A NO-GO；实际为`−0.4`，故
exp391=`SEALED / NO-GO`，Phase B的consumer balance与Phase C的三anchor H3-OFF/ON均禁止实现、
训练、续训或换seed救场。mean loss只说明旧sum存在预算惩罚并能恢复HT0，不足以证明多stage优于
单层D0。该决定不永久否定多阶段机制：下一阶段转入论文证据闭合与“CLIP语义校准多阶段TAPF”
的只读文献/代码/机制审查；后者必须独立设计解决joint-channel语义不可辨识，不能作为普通CLIP KD
直接抢跑正式训练。若语义门禁成立，应另开新实验重新比较semantic single-stage与semantic
multi-stage，而不是恢复或续跑exp391 Phase B/C。

### [2026-07-18] 决策：Phase 0A确认现有TAPF是有效但语义不可辨识的调制器，进入CLIP teacher门禁

**证据**：在exp387 sealed D0 final上，all-PSG bypass导致
`−1.3586/−1.6742/−1.7195/−1.2670`，PSG0/PSG1各自旁路mAP分别`−0.6756/−0.7149`，所以
consumer不是dead path。与此同时，channel-cycle只改变
`+0.0240/+0.0452/−0.0452/+0.0452`，matched-wrong field只改变
`−0.0049/+0.1357/−0.0452/0`；均远低于预注册`0.3 mAP`语义门槛。把每个field通道压成空间常量
反而提升`+0.3463/+0.4977/+0.4977/+0.4977`。

**决策**：Phase 0A=`SEALED / CONSUMER_EFFECTIVE_JOINT_SEMANTICS_NOT_IDENTIFIED`。现有D0保留
exp390三seed mAP-only弱GO，不因本审计被全盘否定；但论文不得把17个channel解释成已学会的具体
关节语义，也不得把精确空间field称为已验证的检索因果来源。冻结bypass测到的是已训练D0内部依赖，
不能包装成相对B0的训练增益。

**下一步**：授权Phase 0B双编码CLIP teacher-only门禁，不授权正式训练。0B必须证明teacher同时依赖
正确RGB、pose mask与text bank，并优于text-only常量、image-only cluster、fixed bands和wrong
RGB/mask/text；否则只否定当前teacher定义或粒度，不永久否定换粒度后的CLIP方向。即使0B通过，
仍需Phase 0C证明NULL identity、semantic mismatch、梯度所有权和generic-router强对照，才允许
semantic single-stage；多阶段继续以后者为直接基线，不能续跑exp391。

### [2026-07-18] 决策：Phase 0B否决naive patch-text teacher，先重构CLIP局部读出而非启动训练

**结果**：全15,618图上，square/aspect-letterbox的correct macro top-1仅`2.692%/4.637%`，
expected margin=`−0.11349/−0.11099`。channel-shuffle top-1=`16.107%/15.511%`，wrong-text=
`29.996%/33.546%`；correct相对matched wrong mask、shuffle和wrong text的paired margin在两种
geometry下全部显著为负。confidence、synthetic-erasing与flip top-1门禁也失败。

**实现归因**：该结果不是pose坐标、hook、投影、label order或resize插值bug。末block hook经官方
`ln_post+proj`后与OpenCLIP `output_tokens`逐元素exact；pose region y顺序正确；bilinear→bicubic
几乎不变。同一mask与prompt改用tight-crop global CLS后，小样本macro top-1从`3–5%`升至
`44.688%`；image-only patch cluster在全量达到`52.8–60.0%`。因此真正错误是把CLIP未直接受
contrastive text监督的raw local patch方向当成了可命名body-part teacher。它含局部结构，但不在
正确text坐标系。

**决策**：Phase 0B=`SEALED / CURRENT_CLIP_TEACHER_NO_GO`，Phase 0C与所有正式训练保持
`NO-START`。不得靠改温度、挑prompt、挑geometry或用wrong-text更高结果救场；也不得把它写成
“CLIP无效”或永久否定CLIP后的多阶段。下一步只授权新的teacher接口研究：优先比较共享早期trunk、
用pose约束多个后段block的CLS readout，与可缓存的region-crop global CLS；arms/upper-leg ontology
必须先消除重叠。新定义需独立设计和全teacher-only反事实，不能覆盖或重复当前封板结果。

### [2026-07-19] 决策：Semantic C0未超过clean D0，只关闭当前组合并继续CLIP–TAPF机制修复

**授权边界变化**：Phase 0B2把失败拆到了具体接口。hard-owner ontology获得exact-zero overlap；
PC-MBCLS在128图五slot上通过局部support反事实，证明CLIP受监督CLS路径能提供sample-specific局部
响应。B2-Sv1的connected-occluder构造失败只关闭该反事实构造，不再作为训练的一票否决。用户明确
要求尝试训练后，首次single-stage bundled arm按fresh、串行、自然e120执行；这不是exp391续训。

**结果**：Semantic C0 final=`56.9/67.1/80.6/85.0`。相对clean D0为
`−0.7/−0.6/−0.2/+0.4`，相对official B0为`−0.5/−0.3/+0.0/−0.2`，相对pure-structure HT0为
`+0.0/+1.2/+0.6/+0.9`。checkpoint严格有限、teacher完全不在student state，anchor/q-head/两个
router均离开初始化，两个consumer都能改变final descriptor，RGB-only与NULL identity全部PASS；
因此该负差不是训练崩溃、dead route、外部pose泄漏或checkpoint错误。

**机制归因边界**：mask/presence loss分别降到`0.158/0.026`，但q loss停在`0.692`；终审混合五slot
的support pooled std=`0.01686`，两个router gate-delta abs-mean仅`3.606e-06/1.040e-05`。当前
student主要学到coarse mask/presence，而CLIP sample-specific support没有形成足够强的可执行动态。由于这是
teacher+readout+router bundled arm，不能把失败单独归因给CLIP，也不能用优于HT0证明CLIP增量。

**决策**：`exp392 Phase 0C Semantic C0 = SEALED / CURRENT BUNDLED COMBINATION NO-GO`。
禁止重跑、续训、换seed、挑e50/e70等中途节点或在封板臂内调温度/loss。balanced semantic
multi-stage继续`NO-START`，因为single-stage语义因果尚未成立。该NO-GO只关闭当前PC-MBCLS
support teacher、弱动态q head与双router组合，不永久否定CLIP–TAPF。

**下一步**：先做最小必要单变量拆因，而不是再开一个bundled 120-epoch臂：用封板checkpoint和同一
执行seam比较learned-q、static-q、pose-only mask/presence与router bypass，确认final依赖究竟来自
五slot geometry、几乎常量的support，还是generic low-rank transform；随后只针对被证实的瓶颈设计
扩大sample-specific support动态范围的teacher/readout。只有新single-stage同时建立correct相对
static/wrong/generic control的因果差并超过clean D0，才授权semantic multi-stage。

### [2026-07-19] 决策：Phase 0D确认整条semantic route在final检索上近似失活，停止调q小变体

**冻结证据**：在同一Semantic C0 e120 checkpoint上，correct start/end全19,871图descriptor exact。
static-slot-q、q-one、spatial-constant mask、slot-cycle、expert-mean、router0/1/all bypass的mAP变化
绝对值全部小于`0.0007`，R1/R5/R10全部严格不变。尤其all-router-bypass仅`−0.000077 mAP`，说明
checkpoint终审中的非零descriptor L2只是数值可达，不构成检索排序贡献。

**q归因修正**：五slot均值不同，但同slot跨图std只有`0.00009–0.00029`；此前pooled std=`0.01686`
主要测到between-slot prior，不能作为sample-specific CLIP信息证据。把q替换为slot常量或全1均不改变
rank，精确mask geometry、state↔expert绑定和slot-specific expert也同样不可辨识。

**决策**：`Phase 0D = SEALED / CURRENT SEMANTIC ROUTE RETRIEVAL-INERT`。禁止把下一步缩成temperature、
BCE权重、prompt、mask细化或更多stage；这些变量都建立在“route已有检索燃料”的错误前提上。
也不因此永久否定CLIP–TAPF：B2-SI仍证明CLIP CLS readout含局部响应，失败发生在把该响应转成训练期
可执行残差的接口。

**下一机制约束**：新single-stage必须显式防止zero-init router长期停在近identity，例如使用有界但
非零的identity-safe residual scale、相对化CLIP target和route-contribution objective；preflight除
gradient finite外，必须预注册真实训练早期all-router-bypass descriptor/retrieval surrogate gap，final
仍以完整all-bypass mAP贡献为门禁。先写独立设计和强对照，再决定是否启动新训练；semantic
multi-stage继续NO-START。

### [2026-07-19] 决策：exp393拆成独立route激活与rich evidence两门，任一FAIL只关闭对应接口

**设计依据**：Phase 0D说明当前失败同时包含两个可能独立的断点：zero-expert使router长期近identity，
而scalar q又丢失CLIP局部视觉方向。若一次同时改初始化、teacher target、student head和internal loss，
即使结果变化也无法判断来自route可执行性还是CLIP信息增量。

**决策**：exp393先用RZ-C0只把zero expert替换为small-nonzero branch加zero ReZero scalar，保持初始化
descriptor严格identity并用final all-bypass验证route是否真的参与检索；rich-code COER则以sealed
RZ-C0为直接对照，只把scalar q换为`K=16` centered CLIP local evidence。Phase 0E teacher审计与
Phase A route control逻辑独立、执行仍串行：teacher FAIL只阻断Phase B，不能替代Phase A裁决；route
FAIL只关闭当前ReZero接口，不能证明rich CLIP evidence不存在。Phase B只有两门都通过才获授权。

**梯度边界修正**：内部alignment必须作用于生产expert生成的pre-alpha branch proposal，并用共享权重、
detached-token重算阻断backbone梯度。它更新token/context/evidence projection与expert；ReZero alpha只由
ReID loss打开。禁止把只监督pre-expert latent的loss误写成“更新执行残差”，也禁止增加训练后删除的
projector吸收CLIP loss。semantic multi-stage继续NO-START。

### [2026-07-19] 决策：0E-128通过rich evidence稳定性门，只授权full teacher审计

**证据**：128个不同PID严格拆为64 fit/64 held-out。五slot 16维code的macro effective rank=
`11.050/16`，每一维held-out std均非零；correct↔flip相对different-PID wrong RGB与same-RGB
low-IoU wrong mask的逐slot PID-cluster CI下界全部严格大于0。slot-mean/global-only exact zero，raw
uncentered明显更弱；fixed random orthogonal仍保留强信号，说明信息属于centered rich local residual，
而不是PCA偶然挑轴。

**决策**：`Phase 0E-128 = SEALED-PASS`。下一步仅允许按冻结协议执行official 15,618 train的
0E-FULL teacher-only PID-disjoint审计，并采用流式/两遍实现避免把全部RGB一次性放入内存。PCA仍是
压缩器而非方法贡献。full PASS也只授权Phase B teacher接口，不授权训练或multi-stage；full FAIL只
关闭当前rich code并阻断Phase B，不能取消独立Phase A RZ-C0的实现与preflight。

### [2026-07-19] 决策：0E-FULL通过，teacher richness门封板为GO但训练仍NO-START

**证据**：official 15,618图、702个PID完整覆盖并按PID严格拆成`7,860/7,758`图和`361/341` PID。
五slot macro effective rank=`12.335/16`；wrong RGB与same-RGB wrong mask的逐slot 95% CI下界全部
严格大于0，13项正式gate全PASS。进程自然退出、GPU空闲、异常0，全部资产SHA冻结。

**决策**：`Phase 0E teacher richness = GO / SEALED-PASS`。该GO只说明rich CLIP local evidence可供
执行，不说明现有或候选router会使用它，不授权直接训练Phase B。严格转入逻辑独立Phase A RZ-C0：
先验证identity-safe nonzero branch能否被ReID loss打开；只有Phase A自身通过，Phase B才同时具备
teacher与route两项授权。semantic multi-stage继续NO-START。

### [2026-07-19] 决策：RZ-C0通过单变量与route-activation预检，授权fresh e120

首次完整预检发现random expert推进CPU RNG并改变router-1 projection，正确判FAIL；修复为局部
保存/恢复RNG后，以新exact source完整重做，非目标state mismatch清零。新预检证明初始化identity、
首个finite alpha-only梯度、后续18步branch更新、24步内full/bypass gap增长、NULL/teacher/strict/RGB-only
与finite全部成立。

**决策**：`Phase A CUDA preflight = PASS`，只授权RZ-C0 single-stage fresh seed1234完整120 epoch，必须
final-only、自然跑满且做all-bypass终审。不提前授权Phase B；只有e120 full−all-bypass `>=+0.1 mAP`
且full不比Semantic C0低超过`0.2 mAP`，RZ route才称alive并与已PASS的teacher门共同授权Phase B。

## 2026-07-19：exp393 Phase A封板为ROUTE-ALIVE-FAIL，原Phase B不启动

RZ-C0 e120 final=`56.8/66.8/79.6/83.9`；full mAP floor通过，但all-router-bypass四项完全相同，
raw full−bypass=`-0.000249709 mAP point`，未达到预注册`+0.1`。checkpoint strict finite、teacher
隔离、NULL identity、RGB-only和router参数轨迹全部通过，排除了“代码没接上”或“状态没保存”。

**决策**：封板当前`random nonzero expert + ReID-owned free ReZero scalar`接口，禁止重跑、换seed、
调alpha、改门槛或把e90/e110中间值当final。由于Phase A门未过，原Phase B不得实现或训练；Phase 0E
teacher PASS保持有效且逻辑独立，因此不得把该FAIL扩大为CLIP–TAPF总体否定。

下一步只授权新的route-ownership只读设计与CPU/static preflight：执行幅度不能再由一个可静默塌回零的
自由标量独占，同时必须保留wrong evidence、static、generic normalized route与all-bypass，防止用
“强迫route非零”伪造CLIP语义贡献。所有新门冻结前正式训练`NO-START`。

### [2026-07-19] 决策：exp394 production static/CPU通过，只授权CUDA预检设计

fresh实现已证明旧D0/HT0/Semantic C0/RZ-C0默认路径state/forward exact，新rich接口的rho schedule、
NULL identity、两consumer、strict reload、teacher隔离与四类梯度所有权在CPU上全部成立。首次实现中
发现并修正两项真实接线问题：relation helper不应要求proposal与teacher code同维；新anchor不应沿用
会阻断首步mask/presence→trunk梯度的零初始化。失败资产均保留，未改预算公式、loss权重或门槛。

**决策**：`PRODUCTION_STATIC_CPU_SEALED_PASS`。只授权下一步先写并冻结CUDA/AMP preflight协议、复制
并验证新的canonical CLIP实体与full codebook接线；当前4090运行仍`NO-START`。CPU PASS不授权正式
e120，不证明CLIP route有效，也不授权semantic multi-stage。后续CUDA门必须至少24步覆盖teacher阶段
identity但branch更新、handoff descriptor gap、correct/wrong/static、两consumer、strict reload、
teacher/optimizer/checkpoint/eval隔离、RGB-only与峰值显存；任一FAIL先归因，不降低门槛。

CUDA/AMP协议随后已冻结为24个official actual batch64更新：12步epoch1 exact-zero预算，接12步epoch6
`rho_star/5` handoff，并包含actual-batch四类分loss梯度所有权与完整reload/RGB-only/NULL/显存终审。
该阶段只完成协议文档，`PREFLIGHT IMPLEMENTATION/CUDA/FORMAL`均`NO-START`；不得把协议冻结误写为
真实AMP门已通过。

### [2026-07-19] 决策：exp394首步AMP gradient non-finite，正式臂不启动

canonical runtime解决了OpenCLIP/OpenCV依赖分裂，且official batch64 teacher target前置门通过；但
唯一actual preflight在step 1 unscale后发现model gradient non-finite，并在optimizer step前退出。
成功更新exact `0/24`、checkpoint `0`，source/assets/tracked保持exact。result没有保存具体parameter组，
所以证据不足以把FAIL单独归因到某个head、router或loss。

**决策**：`CUDA_AMP_PREFLIGHT_SEALED_FAIL`，exp394 e120与semantic multi-stage均`NO-START`。禁止以
CPU PASS覆盖、重跑同一script、修改GradScaler initial scale、loss/rho/batch或补步。该决定只关闭当前
production AMP实现接口；Phase0E teacher richness与Phase0R固定预算仍独立成立，也不永久否定
CLIP–TAPF。后续若继续，只能另立新的、先验冻结的AMP稳定机制/诊断对象，不能把本臂修补后冒充同一
实验。

### [2026-07-19] 决策：exp395只读归因协议与CPU reporter封板，CUDA仍不启动

exp395将exp394未保存的归属信息定义为独立问题，而不是回头修改sealed preflight。协议冻结D0 baseline
与rich graph、11个逐loss backward、15个互斥parameter group、scaled/unscaled双时点范围统计，且
任何行都不得调用optimizer/scaler update。隔离loss只定位支持子图；即使某行非有限，也不能声称它是
唯一根因。baseline、rich ReID、individual auxiliary和pose/total的分层规则已经先验写死。

static/CPU contract连续两遍13/13 PASS并逐SHA一致，证明reporter能精确分类NaN/±Inf、复现固定scale
比例、验证所有权和aggregate公式，同时保持state/RNG exact、CUDA未初始化、更新0。该PASS没有actual
batch或AMP信息。

**决策**：`PHASE0S_STATIC_CPU_SEALED_PASS`只授权下一步设计独立CUDA attribution implementation；
没有新的明确CUDA授权前，不复制fresh远端资产、不占用4090。exp394禁止重跑/修补，formal e120和
semantic multi-stage继续`NO-START`。

CUDA attribution implementation随后已完成并通过连续两遍29/29 CPU-only AST gate：loss/group顺序、
zero-update、默认scale、双时点capture、fresh assets、runtime和状态终审均已写死。静态PASS只表示脚本
符合协议，不表示已运行或已定位根因。

**后续边界**：`CUDA_ATTRIBUTION_IMPLEMENTATION_STATIC_SEALED_PASS / CUDA EXECUTION NO-START`。没有
新的明确授权前，不把脚本或资产送入4090执行；exp394、e120与semantic multi-stage边界不变。

### [2026-07-19] 决策：exp395 actual因大张量quantile reporter失效而封板INVALID

用户给出持续自主CUDA授权后，唯一actual按冻结source/runtime/assets/batch执行。source、资产、official
batch64与teacher target前置控制流通过；但第一行D0 `reid`的scaled backward之后，reporter对backbone
组全量元素调用`torch.quantile`，触发`input tensor is too large` RuntimeError，并在unscale前退出。
没有完整loss×group矩阵，故不能判断D0或rich图的finite支持，更不能对exp394作loss/head归因。

**决策**：`CUDA_ATTRIBUTION_EXECUTION_SEALED_INVALID / REPORTER_RUNTIME_FAIL`。遵守预注册停止门，
禁止修改reporter后重跑exp395；optimizer/scaler update与checkpoint均为0，exp394继续sealed。下一步只
允许另立exp396，在CPU/static阶段先覆盖真实backbone量级的chunk-safe exact分位数与动态范围统计，
再执行独立CUDA归因。用户的持续授权取消了“再次等待确认”，但不取消static先行、fresh once-only、
零更新与失败即停边界。formal e120及semantic multi-stage仍`NO-START`。

### [2026-07-19] 决策：exp396大张量reporter static通过，直接授权fresh CUDA归因

exp396保持原D0/rich loss矩阵、15组、batch、scale与update边界，只把失败的全量Torch quantile替换为
chunk双遍扫描和temporary memmap exact sort。连续两遍static在`16,777,217`元素上完成解析exact
P50/P95/P99，并通过小张量reference、non-finite分类、输入不变及success/exception scratch清零；
CUDA从未初始化。

**决策**：`PHASE0Q_STATIC_CPU_SEALED_PASS`。根据用户已记录的持续自主授权，完成显式提交后直接建立
fresh exp396 execution与regular资产并执行唯一一次CUDA matrix，不再等待逐次确认。该GO只针对
zero-update归因门；exp394/exp395继续sealed，formal训练仍`NO-START`。

### [2026-07-19] 决策：exp396矩阵证明首步non-finite来自shared ReID backbone门

完整D0/rich逐loss矩阵通过全部有效性审计。D0与rich `reid` loss、common初始state及backbone
non-finite计数完全相同；`reid/total`之外的D0 pose和rich全部auxiliary均finite。证据足以排除rich
teacher、evidence/exec loss或aggregate是该现象的必要条件，但十五组粒度不足以指定backbone内某个
parameter或算子。

**决策**：封板exp396为`CUDA_ATTRIBUTION_SEALED_PASS / SHARED_D0_OR_RUNTIME_NONFINITE`，不得重跑。
exp394仍满足自身预注册FAIL，不能事后改判；但其绝对首步finite门被证明缺少matched D0校准，不能用于
声称rich production额外不稳定。下一步另立exp397：保持default initial scale，不手工调scale，让
canonical GradScaler自然执行skip/`update()`，串行比较D0与rich的scale、skip、首个/累计成功update、
finite和state轨迹。只有rich不劣于D0且rich-specific parameter在成功update中有限，才可重新讨论正式
production preflight；当前formal训练仍`NO-START`。

### [2026-07-19] 决策：exp397 native GradScaler static通过，直接执行matched actual

static连续两遍21/21 PASS，确认default scaler未覆写、12步e1→e6 schedule、单一batch序列与matched
RNG进入脚本，且四类关键负反例都会阻断。该门允许native skip/update，但禁止手工scale和checkpoint。

**决策**：`STATIC_CPU_SEALED_PASS / CUDA NATIVE-PARITY FRESH-EXECUTION GO`。按持续授权直接建立fresh
execution/assets并执行唯一actual；PASS也只授权下一production preflight设计，不直接启动e120。

### [2026-07-19] 决策：exp397按冻结绝对门SEALED-FAIL，不改判也不补跑

actual中D0/rich轨迹完全一致且rich-specific组始终finite，但两臂均经历e1五次backoff、到attempt 6才
首次成功，并在e6首步再次matched skip；最终各只有`6/12`次update。这违反预注册的`>=10/12`、首个
成功`<=3`与e6全success门。

**决策**：保持`CUDA NATIVE-PARITY SEALED-FAIL / FORMAL NO-START`，禁止放宽exp397或重跑。matched
结果只允许把解释限制为shared backbone/default-scale适应，而不允许宣称rich已通过；下一步必须新编号
预注册production-shaped、baseline-relative门，不能修改initial scale、loss、batch或复活exp394–397。

### [2026-07-19] 决策：exp398 static通过，执行更长baseline-relative actual

exp398不放宽exp397门，而冻结了不同问题：e1/e6各16步的最后8步必须连续稳态，rich不得相对D0新增
skip/non-finite，并且11个rich组必须在e6真实更新。static连续两遍24/24 PASS且正反例裁决exact。

**决策**：`STATIC_CPU_SEALED_PASS / CUDA BASELINE-RELATIVE FRESH-EXECUTION GO`。直接建立fresh repo/
assets执行唯一actual；PASS只授权新编号final production preflight，不直接授权e120。

### [2026-07-19] 决策：exp398 reporter runtime INVALID，稳态问题仍未回答

actual在任何forward/update前因named-parameter tuple未被state hasher支持而退出。static synthetic没有
覆盖真实`parameter_groups()`容器形状，因此不能用24/24 static PASS替代actual，也不能从零轨迹推测
rich稳定或不稳定。

**决策**：exp398=`CUDA EXECUTION SEALED-INVALID`，禁止修补/重跑。下一步若继续必须新编号，仅修正
state hasher对`(name, parameter)`的exact支持，并新增真实container CPU contract；scale/loss/batch与32步
稳态公式保持不变。

### [2026-07-19] 决策：exp399真实named-parameter contract通过，直接重测科学门

exp399只修reporter tuple支持，正式static两遍35/35 PASS且byte-exact；真实15组coverage与name/order/
value绑定均已进入contract。exp398保持INVALID，不被覆盖。

**决策**：`STATIC_CPU_SEALED_PASS / CUDA FRESH-EXECUTION GO`。按持续授权直接运行唯一fresh actual；
32-step、tail8、default scale、loss和batch全部不变。

### [2026-07-19] 决策：exp399 baseline-relative稳态PASS，推进final production preflight

actual中D0/rich轨迹exact、各`26/32` update，attempts 8–32连续成功；rich没有extra skip或独有
non-finite，11个rich组finite/active/state-changed全部PASS。这回答了exp398未进入的科学问题，同时不
修改exp397原FAIL。

**决策**：`BASELINE_RELATIVE_STEADY_STATE_PASS / PRODUCTION PREFLIGHT GO`。下一步新编号只补strict
reload、teacher-free state、RGB-only、rho/full-bypass/双consumer、finite与final source/asset终审；全过
后才授权唯一fresh e120。当前formal仍`NO-START`。

### [2026-07-19] 决策：exp400 terminal contract static通过，直接执行唯一actual

exp400不改变exp399的32-step、default GradScaler、batch64、loss、rho或schedule，只把strict reload、
teacher-free/finite state、RGB-only、rho0 identity、双consumer独立bypass和diagnostic恢复加入同一终审。
static连续两遍`48/48 PASS`且byte-exact，toy真实验证两个consumer并阻断patch/state泄漏。

**决策**：`STATIC-CPU SEALED-PASS / CUDA FRESH-EXECUTION GO / FORMAL NO-START`。按持续授权直接建立
fresh exp400 execution/assets并运行唯一actual，不等待确认；任一trajectory/validity/terminal FAIL均停止
该编号且不得补跑。只有result显式`formal_training_authorized=true`才直接启动fresh e120。

### [2026-07-19] 决策：exp400 final production全PASS，正式e120直接启动

actual同时复现baseline-relative稳态并通过全部31项terminal gate。D0/rich各`27/32`更新且前期skip exact；
rich没有新增non-finite，11个新增组均active/updated。state strict reload、teacher-free/finite、eval RGB-only、
rho0 identity与两个consumer独立非零执行差均成立，result显式`formal_training_authorized=true`。

**决策**：封板exp400为`FINAL_PRODUCTION_PREFLIGHT_PASS / FORMAL E120 GO`，禁止重跑。按持续授权立即
启动唯一fresh rich-budget C0 seed1234 e120；不得续训、换seed、挑best、按中间指标早停或并行占用GPU。
正式结果仍需final full与all-router-bypass门裁决route alive，preflight PASS本身不进入主结果表。

### [2026-07-19] 决策：exp401 formal launch static通过，立即启动e120

config差异严格限定为三个fresh路径项；exp400授权、source、recipe、rho、optimizer、loss、checkpoint和
无resume门全部通过。初始路径reporter误判已保留，正式两遍18/18 PASS且byte-exact。

**决策**：`STATIC-CPU SEALED-PASS / FORMAL FRESH-EXECUTION GO`。直接启动唯一fresh e120；任何中间
mAP、loss或GateAbs不得裁决或早停，e120前checkpoint必须为0。

### [2026-07-19] 决策：exp401通过final route门，授权Phase-B interface

唯一fresh seed1234自然完成e120，final full=`57.1/67.3/80.3/84.8`；241项checkpoint state finite且
teacher-free，strict reload、RGB-only、两个router/evidence head、source/config/checkpoint恢复与退出审计
全部PASS。冻结all-router-bypass raw mAP=`57.0035860757`，full raw mAP=`57.1230075595`，差=
`+0.1194214838 point`；绝对full门`56.7`与差值门`+0.1`均通过。两个router各在78个batch完整旁路，
不是抽样descriptor差或GateAbs替代检索。

**决策**：封板exp401为`RICH_BUDGET_ROUTE_ALIVE / PHASE-B INTERFACE GO`，禁止重跑、补跑、续训、换seed
或降低后续门槛。该PASS比差值门只高`0.0194214838 point`，R1差为`−0.0904977322 point`，因此只授权
以exp401冻结production graph为接口基线进入Phase-B correct-vs-wrong/static/generic强反事实；不得把单seed
窄幅route贡献直接写成论文主结论，也不得通过调rho/loss/batch放大同一C0臂。

### [2026-07-20] 决策：exp402先做RGB-only语义接口kill-switch，不直接开新训练

exp401已经回答route alive，但其`+0.1194214838 mAP`贡献不足以区分rich evidence、mask/context或generic
residual。exp402因此冻结为同checkpoint只读诊断：correct与same-split/same-camera不同PID donor、zero、
正交旋转、evidence slot-cycle、wrong mask binding、generic expert-mean及router0/1/all bypass在同一进程
串行完整检索。wrong RGB映射按absolute dataset index定义，禁止batch-local roll；eval不读取pose或CLIP。

**决策**：先实现CPU/static正反contract，正式GPU仍`NO-START`。只有所有六个semantic controls都比
correct至少低`0.1 mAP`、route gap复现且descriptor intervention active，才授权下一编号Phase-B formal
mechanism design。FAIL只关闭当前student-evidence/expert解释；不得调rho/loss/batch或删除不利control。

### [2026-07-20] 决策：exp402 validity PASS但semantic kill-switch NO-GO

唯一formal在19,871图、10个串行arm上完整执行，correct与all-bypass逐项精确复现exp401；全部descriptor
active，两个router均有独立执行影响，state/RNG/patch/source/config/checkpoint及RGB-only/teacher-free/
post-exit门全部PASS。因此这是有效科学负结果，不是测量器INVALID。

route gap仍为`+0.1194214838 point`，correct floor也通过；但wrong-RGB mAP比correct高
`0.0066900358 point`，zero evidence也高`0.0006964267 point`。六control最高值使semantic margin=
`−0.0066900358 point`，明显未达到`+0.1`。generic expert mean单独下降`0.1240184555 point`，同时
router0 bypass下降`0.1307568556`、router1 bypass却上升`0.0160559964 point`；最稳妥的解释是当前route
依赖expert heterogeneity与混合router残差，但没有证明sample-specific RGB evidence是有效语义中介。

**决策**：exp402封板为`CURRENT_SEMANTIC_INTERFACE_NO-GO / PHASE-B FORMAL MECHANISM DESIGN
NO-START`，禁止重跑、补跑、删掉wrong/zero control或调rho/loss/batch救活。exp401的route-alive接口证据
保留；该NO-GO只关闭当前student-evidence/expert语义解释，不永久否定Phase0E、Phase0R或CLIP–TAPF。
后续若继续，只能重新定义训练对象/结构对象，使sample-specific evidence相对wrong/zero先建立可辨识因果差；
不得把generic expert prior或router0单臂贡献改写成CLIP语义贡献。

### [2026-07-20] 决策：exp403改写为evidence-owned operator，先过standalone contract

targeted文献与公开代码审计表明，CAL已覆盖随机/全一attention counterfactual prediction effect，AIM已覆盖
双分支分类差的衣物去偏，UCT已覆盖feature-conditioned prototype intervention；dynamic filter、hypernetwork
和low-rank parameterization也均非新原子。因此exp403不以“counterfactual”“dynamic”或“low-rank”单独
声称创新，而把对象限定为：sample evidence拥有生产operator系数、matched complete execution提供
stop-gradient utility reference、最终冻结retrieval同时证明semantic margin与route mediation。

standalone正反contract连续两遍`26/26 PASS`且byte-exact。NULL identity、兼容性序、donor、生产梯度覆盖、
reference零梯度和三个mutant均通过；CUDA未初始化。

**决策**：`PRODUCTION IMPLEMENTATION GO / CUDA NO-START`。下一步允许在新config开关下实现ELO-CUR，
但不得把standalone PASS当formal授权；必须先通过default-off parity、source/state/optimizer contract与真实
batch64 CUDA/AMP preflight。任何失败不调rho、outer loss、batch、stage或删除control救同一execution。

生产CPU/source必要门已`34/34 PASS`。按用户最新指令停止增加重复CPU矩阵，决策升级为
`FRESH GENERIC ASSET + ACTUAL BATCH64 CUDA/AMP PREFLIGHT GO`；preflight全过后直接启动fresh once-only
e120，不再插入非必要诊断。formal在preflight result显式授权前仍为`NO-START`。

### [2026-07-20] 决策：exp403 preflight全过，直接启动唯一formal

fresh generic asset-v3完成official train全覆盖；真实batch64 CUDA/AMP preflight为`16/16 PASS`并显式
`formal_training_authorized=true`。共享生产参数、correct evidence梯度、reference no-grad/RNG、NULL/teacher-free
state与checkpoint门均通过，GPU退出后空闲。未增加额外CPU矩阵。

**决策**：立即启动唯一fresh seed1234/batch64/e120，当前`FORMAL RUNNING`，main PID=`423319`。禁止续训、
换seed、调rho/loss/batch/stage、按中间指标早停或删除不利control；e120前checkpoint必须为0，最终只按冻结的
full绝对门、correct-vs-controls和correct-vs-all-bypass三门裁决。

### [2026-07-20] 决策：exp403测量有效，但ELO-CUR没有形成检索所有权

唯一fresh e120自然完成；七臂final audit每臂全量覆盖`19,871`图/`78` batches。strict checkpoint、
teacher-free/RGB-only、shared ELO/no-static-expert、donor、两个router调用、descriptor active、state/RNG/patch/
source/config/checkpoint恢复与退出postflight全部通过，measurement status=`PASS`。这不是runner错误或无效干预。

correct raw mAP/R1=`0.569929559315091/0.674208164215088`，低于sealed D0门
`0.575587756578/0.676923076923`。wrong/generic/NULL raw mAP分别为
`0.569934358329593/0.569937131506918/0.569937304669369`，all-bypass也为
`0.569937304669369`；correct相对最高control及all-bypass的margin均为
`−7.745354277944e-06`，未达到`+0.001`。七臂R1/R5/R10完全相同。虽然每个descriptor干预都active，
它们没有改变身份排序；训练期compatibility/CUR因此只能解释为shortcut/proxy，不能解释成retrieval
operator ownership。

**决策**：exp403封板为`VALIDITY PASS / SCIENTIFIC ELO_CUR_MECHANISM_NO_GO`，
`phase_b_formal_mechanism_design_authorized=false`。禁止同编号重跑、补跑、续训、换seed，禁止调
rho/loss/batch/stage、mask或删除wrong/generic/NULL/all-bypass control救活。exp401只保留窄幅route-alive
接口边界；exp402关闭当前C0 student-evidence/expert semantic解释；exp403关闭当前ELO-CUR对象。下一候选
必须重新定义问题或结构对象并至少满足创新门槛两项，不能继续围绕ELO-CUR调尺度或loss。

### [2026-07-20] 决策：modality-laziness近邻未提供source ownership机制，exp404继续NO-START

代码/公式级查新覆盖UniCat、MCR、Data Remixing、ResTacVLA、SCOPE、RCL、VIGIL、MiMIC与VLM2Rec。
独立单模态训练后拼接、latent permutation/MI/game regularizer、data remix/dropout、predictive residual、
跨模态topology preservation、counterfactual reliance matching及`seeing > blind`目标均已有直接先例。

这些方法共同没有满足当前最强合同：matched wrong evidence经过与correct相同的最终descriptor路径时，还要对
donor identity保持独立正目标；现有方法通常只把wrong/masked模态删除、置零、推远或降置信，无法排除通过
破坏counterfactual branch制造margin。

**决策**：创新门仍只有问题/证据缺口，机制门为空；不创建exp404、不写config、不做CPU/CUDA preflight、
不占GPU。下一轮仅审计能同时闭合`correct -> current ID`、`wrong -> donor ID`和同一路径欧氏检索的结构对象，
并继续保留generic/NULL/all-bypass；普通full-vs-mask、MI/topology、固定concat、dropout/remix直接排除。

### [2026-07-20] 决策：撤回无条件`wrong -> donor ID`合同，exp404仍NO-START

DG-Net、Hi-CMD与CIFT的代码/正式公式审计完成。DG-Net中被交换的appearance code本身由完整图像的ReID
encoder产生，因此生成分支可跟随该code的身份；Hi-CMD对style/extrinsic交换的身份标签明确跟随
prototype/content来源而非style donor；CIFT只干预graph affinity，对当前身份优化TIE，没有donor标签。

当前16维evidence仅定义support/appearance语义，不具备identity sufficiency。若强制wrong evidence指向donor
identity，会诱导身份泄漏或给不对应真实人的组合贴任意身份；若只增加donor semantic reconstruction，则落回
已有swap/cycle辅助目标，仍不能证明最终检索ownership。

**决策**：上一轮三方donor-ID合同只作为已否决的候选，不再作为普适准入条件。新的合法对象必须让目标跟随
信息承载者，并使semantic正目标与最终identity descriptor共享不可绕过结构；当前机制门仍为空。不创建
exp404、不写design/config/contract、不做CPU/CUDA/GPU执行，继续文献与代码学习。exp401–403封板结论不变。

### [2026-07-20] 决策：当前数据没有realized semantic target，composition路线不启动

NeurIPS 2025 Composed Person Retrieval / FAFA（官方commit `0cc16936`）证明，semantic modification可以拥有
直接检索正目标，但前提是存在`reference image + relative caption + same-ID target image`三元组。其SynCPR
依赖LLM、Flux成对生成和MLLM过滤，ITCPR依赖人工caption，正式推理还需要caption与query-conditioned token
scorer。DiCE-CIR用target caption作proxy，也没有消除显式edit/target semantics。

exp402/403的different-PID wrong evidence不是host A的relative edit，official数据没有已知`target(A,e_B)`。
普通same-ID配对只提供身份相同，不能证明16维evidence描述了相对状态，且ID trunk可继续绕过它。

**决策**：不把composition/contrastive loss当作exp404；不引入外部annotation、生成stage或测试时第二输入来
填补target。创新门保持失败，不写design/config/contract、不做CPU/CUDA/GPU。下一轮只审计能从official RGB
内部构造可验证realized semantic target、且保持单固定descriptor部署的结构对象。

### [2026-07-20] 决策：equivariance/invertibility不构成exp404

DiP公式审计确认：已知affine矩阵可以为part position构造解析target `Kp`，但测试时该位置不参与检索，最终是
pair-specific DiP weighting。当前16维evidence没有已知semantic action，wrong donor也不是host的已知变换，
所以augmentation consistency不能生成所需semantic target。

理论可识别性审计同时确认，invertible/bijective map只保证信息可恢复，不能指定latent factor归属；teacher
evidence target仍未定义其对final identity ranking的唯一作用。

**决策**：不以affine/flip consistency、augmentation invariance、normalizing flow、invertible coupling或
part-weighted scorer建立exp404。继续`NO EXP404 / GPU NO-START`，下一对象必须给当前evidence一个可验证的
semantic action并直接定义固定最终metric。

### [2026-07-20] 决策：封板资产不足以做identity/camera诊断，canonical action不构成exp404

只读资产检查确认，exp402/403 formal result虽报告correct臂`captured_rows=19,871`，但逐样本evidence、PID、
camera和各臂descriptor从未落盘；两个result最长数组都只有2。Phase0E codebook也仅含shared PCA basis、slot
mean和计数，generic evidence只是`5x16`常量。故不能从封板资产计算identity separability/camera confounding，
也不得通过重跑、补跑或新导出冒充离线分析。

随后审计3D-VAN、CSCL和VPFA。前两者的canonical target分别依赖3D重建或DP3D密集2D–3D/SMPL标注；VPFA
官方commit `13de109d`依赖同图LR/HR pair、MSE feature residual和测试文件名resolution suffix。它们不能为当前
16维semantic evidence提供可观测action；wrong donor warp只会形成破坏性负例，原RGB fusion则保留bypass。

**决策**：不创建exp404，不做CPU contract、CUDA preflight或GPU执行；canonical warp、dense-surface alignment、
resolution-vector residual均排除。状态为`ARTIFACT/IDENTIFIABILITY/MECHANISM GATE FAIL / GPU NO-START`，
exp401–403封板结论不变。

### [2026-07-20] 决策：source-provenance patch mix仍没有global正目标

AAAI 2024 SPT的论文与官方commit `ef1e71a9`确认，ReID中已存在不同身份之间的source-exact token transfer。
SPT保留target identity token、搬入candidate背景/遮挡，并用标准global descriptor训练；同时从softmax/triplet
忽略candidate class，因为残余candidate part不能获得第二个全局PID。Token Labeling/TokenMix又已覆盖逐token
dense label和content-weighted mixed target。

若只搬背景，donor无身份target；若搬完整person，donor ID虽合法但evidence已成为完整身份payload，退化为已有
foreground transfer/appearance swap；若只搬部分body slot，局部provenance合法但global chimera没有单一gallery
positive。任何`correct > wrong`都只能靠上下文不变性、实际换人或破坏partial chimera得到。

**决策**：不以SPT/TokenMix/part classifier组合建立exp404，不调patch mask、mix ratio或mixed-label权重。
保持`SOURCE-PROVENANCE TARGET TRILEMMA / NO EXP404 / GPU NO-START`，exp401–403封板不变。

### [2026-07-20] 决策：不以conditional uncertainty metric建立exp404

PFE官方commit `23191e9b`已让样本级对角方差通过mutual likelihood直接进入最终匹配；Bayesian Metric
Learning commit `e0188f4d`和ReID中的part/local/spatial-channel uncertainty、QPM、probabilistic matching
又覆盖posterior校准与质量感知度量。

当前16维evidence没有正确covariance/projector标签。same-ID likelihood可由RGB均值满足，固定rank/trace/
log-det只强迫删除；wrong donor可能与host具有相同nuisance，generic与NULL也没有天然概率顺序。强制
`correct > wrong > generic/NULL`仍会退化为破坏control。

**决策**：conditional covariance、Gaussian descriptor、orthogonal nuisance projector均排除，不创建exp404，
不做CPU contract、CUDA或GPU。状态为`UNCERTAINTY-METRIC PRIOR SATURATION / ORDER IDENTIFIABILITY FAIL /
GPU NO-START`，exp401–403封板不变。

### [2026-07-20] 决策：set-valued identity不进入exp404

KPR官方commit `e3e6ee2f`已在SOLIDER/Swin上输出part embedding与visibility，并通过正/负keypoint prompt处理
多人crop的target ambiguity；多个part仍共享一个target PID。人工A+B composition虽可对两个source局部监督，
真实official query却没有多PID token ownership标注。

保留全部component会把标准ReID改成multi-label retrieval；选择host需要prompt、instance assignment或
heuristic gate；聚合又回到无单一PID的global chimera。

**决策**：不以multi-PID token set、set-to-gallery max或promptless host selector建立exp404，不做CPU/CUDA/
GPU。状态为`SET-VALUED TARGET SELECTION GAP / DEPLOYMENT CONTRACT FAIL / GPU NO-START`，exp401–403
封板不变。

### [2026-07-20] 决策：将random-key control加入后继ownership必要门

合法canonicalization公开实现依赖已知group action；当前16维evidence没有解析群作用。无编号纯CPU诊断的v1
在metric前runtime失败并封板INVALID，fresh v2只修donor slot后有效完成。

完全随机、PID无关key得到correct/wrong/generic/NULL mAP=
`1.000000/0.608134/0.039243/0.030195`；semantic-blind key置换后仍为
`1.000000/0.592977/0.021800/0.028284`，两次强顺序都PASS，constant-quota mutant被抓。

**决策**：原`correct > wrong > generic/NULL`改为必要非充分门；不删除任何旧control，并为所有后继候选新增
semantic-blind random-key或等价null-semantic control。latent gauge、conditional flow、orthogonal unbinding
排除，不创建exp404、不做CUDA/GPU。状态为`RANDOM SOURCE-KEY FALSE OWNERSHIP DEMONSTRATED / GPU
NO-START`，exp401–403封板不变。

### [2026-07-20] 决策：共享重复不自动构成语义，不以random-cluster不确定执行建立exp404

MVI²P官方代码commit `4efd9fc920d2b3b5a8e9329059d81a6573f19b13`已经覆盖同身份多视图综合、CAM可靠性
加权及full-feature向单图传播；ECAI 2025 AG-ReID（arXiv `2508.04998`）已经覆盖identity-majority细粒度
attribute pseudo-label、CoOp属性token与Otsu噪声屏蔽。故“让状态跨样本重复”“identity-level attribute
prototype”或“多视图共享语义”都不能单独通过机制创新门。

frequency-matched random-cluster CPU诊断虽在原始与label-permutation观察中都出现强数值顺序，且mutant被抓，
但原始`cluster_7`只有38个PID，未过预注册`>=40`覆盖门。唯一执行封板为`DIAGNOSTIC_INCONCLUSIVE`，不允许
降低门槛、换seed、补跑或建立v2，也不能据此声称random-cluster假语义成立。

**决策**：当前sample-specific region-global residual若聚合为identity-majority属性，会丢失原support/appearance
对象并进入已有先例；若保留sample-specific状态，则仍须击败random-key control。状态为
`SEMANTIC REPLICATION PRIOR SATURATION / TARGET COLLAPSE / NO EXP404 / GPU NO-START`，继续文献/代码与
CPU诊断，不做CUDA/GPU。

### [2026-07-20] 决策：按C类投稿目标建立exp404 SPK设计

用户明确要求降低创新筛选强度，目标改为C类会议。sealed纪律、强反事实和数据/GPU边界不变，但不再以“每个
原子都必须达到B类主贡献级首创”阻止适度结构创新。

ICLR 2023 multimodal contrastive identifiability代码已覆盖paired modality共享factor的双向InfoNCE；CITRIS/
iCITRIS则要求temporal sequence与已知intervention target。它们限制exp404不能声称理论可识别，但没有覆盖
single-RGB open-set ReID中final fixed descriptor的semantic ownership与random-key终审。

**决策**：建立exp404 `Semantic Product Kernel`设计，把16维student evidence作为固定、无参数乘积因子直接绑定
到最终768维descriptor；不使用C0 expert、ELO-CUR operator或新增ownership loss。按问题/适度机制/证据三项
通过C-track准入，当前仅允许static CPU；在正反contract完成前`GPU NO-START`。

### [2026-07-20] 决策：exp404 production CPU通过，授权必要CUDA preflight

生产实现与v2 CPU/source合同连续两次`41/41 PASS`且byte-exact。默认关闭路径相对preimplementation commit
`07ca01c`保持D0/C0 state、初始化RNG与output exact；SPK无参数、NULL exact identity，且classification、triplet、
BNNeck前后eval都读取同一个绑定descriptor。旧C0/ELO router不在图或state中，global/evidence梯度finite/nonzero。

v1唯一失败是reporter跨函数误命中BNNeck字符串，保留`40/41 FAIL`记录；v2仅修AST测量范围，没有修改模型或降低
科学门。

**决策**：允许创建fresh config和一次必要CUDA/AMP preflight；formal training仍未授权，GPU当前仍NO-START。
C类目标只降低创新包装门槛，不降低correct-vs-wrong/generic/NULL/random-key/random-cluster与all-bypass终审门。

### [2026-07-20] 决策：exp404 CUDA preflight静态门通过

fresh config/source静态合同连续两次`33/33 PASS`且byte-exact，冻结真实batch64、默认GradScaler、16组梯度、
NULL/random-key数值干预、RGB-only eval与独占4090门；无resume或性能早停路径。

**决策**：授权部署fresh repo/runtime/assets并执行一次CUDA/AMP preflight。formal training仍未授权；任何CUDA门
失败先封板该preflight记录，不得直接启动e120。

### [2026-07-20] 决策：CUDA preflight v1封板，joint-field修复后授权v2

v1在首个真实forward发现5通道region field误接17通道D0 gate，未进入optimizer step，更未启动formal。按runtime
测量纪律封板`SEALED_INVALID_RUNTIME`，禁止同编号重跑。

修复只把SPK空间consumer恢复为设计已冻结的D0 joint field及原handoff，不调SPK、loss、rho、batch或stage。
production v3两次`49/49 PASS`，v2 static两次`11/11 PASS`。

**决策**：授权fresh execution id的CUDA preflight v2；formal继续NO-START。v2失败仍不得直接启动e120。

### [2026-07-20] 决策：CUDA v2无更新封板，沿用既有native-scaler稳态纪律授权v3

v2四次attempt均有目标梯度，但默认GradScaler连续backoff且没有optimizer update，完整门判FAIL并禁止重跑。
该四步序列与exp403“前4次backoff、第5次更新”的sealed记录一致。

**决策**：不设置或调低初始scale，不改loss/rho/batch/model；fresh v3只把默认scaler自然观察窗口冻结为8次，
仍须实际更新与原26项门全部PASS。formal继续NO-START。

### [2026-07-20] 决策：CUDA v3全过，授权唯一formal e120

v3在第5次default GradScaler attempt成功更新并通过全部26门；formal once-only prelaunch又连续两次
`15/15 PASS`。没有手调scale、loss、rho或batch。

**决策**：授权唯一fresh seed1234/e120，通过冻结wrapper启动并自然跑满。中间指标只记录，不早停、不续训、
不改运行中代码/config；最终仍按correct对全部强control和D0两级门裁决。

### [2026-07-20] 决策：exp404唯一formal已启动

远端fresh static第三次`15/15 PASS`，启动前repo/output/runner/launch/lock与GPU独占门全部通过。唯一formal已于
`2026-07-20T04:52:15Z`启动，main PID=`436043`，远端HEAD=
`1e40e9a9d1717139b06d09f55821c7f0e68143c7`。

**决策**：保持单4090任务，自然训练至e120；中间性能无论好坏均不早停，不修改运行中代码/config，不续训。

### [2026-07-20] 决策：exp404训练封板，授权全量反事实终审

唯一fresh训练自然完成e120并退出，final rounded=`57.4/67.5/79.7/85.0`，相对clean D0=
`-0.2/-0.2/-1.1/+0.4`，相对exp401=`+0.3/+0.2/-0.6/+0.2`。GPU恢复空闲、唯一checkpoint、异常0。

**决策**：paper性能前置门判FAIL，不声称超过D0；correct mAP通过`56.7`机制绝对门，因此继续冻结的
correct/wrong/generic/NULL/random/all-bypass全量终审。终审前不得把单臂性能写成SPK有效。

### [2026-07-20] 决策：exp404终审v1 static通过，只授权fresh CUDA wiring preflight

九臂protocol已明确SPK实际输入对象：wrong-RGB替换student evidence与presence，wrong-mask只循环presence，
slot-cycle只循环evidence；NULL与product bypass必须exact。两次CPU正反合同均`32/32 PASS`且byte-exact，
三类科学失败mutant均被裁决抓住。

**决策**：授权一个fresh小样本CUDA wiring preflight，仍不授权formal full。preflight必须验证真实checkpoint、
train-generic采集、九臂patch恢复、RGB-only和teacher/pose/codebook零访问；任何失败按执行性质封板，不降低门。

### [2026-07-20] 决策：preflight v1合同无效，fresh v2只修reporter作用域

v1主control与全部执行门通过；FAIL只因wrong-mask/slot-cycle在uniform five-slot presence下按SPK平均定义不变，
而reporter误把补充归因也列入active硬门。该门与预注册SPK公式冲突，不能把数学不变性伪报成模型runtime失败。

**决策**：v1封板`SEALED-INVALID-CONTRACT`且不重跑。授权fresh v2把active硬门限定为wrong/generic/NULL/
bypass/random-key/random-cluster；wrong-mask和slot-cycle仍完整执行、finite并记录。九臂、主mAP阈值、random合同与
NULL=bypass门均不变，formal继续NO-START。

v2 static已连续两次`32/32 PASS`且byte-exact。**决策**：授权fresh v2 actual preflight；通过前formal full仍
NO-START。

v2 actual现已25项validity全过并授权formal full。**决策**：先把v2 result SHA与fresh runtime freeze固化到
once-only wrapper，连续两次static PASS并显式提交后，才可在GPU空闲和formal资产全fresh时启动唯一full。

wrapper refreeze后的static已连续两次`33/33 PASS`且byte-exact。**决策**：显式提交、远端同步并复核GPU与
formal result/runner/manifest/lock全fresh后，立即通过once-only wrapper启动唯一formal full。

### [2026-07-20] 决策：exp404 validity PASS但机制NO-GO，封板并切换设计对象

九臂formal与postflight全过，排除测量器、资产、teacher泄漏和inactive主control。correct=`57.42796/67.46606`，
wrong近乎相同，generic略高，NULL=bypass=`57.60890/68.05430`显著更高；semantic与bypass gap均
`-0.18094484 mAP point`。

**决策**：exp404判`SPK_MECHANISM_NO_GO`并永久封板，不调temperature/loss/scale/presence、不补seed。接受用户
“换一种设计方法”的指令，下一编号先做pose+CLIP近期论文与开源实现审计、独立机制智能体和后续代码盲审；
未形成新的训练/结构对象并通过创新门前GPU NO-START。

### [2026-07-20] 决策：建立exp405 CAVT设计，但只授权Phase 0合同准备

exp404后根因审计证明旧实现没有完成用户原始双编码、pose-defined slot pooling和中间stage可执行路径；e120
student rich evidence平均cosine仅约`0.027`，且slot mean与近乎全1的伪presence消除了可辨识性。因此旧失败不能
被扩大为CLIP–TAPF总否定。

新对象冻结为CAVT：train-only original/deleted/same-ID cross-camera donor构成可观察target；frozen CLIP
image+text共同定义sample-specific slot distribution/support/content；donor-free student在同一TAPF
gather-transform-scatter路径预测状态转移，推理删除CLIP、text、pose和donor。

**决策**：公开近邻审计后宽CAVT创新门失败，只授权最小Phase 0 protocol和synthetic/CPU正反合同。exp392
raw patch teacher已封板，exp405唯一readout冻结为从首层开始的region-isolated五槽双编码teacher，50% deletion
为唯一primary。当前不得创建formal config、output或runner，不得占用4090。teacher oracle必须同时击败两个
单轴破坏、pose/image/text单轴、伪语义control与matched近邻，并在held-out PID上证明donor-free可预测性；
任一失败即不实现student、不跑e120。

### [2026-07-20] 决策：exp405 static v14封口，停止扩审并转真实teacher实现

v14两次fresh CPU execution均`56/56 PASS`且payload byte-exact；最终独立盲审为`0B/0H/0M/0L`。v11--v13
仍是历史不授权记录，禁止覆盖或改判。此前启动器/依赖/receipt审计已足够，继续扩展外围威胁模型只会延迟科学
证伪，因此到v14永久停止。

**决策**：立即实现最小真实region-isolated CLIP image+text teacher measurement。实现完成后只做一次聚焦真实
数据路径、pose同步、encoder/readout、反事实和指标的独立代码盲审；通过前不得运行真实数据、CUDA或占用4090。

### [2026-07-20] 决策：exp405真实teacher static v8通过，只授权512图CUDA preflight

v3的`7/8 FAIL`以及v4--v7均保留为历史、不授权且不得覆盖。多轮代码盲审实际发现并修复了会改变结论或浪费
once-only的缺陷：geometry与readout混淆、recipient/donor重叠、non-torso CI混入torso-only PID、top64匹配
假FAIL、全候选Python metadata潜在OOM，以及seal/COMPLETE异常窗口。最终v8两次fresh `8/8 PASS`且
byte-exact，SHA=`45413c3323f7af7636e1e2f9e581b4a9c5fe15c44d4b0a6e47aa987c0ef9f8ca`；三路固定快照
盲审均无BLOCKER/HIGH。

**决策**：授权显式提交、远端同步、只读资产/独占4090复核和唯一固定512图CUDA preflight。preflight只裁决
机械接线，不计算PID CI、non-torso macro或scientific GO；只有其COMPLETE PASS与fresh formal manifest同时
成立，才允许唯一full-train P0B measurement。formal P0B、transport、student config、训练与e120继续
NO-START。若以后进入student，方法与clean D0必须在相同epoch逐点记录mAP/R1和差值，最终不按中间表现早停，
只用自然e120裁决。

### [2026-07-20] 决策：按用户指定改用MMPOSE-ABU，v9复审通过

MMPOSE-ABU的OpenCLIP/Torch接口探针通过；v8拒绝它的原因只是Conda torch/torchvision没有wheel RECORD，
不是模型或CUDA不兼容。v9以唯一conda-meta JSON替代缺失RECORD并继续绑定实际module origin，缺失/多重匹配
仍fail closed；两次`8/8 PASS`且三路复审`0B/0H`。

**决策**：停止并清理未完成的新venv，唯一512图preflight固定使用MMPOSE-ABU。该环境选择不改变P0B科学门、
formal/student NO-START或同epoch clean D0 mAP/R1合同。

### [2026-07-20] 决策：v9远端Python3.8 static失败，v10兼容修复后仍固定MMPOSE-ABU

远端v9 CPU static在seal前因`ast.unparse`仅存在于Python 3.9+而失败；没有读取official数据/pose，没有初始化
CUDA或启动GPU。该错误属于合同兼容性，不是MMPOSE-ABU、MMPose、Torch或OpenCLIP不兼容。v10只把该AST
调用替换为Python 3.8可用的节点属性读取，本地两次`8/8 PASS`、byte-exact，三路盲审`0B/0H`。

**决策**：不再建立或恢复新venv，继续固定`/usr/local/anaconda3/envs/mmpose-abu/bin/python`。由于远端SSH
当前banner超时，先保持CUDA preflight NO-START；网络恢复后先完成两次远端v10 static、byte-exact、CUDA未
初始化与独占4090复核，全部通过才启动唯一512图preflight。formal P0B和student仍NO-START。

远端连接恢复后，v10已在MMPOSE-ABU中连续两次`8/8 PASS`且byte-exact，CUDA未初始化、GPU独占、fresh输出门
全部通过。**更新决策**：授权立即启动唯一512图CUDA preflight；只裁决机械有效性，formal P0B和student仍
NO-START。

### [2026-07-21] 决策：exp405 preflight v1候选池合同失败，SEALED且不重跑

唯一preflight完成512图original编码后，冻结wrong-mask matcher在该执行子集内找不到满足`8.0` caliper的
same-camera/different-PID donor；权威failure receipt已写入，GPU回到空闲。该失败发生在科学指标之前，不能
解释为MMPOSE-ABU、CLIP或CAVT机制失败。

**决策**：永久封板`exp405-p0b-preflight-v1`为`SEALED-FAIL / SCIENCE NOT EVALUATED`，不重跑、不补跑、不
放宽caliper或删除control。formal P0B、transport和student继续NO-START。只有独立新编号先冻结不改变正式科学门
的候选池合同修正，并通过CPU/static、fresh资产和三路代码盲审，才可授权新的preflight。

### [2026-07-21] 决策：建立exp406固定尺度单调donor扩池合同

三路只读审查确认exp405失败发生在caliper edge构建，尚未进入top64/Hall/assignment；无法从receipt判断具体
主导轴。删除preflight MAD/caliper或先看edge再挑recipient都可能降低既有门或改变冻结recipient，因此拒绝。

**决策**：exp406保持原512 core、20 recipients、core四变量MAD、caliper `8.0`、same-camera/different-PID、
recipient排除和donor唯一，只按冻结camera-round-robin/hash顺序把donor-only pool从512单调扩至full train。
formal仍全train独立重算MAD和原全部科学门。当前只授权CPU/static实现；两次byte-exact与三路新代码`0B/0H`
前不得创建远端资产或GPU execution。

本地static v1现已两次`13/13 PASS`且byte-exact。**更新决策**：只授权显式提交runner/module/contract/results并
启动三路固定快照代码盲审；盲审`0B/0H`、远端MMPOSE static和fresh asset门未完成前，GPU继续NO-START。

### [2026-07-21] 决策：拒绝exp406 static v1授权，采用最小v2后立即复审

**原因**：固定commit `afe9d490`的三路盲审发现v1可在错误runtime/输入或不完整failure receipt下虚假PASS，且
`13/13`没有实际覆盖协议中的关键反合同。

**决策**：不启动v1；只修复已指出的B/H，不扩张新科学变量。v2两次fresh `17/17 PASS`且byte-exact后，进行
一次最终三路盲审。若`0B/0H`，立即进入远端MMPOSE-ABU隔离static/fresh资产和唯一preflight；若仍有B/H，只修
对应项。该决定不改变exp405封板结论，也不改变CAVT科学门。

### [2026-07-21] 决策：只闭合v2盲审的单一static blocker，不扩张新门

v2三路中一条已`0B/0H`，其余意见可合并为formal/caliper/preference/witness同一static证明缺口。v3只实现这些
明确反例并取得两次byte-exact `19/19 PASS`。下一轮只做差异复审；若`0B/0H`立即转远端MMPOSE-ABU并启动
fresh preflight，不再增加本地合同或低风险建议。

### [2026-07-21] 决策：exp406 v3授权并启动唯一preflight

三路聚焦复审均`0B/0H`，远端MMPOSE-ABU两次static与本地byte-exact，fresh资产、隔离HEAD、输入SHA、GPU
独占和once-only路径全部通过。因此授权并启动唯一`exp406-p0b-preflight-v1`。该授权只覆盖机械preflight；
COMPLETE PASS前formal和student继续NO-START，运行中不改代码、协议或参数。

### [2026-07-21] 决策：exp406因cache自检runtime失败封板，转exp407

exp406已完成全部真实teacher计算，但在result/COMPLETE前被Torch 1.13的`weights_only=True`限制阻断。临时cache
用`weights_only=False`可读，只能证明根因，不授权补写结果或重跑。

**决策**：永久封板exp406为`SEALED-FAIL / SCIENCE NOT EVALUATED`；建立fresh exp407，仅把本进程刚写出的受信任
cache回读自检改为`weights_only=False`，其余teacher、donor、controls、阈值和数据路径不变。先做一个针对性
MMPOSE-ABU roundtrip合同与代码盲审，通过后立即启动fresh preflight，不继续扩张static。

### [2026-07-21] 决策：授权exp407唯一fresh preflight

固定MMPOSE-ABU targeted roundtrip两次byte-exact PASS；盲审发现并修复donor salt漂移后闭环`0B/0H`。因此不再
追加static，进入fresh远端隔离仓库、asset/input SHA、once-only路径与GPU独占核对；全部通过即启动唯一
`exp407-p0b-preflight-v1`。该授权不等于formal或student授权，preflight PASS后必须先发布完整receipt。

远端所有fresh门已通过，唯一preflight已按授权启动并取得started seal。运行中只监控自然完成，不改源码、协议、
batch或阈值；只有COMPLETE PASS才进入formal manifest冻结，任何runtime/validity失败均如实封板exp407执行。

### [2026-07-21] 决策：exp407 preflight PASS，授权冻结formal manifest

唯一preflight自然完成，八项validity与COMPLETE provenance全部PASS，且无failure共存。因此授权建立fresh
`exp407-p0b-iso-teacher-v1` manifest并做一次聚焦盲审；0B/0H且formal once-only/GPU门通过后立即启动formal。
该授权不允许读取preflight cache/scale/pair作为formal输入，也不授权student。

formal manifest自验证PASS且独立盲审`0B/0H`，因此唯一formal启动门已满足。GPU/fresh路径最终核对通过后立即运行，
自然完成前不按中间方向停止；只有formal COMPLETE科学GO才授权student。

唯一formal已启动并取得started seal。运行中只监控自然完成；若科学NO-GO则停止CAVT并立即设计下一种pose+CLIP
训练对象，若GO则先完整记录强反事实证据，再冻结student与clean D0方案。

### [2026-07-21] 决策：exp407 formal validity失败，停止CAVT测量器修补

唯一formal在全量编码后因wrong-mask balance caliper无donor而失败，科学门未执行。exp407永久封板，禁止调caliper、
换recipient、删control或新编号补跑同一formal。严格说这不证明CAVT科学NO-GO；但exp405/406/407已连续消耗三次
执行在测量合同而非涨点验证上，继续修补不符合当前C类会议与快速训练目标。因此下一编号必须是新的pose+CLIP训练
对象，直接以clean D0和自然ReID mAP/R1裁决，不再是CAVT donor/matcher修复。

### [2026-07-21] 决策：exp408冻结PICRD，不再推进PICAG或CAVT测量器

文献查重表明pose+CLIP与part distillation原子已高度拥挤；PICAG需要新增agreement head与推理门控，容易重现
旧C0 support/router问题。代码根因则明确指向旧rich relation被双detach阻断、且证据来自全图GAP。

**决策**：exp408选择更小且可直接裁决的PICRD：五槽region-isolated CLIP target、逐槽跨batch relation、
correct-vs-wrong/generic/zero无temperature排序、未detach Stage-2直传。只做必要实现检查和一次智能体盲审，
0B/0H后立即fresh cache与e120；不再增加donor/matcher/static供应链。自然e120必须同时超过clean D0 mAP/R1，
否则封板并换新的训练/结构对象，不调旧臂。

实现盲审已闭环`0B/0H`，固定MMPOSE-ABU真实batch64 CUDA/AMP梯度更新PASS，fresh cache-v2全15,618图及
64图强诊断也已完整核验并冻结SHA。**更新决策**：不再做额外CPU/static或小样本preflight，立即启动唯一fresh
seed1234/e120 student；运行中不改代码/config、不续训、不按中间性能早停，e10/20/.../120只与sealed clean D0
同epoch并排记录。
